"""Parameter recovery with the true graph fixed, in the DDE paper's format.

The DDE paper (Lee and Gu, 2025) reports the 2-layer simulation accuracy as
RMSE(Theta) over 100 replicates in Figure 4 and Tables S.9-S.11, with the
graph estimated by SAEM or PEM.  This script fits the same designs with the
true graph supplied as ``G_list`` (the confirmatory setting of the paper's
TIMSS analysis), so only the continuous parameters are estimated, and
reports RMSE(Theta) next to the paper's values.

Designs follow ``Sim_sid_{Bernoulli,Poisson,Normal}.m``: strictly
identifiable coefficients B_s ((J, K1, K2) = (18, 6, 2), (54, 18, 6),
(90, 30, 10); cross coefficients -4/3 as in the MATLAB code, the graph is the
same for either sign), prevalences 0.5 and unit Normal variances.

Arms: ``em`` is the exact EM on the fixed graph (the default for layers of
width <= 10, here (18, 6, 2); package defaults, including ``n_init``);
``saem`` is the fixed-width SAEM kernel with the graph held fixed (C = 1,
the package defaults otherwise).  ``--arms`` restricts the run.

RMSE(Theta) per replicate is the root mean square error over the
prevalences, both coefficient matrices including intercepts and, for Normal,
the variances (the MATLAB Normal script squares the summed variance error;
this script squares each entry).  The paper aggregates
``sqrt(mean(rmoutliers(mse)))``; both that and the plain
``sqrt(mean(mse))`` are reported.  No threshold or column alignment is
applied: the graph labels the latent columns.

    python scripts/evaluate_fixed_graph_paper.py --n-jobs 8

Outputs under ``Results/fixed_graph_paper_v1``: ``replicates.csv``,
``summary.csv``, ``paper_comparison.csv``, ``metadata.json``.  Re-running
resumes from completed replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.special import erfcinv, expit

DESIGNS = {"s18": (18, 6, 2), "s54": (54, 18, 6), "s90": (90, 30, 10)}
FAMILIES = ("bernoulli", "poisson", "normal")
N_GRID = (500, 1000, 2000, 4000, 8000)
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "Results" / "fixed_graph_paper_v1"

# Tables S.9-S.11 (B_s rows): RMSE(Theta) with the graph estimated.
PAPER_RMSE = {
    ("bernoulli", "s18", "SAEM"): (0.498, 0.431, 0.386, 0.359, 0.341),
    ("bernoulli", "s18", "PEM"): (0.404, 0.304, 0.231, 0.203, 0.184),
    ("bernoulli", "s54", "SAEM"): (0.392, 0.289, 0.231, 0.213, 0.208),
    ("bernoulli", "s90", "SAEM"): (0.356, 0.240, 0.204, 0.185, 0.177),
    ("poisson", "s18", "SAEM"): (0.281, 0.258, 0.245, 0.243, 0.242),
    ("poisson", "s18", "PEM"): (0.219, 0.158, 0.108, 0.080, 0.061),
    ("poisson", "s54", "SAEM"): (0.289, 0.168, 0.153, 0.147, 0.145),
    ("poisson", "s90", "SAEM"): (0.281, 0.195, 0.125, 0.117, 0.116),
    ("normal", "s18", "SAEM"): (0.189, 0.156, 0.147, 0.137, 0.133),
    ("normal", "s18", "PEM"): (0.170, 0.128, 0.080, 0.063, 0.054),
    ("normal", "s54", "SAEM"): (0.117, 0.090, 0.072, 0.063, 0.057),
    ("normal", "s90", "SAEM"): (0.108, 0.074, 0.054, 0.045, 0.041),
}

REPLICATE_FIELDS = [
    "family", "design", "arm", "N", "replicate", "seed", "status", "error",
    "mse_theta", "rmse_theta", "layer_rmse", "prop_rmse", "gamma_rmse",
    "latent_accuracy", "iterations", "seconds", "graph_algorithm", "best_start",
]
SUMMARY_FIELDS = [
    "family", "design", "arm", "N", "replicates", "failed",
    "rmse_theta_paper", "rmse_theta_all", "rmse_theta_median", "outliers_removed",
    "layer1_rmse", "layer2_rmse", "latent1_accuracy", "latent2_accuracy",
    "seconds_mean", "iterations_mean",
]


def bs_layer(K: int) -> np.ndarray:
    """``Sim_sid_*.m`` B_s coefficients (intercept first) of one 3K x K layer."""
    slopes = np.zeros((3 * K, K))
    half = K // 2
    for k in range(K):
        slopes[k, k] = 4.0
        slopes[k + K, k] = 4.0
    for k in range(half):
        slopes[k + 2 * K, k] = 4.0
        slopes[k + 2 * K + half, k] = -4.0 / 3.0
        slopes[k + 2 * K, k + half] = -4.0 / 3.0
        slopes[k + 2 * K + half, k + half] = 4.0
    intercept = np.concatenate([np.full(K, -2.0), np.full(K, -4.0), np.full(K, -2.0)])
    return np.column_stack([intercept, slopes])


def simulate(family: str, design: str, N: int, seed: int):
    J, K1, K2 = DESIGNS[design]
    rng = np.random.default_rng(seed)
    B_list = [bs_layer(K1), bs_layer(K2)]
    A2 = rng.binomial(1, 0.5, size=(N, K2))
    A1 = rng.binomial(1, expit(B_list[1][:, 0] + A2 @ B_list[1][:, 1:].T))
    eta = B_list[0][:, 0] + A1 @ B_list[0][:, 1:].T
    if family == "bernoulli":
        X = rng.binomial(1, expit(eta))
    elif family == "poisson":
        X = rng.poisson(np.exp(eta))
    else:
        X = eta + rng.normal(size=eta.shape)
    return X, B_list, [A1, A2]


def seed_for(family: str, design: str, N: int, replicate: int) -> int:
    return int(
        10_000_000 * (1 + FAMILIES.index(family))
        + 1_000_000 * (1 + list(DESIGNS).index(design))
        + 100 * N
        + replicate
    )


def run_one(task):
    family, design, arm, N, replicate = task
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from dde import fit

    seed = seed_for(family, design, N, replicate)
    row = {
        "family": family, "design": design, "arm": arm, "N": N,
        "replicate": replicate, "seed": seed, "status": "ok", "error": "",
    }
    try:
        X, B_true, A_true = simulate(family, design, N, seed)
        G = [B[:, 1:] != 0 for B in B_true]
        kwargs = dict(method="DDE", family=family, D=2, G_list=G, verbose=False)
        if arm == "saem":
            kwargs.update(C=1, graph_algorithm="saem")
        else:
            kwargs.update(graph_algorithm="em")
        np.random.seed(seed % (2**32))
        started = time.perf_counter()
        result = fit(X, **kwargs)
        seconds = time.perf_counter() - started
        prop_err = np.ravel(result.prop - 0.5)
        layer_err = [np.ravel(B - Bt) for B, Bt in zip(result.B_list, B_true)]
        pieces = [prop_err] + layer_err
        gamma_rmse = ""
        if family == "normal":
            gamma_err = np.ravel(result.gamma - 1.0)
            pieces.append(gamma_err)
            gamma_rmse = float(np.sqrt(np.mean(gamma_err**2)))
        mse = float(np.mean(np.concatenate(pieces) ** 2))
        row.update(
            mse_theta=mse,
            rmse_theta=math.sqrt(mse),
            layer_rmse=json.dumps([float(np.sqrt(np.mean(e**2))) for e in layer_err]),
            prop_rmse=float(np.sqrt(np.mean(prop_err**2))),
            gamma_rmse=gamma_rmse,
            latent_accuracy=json.dumps(
                [float(np.mean(a == t)) for a, t in zip(result.A_list, A_true)]
            ),
            iterations=int(result.diagnostics["iterations"]),
            seconds=seconds,
            graph_algorithm=result.diagnostics.get("graph_algorithm", ""),
            best_start=result.diagnostics.get("best_start", ""),
        )
    except Exception as exc:  # pragma: no cover - recorded per replicate
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def matlab_rmoutliers(values: np.ndarray):
    """MATLAB ``rmoutliers`` default: drop values beyond 3 scaled MADs of the median."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    scale = -1.0 / (math.sqrt(2.0) * erfcinv(1.5)) * np.median(np.abs(values - median))
    keep = np.abs(values - median) <= 3.0 * scale
    return values[keep], int(np.sum(~keep))


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["family"], row["design"], row["arm"], int(row["N"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (family, design, arm, N), group in sorted(
        groups.items(), key=lambda item: (FAMILIES.index(item[0][0]), item[0][1], item[0][2], item[0][3])
    ):
        ok = [row for row in group if row["status"] == "ok"]
        if not ok:
            continue
        mse = np.array([float(row["mse_theta"]) for row in ok])
        kept, removed = matlab_rmoutliers(mse)
        layers = np.array([json.loads(row["layer_rmse"]) for row in ok])
        latent = np.array([json.loads(row["latent_accuracy"]) for row in ok])
        summary.append({
            "family": family, "design": design, "arm": arm, "N": N,
            "replicates": len(ok), "failed": len(group) - len(ok),
            "rmse_theta_paper": math.sqrt(float(np.mean(kept))),
            "rmse_theta_all": math.sqrt(float(np.mean(mse))),
            "rmse_theta_median": math.sqrt(float(np.median(mse))),
            "outliers_removed": removed,
            "layer1_rmse": float(np.mean(layers[:, 0])),
            "layer2_rmse": float(np.mean(layers[:, 1])),
            "latent1_accuracy": float(np.mean(latent[:, 0])),
            "latent2_accuracy": float(np.mean(latent[:, 1])),
            "seconds_mean": float(np.mean([float(row["seconds"]) for row in ok])),
            "iterations_mean": float(np.mean([float(row["iterations"]) for row in ok])),
        })
    return summary


def paper_comparison(summary):
    rows = []
    for item in summary:
        for algorithm in ("SAEM", "PEM"):
            paper = PAPER_RMSE.get((item["family"], item["design"], algorithm))
            if paper is None:
                continue
            value = paper[N_GRID.index(item["N"])]
            rows.append({
                "family": item["family"], "design": item["design"], "arm": item["arm"],
                "N": item["N"], "paper_algorithm": algorithm,
                "paper_rmse_theta": value,
                "fixed_graph_rmse_theta": item["rmse_theta_paper"],
                "ratio": item["rmse_theta_paper"] / value,
            })
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--designs", nargs="+", default=list(DESIGNS))
    parser.add_argument("--N", nargs="+", type=int, default=list(N_GRID))
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--arms", nargs="+", default=["em", "saem"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    replicate_path = args.out / "replicates.csv"

    rows = []
    if replicate_path.exists() and not args.no_resume:
        with open(replicate_path, newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["status"] == "ok"]
    done = {(r["family"], r["design"], r["arm"], int(r["N"]), int(r["replicate"])) for r in rows}
    tasks = []
    for family in args.families:
        for design in args.designs:
            arms = ("em", "saem") if design == "s18" else ("saem",)
            for arm in (arm for arm in arms if arm in args.arms):
                for N in args.N:
                    for replicate in range(args.replicates):
                        key = (family, design, arm, N, replicate)
                        if key not in done:
                            tasks.append(key)
    # Slow cells first so the pool stays busy at the end.
    tasks.sort(key=lambda t: (-DESIGNS[t[1]][0] * t[3], t))
    metadata = {
        "started": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "designs": DESIGNS, "families": args.families, "N": args.N,
        "replicates": args.replicates, "arms": {"s18": ["em", "saem"], "other": ["saem"]},
        "truth": "Sim_sid_*.m B_s (cross coefficients -4/3), prop 0.5, Normal variance 1",
        "metric": "RMSE(Theta) over prop, B1, B2 (with intercepts) and Normal variances; "
                  "paper aggregate sqrt(mean(rmoutliers(mse)))",
        "paper_tables": "Tables S.9-S.11 (graph estimated)",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"{len(tasks)} fits to run ({len(rows)} already done)", flush=True)

    def flush():
        write_csv(replicate_path, rows, REPLICATE_FIELDS)
        summary = summarize(rows)
        write_csv(args.out / "summary.csv", summary, SUMMARY_FIELDS)
        write_csv(
            args.out / "paper_comparison.csv", paper_comparison(summary),
            ["family", "design", "arm", "N", "paper_algorithm", "paper_rmse_theta",
             "fixed_graph_rmse_theta", "ratio"],
        )

    completed = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.n_jobs), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                flush()
                print(f"{completed}/{len(tasks)} done", flush=True)
    flush()


if __name__ == "__main__":
    main()
