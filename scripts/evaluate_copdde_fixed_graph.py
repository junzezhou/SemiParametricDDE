"""CopDDE parameter recovery with the true graph fixed, in the CopDDE paper's format.

The CopDDE paper reports weight-matrix accuracy as the average entry-wise MSE
of B^(1) and B^(2) (B^(3) for D = 3) across sample sizes (Figures 4, C.3,
C.9 and C.10; computed in ``Sim_Overspecified.m`` and ``Deeper_Sims.m``),
with the graph estimated (RL CSP, RL Gibbs, RL SAEM).  This script fits the
paper's designs with the true graph supplied as ``G_list`` and reports the
same metric next to the graph-estimated RL CSP fits of the MATLAB baseline
(``Sim_CopDDE_baseline.m``: the unmodified paper code, 30 replicates,
N = 500-8000, which reproduces the paper's Tables 1, S.1 and S.3).

Designs: the truth of the baseline's ``settings/<scenario>.mat``:
``d2_j50`` (J = 50, K = (10, 3); Table S.1), ``d2_j100`` (J = 100,
K = (10, 3); Table 1 and Figure 4) and ``d3_j108`` (J = 108, K = (18, 6, 2);
Table S.3 and Figure C.10).  Data are generated in Python with the paper's
construction (``generate_X_Cop_marg_emp_D``) and the seeds of
``scripts/copdde_python_sim.py``; the fit is ``dde.fit(method="CopDDE",
G_list=...)`` with the package defaults.

Metric, as in the MATLAB scripts: B^(1) rows are standardized by the fitted
latent Gaussian ``Z``'s empirical means and standard deviations
(``rescale_B1`` with ``emp = true``), estimates are padded with zeros to the
paper's maximal widths K_max (the truth likewise), and the squared entry
errors are averaged per layer and then over replicates.  The D = 3 truth
B^(1) is standardized by the replicate's true ``Z`` (``Deeper_Sims.m``).
The MSE on the true-width block (the scripts' "active columns" evaluation)
is reported too.

Alignment: the paper aligns columns to the truth by a Hungarian assignment
(``format_estimates.m``) because CSP labels are arbitrary.  A fixed graph
gives the labels, but a binary graph leaves each latent variable's 0/1
coding free and makes latent variables with identical graph columns (and
rows above) interchangeable: the top-layer variables 1 and 2 of both D = 2
designs.  Estimates are mapped to the truth by these exact
reparametrizations: within each group of interchangeable variables the
Hungarian assignment on |sum of the estimates over the truth's positive
entries| (the paper's criterion), then a variable's coding is flipped when
that sum is negative.

    python scripts/evaluate_copdde_fixed_graph.py --n-jobs 10

Outputs under ``Results/copdde_fixed_graph_v1``: ``rows.jsonl`` (one fit per
line), ``fits/*.npz`` (aligned estimates), ``summary.csv``,
``metadata.json``.  Re-running resumes from completed fits.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "DDE-Copula-m" / "Results" / "copdde_baseline" / "settings"
BASELINE = ROOT / "DDE-Copula-m" / "Results" / "copdde_baseline"
DEFAULT_OUT = ROOT / "Results" / "copdde_fixed_graph_v1"
SCENARIOS = ("d2_j50", "d2_j100", "d3_j108")
N_GRID = (500, 1000, 2000, 4000, 8000)
# Graph-estimated comparison: the paper's two-layer routine for D = 2, the D-layer one for D = 3.
BASELINE_ALGO = {"d2_j50": "csp_2layer", "d2_j100": "csp_2layer", "d3_j108": "csp_D"}


def seed_for(N: int, rep: int) -> int:
    return 7_000_000 + 1000 * int(N) + int(rep)


def interchangeable_groups(G_list, d):
    """Latent variables of layer ``d`` whose graph column and row above coincide."""
    above = G_list[d + 1] if d + 1 < len(G_list) else None
    groups = {}
    for k in range(G_list[d].shape[1]):
        key = (G_list[d][:, k].tobytes(), None if above is None else above[k].tobytes())
        groups.setdefault(key, []).append(k)
    return list(groups.values())


def align_to_truth(B_list, prop, T_list, G_list):
    """Map a fixed-graph estimate to the truth's coding and interchangeable labels.

    Returns the aligned ``(B_list, prop, A_maps)``; ``A_maps[d]`` is
    ``(permutation, flipped)`` for layer ``d``.  Flipping latent ``k`` of
    layer ``d`` (``a -> 1 - a``) moves its slopes into the intercepts and
    negates them, and negates its own logistic row in ``B_list[d + 1]`` (or
    replaces its prevalence ``p`` by ``1 - p`` at the top).
    """
    B = [np.asarray(b, dtype=float).copy() for b in B_list]
    prop = np.asarray(prop, dtype=float).copy()
    D = len(B)
    maps = []
    for d in range(D):
        K = B[d].shape[1] - 1
        positive = [T_list[d][:, k + 1] > 0 for k in range(K)]
        perm = np.arange(K)
        for group in interchangeable_groups(G_list, d):
            if len(group) > 1:
                cost = np.array([[-abs(B[d][positive[k], c + 1].sum()) for c in group] for k in group])
                _, chosen = linear_sum_assignment(cost)
                perm[group] = np.asarray(group)[chosen]
        B[d] = np.column_stack((B[d][:, 0], B[d][:, 1:][:, perm]))
        if d + 1 < D:
            B[d + 1] = B[d + 1][perm, :]
        else:
            prop = prop[perm]
        flipped = np.array([B[d][positive[k], k + 1].sum() < 0 for k in range(K)])
        for k in np.flatnonzero(flipped):
            B[d][:, 0] += B[d][:, k + 1]
            B[d][:, k + 1] *= -1.0
            if d + 1 < D:
                B[d + 1][k, :] *= -1.0
            else:
                prop[k] = 1.0 - prop[k]
        maps.append((perm.tolist(), np.flatnonzero(flipped).tolist()))
    return B, prop, maps


def padded(B, shape):
    out = np.zeros(shape)
    out[: B.shape[0], : B.shape[1]] = B
    return out


def run_one(task):
    scenario, N, rep, out_dir = task
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copdde_paper_metrics import load_truth
    from dde import fit
    from dde.copula import rescale_B1_D
    from dde.simulation.generate_copula_D import generate_X_Cop_marg_emp_D

    seed = seed_for(N, rep)
    row = {"scenario": scenario, "N": int(N), "rep": int(rep), "seed": seed, "status": "ok", "error": ""}
    try:
        truth = load_truth(SETTINGS / f"{scenario}.mat")
        D = truth["D"]
        gen = generate_X_Cop_marg_emp_D(
            N, truth["prop_true"], truth["lambdas"], truth["gen_gamma"], truth["gen_B"], random_state=seed
        )
        X = gen["X"]
        J = X.shape[1]
        G_list = [B[:, 1:] != 0 for B in truth["gen_B"]]
        if D == 2:
            T_list = [truth["gen_B"][0], truth["gen_B"][1]]
        else:
            B1_scaled = rescale_B1_D(
                truth["prop_true"], truth["gen_B"], truth["gen_gamma"], emp=True, Z=gen["Z"]
            )[0][0]
            T_list = [B1_scaled] + [B.copy() for B in truth["gen_B"][1:]]
        K_max = truth["K_max"]
        shapes = [(J, K_max[0] + 1)] + [(K_max[d - 1], K_max[d] + 1) for d in range(1, D)]

        np.random.seed(seed % (2**32))
        started = time.perf_counter()
        result = fit(X, method="CopDDE", G_list=[G.astype(int) for G in G_list],
                     random_state=seed, verbose=False)
        seconds = time.perf_counter() - started

        B_aligned, prop_aligned, maps = align_to_truth(result.B_list, result.prop, T_list, G_list)
        B_scaled, gamma_scaled, _, _ = rescale_B1_D(
            prop_aligned, B_aligned, result.gamma, emp=True, Z=result.transformed
        )
        mse_padded = [
            float(np.mean((padded(B_scaled[d], shapes[d]) - padded(T_list[d], shapes[d])) ** 2))
            for d in range(D)
        ]
        mse_active = [float(np.mean((B_scaled[d] - T_list[d]) ** 2)) for d in range(D)]
        graph_start = result.diagnostics.get("graph_start") or {}
        logliks = np.asarray(graph_start.get("deep_em_start_logliks", []), dtype=float)
        row.update(
            mse_padded=mse_padded,
            mse_active=mse_active,
            prop_abs_error=float(np.max(np.abs(prop_aligned - truth["prop_true"]))),
            permutations=[m[0] for m in maps],
            flipped=[m[1] for m in maps],
            iterations=int(result.diagnostics["iterations"]),
            seconds=seconds,
            deep_em_best_start=graph_start.get("deep_em_best_start"),
            deep_em_starts_at_best=int(np.sum(logliks > logliks.max() - 0.5)) if logliks.size else None,
            deep_em_starts=int(logliks.size),
        )
        np.savez_compressed(
            Path(out_dir) / "fits" / f"{scenario}_N{N}_r{rep}.npz",
            **{f"B{d + 1}": B_scaled[d] for d in range(D)},
            prop=prop_aligned, gamma=gamma_scaled,
        )
    except Exception as exc:  # pragma: no cover - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def baseline_means(scenario):
    """Mean per-layer MSE of the MATLAB RL CSP baseline (graph estimated), by N."""
    path = BASELINE / f"summary_{scenario}_{BASELINE_ALGO[scenario]}.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "ok"]
    for N in sorted({int(float(r["N"])) for r in rows}):
        cell = [r for r in rows if int(float(r["N"])) == N]
        out[N] = {
            "n": len(cell),
            "mse": [float(np.mean([float(r[f"mseB_{d}"]) for r in cell])) for d in (1, 2, 3)],
        }
    return out


def summarize(rows):
    summary = []
    for scenario in SCENARIOS:
        base = baseline_means(scenario)
        for N in N_GRID:
            cell = [r for r in rows if r["scenario"] == scenario and r["N"] == N and r["status"] == "ok"]
            if not cell:
                continue
            D = len(cell[0]["mse_padded"])
            entry = {"scenario": scenario, "N": N, "replicates": len(cell),
                     "median_seconds": float(np.median([r["seconds"] for r in cell]))}
            for d in range(D):
                padded_values = np.array([r["mse_padded"][d] for r in cell])
                entry[f"mse_B{d + 1}"] = float(np.mean(padded_values))
                entry[f"median_mse_B{d + 1}"] = float(np.median(padded_values))
                entry[f"active_mse_B{d + 1}"] = float(np.mean([r["mse_active"][d] for r in cell]))
                if N in base:
                    entry[f"csp_mse_B{d + 1}"] = base[N]["mse"][d]
                    entry["csp_replicates"] = base[N]["n"]
            summary.append(entry)
    return summary


SUMMARY_FIELDS = [
    "scenario", "N", "replicates",
    "mse_B1", "mse_B2", "mse_B3", "csp_mse_B1", "csp_mse_B2", "csp_mse_B3", "csp_replicates",
    "median_mse_B1", "median_mse_B2", "median_mse_B3",
    "active_mse_B1", "active_mse_B2", "active_mse_B3", "median_seconds",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--N", nargs="+", type=int, default=list(N_GRID))
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    (args.out / "fits").mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rows.jsonl"
    rows = []
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["status"] == "ok"]
    done = {(r["scenario"], r["N"], r["rep"]) for r in rows}
    tasks = [(s, N, rep, str(args.out)) for s in args.scenarios for N in args.N
             for rep in range(1, args.reps + 1) if (s, N, rep) not in done]
    # Slow cells first so the pool stays busy at the end.
    tasks.sort(key=lambda t: (-t[1] * (3 if t[0] == "d3_j108" else 1), t))
    metadata = {
        "started": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "scenarios": args.scenarios, "N": args.N, "reps": args.reps,
        "fit": "dde.fit(method='CopDDE', G_list=true graph) with package defaults",
        "metric": "per-layer entry-wise MSE, B1 standardized by the fitted Z (rescale_B1, emp=true), "
                  "padded to K_max; mean over replicates (paper Figures 4, C.3, C.9, C.10)",
        "comparison": "MATLAB RL CSP baseline (graph estimated), "
                      + ", ".join(f"{s}: {a}" for s, a in BASELINE_ALGO.items()),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"{len(tasks)} fits to run ({len(rows)} already done)", flush=True)

    def flush():
        summary = summarize(rows)
        with open(args.out / "summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)

    completed = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.n_jobs), mp_context=multiprocessing.get_context("spawn")
    ) as executor, open(rows_path, "a") as handle:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if row["status"] == "ok":
                rows.append(row)
            else:
                print("FAILED", row["scenario"], row["N"], row["rep"], row["error"], flush=True)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                flush()
                print(f"{completed}/{len(tasks)} done", flush=True)
    flush()


if __name__ == "__main__":
    main()
