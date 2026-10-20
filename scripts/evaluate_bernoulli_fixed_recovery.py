"""Paper-parameter recovery experiment for fixed-width Bernoulli DDE.

The D=2 scenario exactly uses the sparse identifiable coefficients ``B_s``
from equation (S.36) of the DDE paper. The D=3, D=4, and D=5 scenarios are
explicitly labelled recursive extensions of the same layerwise construction;
the paper does not report these deeper Bernoulli experiments.

The Bernoulli recovery workflow uses fixed ``tau=0.3`` for both TLP fitting
and graph scoring.  This follows the paired tau-attribution experiment, which
showed that the decreasing paper schedule overselects weak false edges in the
current fixed-width kernel.

Defaults run the requested full experiment:

    python scripts/evaluate_bernoulli_fixed_recovery.py --n-jobs 8

Outputs are checkpointed under ``results/bernoulli_fixed_tau03_rmse_tol005``:

* ``replicates.csv``: one row per fit;
* ``summary.csv``: one row per scenario and sample size;
* ``paper_comparison.csv``: D=2 Python-minus-paper comparisons;
* ``metadata.json``: settings and metric definitions.

Re-running the same command resumes from completed replicate rows. Pass
``--no-resume`` to replace an existing run.

The fixed-kernel controls ``--sa-burn-in``, ``--binarize`` and
``--warm-start`` override the package defaults for ablations (``sa_burn_in``,
``fixed_init_binarize`` and ``fixed_warm_start`` of :func:`dde.fit`); the
values used are recorded per replicate.  The deep-layer check behind the
Bernoulli defaults is ``Results/bernoulli_fixed_recon/REPORT.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from scipy.special import expit
from threadpoolctl import threadpool_limits

from dde import fit
from dde.algorithms.dde import DEFAULT_PARAMETRIC_CONVERGENCE_TOL


SCENARIOS = {
    "d2": {
        "label": "D2_paper_Bs_18_6_2",
        "D": 2,
        "J": 18,
        "K_list": (6, 2),
        "paper_reported": True,
    },
    "d3": {
        "label": "D3_Bs_extension_54_18_6_2",
        "D": 3,
        "J": 54,
        "K_list": (18, 6, 2),
        "paper_reported": False,
    },
    "d4": {
        "label": "D4_Bs_extension_162_54_18_6_2",
        "D": 4,
        "J": 162,
        "K_list": (54, 18, 6, 2),
        "paper_reported": False,
    },
    "d5": {
        "label": "D5_Bs_extension_486_162_54_18_6_2",
        "D": 5,
        "J": 486,
        "K_list": (162, 54, 18, 6, 2),
        "paper_reported": False,
    },
}

PAPER_D2_SAEM = {
    500: {"graph_accuracy": 0.899, "rmse_theta": 0.498},
    1000: {"graph_accuracy": 0.931, "rmse_theta": 0.431},
    2000: {"graph_accuracy": 0.956, "rmse_theta": 0.386},
    4000: {"graph_accuracy": 0.974, "rmse_theta": 0.359},
    8000: {"graph_accuracy": 0.985, "rmse_theta": 0.341},
}

RAW_FIELDS = [
    "scenario", "D", "J", "K_list", "N", "replicate", "data_seed",
    "fit_seed", "penalty", "tau", "C", "tol", "max_iter", "sa_burn_in",
    "fixed_init_binarize", "fixed_warm_start", "status",
    "error", "finite", "iterations", "hit_iteration_limit", "seconds",
    "graph_accuracy", "sensitivity", "specificity", "false_positive_rate",
    "false_negative_rate", "true_positive", "true_negative", "false_positive",
    "false_negative", "parameter_mse", "parameter_rmse", "latent_accuracy",
    "layer_graph_accuracy", "layer_parameter_rmse", "layer_latent_accuracy",
    "alignment_permutations",
]

SUMMARY_FIELDS = [
    "scenario", "D", "J", "K_list", "N", "requested_replicates",
    "successful_replicates", "failed_replicates", "graph_accuracy_mean",
    "graph_accuracy_sd", "graph_accuracy_se", "sensitivity_mean",
    "specificity_mean", "parameter_rmse_paper", "parameter_rmse_all",
    "removed_mse_outliers", "latent_accuracy_mean",
    "layer_graph_accuracy_mean", "layer_parameter_rmse_mean",
    "layer_latent_accuracy_mean",
    "iterations_mean", "iteration_limit_fraction", "seconds_mean",
    "seconds_total",
]

COMPARISON_FIELDS = [
    "scenario", "N", "python_graph_accuracy", "paper_graph_accuracy",
    "graph_accuracy_delta", "python_rmse_theta", "paper_rmse_theta",
    "rmse_theta_delta", "paper_value_available",
]


def _paper_sparse_coefficients(n_parent: int) -> np.ndarray:
    """Return equation (S.36) coefficients for one 3K-by-K layer."""
    n_parent = int(n_parent)
    if n_parent < 1 or n_parent % 2:
        raise ValueError("The paper B_s construction requires a positive even width.")
    n_child = 3 * n_parent
    slopes = np.zeros((n_child, n_parent), dtype=float)
    slopes[:n_parent] = 4.0 * np.eye(n_parent)
    slopes[n_parent : 2 * n_parent] = 4.0 * np.eye(n_parent)

    third = 4.0 * np.eye(n_parent)
    half = n_parent // 2
    for row in range(n_parent):
        third[row, (row + half) % n_parent] = 4.0 / 3.0
    slopes[2 * n_parent :] = third

    intercept = np.concatenate(
        [
            np.full(n_parent, -2.0),
            np.full(n_parent, -4.0),
            np.full(n_parent, -2.0),
        ]
    )
    return np.column_stack([intercept, slopes])


def build_scenario(scenario_key: str) -> dict:
    """Construct one exact B_s hierarchy and its graph matrices."""
    spec = SCENARIOS[scenario_key]
    widths = spec["K_list"]
    child_widths = (spec["J"],) + widths[:-1]
    if any(child != 3 * parent for child, parent in zip(child_widths, widths)):
        raise ValueError("Every B_s layer must have exactly three children per parent.")
    B_list = [_paper_sparse_coefficients(width) for width in widths]
    return {
        **spec,
        "prop": np.full(widths[-1], 0.5, dtype=float),
        "B_list": B_list,
        "G_list": [B[:, 1:] != 0.0 for B in B_list],
    }


def simulate_bernoulli_dde(scenario: dict, N: int, seed: int):
    """Simulate X and all latent layers from the specified hierarchy."""
    rng = np.random.default_rng(int(seed))
    widths = scenario["K_list"]
    B_list = scenario["B_list"]
    A_list = [None] * scenario["D"]
    A_list[-1] = rng.binomial(
        1, scenario["prop"], size=(N, widths[-1])
    ).astype(np.uint8)
    for layer in range(scenario["D"] - 2, -1, -1):
        eta = (
            B_list[layer + 1][:, 0][None, :]
            + A_list[layer + 1] @ B_list[layer + 1][:, 1:].T
        )
        A_list[layer] = rng.binomial(1, expit(eta)).astype(np.uint8)
    observed_eta = (
        B_list[0][:, 0][None, :]
        + A_list[0] @ B_list[0][:, 1:].T
    )
    X = rng.binomial(1, expit(observed_eta)).astype(np.uint8)
    return X, A_list


def _threshold_all_coefficients(B_list, tau: float) -> list[np.ndarray]:
    """Match the MATLAB reporting script, which thresholds complete B matrices."""
    thresholded = []
    for coefficient in B_list:
        value = np.asarray(coefficient, dtype=float).copy()
        value[np.abs(value) < tau] = 0.0
        thresholded.append(value)
    return thresholded


def align_hierarchy(prop, B_list, A_list, B_true):
    """Apply the paper's bottom-up squared-L2 Hungarian alignment."""
    prop_aligned = np.asarray(prop, dtype=float).copy()
    B_aligned = [np.asarray(B, dtype=float).copy() for B in B_list]
    A_aligned = [np.asarray(A, dtype=np.uint8).copy() for A in A_list]
    permutations = []

    for layer, truth in enumerate(B_true):
        estimated_columns = B_aligned[layer][:, 1:].T
        true_columns = np.asarray(truth, dtype=float)[:, 1:].T
        cost = np.sum(
            (true_columns[:, None, :] - estimated_columns[None, :, :]) ** 2,
            axis=2,
        )
        true_indices, estimate_indices = linear_sum_assignment(cost)
        order = estimate_indices[np.argsort(true_indices)]
        permutations.append(order.astype(int).tolist())

        B_aligned[layer][:, 1:] = B_aligned[layer][:, 1:][:, order]
        A_aligned[layer] = A_aligned[layer][:, order]
        if layer + 1 < len(B_aligned):
            B_aligned[layer + 1] = B_aligned[layer + 1][order, :]
        else:
            prop_aligned = prop_aligned[order]

    return prop_aligned, B_aligned, A_aligned, permutations


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def calculate_metrics(prop, B_list, A_list, scenario, A_true, tau):
    """Calculate paper metrics after thresholding and bottom-up alignment."""
    B_reported = _threshold_all_coefficients(B_list, tau)
    prop, B_reported, A_list, permutations = align_hierarchy(
        prop, B_reported, A_list, scenario["B_list"]
    )

    estimated_graph = np.concatenate(
        [(B[:, 1:] != 0.0).ravel() for B in B_reported]
    )
    true_graph = np.concatenate([G.ravel() for G in scenario["G_list"]])
    tp = int(np.sum(estimated_graph & true_graph))
    tn = int(np.sum(~estimated_graph & ~true_graph))
    fp = int(np.sum(estimated_graph & ~true_graph))
    fn = int(np.sum(~estimated_graph & true_graph))

    parameter_errors = [(prop - scenario["prop"]).ravel() ** 2]
    parameter_errors.extend(
        (estimated - truth).ravel() ** 2
        for estimated, truth in zip(B_reported, scenario["B_list"])
    )
    parameter_mse = float(np.mean(np.concatenate(parameter_errors)))

    layer_graph_accuracy = [
        float(np.mean((estimated[:, 1:] != 0.0) == truth_graph_layer))
        for estimated, truth_graph_layer in zip(B_reported, scenario["G_list"])
    ]
    layer_parameter_rmse = [
        float(np.sqrt(np.mean((estimated - truth) ** 2)))
        for estimated, truth in zip(B_reported, scenario["B_list"])
    ]
    layer_latent_accuracy = [
        float(np.mean(estimated == truth))
        for estimated, truth in zip(A_list, A_true)
    ]
    latent_correct = sum(
        int(np.sum(estimated == truth))
        for estimated, truth in zip(A_list, A_true)
    )
    latent_total = sum(truth.size for truth in A_true)

    return {
        "graph_accuracy": float(np.mean(estimated_graph == true_graph)),
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "false_positive_rate": _safe_ratio(fp, fp + tn),
        "false_negative_rate": _safe_ratio(fn, fn + tp),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "parameter_mse": parameter_mse,
        "parameter_rmse": math.sqrt(parameter_mse),
        "latent_accuracy": _safe_ratio(latent_correct, latent_total),
        "layer_graph_accuracy": json.dumps(layer_graph_accuracy),
        "layer_parameter_rmse": json.dumps(layer_parameter_rmse),
        "layer_latent_accuracy": json.dumps(layer_latent_accuracy),
        "alignment_permutations": json.dumps(permutations),
    }


def _seed_pair(base_seed: int, scenario_key: str, N: int, replicate: int):
    scenario_offset = {
        "d2": 10_000_000,
        "d3": 20_000_000,
        "d4": 30_000_000,
        "d5": 40_000_000,
    }[scenario_key]
    data_seed = int(base_seed + scenario_offset + 1000 * N + replicate)
    fit_seed = int(data_seed + 500_000_000)
    return data_seed, fit_seed


def run_one(task: tuple) -> dict:
    scenario_key, N, replicate, settings = task
    scenario = build_scenario(scenario_key)
    data_seed, fit_seed = _seed_pair(
        settings["base_seed"], scenario_key, N, replicate
    )
    penalty = float(N) ** 0.25
    tau = float(settings["tau"])
    tolerance = settings["tol"]
    common = {
        "scenario": scenario["label"],
        "D": scenario["D"],
        "J": scenario["J"],
        "K_list": json.dumps(scenario["K_list"]),
        "N": int(N),
        "replicate": int(replicate),
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        "penalty": penalty,
        "tau": tau,
        "C": settings["C"],
        "tol": tolerance,
        "max_iter": settings["max_iter"],
        "sa_burn_in": "" if settings["sa_burn_in"] is None else settings["sa_burn_in"],
        "fixed_init_binarize": settings["binarize"] or "",
        "fixed_warm_start": settings["warm_start"] or "",
    }
    kernel_options = {
        key: value
        for key, value in (
            ("sa_burn_in", settings["sa_burn_in"]),
            ("fixed_init_binarize", settings["binarize"]),
            ("fixed_warm_start", settings["warm_start"]),
        )
        if value is not None
    }
    started = time.perf_counter()
    try:
        X, A_true = simulate_bernoulli_dde(scenario, N, data_seed)
        np.random.seed(fit_seed)
        with threadpool_limits(limits=1):
            result = fit(
                X,
                method="DDE",
                family="Bernoulli",
                D=scenario["D"],
                dimension_method="fixed",
                K_list=scenario["K_list"],
                pen_list=[penalty] * scenario["D"],
                tau=tau,
                C=settings["C"],
                tol=tolerance,
                max_it=settings["max_iter"],
                initialize="data",
                epsilon=settings["epsilon"],
                verbose=False,
                **kernel_options,
            )
        for key in ("sa_burn_in", "fixed_init_binarize", "fixed_warm_start"):
            common[key] = result.diagnostics[key]
        metrics = calculate_metrics(
            result.prop, result.B_list, result.A_list, scenario, A_true, tau
        )
        finite = bool(
            np.all(np.isfinite(result.prop))
            and all(np.all(np.isfinite(B)) for B in result.B_list)
            and all(np.isfinite(float(metrics[key])) for key in (
                "graph_accuracy", "parameter_mse", "latent_accuracy"
            ))
        )
        return {
            **common,
            "status": "ok" if finite else "nonfinite",
            "error": "",
            "finite": finite,
            "iterations": int(result.diagnostics["iterations"]),
            "hit_iteration_limit": bool(
                result.diagnostics["iterations"] >= settings["max_iter"]
            ),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
    except Exception as exc:  # Preserve failed replicates in the audit trail.
        empty = {key: "" for key in RAW_FIELDS}
        return {
            **empty,
            **common,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "finite": False,
            "seconds": time.perf_counter() - started,
        }


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _matlab_rmoutliers(values: np.ndarray):
    """Approximate MATLAB rmoutliers default: median +/- 3 scaled MAD."""
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    scaled_mad = 1.482602218505602 * float(np.median(np.abs(values - median)))
    if not np.isfinite(scaled_mad) or scaled_mad == 0.0:
        return values, 0
    keep = np.abs(values - median) <= 3.0 * scaled_mad
    if not np.any(keep):
        return values, 0
    return values[keep], int(np.sum(~keep))


def summarize(rows: list[dict], requested_replicates: int):
    summary_rows = []
    sample_sizes = sorted({int(row["N"]) for row in rows})
    for scenario in SCENARIOS.values():
        for N in sample_sizes:
            group = [
                row for row in rows
                if row["scenario"] == scenario["label"] and int(row["N"]) == N
            ]
            if not group:
                continue
            successful = [row for row in group if row["status"] == "ok"]
            base = {
                "scenario": scenario["label"],
                "D": scenario["D"],
                "J": scenario["J"],
                "K_list": json.dumps(scenario["K_list"]),
                "N": N,
                "requested_replicates": requested_replicates,
                "successful_replicates": len(successful),
                "failed_replicates": len(group) - len(successful),
            }
            if not successful:
                summary_rows.append(base)
                continue
            graph = _float_values(successful, "graph_accuracy")
            mse = _float_values(successful, "parameter_mse")
            filtered_mse, removed = _matlab_rmoutliers(mse)
            layer_graph = np.asarray([
                json.loads(row["layer_graph_accuracy"]) for row in successful
            ])
            layer_latent = np.asarray([
                json.loads(row["layer_latent_accuracy"]) for row in successful
            ])
            layer_rmse = np.asarray([
                json.loads(row["layer_parameter_rmse"]) for row in successful
                if row.get("layer_parameter_rmse")
            ])
            iterations = _float_values(successful, "iterations")
            seconds = _float_values(successful, "seconds")
            summary_rows.append({
                **base,
                "graph_accuracy_mean": float(np.mean(graph)),
                "graph_accuracy_sd": float(np.std(graph, ddof=1)) if len(graph) > 1 else 0.0,
                "graph_accuracy_se": (
                    float(np.std(graph, ddof=1) / np.sqrt(len(graph)))
                    if len(graph) > 1 else 0.0
                ),
                "sensitivity_mean": float(np.mean(_float_values(successful, "sensitivity"))),
                "specificity_mean": float(np.mean(_float_values(successful, "specificity"))),
                "parameter_rmse_paper": math.sqrt(float(np.mean(filtered_mse))),
                "parameter_rmse_all": math.sqrt(float(np.mean(mse))),
                "removed_mse_outliers": removed,
                "latent_accuracy_mean": float(np.mean(_float_values(successful, "latent_accuracy"))),
                "layer_graph_accuracy_mean": json.dumps(
                    np.mean(layer_graph, axis=0).tolist()
                ),
                "layer_parameter_rmse_mean": json.dumps(
                    np.mean(layer_rmse, axis=0).tolist() if layer_rmse.size else []
                ),
                "layer_latent_accuracy_mean": json.dumps(
                    np.mean(layer_latent, axis=0).tolist()
                ),
                "iterations_mean": float(np.mean(iterations)),
                "iteration_limit_fraction": float(np.mean([
                    str(row["hit_iteration_limit"]).lower() in {"true", "1"}
                    for row in successful
                ])),
                "seconds_mean": float(np.mean(seconds)),
                "seconds_total": float(np.sum(seconds)),
            })
    return summary_rows


def compare_with_paper(summary_rows: list[dict]):
    rows = []
    d2_label = SCENARIOS["d2"]["label"]
    for summary in summary_rows:
        if summary["scenario"] != d2_label:
            continue
        N = int(summary["N"])
        paper = PAPER_D2_SAEM.get(N)
        python_graph = summary.get("graph_accuracy_mean", "")
        python_rmse = summary.get("parameter_rmse_paper", "")
        rows.append({
            "scenario": d2_label,
            "N": N,
            "python_graph_accuracy": python_graph,
            "paper_graph_accuracy": "" if paper is None else paper["graph_accuracy"],
            "graph_accuracy_delta": (
                "" if paper is None or python_graph == ""
                else float(python_graph) - paper["graph_accuracy"]
            ),
            "python_rmse_theta": python_rmse,
            "paper_rmse_theta": "" if paper is None else paper["rmse_theta"],
            "rmse_theta_delta": (
                "" if paper is None or python_rmse == ""
                else float(python_rmse) - paper["rmse_theta"]
            ),
            "paper_value_available": paper is not None,
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=["d2", "d3"]
    )
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int,
        default=[500, 1000, 2000, 4000, 8000, 16000],
    )
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_PARAMETRIC_CONVERGENCE_TOL,
        help="Postprocessed parameter-update RMSE threshold.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.3,
        help="Fixed TLP fitting and graph-scoring threshold (default: 0.3).",
    )
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--sa-burn-in", type=int, default=None,
        help="Unit-step SAEM burn-in sweeps (default: the package default).",
    )
    parser.add_argument(
        "--binarize", choices=["projection", "anchor_lca"], default=None,
        help="Initial latent-state rule (default: the package default).",
    )
    parser.add_argument(
        "--warm-start", choices=["none", "layerwise"], default=None,
        help="Layer-wise warm start (default: the package default; "
        "'layerwise' hurts binary data and is kept for ablations only).",
    )
    parser.add_argument("--base-seed", type=int, default=20260901)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/bernoulli_fixed_tau03_rmse_tol005"),
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replicates < 1 or args.n_jobs < 1 or args.C < 1 or args.max_iter < 1:
        raise ValueError("replicates, n_jobs, C, and max_iter must be positive.")
    if not np.isfinite(args.tol) or args.tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    if not np.isfinite(args.tau) or args.tau <= 0:
        raise ValueError("tau must be finite and positive.")
    if any(N < 1 for N in args.sample_sizes):
        raise ValueError("sample sizes must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "replicates.csv"
    if raw_path.exists() and not args.resume:
        raw_path.unlink()
    rows = _read_existing(raw_path) if args.resume else []
    if rows and any(
        float(row["tol"]) != args.tol
        or int(row["max_iter"]) != args.max_iter
        or int(row["C"]) != args.C
        or not np.isclose(float(row["tau"]), args.tau)
        for row in rows
    ):
        raise ValueError(
            "Existing recovery rows use incompatible tol, max_iter, C, or tau; "
            "select a new output directory or pass --no-resume."
        )
    requested_options = {
        "sa_burn_in": args.sa_burn_in,
        "fixed_init_binarize": args.binarize,
        "fixed_warm_start": args.warm_start,
    }
    for key, value in requested_options.items():
        if value is None:
            continue
        if any(row.get(key, "") not in ("", str(value)) for row in rows):
            raise ValueError(
                f"Existing recovery rows use a different {key}; "
                "select a new output directory or pass --no-resume."
            )
    completed = {
        (row["scenario"], int(row["N"]), int(row["replicate"])) for row in rows
    }
    settings = {
        "C": args.C,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "tau": args.tau,
        "epsilon": args.epsilon,
        "base_seed": args.base_seed,
        "sa_burn_in": args.sa_burn_in,
        "binarize": args.binarize,
        "warm_start": args.warm_start,
    }
    tasks = []
    for key in args.scenarios:
        label = SCENARIOS[key]["label"]
        for N in args.sample_sizes:
            for replicate in range(1, args.replicates + 1):
                if (label, N, replicate) not in completed:
                    tasks.append((key, N, replicate, settings))

    print(
        f"Bernoulli fixed-tau recovery: {len(tasks)} pending / "
        f"{len(tasks) + len(completed)} total fits; n_jobs={args.n_jobs}",
        flush=True,
    )
    started = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            for number, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                if number % 10 == 0 or number == len(tasks):
                    rows.sort(key=lambda item: (
                        item["scenario"], int(item["N"]), int(item["replicate"])
                    ))
                    _write_csv(raw_path, rows, RAW_FIELDS)
                    elapsed = time.perf_counter() - started
                    print(
                        f"completed {number}/{len(tasks)} new fits "
                        f"({elapsed / 60.0:.1f} min); last="
                        f"{row['scenario']} N={row['N']} rep={row['replicate']} "
                        f"status={row['status']}",
                        flush=True,
                    )

    rows.sort(key=lambda item: (
        item["scenario"], int(item["N"]), int(item["replicate"])
    ))
    _write_csv(raw_path, rows, RAW_FIELDS)
    summary_rows = summarize(rows, args.replicates)
    _write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    comparison_rows = compare_with_paper(summary_rows)
    _write_csv(
        output_dir / "paper_comparison.csv", comparison_rows, COMPARISON_FIELDS
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scenarios": {
            key: {**value, "K_list": list(value["K_list"])}
            for key, value in SCENARIOS.items() if key in args.scenarios
        },
        "sample_sizes": args.sample_sizes,
        "replicates": args.replicates,
        "n_jobs": args.n_jobs,
        "settings": {
            **settings,
            "initialization": "data",
            "penalty": "N**0.25 for every layer",
            "tau": f"fixed {args.tau}",
            "tol": args.tol,
            "tol_definition": "postprocessed parameter-update RMSE",
            "family": "Bernoulli",
            "dimension_method": "fixed",
            "kernel_options": "package defaults unless --sa-burn-in/--binarize/--warm-start are given; the values used are in replicates.csv",
        },
        "metrics": {
            "graph_accuracy": "Entrywise accuracy over all slope support matrices after bottom-up Hungarian alignment.",
            "parameter_mse": "Per-replicate mean squared error over prop and all coefficient matrices, including intercepts.",
            "parameter_rmse_paper": "sqrt(mean(parameter_mse after median +/- 3 scaled-MAD outlier removal)).",
            "alignment": "Squared-L2 column assignment from the observed layer upward; corresponding rows of the next coefficient matrix are permuted.",
        },
        "paper_reference": {
            "parameters": "Equation (S.36), sparse identifiable B_s; positive 4/3 paired loading.",
            "D2_graph_accuracy": "Table S.6, Bernoulli B_s (18,6,2), SAEM.",
            "D2_rmse": "Table S.9, Bernoulli B_s (18,6,2), SAEM.",
            "D3_note": "No D=3 Bernoulli result is reported; this scenario recursively extends equation (S.36) to (54,18,6,2).",
            "D4_note": "No D=4 Bernoulli result is reported; this scenario recursively extends equation (S.36) to (162,54,18,6,2).",
            "D5_note": "No D=5 Bernoulli result is reported; this scenario recursively extends equation (S.36) to (486,162,54,18,6,2).",
            "N16000_note": "The paper tables end at N=8000; N=16000 is an added extrapolation.",
        },
        "completed_rows": len(rows),
        "successful_rows": sum(row["status"] == "ok" for row in rows),
        "failed_rows": sum(row["status"] != "ok" for row in rows),
        "wall_seconds_this_invocation": time.perf_counter() - started,
    }
    metadata_path = output_dir / "metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)

    print(f"results written to {output_dir}", flush=True)
    for row in summary_rows:
        print(
            f"{row['scenario']} N={row['N']}: "
            f"ok={row['successful_replicates']}, "
            f"Acc(G)={row.get('graph_accuracy_mean', '')}, "
            f"RMSE={row.get('parameter_rmse_paper', '')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
