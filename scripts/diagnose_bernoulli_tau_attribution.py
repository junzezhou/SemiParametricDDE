"""Paired attribution experiment for the Bernoulli DDE tau schedule.

The experiment separates two uses of ``tau``:

1. the fitting value used by the TLP penalty and coefficient postprocessor;
2. the scoring threshold used to turn fitted slopes into a graph.

For every simulated dataset and fit seed, the model is fitted twice: once with
the paper schedule ``3*N**(-0.3)`` and once with fixed ``tau=0.3``.  Every fit
is then scored at both thresholds.  The resulting 2-by-2 paired design
distinguishes a reporting-threshold effect from a fitting-path effect.
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
from threadpoolctl import threadpool_limits

from dde import fit
from dde.algorithms.dde import DEFAULT_PARAMETRIC_CONVERGENCE_TOL

import evaluate_bernoulli_fixed_recovery as recovery


FIT_TAU_MODES = ("dynamic", "fixed_0.3")
SCORE_TAU_MODES = ("dynamic", "fixed_0.3")

RAW_FIELDS = [
    "scenario", "D", "J", "K_list", "N", "replicate", "data_seed",
    "fit_seed", "penalty", "fit_tau_mode", "fit_tau", "score_tau_mode",
    "score_tau", "C", "tol", "max_iter", "status", "error", "finite",
    "iterations", "hit_iteration_limit", "seconds", "graph_accuracy",
    "balanced_accuracy", "sensitivity", "specificity",
    "false_positive_rate", "false_negative_rate", "true_positive",
    "true_negative", "false_positive", "false_negative", "active_edges",
    "parameter_mse", "parameter_rmse", "latent_accuracy",
    "layer_graph_accuracy", "layer_sensitivity", "layer_specificity",
    "layer_true_positive", "layer_true_negative", "layer_false_positive",
    "layer_false_negative", "layer_active_edges", "alignment_permutations",
]

SUMMARY_FIELDS = [
    "scenario", "D", "J", "K_list", "N", "fit_tau_mode", "fit_tau",
    "score_tau_mode", "score_tau", "requested_replicates",
    "successful_replicates", "failed_replicates", "graph_accuracy_mean",
    "graph_accuracy_se", "balanced_accuracy_mean", "sensitivity_mean",
    "specificity_mean", "false_positive_mean", "false_negative_mean",
    "active_edges_mean", "parameter_rmse_all", "latent_accuracy_mean",
    "layer_graph_accuracy_mean", "layer_sensitivity_mean",
    "layer_specificity_mean", "layer_false_positive_mean",
    "layer_false_negative_mean", "layer_active_edges_mean",
    "iterations_mean", "iteration_limit_fraction", "seconds_mean",
]

PAIRED_FIELDS = [
    "scenario", "D", "N", "comparison", "left_cell", "right_cell",
    "successful_pairs", "metric", "left_mean", "right_mean",
    "right_minus_left_mean", "right_minus_left_sd", "right_minus_left_se",
]


def _tau_values(N: int, fixed_tau: float) -> dict[str, float]:
    return {
        "dynamic": 3.0 * float(N) ** -0.3,
        "fixed_0.3": float(fixed_tau),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _score_fit(prop, B_list, A_list, scenario, A_true, score_tau: float):
    B_reported = recovery._threshold_all_coefficients(B_list, score_tau)
    prop_aligned, B_aligned, A_aligned, permutations = recovery.align_hierarchy(
        prop, B_reported, A_list, scenario["B_list"]
    )

    layer_graph_accuracy = []
    layer_sensitivity = []
    layer_specificity = []
    layer_tp = []
    layer_tn = []
    layer_fp = []
    layer_fn = []
    layer_active = []
    for estimated, truth in zip(B_aligned, scenario["G_list"]):
        graph = estimated[:, 1:] != 0.0
        tp = int(np.sum(graph & truth))
        tn = int(np.sum(~graph & ~truth))
        fp = int(np.sum(graph & ~truth))
        fn = int(np.sum(~graph & truth))
        layer_graph_accuracy.append(float(np.mean(graph == truth)))
        layer_sensitivity.append(_safe_ratio(tp, tp + fn))
        layer_specificity.append(_safe_ratio(tn, tn + fp))
        layer_tp.append(tp)
        layer_tn.append(tn)
        layer_fp.append(fp)
        layer_fn.append(fn)
        layer_active.append(tp + fp)

    tp = int(sum(layer_tp))
    tn = int(sum(layer_tn))
    fp = int(sum(layer_fp))
    fn = int(sum(layer_fn))
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)

    parameter_errors = [(prop_aligned - scenario["prop"]).ravel() ** 2]
    parameter_errors.extend(
        (estimated - truth).ravel() ** 2
        for estimated, truth in zip(B_aligned, scenario["B_list"])
    )
    parameter_mse = float(np.mean(np.concatenate(parameter_errors)))

    latent_correct = sum(
        int(np.sum(estimated == truth))
        for estimated, truth in zip(A_aligned, A_true)
    )
    latent_total = sum(truth.size for truth in A_true)

    return {
        "graph_accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": _safe_ratio(fp, fp + tn),
        "false_negative_rate": _safe_ratio(fn, fn + tp),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "active_edges": tp + fp,
        "parameter_mse": parameter_mse,
        "parameter_rmse": math.sqrt(parameter_mse),
        "latent_accuracy": _safe_ratio(latent_correct, latent_total),
        "layer_graph_accuracy": json.dumps(layer_graph_accuracy),
        "layer_sensitivity": json.dumps(layer_sensitivity),
        "layer_specificity": json.dumps(layer_specificity),
        "layer_true_positive": json.dumps(layer_tp),
        "layer_true_negative": json.dumps(layer_tn),
        "layer_false_positive": json.dumps(layer_fp),
        "layer_false_negative": json.dumps(layer_fn),
        "layer_active_edges": json.dumps(layer_active),
        "alignment_permutations": json.dumps(permutations),
    }


def run_one(task: tuple) -> list[dict]:
    scenario_key, N, replicate, fit_tau_mode, settings = task
    scenario = recovery.build_scenario(scenario_key)
    data_seed, fit_seed = recovery._seed_pair(
        settings["base_seed"], scenario_key, N, replicate
    )
    tau_values = _tau_values(N, settings["fixed_tau"])
    fit_tau = tau_values[fit_tau_mode]
    penalty = float(N) ** 0.25
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
        "fit_tau_mode": fit_tau_mode,
        "fit_tau": fit_tau,
        "C": settings["C"],
        "tol": settings["tol"],
        "max_iter": settings["max_iter"],
    }
    started = time.perf_counter()
    try:
        X, A_true = recovery.simulate_bernoulli_dde(scenario, N, data_seed)
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
                tau=fit_tau,
                C=settings["C"],
                tol=settings["tol"],
                max_it=settings["max_iter"],
                initialize="data",
                epsilon=settings["epsilon"],
                verbose=False,
            )
        elapsed = time.perf_counter() - started
        rows = []
        for score_tau_mode in SCORE_TAU_MODES:
            score_tau = tau_values[score_tau_mode]
            metrics = _score_fit(
                result.prop,
                result.B_list,
                result.A_list,
                scenario,
                A_true,
                score_tau,
            )
            finite = bool(
                np.all(np.isfinite(result.prop))
                and all(np.all(np.isfinite(B)) for B in result.B_list)
                and all(np.isfinite(float(metrics[key])) for key in (
                    "graph_accuracy", "parameter_mse", "latent_accuracy"
                ))
            )
            rows.append({
                **common,
                "score_tau_mode": score_tau_mode,
                "score_tau": score_tau,
                "status": "ok" if finite else "nonfinite",
                "error": "",
                "finite": finite,
                "iterations": int(result.diagnostics["iterations"]),
                "hit_iteration_limit": bool(
                    result.diagnostics["iterations"] >= settings["max_iter"]
                ),
                "seconds": elapsed,
                **metrics,
            })
        return rows
    except Exception as exc:
        elapsed = time.perf_counter() - started
        rows = []
        for score_tau_mode in SCORE_TAU_MODES:
            empty = {key: "" for key in RAW_FIELDS}
            rows.append({
                **empty,
                **common,
                "score_tau_mode": score_tau_mode,
                "score_tau": tau_values[score_tau_mode],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "finite": False,
                "seconds": elapsed,
            })
        return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _json_mean(rows: list[dict], key: str) -> str:
    values = np.asarray([json.loads(row[key]) for row in rows], dtype=float)
    return json.dumps(np.mean(values, axis=0).tolist())


def summarize(rows: list[dict], requested_replicates: int) -> list[dict]:
    output = []
    groups = sorted({
        (row["scenario"], int(row["N"]), row["fit_tau_mode"], row["score_tau_mode"])
        for row in rows
    })
    scenario_by_label = {
        value["label"]: value for value in recovery.SCENARIOS.values()
    }
    for scenario_label, N, fit_mode, score_mode in groups:
        group = [
            row for row in rows
            if row["scenario"] == scenario_label
            and int(row["N"]) == N
            and row["fit_tau_mode"] == fit_mode
            and row["score_tau_mode"] == score_mode
        ]
        successful = [row for row in group if row["status"] == "ok"]
        spec = scenario_by_label[scenario_label]
        base = {
            "scenario": scenario_label,
            "D": spec["D"],
            "J": spec["J"],
            "K_list": json.dumps(spec["K_list"]),
            "N": N,
            "fit_tau_mode": fit_mode,
            "fit_tau": group[0]["fit_tau"],
            "score_tau_mode": score_mode,
            "score_tau": group[0]["score_tau"],
            "requested_replicates": requested_replicates,
            "successful_replicates": len(successful),
            "failed_replicates": len(group) - len(successful),
        }
        if not successful:
            output.append(base)
            continue
        graph = np.asarray([float(row["graph_accuracy"]) for row in successful])
        mse = np.asarray([float(row["parameter_mse"]) for row in successful])
        output.append({
            **base,
            "graph_accuracy_mean": float(np.mean(graph)),
            "graph_accuracy_se": (
                float(np.std(graph, ddof=1) / np.sqrt(graph.size))
                if graph.size > 1 else 0.0
            ),
            "balanced_accuracy_mean": _mean(successful, "balanced_accuracy"),
            "sensitivity_mean": _mean(successful, "sensitivity"),
            "specificity_mean": _mean(successful, "specificity"),
            "false_positive_mean": _mean(successful, "false_positive"),
            "false_negative_mean": _mean(successful, "false_negative"),
            "active_edges_mean": _mean(successful, "active_edges"),
            "parameter_rmse_all": math.sqrt(float(np.mean(mse))),
            "latent_accuracy_mean": _mean(successful, "latent_accuracy"),
            "layer_graph_accuracy_mean": _json_mean(successful, "layer_graph_accuracy"),
            "layer_sensitivity_mean": _json_mean(successful, "layer_sensitivity"),
            "layer_specificity_mean": _json_mean(successful, "layer_specificity"),
            "layer_false_positive_mean": _json_mean(successful, "layer_false_positive"),
            "layer_false_negative_mean": _json_mean(successful, "layer_false_negative"),
            "layer_active_edges_mean": _json_mean(successful, "layer_active_edges"),
            "iterations_mean": _mean(successful, "iterations"),
            "iteration_limit_fraction": float(np.mean([
                str(row["hit_iteration_limit"]).lower() in {"true", "1"}
                for row in successful
            ])),
            "seconds_mean": _mean(successful, "seconds"),
        })
    return output


COMPARISONS = {
    "scoring_effect_with_dynamic_fit": (
        ("dynamic", "dynamic"), ("dynamic", "fixed_0.3")
    ),
    "scoring_effect_with_fixed_fit": (
        ("fixed_0.3", "dynamic"), ("fixed_0.3", "fixed_0.3")
    ),
    "fitting_effect_at_dynamic_score": (
        ("dynamic", "dynamic"), ("fixed_0.3", "dynamic")
    ),
    "fitting_effect_at_fixed_score": (
        ("dynamic", "fixed_0.3"), ("fixed_0.3", "fixed_0.3")
    ),
    "total_fixed_vs_paper_dynamic": (
        ("dynamic", "dynamic"), ("fixed_0.3", "fixed_0.3")
    ),
}

PAIRED_METRICS = (
    "graph_accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "false_positive", "false_negative", "active_edges", "parameter_rmse",
    "iterations",
)


def paired_summary(rows: list[dict]) -> list[dict]:
    successful = [row for row in rows if row["status"] == "ok"]
    lookup = {
        (
            row["scenario"], int(row["N"]), int(row["replicate"]),
            row["fit_tau_mode"], row["score_tau_mode"],
        ): row
        for row in successful
    }
    output = []
    scenario_labels = sorted({row["scenario"] for row in successful})
    sample_sizes = sorted({int(row["N"]) for row in successful})
    scenario_by_label = {
        value["label"]: value for value in recovery.SCENARIOS.values()
    }
    for scenario_label in scenario_labels:
        for N in sample_sizes:
            replicates = sorted({
                int(row["replicate"]) for row in successful
                if row["scenario"] == scenario_label and int(row["N"]) == N
            })
            for comparison, (left_cell, right_cell) in COMPARISONS.items():
                pairs = []
                for replicate in replicates:
                    left = lookup.get((scenario_label, N, replicate, *left_cell))
                    right = lookup.get((scenario_label, N, replicate, *right_cell))
                    if left is not None and right is not None:
                        pairs.append((left, right))
                if not pairs:
                    continue
                for metric in PAIRED_METRICS:
                    left_values = np.asarray([
                        float(left[metric]) for left, _ in pairs
                    ])
                    right_values = np.asarray([
                        float(right[metric]) for _, right in pairs
                    ])
                    differences = right_values - left_values
                    output.append({
                        "scenario": scenario_label,
                        "D": scenario_by_label[scenario_label]["D"],
                        "N": N,
                        "comparison": comparison,
                        "left_cell": "/".join(left_cell),
                        "right_cell": "/".join(right_cell),
                        "successful_pairs": len(pairs),
                        "metric": metric,
                        "left_mean": float(np.mean(left_values)),
                        "right_mean": float(np.mean(right_values)),
                        "right_minus_left_mean": float(np.mean(differences)),
                        "right_minus_left_sd": (
                            float(np.std(differences, ddof=1))
                            if differences.size > 1 else 0.0
                        ),
                        "right_minus_left_se": (
                            float(np.std(differences, ddof=1) / np.sqrt(differences.size))
                            if differences.size > 1 else 0.0
                        ),
                    })
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", nargs="+", choices=recovery.SCENARIOS,
        default=["d2", "d3"],
    )
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int,
        default=[2000, 8000, 16000],
    )
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--fixed-tau", type=float, default=0.3)
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument(
        "--tol", type=float, default=DEFAULT_PARAMETRIC_CONVERGENCE_TOL
    )
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--base-seed", type=int, default=20260901)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/bernoulli_tau_attribution"),
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replicates < 1 or args.n_jobs < 1 or args.C < 1 or args.max_iter < 1:
        raise ValueError("replicates, n_jobs, C, and max_iter must be positive.")
    if any(N < 1 for N in args.sample_sizes):
        raise ValueError("sample sizes must be positive.")
    if not np.isfinite(args.fixed_tau) or args.fixed_tau <= 0.0:
        raise ValueError("fixed_tau must be finite and positive.")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        raise ValueError("tol must be finite and nonnegative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "replicates.csv"
    if raw_path.exists() and not args.resume:
        raw_path.unlink()
    rows = _read_csv(raw_path) if args.resume else []

    expected_settings = (args.C, args.max_iter, args.tol)
    if rows and any(
        (int(row["C"]), int(row["max_iter"]), float(row["tol"]))
        != expected_settings
        or not np.isclose(
            float(row["fit_tau"]),
            _tau_values(int(row["N"]), args.fixed_tau)[row["fit_tau_mode"]],
        )
        or not np.isclose(
            float(row["score_tau"]),
            _tau_values(int(row["N"]), args.fixed_tau)[row["score_tau_mode"]],
        )
        for row in rows
    ):
        raise ValueError(
            "Existing rows use incompatible C, max_iter, tol, or tau values; "
            "select a new output directory or pass --no-resume."
        )

    completed_cells = {
        (
            row["scenario"], int(row["N"]), int(row["replicate"]),
            row["fit_tau_mode"], row["score_tau_mode"],
        )
        for row in rows
    }
    settings = {
        "fixed_tau": args.fixed_tau,
        "C": args.C,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "epsilon": args.epsilon,
        "base_seed": args.base_seed,
    }
    tasks = []
    for scenario_key in args.scenarios:
        label = recovery.SCENARIOS[scenario_key]["label"]
        for N in args.sample_sizes:
            for replicate in range(1, args.replicates + 1):
                for fit_tau_mode in FIT_TAU_MODES:
                    expected = {
                        (label, N, replicate, fit_tau_mode, score_mode)
                        for score_mode in SCORE_TAU_MODES
                    }
                    if not expected.issubset(completed_cells):
                        rows = [
                            row for row in rows
                            if not (
                                row["scenario"] == label
                                and int(row["N"]) == N
                                and int(row["replicate"]) == replicate
                                and row["fit_tau_mode"] == fit_tau_mode
                            )
                        ]
                        tasks.append((
                            scenario_key, N, replicate, fit_tau_mode, settings
                        ))

    print(
        f"Bernoulli tau attribution: {len(tasks)} pending fits; "
        f"each fit has {len(SCORE_TAU_MODES)} scoring cells; n_jobs={args.n_jobs}",
        flush=True,
    )
    started = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            for number, future in enumerate(as_completed(futures), start=1):
                new_rows = future.result()
                rows.extend(new_rows)
                if number % 4 == 0 or number == len(tasks):
                    rows.sort(key=lambda row: (
                        row["scenario"], int(row["N"]), int(row["replicate"]),
                        row["fit_tau_mode"], row["score_tau_mode"],
                    ))
                    _write_csv(raw_path, rows, RAW_FIELDS)
                    last = new_rows[0]
                    print(
                        f"completed {number}/{len(tasks)} fits "
                        f"({(time.perf_counter() - started) / 60.0:.1f} min); "
                        f"last={last['scenario']} N={last['N']} "
                        f"rep={last['replicate']} fit_tau={last['fit_tau_mode']} "
                        f"status={last['status']}",
                        flush=True,
                    )

    rows.sort(key=lambda row: (
        row["scenario"], int(row["N"]), int(row["replicate"]),
        row["fit_tau_mode"], row["score_tau_mode"],
    ))
    _write_csv(raw_path, rows, RAW_FIELDS)
    summary_rows = summarize(rows, args.replicates)
    _write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    paired_rows = paired_summary(rows)
    _write_csv(output_dir / "paired_differences.csv", paired_rows, PAIRED_FIELDS)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sample_sizes": args.sample_sizes,
        "replicates": args.replicates,
        "n_jobs": args.n_jobs,
        "design": {
            "fit_tau": {
                "dynamic": "3*N**(-0.3)",
                "fixed_0.3": args.fixed_tau,
            },
            "score_tau": {
                "dynamic": "3*N**(-0.3)",
                "fixed_0.3": args.fixed_tau,
            },
            "pairing": "same X, initialization inputs, and fit seed within replicate",
        },
        "settings": settings,
        "completed_score_rows": len(rows),
        "successful_score_rows": sum(row["status"] == "ok" for row in rows),
        "failed_score_rows": sum(row["status"] != "ok" for row in rows),
        "wall_seconds_this_invocation": time.perf_counter() - started,
    }
    metadata_path = output_dir / "metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)

    print(f"results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
