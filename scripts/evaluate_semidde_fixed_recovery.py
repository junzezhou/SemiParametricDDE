"""Run the fixed-width SemiDDE simulation used by the example notebook.

The defaults reproduce the settings in ``examples/02_simulation_SemiDDE`` for
the D=2 and D=3 designs:

* widths ``(6, 2)`` and ``(18, 6, 2)`` with ``J=3*K_1``;
* log-normal observed margins, ``Y = exp(X)``;
* ``lambda_d = N**(1/4)``, ``tau=0.3``, ``tol=K_D/2``;
* one Monte Carlo draw, at most 200 outer iterations and 50 inner iterations;
* seeds ``150 + replicate`` for each sample size, as in the notebook.

Replicate metrics are checkpointed after every completed fit, so rerunning the
same command safely resumes an interrupted experiment.
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
from dde.identifiability import build_true_params
from dde.simulation import generate_X_normal
from dde.utils import thres


SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000)
SCENARIOS = {
    "d2": {"label": "D2_fixed_lognormal", "D": 2, "K_list": (6, 2), "J": 18},
    "d3": {
        "label": "D3_fixed_lognormal",
        "D": 3,
        "K_list": (18, 6, 2),
        "J": 54,
    },
}

RAW_FIELDS = [
    "scenario",
    "D",
    "J",
    "K_list",
    "N",
    "replicate",
    "seed",
    "status",
    "error",
    "finite",
    "max_outer_iter_budget",
    "iterations",
    "seconds",
    "graph_accuracy",
    "parameter_mse",
    "parameter_rmse",
    "transform_rmse",
    "layer_graph_accuracy",
]

SUMMARY_FIELDS = [
    "scenario",
    "D",
    "J",
    "K_list",
    "N",
    "requested_replicates",
    "successful_replicates",
    "failed_replicates",
    "graph_accuracy_mean",
    "graph_accuracy_sd",
    "graph_accuracy_se",
    "parameter_rmse",
    "parameter_rmse_mean",
    "transform_rmse_mean",
    "iterations_mean",
    "seconds_mean",
    "seconds_total",
    "layer_graph_accuracy_mean",
]


def build_scenario(key: str) -> dict:
    spec = dict(SCENARIOS[key])
    prop, B_list, G_list, gamma = build_true_params(
        spec["K_list"], spec["J"], 4
    )
    spec.update(
        prop=np.asarray(prop, dtype=float),
        B_list=[np.asarray(value, dtype=float) for value in B_list],
        G_list=[np.asarray(value, dtype=bool) for value in G_list],
        gamma=np.asarray(gamma, dtype=float),
    )
    return spec


def _calculate_metrics(result, scenario, X_true, tau):
    B_est = [thres(value, tau) for value in result.B_list]
    layer_accuracy = []
    correct = 0
    entries = 0
    for estimated, truth in zip(B_est, scenario["G_list"]):
        estimated_graph = estimated[:, 1:] != 0.0
        layer_correct = int(np.sum(estimated_graph == truth))
        layer_accuracy.append(float(layer_correct / truth.size))
        correct += layer_correct
        entries += int(truth.size)

    squared_error = float(np.sum((result.prop - scenario["prop"]) ** 2))
    parameter_count = int(scenario["prop"].size)
    for estimated, truth in zip(B_est, scenario["B_list"]):
        squared_error += float(np.sum((estimated - truth) ** 2))
        parameter_count += int(truth.size)
    squared_error += float(np.sum((result.gamma - scenario["gamma"]) ** 2))
    parameter_count += int(scenario["gamma"].size)
    parameter_mse = squared_error / parameter_count

    transform_rmse = float(
        np.sqrt(np.mean((np.asarray(result.transformed) - X_true) ** 2))
    )
    finite = bool(
        np.isfinite(parameter_mse)
        and np.isfinite(transform_rmse)
        and all(np.all(np.isfinite(value)) for value in result.B_list)
        and np.all(np.isfinite(result.prop))
        and np.all(np.isfinite(result.gamma))
    )
    return {
        "finite": finite,
        "graph_accuracy": float(correct / entries),
        "parameter_mse": float(parameter_mse),
        "parameter_rmse": float(math.sqrt(parameter_mse)),
        "transform_rmse": transform_rmse,
        "layer_graph_accuracy": json.dumps(layer_accuracy),
    }


def _fit_one(task: tuple) -> dict:
    scenario_key, N, replicate, args_dict = task
    scenario = build_scenario(scenario_key)
    seed = int(args_dict["base_seed"] + replicate)
    common = {
        "scenario": scenario["label"],
        "D": scenario["D"],
        "J": scenario["J"],
        "K_list": json.dumps(scenario["K_list"]),
        "N": int(N),
        "replicate": int(replicate),
        "seed": seed,
    }
    started = time.perf_counter()
    if N >= 8000:
        max_outer_iter = int(args_dict["n8000_max_outer_iter"])
    elif N >= 4000:
        max_outer_iter = int(args_dict["n4000_max_outer_iter"])
    else:
        max_outer_iter = int(args_dict["max_outer_iter"])
    try:
        # The original notebook resets NumPy before both data generation and
        # fitting; retain that exact convention for comparable replicates.
        np.random.seed(seed)
        X_true, _ = generate_X_normal(
            N, scenario["prop"], scenario["B_list"], scenario["gamma"]
        )
        Y = np.exp(X_true)
        with threadpool_limits(limits=1):
            result = fit(
                Y,
                model="SemiDDE",
                D=scenario["D"],
                K_list=scenario["K_list"],
                dimension_method="fixed",
                initialize="oracle",
                B_list_true=scenario["B_list"],
                G_list_true=scenario["G_list"],
                pen_list=[N ** 0.25] * scenario["D"],
                tau=args_dict["tau"],
                tol=scenario["K_list"][-1] / 2,
                C=1,
                epsilon=args_dict["epsilon"],
                max_outer_iter=max_outer_iter,
                max_inner_iter=args_dict["max_inner_iter"],
                verbose=False,
            )
        metrics = _calculate_metrics(
            result, scenario, X_true, args_dict["tau"]
        )
        return {
            **common,
            "status": "ok" if metrics["finite"] else "nonfinite",
            "error": "",
            "max_outer_iter_budget": max_outer_iter,
            "iterations": int(result.diagnostics["iterations"]),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
    except Exception as exc:  # Keep long runs resumable and auditable.
        empty = {field: "" for field in RAW_FIELDS}
        return {
            **empty,
            **common,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "finite": False,
            "max_outer_iter_budget": max_outer_iter,
            "seconds": time.perf_counter() - started,
        }


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


def _float_values(rows: list[dict], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def summarize(rows: list[dict], scenarios: list[str], sample_sizes, replicates):
    summaries = []
    for scenario_key in scenarios:
        scenario = SCENARIOS[scenario_key]
        for N in sample_sizes:
            group = [
                row
                for row in rows
                if row["scenario"] == scenario["label"] and int(row["N"]) == N
            ]
            successful = [row for row in group if row["status"] == "ok"]
            base = {
                "scenario": scenario["label"],
                "D": scenario["D"],
                "J": scenario["J"],
                "K_list": json.dumps(scenario["K_list"]),
                "N": int(N),
                "requested_replicates": int(replicates),
                "successful_replicates": len(successful),
                "failed_replicates": len(group) - len(successful),
            }
            if not successful:
                summaries.append(base)
                continue
            graph = _float_values(successful, "graph_accuracy")
            mse = _float_values(successful, "parameter_mse")
            rmse = _float_values(successful, "parameter_rmse")
            transform = _float_values(successful, "transform_rmse")
            seconds = _float_values(successful, "seconds")
            layer = np.asarray(
                [json.loads(row["layer_graph_accuracy"]) for row in successful]
            )
            summaries.append(
                {
                    **base,
                    "graph_accuracy_mean": float(np.mean(graph)),
                    "graph_accuracy_sd": (
                        float(np.std(graph, ddof=1)) if len(graph) > 1 else 0.0
                    ),
                    "graph_accuracy_se": (
                        float(np.std(graph, ddof=1) / np.sqrt(len(graph)))
                        if len(graph) > 1
                        else 0.0
                    ),
                    # This is the notebook's definition: sqrt(mean MSE).
                    "parameter_rmse": float(np.sqrt(np.mean(mse))),
                    "parameter_rmse_mean": float(np.mean(rmse)),
                    "transform_rmse_mean": float(np.mean(transform)),
                    "iterations_mean": float(
                        np.mean(_float_values(successful, "iterations"))
                    ),
                    "seconds_mean": float(np.mean(seconds)),
                    "seconds_total": float(np.sum(seconds)),
                    "layer_graph_accuracy_mean": json.dumps(
                        np.mean(layer, axis=0).tolist()
                    ),
                }
            )
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=["d2", "d3"]
    )
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int, default=list(SAMPLE_SIZES)
    )
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--base-seed", type=int, default=150)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--max-outer-iter", type=int, default=200)
    parser.add_argument("--n4000-max-outer-iter", type=int, default=100)
    parser.add_argument("--n8000-max-outer-iter", type=int, default=50)
    parser.add_argument("--max-inner-iter", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/semidde_fixed_d2_d3_r100"),
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replicates < 1 or any(value < 1 for value in args.sample_sizes):
        raise SystemExit("replicates and sample sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "replicates.csv"
    summary_path = args.output_dir / "summary.csv"

    existing = _read_csv(raw_path) if args.resume else []
    selected_labels = {SCENARIOS[key]["label"] for key in args.scenarios}
    existing = [
        row
        for row in existing
        if row.get("scenario") in selected_labels
        and int(row.get("N", 0)) in args.sample_sizes
        and int(row.get("replicate", 0)) <= args.replicates
    ]
    # Rows produced before the size-dependent budget was introduced all used
    # the original (small-sample) outer-iteration budget. Backfill it so the
    # final audit table records the actual budget for every replicate.
    for row in existing:
        if not row.get("max_outer_iter_budget"):
            row["max_outer_iter_budget"] = args.max_outer_iter
    completed = {
        (row["scenario"], int(row["N"]), int(row["replicate"]))
        for row in existing
    }
    args_dict = {
        "base_seed": args.base_seed,
        "tau": args.tau,
        "epsilon": args.epsilon,
        "max_outer_iter": args.max_outer_iter,
        "n4000_max_outer_iter": args.n4000_max_outer_iter,
        "n8000_max_outer_iter": args.n8000_max_outer_iter,
        "max_inner_iter": args.max_inner_iter,
    }
    tasks = []
    for key in args.scenarios:
        label = SCENARIOS[key]["label"]
        for N in args.sample_sizes:
            for replicate in range(1, args.replicates + 1):
                if (label, N, replicate) not in completed:
                    tasks.append((key, N, replicate, args_dict))

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "sample_sizes": args.sample_sizes,
        "replicates": args.replicates,
        "workers": args.workers,
        "base_seed": args.base_seed,
        "seed_rule": "base_seed + replicate (reset for each D and N)",
        "observed_transform": "Y = exp(X) (log-normal margins, g_mode=1)",
        "max_val": 4,
        "penalty": "N**(1/4) for every layer",
        "tau": args.tau,
        "tol": "K_D/2 (=1 for both scenarios)",
        "C": 1,
        "epsilon": args.epsilon,
        "max_outer_iter_schedule": {
            "N<=2000": args.max_outer_iter,
            "N=4000": args.n4000_max_outer_iter,
            "N>=8000": args.n8000_max_outer_iter,
        },
        "max_inner_iter": args.max_inner_iter,
        "initialization": (
            "oracle-labelled spectral initialization, matching the notebook's "
            "Normal_init(X, K1, K_list, B_true, G_true, epsilon) call"
        ),
        "metric_definitions": {
            "graph_accuracy": "fraction of slope support entries correctly recovered after tau thresholding",
            "parameter_rmse": "sqrt(mean replicate parameter MSE), matching the notebook",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    rows = list(existing)
    if tasks:
        print(f"Scheduling {len(tasks)} fits with {args.workers} workers", flush=True)
        # Reuse worker processes.  Importing the scientific stack afresh for
        # every replicate dominates this particular fixed-width experiment;
        # the fit itself does not retain cross-replicate state.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_fit_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                rows.sort(
                    key=lambda value: (
                        int(value["D"]),
                        int(value["N"]),
                        int(value["replicate"]),
                    )
                )
                _write_csv(raw_path, rows, RAW_FIELDS)
                print(
                    f"[{index}/{len(tasks)}] {row['scenario']} N={row['N']} "
                    f"rep={row['replicate']} status={row['status']} "
                    f"seconds={float(row['seconds']):.1f}",
                    flush=True,
                )
    elif not raw_path.exists():
        _write_csv(raw_path, rows, RAW_FIELDS)

    # Rewrite even on a no-op resume so legacy rows receive newly introduced
    # provenance columns such as max_outer_iter_budget.
    _write_csv(raw_path, rows, RAW_FIELDS)
    summaries = summarize(
        rows, args.scenarios, args.sample_sizes, args.replicates
    )
    _write_csv(summary_path, summaries, SUMMARY_FIELDS)
    actual_budgets = {}
    for row in rows:
        key = f"D={int(row['D'])},N={int(row['N'])}"
        actual_budgets.setdefault(key, set()).add(
            int(float(row["max_outer_iter_budget"]))
        )
    metadata["actual_max_outer_iter_budgets"] = {
        key: sorted(values) for key, values in sorted(actual_budgets.items())
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    failures = sum(row["status"] != "ok" for row in rows)
    print(
        f"Finished {len(rows)} rows with {failures} failures; summary: {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
