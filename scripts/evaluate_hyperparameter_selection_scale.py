"""Benchmark full-grid hyperparameter selection across all ten public kernels.

The benchmark covers N=(500, 1000, 2000, 4000, 8000), D=(2, 3), and:

* Normal, Bernoulli, and Poisson DDE with fixed and CSP dimensions;
* SemiDDE with fixed and CSP dimensions; and
* CopDDE with fixed and CSP dimensions.

Every task calls :func:`dde.select_hyperparameters`. Results are checkpointed
after each task so an interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dde import select_hyperparameters

# Reuse the package's established recovery designs rather than introducing a
# separate benchmark-only data-generating process.
from evaluate_bernoulli_csp_recovery import (
    build_scenario as build_bernoulli_scenario,
    calculate_metrics as bernoulli_metrics,
    simulate as simulate_bernoulli,
)
from normal_csp_scenarios import (
    _scenario_d2 as normal_scenario_d2,
    _scenario_d3 as normal_scenario_d3,
    _simulate as simulate_normal,
)
from evaluate_poisson_csp_recovery import (
    _poisson_fit_metrics as poisson_metrics,
    build_scenario as build_poisson_scenario,
    simulate as simulate_poisson,
)
from evaluate_semidde_csp_recovery import (
    _scenario as build_semidde_scenario,
    _simulate_semidde as simulate_semidde,
)


SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000)
KERNELS = (
    "dde_normal_fixed", "dde_normal_csp",
    "dde_bernoulli_fixed", "dde_bernoulli_csp",
    "dde_poisson_fixed", "dde_poisson_csp",
    "semidde_fixed", "semidde_csp",
    "copula_fixed", "copula_csp",
)
FIELDS = (
    "kernel", "method", "family", "dimension_method", "criterion",
    "D", "J", "N", "replicate", "seed", "status", "error",
    "true_K", "max_K", "fitted_K", "dimension_exact", "dimension_l1",
    "underselected", "overselected", "candidate_count", "best_index",
    "best_score", "best_score_sd", "selected_pen_list", "selected_t_spike",
    "selected_tau", "selected_temp", "selected_lambda_exp",
    "search_seconds", "candidate_fit_seconds", "candidate_score_seconds",
    "selected_candidate_seconds", "iterations", "finite",
    "graph_accuracy", "predictive_metric", "predictive_metric_name",
)


class TaskTimeoutError(TimeoutError):
    """Raised when one full-grid benchmark cell exceeds its wall-time budget."""


def _raise_task_timeout(signum, frame):
    del signum, frame
    raise TaskTimeoutError("full-grid search exceeded the per-task wall-time budget")


def _scenario_and_data(kernel: str, depth: int, N: int, seed: int):
    key = f"d{depth}"
    if kernel.startswith("dde_normal"):
        scenario = normal_scenario_d2() if depth == 2 else normal_scenario_d3()
        observed, latent = simulate_normal(scenario, N, seed)
    elif kernel.startswith("dde_bernoulli"):
        scenario = build_bernoulli_scenario(key)
        observed, latent = simulate_bernoulli(scenario, N, seed)
    elif kernel.startswith("dde_poisson"):
        scenario = build_poisson_scenario(key)
        observed, latent = simulate_poisson(scenario, N, seed)
    else:
        scenario = build_semidde_scenario(key)
        observed = simulate_semidde(scenario, N, seed)
        latent = None
    return scenario, np.asarray(observed), latent


def _kernel_spec(kernel: str):
    if kernel.startswith("dde_normal"):
        return "DDE", "Normal"
    if kernel.startswith("dde_bernoulli"):
        return "DDE", "Bernoulli"
    if kernel.startswith("dde_poisson"):
        return "DDE", "Poisson"
    if kernel.startswith("semidde"):
        return "SemiDDE", None
    return "CopDDE", None


def _base_fit_kwargs(kernel: str, args):
    common = {"C": 1, "verbose": False}
    if kernel.startswith("dde_"):
        common.update(
            max_iter=args.dde_iterations,
            optimizer_maxiter=args.optimizer_iterations,
        )
    elif kernel.startswith("semidde"):
        common.update(
            max_iter=args.semi_inner_iterations,
            max_outer_iter=args.semi_outer_iterations,
            max_enum_bits=12,
            mc_samples=args.score_mc_samples,
        )
        if kernel.endswith("csp"):
            common["optimizer_maxiter"] = args.optimizer_iterations
    else:
        common.update(
            max_iter=args.copula_iterations,
            optimizer_maxiter=args.optimizer_iterations,
        )
        if kernel.endswith("fixed"):
            common["tol"] = 0.005
    return common


def _json_value(value):
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _predictive_metrics(kernel, result, observed, latent, scenario, tau):
    graph_accuracy = ""
    metric = ""
    metric_name = ""
    if kernel.startswith("dde_bernoulli"):
        values = bernoulli_metrics(result, observed, latent, scenario, tau)
        graph_accuracy = values["graph_accuracy"]
        metric = values["conditional_log_loss"]
        metric_name = "conditional_log_loss"
    elif kernel.startswith("dde_poisson"):
        values = poisson_metrics(result, observed, latent, scenario)
        metric = values["conditional_nll"]
        metric_name = "conditional_nll"
    elif kernel.startswith("dde_normal"):
        B = np.asarray(result.B_list[0], dtype=float)
        mean = np.column_stack([np.ones(observed.shape[0]), result.A_list[0]]) @ B.T
        metric = float(np.sqrt(np.mean((observed - mean) ** 2)))
        metric_name = "conditional_rmse"
    elif kernel.startswith("copula"):
        metric = -float(result.diagnostics["cv_table"][result.diagnostics["hyperparameter_selection"].best_index]["mean_score"])
        metric_name = "pmse"
    return graph_accuracy, metric, metric_name


def run_task(kernel: str, depth: int, N: int, replicate: int, args):
    seed = int(args.seed + 10_000_000 * depth + 10_000 * N + replicate)
    method, family = _kernel_spec(kernel)
    dimension_method = "fixed" if kernel.endswith("fixed") else "csp"
    criterion = "pmse" if method == "CopDDE" else "ebic"
    scenario, observed, latent = _scenario_and_data(kernel, depth, N, seed)
    true_K = tuple(int(value) for value in scenario["K_true"])
    max_K = tuple(int(value) for value in scenario["K_max"])
    fixed_hp = {"K_list": list(true_K)} if dimension_method == "fixed" else None
    started = time.perf_counter()
    previous_alarm_handler = None
    if args.task_timeout_seconds > 0:
        previous_alarm_handler = signal.signal(signal.SIGALRM, _raise_task_timeout)
        signal.setitimer(signal.ITIMER_REAL, args.task_timeout_seconds)
    try:
        with threadpool_limits(limits=1):
            result = select_hyperparameters(
                observed,
                D=depth,
                method=method,
                family=family,
                dimension_method=dimension_method,
                criterion=criterion,
                fixed_hp=fixed_hp,
                base_fit_kwargs=_base_fit_kwargs(kernel, args),
                random_state=seed,
                n_splits=args.pmse_folds,
                n_repeats=args.pmse_repeats,
                pmse_n_estimators=args.pmse_trees,
                n_jobs=args.pmse_jobs,
                refit=True,
            )
        elapsed = time.perf_counter() - started
        diagnostics = result.diagnostics
        selection = diagnostics["hyperparameter_selection"]
        rows = diagnostics["cv_table"]
        best = rows[selection.best_index]
        fitted_K = tuple(int(value) for value in result.K_list)
        l1 = int(sum(abs(a - b) for a, b in zip(fitted_K, true_K)))
        finite = bool(
            np.isfinite(best["mean_score"])
            and all(np.all(np.isfinite(B)) for B in result.B_list)
        )
        graph_accuracy, metric, metric_name = _predictive_metrics(
            kernel, result, observed, latent, scenario, float(best.get("tau", 0.0))
        )
        timing = diagnostics.get("timing_table", [])
        row = {
            "kernel": kernel, "method": method, "family": family or "",
            "dimension_method": dimension_method, "criterion": criterion,
            "D": depth, "J": observed.shape[1], "N": N,
            "replicate": replicate, "seed": seed, "status": "ok", "error": "",
            "true_K": _json_value(true_K), "max_K": _json_value(max_K),
            "fitted_K": _json_value(fitted_K), "dimension_exact": int(fitted_K == true_K),
            "dimension_l1": l1,
            "underselected": int(any(a < b for a, b in zip(fitted_K, true_K))),
            "overselected": int(any(a > b for a, b in zip(fitted_K, true_K))),
            "candidate_count": len(rows), "best_index": selection.best_index,
            "best_score": best["mean_score"], "best_score_sd": best["std_score"],
            "selected_pen_list": _json_value(best.get("pen_list")),
            "selected_t_spike": _json_value(best.get("t_spike")),
            "selected_tau": best.get("tau", ""), "selected_temp": best.get("temp", ""),
            "selected_lambda_exp": _json_value(best.get("lambda_exp")),
            "search_seconds": elapsed,
            "candidate_fit_seconds": float(sum(row.get("fit_seconds", 0.0) for row in timing)),
            "candidate_score_seconds": float(sum(row.get("score_seconds", 0.0) for row in timing)),
            "selected_candidate_seconds": best.get("elapsed_seconds", ""),
            "iterations": result.diagnostics.get("iterations", ""),
            "finite": int(finite), "graph_accuracy": graph_accuracy,
            "predictive_metric": metric, "predictive_metric_name": metric_name,
        }
        return row
    except Exception as exc:
        status = "timeout" if isinstance(exc, TaskTimeoutError) else "error"
        return {
            "kernel": kernel, "method": method, "family": family or "",
            "dimension_method": dimension_method, "criterion": criterion,
            "D": depth, "J": observed.shape[1], "N": N,
            "replicate": replicate, "seed": seed, "status": status,
            "error": f"{type(exc).__name__}: {exc}",
            "true_K": _json_value(true_K), "max_K": _json_value(max_K),
            "search_seconds": time.perf_counter() - started,
        }
    finally:
        if args.task_timeout_seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)


def _read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _key(row):
    return row["kernel"], int(row["D"]), int(row["N"]), int(row["replicate"])


def _write_report(path: Path, rows, args):
    successful = [row for row in rows if row.get("status") == "ok"]
    timed_out = [row for row in rows if row.get("status") == "timeout"]
    failed = [row for row in rows if row.get("status") == "error"]
    lines = [
        "# Full-grid hyperparameter-selection scale benchmark", "",
        f"- Tasks completed: {len(rows)}/{len(KERNELS) * 2 * len(SAMPLE_SIZES) * args.replicates}",
        f"- Successful: {len(successful)}; timed out: {len(timed_out)}; failed: {len(failed)}",
        f"- Per-task wall-time limit: {args.task_timeout_seconds:.0f}s (0 disables it)",
        f"- Iteration budgets: DDE={args.dde_iterations}, SemiDDE={args.semi_outer_iterations}x{args.semi_inner_iterations}, CopDDE={args.copula_iterations}",
        f"- CopDDE pMSE: repeats={args.pmse_repeats}, folds={args.pmse_folds}, trees={args.pmse_trees}",
        "", "| Kernel | D | N | candidates | fitted K | exact | score | seconds |", "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted(successful, key=lambda value: _key(value)):
        lines.append(
            f"| {row['kernel']} | {row['D']} | {row['N']} | {row['candidate_count']} | "
            f"{row['fitted_K']} | {row['dimension_exact']} | {float(row['best_score']):.6g} | {float(row['search_seconds']):.2f} |"
        )
    if timed_out:
        lines.extend(["", "## Timed-out full grids", "", "| Kernel | D | N | elapsed lower bound (s) |", "|---|---:|---:|---:|"])
        for row in sorted(timed_out, key=lambda value: _key(value)):
            lines.append(
                f"| {row['kernel']} | {row['D']} | {row['N']} | {float(row['search_seconds']):.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/hyperparameter_selection_scale"))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--depths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=KERNELS)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--dde-iterations", type=int, default=20)
    parser.add_argument("--semi-outer-iterations", type=int, default=2)
    parser.add_argument("--semi-inner-iterations", type=int, default=20)
    parser.add_argument("--copula-iterations", type=int, default=10)
    parser.add_argument("--optimizer-iterations", type=int, default=30)
    parser.add_argument("--score-mc-samples", type=int, default=4096)
    parser.add_argument("--pmse-repeats", type=int, default=2)
    parser.add_argument("--pmse-folds", type=int, default=3)
    parser.add_argument("--pmse-trees", type=int, default=30)
    parser.add_argument("--pmse-jobs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--task-timeout-seconds", type=float, default=0.0,
        help="Mark a full-grid cell as timed out after this many wall seconds; 0 disables the limit.",
    )
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.task_timeout_seconds < 0:
        parser.error("--task-timeout-seconds cannot be negative")

    raw_path = args.output / "fits.csv"
    report_path = args.output / "REPORT.md"
    rows = _read_rows(raw_path)
    completed = {_key(row) for row in rows if row.get("status") in {"ok", "timeout"}}
    tasks = [
        (kernel, depth, N, replicate)
        for kernel in args.kernels for depth in args.depths
        for N in args.sample_sizes for replicate in range(args.replicates)
        if (kernel, depth, N, replicate) not in completed
    ]
    # Cover the previously unmeasured Copula CSP path before returning to the
    # remaining long-running fixed-width cells.  Within a path, smaller cells
    # provide useful completed measurements before the timeout boundary.
    path_priority = {"copula_csp": 0, "semidde_fixed": 1, "copula_fixed": 2}
    tasks.sort(
        key=lambda task: (
            path_priority.get(task[0], 1), task[1], task[2], task[3], task[0],
        )
    )
    print(f"pending tasks: {len(tasks)}", flush=True)
    if args.workers == 1:
        completed_rows = ((task, run_task(*task, args)) for task in tasks)
        for index, (task, row) in enumerate(completed_rows, start=1):
            rows = [existing for existing in rows if _key(existing) != task]
            rows.append(row)
            rows.sort(key=_key)
            _write_rows(raw_path, rows)
            _write_report(report_path, rows, args)
            print(
                f"[{index}/{len(tasks)}] {task}: {row['status']} "
                f"({float(row['search_seconds']):.2f}s)", flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task, *task, args): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "kernel": task[0], "D": task[1], "N": task[2],
                        "replicate": task[3], "status": "error",
                        "error": f"worker {type(exc).__name__}: {exc}",
                        "search_seconds": 0.0,
                    }
                rows = [existing for existing in rows if _key(existing) != task]
                rows.append(row)
                rows.sort(key=_key)
                _write_rows(raw_path, rows)
                _write_report(report_path, rows, args)
                print(
                    f"[{index}/{len(tasks)}] {task}: {row['status']} "
                    f"({float(row['search_seconds']):.2f}s)", flush=True,
                )
    _write_report(report_path, rows, args)
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
