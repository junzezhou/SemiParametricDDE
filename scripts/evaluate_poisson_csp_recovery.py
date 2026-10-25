"""Evaluate Poisson-DDE CSP dimension recovery under maximum-width starts.

The generating coefficients follow the sparse construction used by the
MATLAB Poisson simulation. Extra unit-rate observed coordinates make the
identifiable CSP initialization genuinely wider than the true hierarchy:

* D=2: true (6, 2), maximum (9, 3), J=27;
* D=3: true (18, 6, 2), maximum (21, 7, 2), J=64.
* D=4: true (54, 18, 6, 2), maximum (64, 21, 7, 2), J=192.

The primary outputs are the exact dimension-shrinkage recovery rate and the
mean L1 distance between fitted and true width vectors. Fits are checkpointed
after completion, and rerunning the command resumes unfinished cells.

Fits use the Poisson CSP defaults of ``fit`` (the copula kernel: sequential
mean-field transition, sqrt(N) penalty in every layer, every layer
thresholded, dense BIC deletion test for the deepest layer, 100 iterations
without early stop).  ``--max-iter``, ``--tol`` and ``--optimizer-maxiter``
override the kernel's budget when given.  The MCEM results of earlier runs
are reproducible from the git tag ``csp-mcem-last``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit, gammaln
from threadpoolctl import threadpool_limits

from dde import fit
from dde.algorithms.csp import maximal_identifiable_widths


SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000)
SCENARIO_SPECS = {
    "d2": {
        "name": "D2_MATLAB_sparse_overspecified_J27",
        "D": 2,
        "J": 27,
        "K_true": (6, 2),
        "extra_observed": 9,
        "t_spike": (0.03, 0.05),
        "temperature": 0.9,
    },
    "d3": {
        "name": "D3_MATLAB_sparse_overspecified_J64",
        "D": 3,
        "J": 64,
        "K_true": (18, 6, 2),
        "extra_observed": 10,
        "t_spike": (0.01, 0.03, 0.005),
        "temperature": 0.9,
    },
    "d4": {
        "name": "D4_MATLAB_sparse_overspecified_J192",
        "D": 4,
        "J": 192,
        "K_true": (54, 18, 6, 2),
        "extra_observed": 30,
        "t_spike": (0.005, 0.01, 0.03, 0.005),
        "temperature": 0.9,
    },
}


def _matlab_sparse_coefficients(n_parent: int) -> np.ndarray:
    """Return the sparse 3K-by-K coefficients from the MATLAB simulation."""
    n_parent = int(n_parent)
    if n_parent < 1 or n_parent % 2:
        raise ValueError(
            "The MATLAB construction requires a positive even width."
        )
    slopes = np.zeros((3 * n_parent, n_parent), dtype=float)
    slopes[:n_parent] = 4.0 * np.eye(n_parent)
    slopes[n_parent : 2 * n_parent] = 4.0 * np.eye(n_parent)
    third = 4.0 * np.eye(n_parent)
    half = n_parent // 2
    for row in range(n_parent):
        third[row, (row + half) % n_parent] = -4.0 / 3.0
    slopes[2 * n_parent :] = third
    intercept = np.concatenate(
        [
            np.full(n_parent, -2.0),
            np.full(n_parent, -4.0),
            np.full(n_parent, -2.0),
        ]
    )
    return np.column_stack([intercept, slopes])


def build_scenario(key: str) -> dict:
    """Build one arbitrary-depth Poisson hierarchy and its maximum widths."""
    spec = SCENARIO_SPECS[key]
    widths = spec["K_true"]
    B_true = [_matlab_sparse_coefficients(width) for width in widths]
    extra = int(spec["extra_observed"])
    if extra:
        # Intercept zero and zero slopes give independent Poisson(1) noise.
        B_true[0] = np.row_stack(
            [B_true[0], np.zeros((extra, widths[0] + 1), dtype=float)]
        )
    if B_true[0].shape[0] != spec["J"]:
        raise AssertionError("Scenario observed width is inconsistent.")
    maximum = maximal_identifiable_widths(spec["J"], spec["D"])
    if not any(true < cap for true, cap in zip(widths, maximum)):
        raise AssertionError("A recovery scenario must start overspecified.")
    if any(true > cap for true, cap in zip(widths, maximum)):
        raise AssertionError(
            "True widths must not exceed their maximum widths."
        )
    return {
        **spec,
        "K_max": tuple(int(value) for value in maximum),
        "B_true": B_true,
        "prop": np.full(widths[-1], 0.5, dtype=float),
    }


def simulate(scenario: dict, N: int, seed: int):
    """Simulate binary latent layers and log-link Poisson observations."""
    rng = np.random.default_rng(int(seed))
    A_true = [None] * scenario["D"]
    A_true[-1] = rng.binomial(
        1, scenario["prop"], size=(N, scenario["K_true"][-1])
    ).astype(np.uint8)
    for layer in range(scenario["D"] - 2, -1, -1):
        B = scenario["B_true"][layer + 1]
        eta = B[:, 0][None, :] + A_true[layer + 1] @ B[:, 1:].T
        A_true[layer] = rng.binomial(1, expit(eta)).astype(np.uint8)
    B1 = scenario["B_true"][0]
    eta = B1[:, 0][None, :] + A_true[0] @ B1[:, 1:].T
    X = rng.poisson(np.exp(eta)).astype(float)
    return X, A_true


def _padded_truth_initialization(scenario, A_true, seed):
    """Embed true parameters and states inside maximum-width CSP arrays."""
    rng = np.random.default_rng(int(seed))
    K_true = scenario["K_true"]
    K_max = scenario["K_max"]
    B_true = scenario["B_true"]
    B_start = [np.zeros((scenario["J"], K_max[0] + 1), dtype=float)]
    B_start[0][:, 0] = B_true[0][:, 0]
    B_start[0][:, 1 : K_true[0] + 1] = B_true[0][:, 1:]
    for layer in range(1, scenario["D"]):
        coefficient = np.zeros(
            (K_max[layer - 1], K_max[layer] + 1), dtype=float
        )
        coefficient[: K_true[layer - 1], 0] = B_true[layer][:, 0]
        coefficient[: K_true[layer - 1], 1 : K_true[layer] + 1] = B_true[
            layer
        ][:, 1:]
        B_start.append(coefficient)

    A_start = []
    for truth, true_width, maximum_width in zip(A_true, K_true, K_max):
        state = rng.integers(
            0, 2, size=(truth.shape[0], maximum_width), dtype=np.uint8
        )
        state[:, :true_width] = truth
        A_start.append(state)
    prop_start = np.full(K_max[-1], 0.5, dtype=float)
    prop_start[: K_true[-1]] = scenario["prop"]
    return prop_start, B_start, A_start


def dimension_recovery_metrics(fitted, truth) -> dict:
    """Return the two primary width-selection metrics for one fit."""
    fitted = tuple(int(value) for value in fitted)
    truth = tuple(int(value) for value in truth)
    if len(fitted) != len(truth):
        raise ValueError(
            "Fitted and true dimension vectors must have equal depth."
        )
    shallow_depth = min(2, len(truth))
    metrics = {
        "dimension_exact": int(fitted == truth),
        "dimension_l1": int(
            sum(
                abs(estimate - target)
                for estimate, target in zip(fitted, truth)
            )
        ),
        "shallow_dimension_exact": int(
            fitted[:shallow_depth] == truth[:shallow_depth]
        ),
        "shallow_dimension_l1": int(
            sum(
                abs(estimate - target)
                for estimate, target in zip(
                    fitted[:shallow_depth], truth[:shallow_depth]
                )
            )
        ),
    }
    for layer, (estimate, target) in enumerate(zip(fitted, truth), start=1):
        metrics[f"layer_{layer}_fit"] = estimate
        metrics[f"layer_{layer}_error"] = estimate - target
        metrics[f"layer_{layer}_abs_error"] = abs(estimate - target)
        metrics[f"layer_{layer}_exact"] = int(estimate == target)
    return metrics


def _poisson_fit_metrics(result, X, A_true, scenario):
    fitted = tuple(int(value) for value in result.K_list)
    dimension = dimension_recovery_metrics(fitted, scenario["K_true"])
    B1 = np.asarray(result.B_list[0], dtype=float)
    eta = B1[:, 0][None, :] + result.A_list[0] @ B1[:, 1:].T
    conditional_nll = float(np.mean(np.exp(eta) - X * eta + gammaln(X + 1.0)))
    oracle_B1 = scenario["B_true"][0]
    oracle_eta = oracle_B1[:, 0][None, :] + A_true[0] @ oracle_B1[:, 1:].T
    oracle_nll = float(
        np.mean(np.exp(oracle_eta) - X * oracle_eta + gammaln(X + 1.0))
    )
    return {
        "K_fit": json.dumps(fitted),
        **dimension,
        "underselected": int(
            any(
                value < target
                for value, target in zip(fitted, scenario["K_true"])
            )
        ),
        "overselected": int(
            any(
                value > target
                for value, target in zip(fitted, scenario["K_true"])
            )
        ),
        "conditional_nll": conditional_nll,
        "oracle_conditional_nll": oracle_nll,
    }


def _seed_pair(base_seed, scenario_key, N, replicate):
    depth_offset = {
        "d2": 20_000_000,
        "d3": 30_000_000,
        "d4": 40_000_000,
    }[scenario_key]
    data_seed = int(base_seed + depth_offset + 1000 * N + replicate)
    return data_seed, data_seed + 500_000_000


def run_one(task):
    scenario_key, N, replicate, settings = task
    scenario = build_scenario(scenario_key)
    data_seed, fit_seed = _seed_pair(
        settings["base_seed"], scenario_key, N, replicate
    )
    common = {
        "scenario": scenario["name"],
        "D": scenario["D"],
        "J": scenario["J"],
        "K_true": json.dumps(scenario["K_true"]),
        "K_max": json.dumps(scenario["K_max"]),
        "N": int(N),
        "replicate": int(replicate),
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        "settings_id": settings["settings_id"],
    }
    started = time.perf_counter()
    try:
        X, A_true = simulate(scenario, N, data_seed)
        initialization_arguments = {}
        if settings["initialization"] == "truth":
            prop, B_list, A_list = _padded_truth_initialization(
                scenario, A_true, data_seed + 101
            )
            initialization_arguments = {
                "prop_in": prop,
                "B_list_in": B_list,
                "A_list_in": A_list,
            }
        overrides = {
            name: settings[key]
            for name, key in (("max_it", "max_iter"), ("tol", "tol"), ("optimizer_maxiter", "optimizer_maxiter"))
            if settings[key] is not None
        }
        with threadpool_limits(limits=1):
            result = fit(
                X,
                method="DDE",
                family="Poisson",
                D=scenario["D"],
                dimension_method="csp",
                t_spike=(
                    np.asarray(scenario["t_spike"])
                    * settings["t_spike_multiplier"]
                ),
                temp=scenario["temperature"],
                tau=settings["tau"],
                C=settings["C"],
                random_state=fit_seed,
                verbose=False,
                **overrides,
                **initialization_arguments,
            )
        metrics = _poisson_fit_metrics(result, X, A_true, scenario)
        finite = bool(
            np.all(np.isfinite(result.prop))
            and all(np.all(np.isfinite(B)) for B in result.B_list)
            and np.isfinite(metrics["conditional_nll"])
        )
        return {
            **common,
            "status": "ok" if finite else "nonfinite",
            "error": "",
            "finite": finite,
            "iterations": int(result.diagnostics["iterations"]),
            "hit_iteration_limit": int(
                result.diagnostics["iterations"] >= result.diagnostics["max_iterations"]
            ),
            "stopped_by_convergence": int(
                result.diagnostics["stopped_by_convergence"]
            ),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
    except Exception as exc:
        return {
            **common,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "finite": False,
            "seconds": time.perf_counter() - started,
        }


def _write_csv(path: Path, rows: list[dict]):
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.mean(values)) if values else float("nan")


def summarize(rows):
    """Aggregate exact-recovery rates and mean dimension L1 by design cell."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], int(row["N"]))].append(row)
    summary = []
    for (scenario, N), group in sorted(groups.items()):
        successful = [row for row in group if row["status"] == "ok"]
        width_counts = defaultdict(int)
        for row in successful:
            width_counts[row["K_fit"]] += 1
        summary.append(
            {
                "scenario": scenario,
                "D": int(group[0]["D"]),
                "N": N,
                "fits": len(group),
                "successful_fits": len(successful),
                "failed_fits": len(group) - len(successful),
                "dimension_shrinkage_accuracy": _mean(
                    successful, "dimension_exact"
                ),
                "mean_dimension_l1": _mean(successful, "dimension_l1"),
                "shallow_dimension_accuracy": _mean(
                    successful, "shallow_dimension_exact"
                ),
                "shallow_mean_dimension_l1": _mean(
                    successful, "shallow_dimension_l1"
                ),
                "underselected_rate": _mean(successful, "underselected"),
                "overselected_rate": _mean(successful, "overselected"),
                "conditional_nll_mean": _mean(successful, "conditional_nll"),
                "oracle_conditional_nll_mean": _mean(
                    successful, "oracle_conditional_nll"
                ),
                "iterations_mean": _mean(successful, "iterations"),
                "iteration_limit_rate": _mean(
                    successful, "hit_iteration_limit"
                ),
                "converged_rate": _mean(
                    successful, "stopped_by_convergence"
                ),
                "seconds_mean": _mean(successful, "seconds"),
                "K_fit_counts": json.dumps(width_counts, sort_keys=True),
            }
        )
        for layer in range(1, int(group[0]["D"]) + 1):
            summary[-1][f"layer_{layer}_mean_fit"] = _mean(
                successful, f"layer_{layer}_fit"
            )
            summary[-1][f"layer_{layer}_mean_error"] = _mean(
                successful, f"layer_{layer}_error"
            )
            summary[-1][f"layer_{layer}_mae"] = _mean(
                successful, f"layer_{layer}_abs_error"
            )
            summary[-1][f"layer_{layer}_exact_rate"] = _mean(
                successful, f"layer_{layer}_exact"
            )
    return summary


def _write_report(path: Path, summary: list[dict], settings: dict):
    lines = [
        "# Poisson CSP dimension-recovery test",
        "",
        "| Scenario | N | OK | Shallow exact | Shallow L1 | All exact | All L1 | K1 exact | K2 exact | Under | Over | Iter. | Limit | Conv. | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {scenario} | {N} | {successful_fits}/{fits} | "
            "{shallow_dimension_accuracy:.3f} | "
            "{shallow_mean_dimension_l1:.3f} | "
            "{dimension_shrinkage_accuracy:.3f} | {mean_dimension_l1:.3f} | "
            "{layer_1_exact_rate:.3f} | {layer_2_exact_rate:.3f} | "
            "{underselected_rate:.3f} | {overselected_rate:.3f} | "
            "{iterations_mean:.1f} | {iteration_limit_rate:.3f} | "
            "{converged_rate:.3f} | {seconds_mean:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The exact-shrinkage metric equals one only when every fitted layer "
            "width equals its generating width. Dimension L1 is the sum of the "
            "absolute layer-wise width errors. Shallow metrics use only the "
            "first two latent layers.",
            "",
            "## Settings",
            "",
            "```json",
            json.dumps(settings, indent=2),
            "```",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIO_SPECS, default=["d2", "d3"]
    )
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int, default=list(SAMPLE_SIZES)
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1)
    )
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=None, help="kernel default when omitted (100)")
    parser.add_argument("--optimizer-maxiter", type=int, default=None, help="kernel default when omitted (15)")
    parser.add_argument("--tol", type=float, default=None, help="kernel default when omitted (0: no early stop)")
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--t-spike-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--initialization", choices=("data", "truth"), default="data"
    )
    parser.add_argument("--base-seed", type=int, default=20260916)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/poisson_csp_recovery"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replicates < 1 or args.workers < 1 or args.C < 1:
        raise ValueError("replicates, workers, and C must be positive.")
    if any(value is not None and value < 1 for value in (args.max_iter, args.optimizer_maxiter)):
        raise ValueError("iteration budgets must be positive.")
    if any(N < 1 for N in args.sample_sizes):
        raise ValueError("sample sizes must be positive.")
    if not np.isfinite(args.t_spike_multiplier) or args.t_spike_multiplier <= 0:
        raise ValueError("t_spike_multiplier must be finite and positive.")

    settings = {
        "backend": "copula",
        "C": args.C,
        "max_iter": args.max_iter,
        "optimizer_maxiter": args.optimizer_maxiter,
        "tol": args.tol,
        "tau": args.tau,
        "t_spike_multiplier": args.t_spike_multiplier,
        "initialization": args.initialization,
        "base_seed": args.base_seed,
    }
    settings["settings_id"] = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = output_dir / "fits.csv"
    rows = _read_csv(fits_path) if args.resume else []
    existing_settings = {row.get("settings_id", "") for row in rows}
    if rows and existing_settings != {settings["settings_id"]}:
        raise ValueError(
            "The output directory contains fits from different settings; "
            "use a new --output-dir or rerun with --no-resume."
        )
    for row in rows:
        if row.get("status") == "ok" and row.get("K_fit"):
            row.update(
                dimension_recovery_metrics(
                    json.loads(row["K_fit"]), json.loads(row["K_true"])
                )
            )
    completed = {
        (row["scenario"], int(row["N"]), int(row["replicate"])) for row in rows
    }
    tasks = []
    for key in args.scenarios:
        scenario = build_scenario(key)
        for N in args.sample_sizes:
            for replicate in range(1, args.replicates + 1):
                identity = (scenario["name"], int(N), replicate)
                if identity not in completed:
                    tasks.append((key, int(N), replicate, settings))

    print(
        f"Poisson CSP recovery: {len(tasks)} pending; workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(
            max_workers=args.workers, max_tasks_per_child=1
        ) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                rows.sort(
                    key=lambda value: (
                        value["scenario"],
                        int(value["N"]),
                        int(value["replicate"]),
                    )
                )
                _write_csv(fits_path, rows)
                print(
                    f"[{index}/{len(tasks)}] {row['scenario']} N={row['N']} "
                    f"rep={row['replicate']} status={row['status']} "
                    f"K={row.get('K_fit', '')} sec={float(row['seconds']):.2f}",
                    flush=True,
                )

    summary = summarize(rows)
    _write_csv(output_dir / "summary.csv", summary)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sample_sizes": args.sample_sizes,
        "replicates": args.replicates,
        "workers": args.workers,
        "settings": settings,
        "completed_rows": len(rows),
        "successful_rows": sum(row["status"] == "ok" for row in rows),
        "failed_rows": sum(row["status"] != "ok" for row in rows),
        "wall_seconds_this_invocation": time.perf_counter() - started,
    }
    metadata_path = output_dir / "metadata.json"
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    _write_report(output_dir / "REPORT.md", summary, settings)
    print(f"results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
