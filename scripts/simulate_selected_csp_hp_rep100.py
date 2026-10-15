"""Replicate CSP recovery under each setting's previously selected HP.

The 50 selected candidates are frozen from Results/hp_selection_validation/fits.csv.
Each replicate regenerates independent data under the same simulation scenario
and calls the public ``dde.fit`` API once, without another grid search.
Cell JSON checkpoints make an interrupted 5,000-fit run resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dde import fit
from dde.model_selection import (
    _resolve_copula_defaults,
    _resolve_dde_defaults,
    _resolve_semidde_csp_defaults,
    _with_copula_method_defaults,
)

from evaluate_hp_selection_validation import _data


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Results" / "hp_selection_validation" / "fits.csv"
DEFAULT_OUTPUT = ROOT / "Results" / "hp_selection_validation" / "csp_selected_hp_rep100"
KERNELS = ("dde_normal_csp", "dde_bernoulli_csp", "dde_poisson_csp",
           "semidde_csp", "copula_csp")
DEPTHS = (2, 3)
SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_candidates(source: Path) -> dict[tuple[str, int, int], dict]:
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        kernel = row["kernel"]
        if kernel not in KERNELS:
            continue
        key = (kernel, int(row["D"]), int(row["N"]))
        if key in selected or row["status"] != "ok":
            raise ValueError(f"Missing or duplicate successful selected candidate: {key}")
        hp = json.loads(row["selected_hyperparameters"])
        if set(hp) != {"temp", "tau", "t_spike"} or len(hp["t_spike"]) != key[1]:
            raise ValueError(f"Unexpected selected CSP candidate: {key}: {hp}")
        if not all(np.isfinite(float(value)) for value in (hp["temp"], hp["tau"], *hp["t_spike"])):
            raise ValueError(f"Non-finite selected candidate: {key}")
        selected[key] = {
            "hp": hp, "selection_seed": int(row["seed"]),
            "selection_fitted_K": json.loads(row["fitted_K"]),
            "selection_best_index": int(row["best_index"]),
        }
    expected = {(k, d, n) for k in KERNELS for d in DEPTHS for n in SAMPLE_SIZES}
    if set(selected) != expected:
        raise ValueError(f"Selected-HP grid mismatch: missing={expected-set(selected)}, extra={set(selected)-expected}")
    return selected


def _seed(kernel: str, depth: int, sample_size: int, replicate: int, base: int) -> int:
    return int(base + KERNELS.index(kernel) * 100_000_000
               + depth * 10_000_000 + sample_size * 100 + replicate)


def _fit_kwargs(X: np.ndarray, kernel: str, depth: int, hp: dict, seed: int,
                *, dde_iterations: int, semi_outer_iterations: int,
                semi_inner_iterations: int, copula_iterations: int,
                optimizer_iterations: int, score_mc_samples: int) -> dict:
    if kernel.startswith("dde_"):
        defaults, _ = _resolve_dde_defaults(X, depth, {}, dimension_method="csp")
        controls = {"max_it": dde_iterations, "optimizer_maxiter": optimizer_iterations}
    elif kernel == "semidde_csp":
        defaults, _ = _resolve_semidde_csp_defaults(X, depth)
        controls = {"max_outer_iter": semi_outer_iterations,
                    "max_inner_iter": semi_inner_iterations,
                    "max_enum_bits": 12, "mc_samples": score_mc_samples,
                    "optimizer_maxiter": optimizer_iterations}
    else:
        defaults, _ = _resolve_copula_defaults(X, depth, {})
        controls = {"it": copula_iterations, "optimizer_maxiter": optimizer_iterations}
    kwargs = {**defaults, **controls, **hp,
              "dimension_method": "csp", "random_state": seed}
    if kernel == "copula_csp":
        kwargs = _with_copula_method_defaults(kwargs, depth)
    return kwargs


def _task_filename(kernel: str, depth: int, sample_size: int, replicate: int) -> str:
    return f"{kernel}_d{depth}_n{sample_size}_r{replicate:03d}.json"


def _run_task(task: tuple, config: dict) -> dict:
    kernel, depth, sample_size, replicate, hp = task
    seed = _seed(kernel, depth, sample_size, replicate, config["seed"])
    row = {"kernel": kernel, "D": depth, "N": sample_size,
           "replicate": replicate, "seed": seed, "hp": hp,
           "status": "error", "error": ""}
    started = time.perf_counter()
    try:
        X, truth = _data(kernel, depth, sample_size, seed)
        true_K = tuple(map(int, truth["K_true"]))
        row.update(J=int(X.shape[1]), scenario=truth["name"], true_K=list(true_K))
        # Some legacy kernels use NumPy's global RNG in addition to the
        # random_state argument. Resetting per task makes scheduling immaterial.
        np.random.seed(seed)
        with threadpool_limits(limits=1):
            kwargs = _fit_kwargs(
                X, kernel, depth, hp, seed,
                dde_iterations=config["dde_iterations"],
                semi_outer_iterations=config["semi_outer_iterations"],
                semi_inner_iterations=config["semi_inner_iterations"],
                copula_iterations=config["copula_iterations"],
                optimizer_iterations=config["optimizer_iterations"],
                score_mc_samples=config["score_mc_samples"],
            )
            model = "DDE" if kernel.startswith("dde_") else (
                "SemiDDE" if kernel == "semidde_csp" else "CopDDE")
            family = kernel.split("_")[1].title() if model == "DDE" else None
            result = fit(X, method=model, family=family, D=depth, **kwargs)
        fitted_K = tuple(map(int, result.K_list))
        if len(fitted_K) != depth or any(width < 0 for width in fitted_K):
            raise ValueError(f"Invalid fitted widths: {fitted_K}")
        differences = [a - b for a, b in zip(fitted_K, true_K)]
        row.update(
            status="ok", fitted_K=list(fitted_K),
            exact=int(fitted_K == true_K),
            dimension_l1=int(sum(abs(value) for value in differences)),
            under_width=int(sum(max(-value, 0) for value in differences)),
            over_width=int(sum(max(value, 0) for value in differences)),
            iterations=result.diagnostics.get("iterations"),
            outer_iterations=result.diagnostics.get("outer_iterations"),
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = float(time.perf_counter() - started)
    return row


def _load_checkpoints(output: Path) -> dict[tuple[str, int, int, int], dict]:
    rows = {}
    for path in (output / "cells").glob("*.json"):
        row = json.loads(path.read_text())
        key = (row["kernel"], int(row["D"]), int(row["N"]), int(row["replicate"]))
        if key in rows:
            raise ValueError(f"Duplicate checkpoint: {key}")
        rows[key] = row
    return rows


def _snapshot(output: Path, rows: dict, selected: dict, replicates: int) -> None:
    ordered = [rows[key] for key in sorted(rows)]
    path = output / "fits.csv"
    fields = ("kernel", "D", "N", "replicate", "seed", "scenario", "J",
              "true_K", "fitted_K", "exact", "dimension_l1", "under_width",
              "over_width", "iterations", "outer_iterations", "seconds",
              "status", "error", "hp")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: json.dumps(row[field]) if field in {"hp", "true_K", "fitted_K"} and field in row
                             else row.get(field, "") for field in fields})
    os.replace(temporary, path)

    grouped = []
    for (kernel, depth, n), candidate in sorted(selected.items()):
        sample = [rows[(kernel, depth, n, rep)] for rep in range(replicates)
                  if (kernel, depth, n, rep) in rows]
        good = [row for row in sample if row["status"] == "ok"]
        exact = sum(row["exact"] for row in good)
        l1 = [row["dimension_l1"] for row in good]
        durations = [row["seconds"] for row in sample]
        grouped.append({
            "kernel": kernel, "D": depth, "N": n,
            "requested": replicates, "completed": len(sample),
            "successful": len(good), "failed": len(sample) - len(good),
            "exact_count": exact,
            "exact_rate_all": exact / len(sample) if sample else "",
            "exact_rate_success": exact / len(good) if good else "",
            "mean_dimension_l1": float(np.mean(l1)) if l1 else "",
            "mean_seconds": float(np.mean(durations)) if durations else "",
            "p95_seconds": float(np.percentile(durations, 95)) if durations else "",
            "selected_hp": json.dumps(candidate["hp"], sort_keys=True),
        })
    path = output / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    os.replace(temporary, path)
    (output / "progress.json").write_text(json.dumps({
        "completed": len(rows), "requested": len(selected) * replicates,
        "successful": sum(row["status"] == "ok" for row in rows.values()),
        "failed": sum(row["status"] != "ok" for row in rows.values()),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=KERNELS)
    parser.add_argument("--depths", nargs="+", type=int, choices=DEPTHS, default=DEPTHS)
    parser.add_argument("--sample-sizes", nargs="+", type=int, choices=SAMPLE_SIZES, default=SAMPLE_SIZES)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--dde-iterations", type=int, default=50)
    parser.add_argument("--semi-outer-iterations", type=int, default=10)
    parser.add_argument("--semi-inner-iterations", type=int, default=50)
    parser.add_argument("--copula-iterations", type=int, default=10)
    parser.add_argument("--optimizer-iterations", type=int, default=30)
    parser.add_argument("--score-mc-samples", type=int, default=4096)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.replicates < 1 or args.workers < 1:
        parser.error("replicates and workers must be positive")
    source, output = args.source.resolve(), args.output.resolve()
    selected = _selected_candidates(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cells").mkdir(exist_ok=True)
    config = {key: list(value) if key in {"kernels", "depths", "sample_sizes"} else value
              for key, value in vars(args).items()
              if key not in {"source", "output", "retry_errors"}}
    config.update(source=str(source), source_sha256=_digest(source), output=str(output),
                  scenario_code_sha256=_digest(Path(__file__).with_name("evaluate_hp_selection_validation.py")))
    config_path = output / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old != config:
            raise ValueError("Existing checkpoint config differs; use a new output directory")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    selected_path = output / "selected_hp.json"
    frozen = [{"kernel": k, "D": d, "N": n, **selected[k, d, n]}
              for k, d, n in sorted(selected)]
    if selected_path.exists():
        if json.loads(selected_path.read_text()) != frozen:
            raise ValueError("Existing selected HP snapshot differs")
    else:
        selected_path.write_text(json.dumps(frozen, indent=2))

    rows = _load_checkpoints(output)
    tasks = []
    for kernel in args.kernels:
        for depth in args.depths:
            for n in args.sample_sizes:
                for rep in range(args.replicates):
                    key = (kernel, depth, n, rep)
                    if key not in rows or (args.retry_errors and rows[key]["status"] != "ok"):
                        tasks.append((kernel, depth, n, rep, selected[kernel, depth, n]["hp"]))
    # Start longer D=3 and larger-N fits first to reduce the final tail.
    tasks.sort(key=lambda task: (task[1], task[2],
                                 KERNELS.index(task[0])), reverse=True)
    print(f"Pending {len(tasks)} fits; checkpointed {len(rows)}", flush=True)
    if not tasks:
        _snapshot(output, rows, selected, args.replicates)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_task, task, config): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            key = (row["kernel"], row["D"], row["N"], row["replicate"])
            path = output / "cells" / _task_filename(*key)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(row, indent=2))
            os.replace(temporary, path)
            rows[key] = row
            if index % 25 == 0 or index == len(tasks):
                _snapshot(output, rows, selected, args.replicates)
                print(f"Completed {len(rows)} / {len(selected)*args.replicates}; "
                      f"errors {sum(r['status'] != 'ok' for r in rows.values())}", flush=True)


if __name__ == "__main__":
    main()
