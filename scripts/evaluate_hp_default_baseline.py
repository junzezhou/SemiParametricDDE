"""Default-hyperparameter baseline for the fixed-path cells of an HP sweep.

For every successful fixed-path cell of ``<sweep>/fits.csv`` this regenerates
the cell's data (same kernel, depth, N and seed) and fits the model once with
the package defaults that ``select_hyperparameters`` would otherwise tune over
(``pen_list = N**(2/8)`` on every layer, ``tau = 0.3``; CopDDE
``tau = max(0.3, 3 N^-0.3)``), using the sweep's own iteration budgets.  The
aligned raw and thresholded parameter RMSE are written to
``<sweep>/default_baseline.csv`` so the selected hyperparameters can be
compared with the untuned defaults on identical data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from dde import select_hyperparameters
from dde.utils.hyperparameter_selection import copula_tau_default

from evaluate_hp_selection_validation import _aligned_mse, _data, _fit_kwargs

FIELDS = ["kernel", "D", "N", "replicate", "seed", "status", "pen_exponent", "tau",
          "parameter_rmse_raw", "parameter_rmse_thresholded", "seconds", "error"]


def run_cell(task):
    kernel, depth, N, replicate, seed, config = task
    args = SimpleNamespace(**config)
    method = "DDE" if kernel.startswith("dde_") else ("SemiDDE" if kernel.startswith("semidde") else "CopDDE")
    family = kernel.split("_")[1].title() if method == "DDE" else None
    row = {"kernel": kernel, "D": depth, "N": N, "replicate": replicate, "seed": seed,
           "status": "error", "pen_exponent": 0.25, "error": ""}
    started = time.perf_counter()
    try:
        X, truth = _data(kernel, depth, N, seed)
        K_true = tuple(map(int, truth["K_true"]))
        tau = copula_tau_default(N) if method == "CopDDE" else 0.3
        candidate = {"pen_list": [float(N ** 0.25)] * depth, "tau": tau}
        np.random.seed(seed)
        with threadpool_limits(limits=1):
            result = select_hyperparameters(
                X, D=depth, method=method, family=family, dimension_method="fixed",
                criterion="pmse" if method == "CopDDE" else "ebic",
                fixed_hp={"K_list": K_true}, candidates=[candidate],
                base_fit_kwargs=_fit_kwargs(kernel, args), random_state=seed,
                n_splits=args.pmse_folds, n_repeats=args.pmse_repeats,
                pmse_n_estimators=args.pmse_trees, n_jobs=1, refit=True)
        mse_raw, mse_thr = _aligned_mse(result, truth, float(tau), copula=method == "CopDDE")
        row.update(status="ok", tau=float(tau), parameter_rmse_raw=float(np.sqrt(mse_raw)),
                   parameter_rmse_thresholded=float(np.sqrt(mse_thr)))
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--kernels", nargs="+", default=["dde_normal_fixed", "dde_bernoulli_fixed", "dde_poisson_fixed", "semidde_fixed"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="only the first n pending cells (smoke test)")
    args = parser.parse_args()
    config = json.loads((args.sweep / "config.json").read_text())
    out_path = args.sweep / "default_baseline.csv"
    done = {}
    if out_path.exists():
        with out_path.open() as f:
            for r in csv.DictReader(f):
                if r["status"] == "ok":
                    done[(r["kernel"], int(r["D"]), int(r["N"]), int(r["replicate"]))] = r
    with (args.sweep / "fits.csv").open() as f:
        cells = [r for r in csv.DictReader(f) if r["status"] == "ok" and r["kernel"] in args.kernels]
    tasks = [(r["kernel"], int(r["D"]), int(r["N"]), int(r["replicate"]), int(r["seed"]), config)
             for r in cells if (r["kernel"], int(r["D"]), int(r["N"]), int(r["replicate"])) not in done]
    tasks.sort(key=lambda t: (t[1], t[2]))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(done)} cached, {len(tasks)} to run", flush=True)
    rows = list(done.values())

    def store(row):
        rows.append(row)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, restval=""); w.writeheader(); w.writerows(rows)
        print(f"{row['kernel']} D={row['D']} N={row['N']} r={row['replicate']}: {row['status']} rmse_thr={row.get('parameter_rmse_thresholded', '')} {row['seconds']:.1f}s {row['error']}", flush=True)

    if args.workers <= 1:
        for t in tasks:
            store(run_cell(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(run_cell, t) for t in tasks]):
                store(fut.result())


if __name__ == "__main__":
    main()
