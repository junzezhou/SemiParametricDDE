"""Checkpointed, full-grid validation of all ten hyperparameter-selection paths.

Each cell calls the public select_hyperparameters API with its default Cartesian
grid.  The iteration budgets are explicit so this screening experiment remains
auditable; they are not claims of paper-equivalent convergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from dde import select_hyperparameters
from dde.simulation import generate_X_Cop_marg_emp_D, generate_X_normal
from dde.utils import thres

from evaluate_bernoulli_csp_recovery import build_scenario as bernoulli_csp_scenario
from evaluate_bernoulli_csp_recovery import simulate as bernoulli_simulate
from evaluate_bernoulli_fixed_recovery import build_scenario as bernoulli_fixed_scenario
from evaluate_bernoulli_fixed_recovery import _paper_sparse_coefficients
from evaluate_bernoulli_fixed_recovery import simulate_bernoulli_dde
from normal_csp_scenarios import _scenario_d2, _scenario_d3, _simulate
from evaluate_poisson_csp_recovery import build_scenario as poisson_csp_scenario
from evaluate_poisson_csp_recovery import simulate as poisson_simulate
from evaluate_semidde_csp_recovery import _scenario as semi_csp_scenario
from evaluate_semidde_csp_recovery import _simulate_semidde
from evaluate_semidde_fixed_recovery import build_scenario as semi_fixed_scenario


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Results" / "hp_selection_validation"
N_VALUES = (500, 1000, 2000, 4000, 8000)
KERNELS = (
    "dde_normal_fixed", "dde_normal_csp",
    "dde_bernoulli_fixed", "dde_bernoulli_csp",
    "dde_poisson_fixed", "dde_poisson_csp",
    "semidde_fixed", "semidde_csp",
    "copula_fixed", "copula_csp",
)


@lru_cache(maxsize=2)
def _normal_paper_truth(depth: int):
    """Equation S.36 sparse loadings and unit Normal dispersion."""
    K = (6, 2) if depth == 2 else (18, 6, 2)
    J = 3 * K[0]
    prop = np.full(K[-1], 0.5)
    B = [_paper_sparse_coefficients(k) for k in K]
    gamma = np.ones(J)
    return K, J, prop, B, gamma


@lru_cache(maxsize=2)
def _semi_fixed_truth(depth: int):
    return semi_fixed_scenario(f"d{depth}")


def _copula_truth(depth: int) -> dict:
    records = ROOT / "Results" / "copdde_MATLAB_reference_results" / "setting_records"
    if depth == 2:
        with h5py.File(records / "Sim_Overspecified_true_settings.mat") as file:
            # J=100 is the setting reported in the current SemiDDE manuscript
            # (true widths 10/3, initial CSP widths 33/11).
            truth = file[file["truth_by_J"][0, 1]]
            B = [np.asarray(truth["B1_true_scale"]).T, np.asarray(truth["B2_true"]).T]
            gamma = np.asarray(truth["gamma_scale"]).ravel()
            lambdas = np.asarray(truth["lambdas"]).ravel()
            prop = np.asarray(truth["prop_true"]).ravel()
    else:
        with h5py.File(records / "Deeper_Sims_true_settings.mat") as file:
            truth = file["truth"]
            B = [np.asarray(truth[key]).T for key in ("B1_true_unscale", "B2_true", "B3_true")]
            gamma = np.asarray(truth["gamma_true"]).ravel()
            lambdas = np.asarray(truth["lambdas"]).ravel()
            prop = np.asarray(truth["prop_true"]).ravel()
    return {
        "name": f"copula_paper_d{depth}_j{len(lambdas)}",
        "K_true": tuple(int(value.shape[1] - 1) for value in B),
        "B_true": B, "gamma": gamma, "prop": prop, "lambdas": lambdas,
    }


def _data(kernel: str, depth: int, N: int, seed: int):
    key = f"d{depth}"
    if kernel == "dde_normal_fixed":
        K, J, prop, B, gamma = _normal_paper_truth(depth)
        np.random.seed(seed)
        X, _ = generate_X_normal(N, prop, B, gamma)
        scenario = {"name": f"normal_paper_d{depth}", "K_true": K,
                    "B_true": B, "prop": prop, "gamma": gamma}
    elif kernel == "dde_normal_csp":
        scenario = _scenario_d2() if depth == 2 else _scenario_d3()
        X, _ = _simulate(scenario, N, seed)
        scenario = {**scenario, "B_true": scenario["B_true"],
                    "gamma": scenario["gamma"]}
    elif kernel == "dde_bernoulli_fixed":
        scenario = bernoulli_fixed_scenario(key)
        X, _ = simulate_bernoulli_dde(scenario, N, seed)
        scenario = {**scenario, "name": scenario["label"],
                    "K_true": scenario["K_list"],
                    "B_true": scenario["B_list"]}
    elif kernel == "dde_bernoulli_csp":
        scenario = bernoulli_csp_scenario(key)
        X, _ = bernoulli_simulate(scenario, N, seed)
    elif kernel == "dde_poisson_fixed":
        K = (6, 2) if depth == 2 else (18, 6, 2)
        scenario = {"name": f"poisson_sparse_d{depth}", "D": depth,
                    "K_true": K,
                    "B_true": [_paper_sparse_coefficients(k) for k in K],
                    "prop": np.full(K[-1], 0.5)}
        X, _ = poisson_simulate(scenario, N, seed)
    elif kernel == "dde_poisson_csp":
        scenario = poisson_csp_scenario(key)
        X, _ = poisson_simulate(scenario, N, seed)
    elif kernel == "semidde_fixed":
        scenario = _semi_fixed_truth(depth)
        np.random.seed(seed)
        latent, _ = generate_X_normal(N, scenario["prop"], scenario["B_list"], scenario["gamma"])
        X = np.exp(latent)
        scenario = {**scenario, "name": scenario["label"],
                    "K_true": scenario["K_list"], "B_true": scenario["B_list"]}
    elif kernel == "semidde_csp":
        scenario = semi_csp_scenario(key)
        X = _simulate_semidde(scenario, N, seed)
        scenario = {**scenario, "name": f"semi_csp_d{depth}",
                    "B_true": scenario["B_list"]}
    else:
        scenario = _copula_truth(depth)
        generated = generate_X_Cop_marg_emp_D(
            N, scenario["prop"], scenario["lambdas"], scenario["gamma"],
            scenario["B_true"], random_state=seed,
        )
        X = generated["X"]
        # The rank likelihood identifies B1 only up to a row-wise affine
        # transformation. Use the generated latent Z to fix that scale.
        Z = generated["Z"]
        center, scale = Z.mean(axis=0), Z.std(axis=0)
        B = [b.copy() for b in scenario["B_true"]]
        B[0][:, 0] = (B[0][:, 0] - center) / scale
        B[0][:, 1:] /= scale[:, None]
        scenario["B_true_standardized"] = B
        scenario["gamma_standardized"] = scenario["gamma"] / scale**2
    return np.asarray(X), scenario


def _aligned_mse(result, truth: dict, tau: float, *, copula: bool) -> tuple[float, float]:
    """Align latent labels; return raw and thresholded all-parameter MSE."""
    target_B = truth.get("B_true_standardized", truth["B_true"])
    target_gamma = truth.get("gamma_standardized", truth.get("gamma"))
    B_raw = [np.asarray(b, float).copy() for b in result.B_list]
    if len(B_raw) != len(target_B) or any(a.shape != b.shape for a, b in zip(B_raw, target_B)):
        raise ValueError("Fixed-width fit and truth parameter shapes differ")
    prop = np.asarray(result.prop, float).copy()
    for layer, target in enumerate(target_B):
        cost = np.sum((target[:, 1:].T[:, None, :] - B_raw[layer][:, 1:].T[None, :, :])**2, axis=2)
        true_idx, fitted_idx = linear_sum_assignment(cost)
        order = fitted_idx[np.argsort(true_idx)]
        B_raw[layer][:, 1:] = B_raw[layer][:, 1:][:, order]
        if layer + 1 < len(B_raw):
            B_raw[layer + 1] = B_raw[layer + 1][order, :]
        else:
            prop = prop[order]

    def mse(B):
        errors = [(prop - np.asarray(truth["prop"]))**2]
        errors.extend((a - np.asarray(b))**2 for a, b in zip(B, target_B))
        if target_gamma is not None:
            errors.append((np.asarray(result.gamma) - np.asarray(target_gamma))**2)
        return float(np.mean(np.concatenate([e.ravel() for e in errors])))

    raw = mse(B_raw)
    thresholded = mse([thres(b, tau) for b in B_raw])
    return raw, thresholded


def _fit_kwargs(kernel: str, args) -> dict:
    common = {"C": 1, "verbose": False}
    if kernel.startswith("dde_"):
        common.update(max_iter=args.dde_iterations)
        if kernel != "dde_normal_csp":
            # Normal CSP resolves its own validated optimizer budget (15
            # L-BFGS-B steps on the deepest layer); the MCEM kernels and the
            # fixed paths take the explicit budget.
            common["optimizer_maxiter"] = args.optimizer_iterations
    elif kernel.startswith("semidde"):
        common.update(max_iter=args.semi_inner_iterations,
                      max_outer_iter=args.semi_outer_iterations,
                      max_enum_bits=12, mc_samples=args.score_mc_samples)
        if kernel.endswith("csp"):
            common["optimizer_maxiter"] = args.optimizer_iterations
    else:
        common.update(max_iter=args.copula_iterations,
                      optimizer_maxiter=args.copula_optimizer_iterations)
        if kernel.endswith("fixed"):
            common["tol"] = 0.005
    return common


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_cell(task, args):
    kernel, depth, N, replicate = task
    seed = int(args.seed + 10_000_000 * depth + 10_000 * N + replicate)
    dimension = "fixed" if kernel.endswith("fixed") else "csp"
    method = "DDE" if kernel.startswith("dde_") else ("SemiDDE" if kernel.startswith("semidde") else "CopDDE")
    family = kernel.split("_")[1].title() if method == "DDE" else None
    criterion = "pmse" if method == "CopDDE" else "ebic"
    basename = f"{kernel}_d{depth}_n{N}_r{replicate}"
    row = {"kernel": kernel, "D": depth, "N": N, "replicate": replicate,
           "seed": seed, "method": method, "family": family or "",
           "dimension_method": dimension, "criterion": criterion, "status": "error"}
    started = time.perf_counter()
    try:
        X, truth = _data(kernel, depth, N, seed)
        K_true = tuple(map(int, truth["K_true"]))
        row.update(J=X.shape[1], scenario=truth["name"], true_K=json.dumps(K_true))
        # The parametric fixed kernels use NumPy's legacy RNG internally.
        # Reset it per cell so process scheduling cannot change the fit.
        np.random.seed(seed)
        with threadpool_limits(limits=1):
            result = select_hyperparameters(
                X, D=depth, method=method, family=family,
                dimension_method=dimension, criterion=criterion,
                fixed_hp={"K_list": K_true} if dimension == "fixed" else None,
                base_fit_kwargs=_fit_kwargs(kernel, args), random_state=seed,
                n_splits=args.pmse_folds, n_repeats=args.pmse_repeats,
                pmse_n_estimators=args.pmse_trees, n_jobs=1, refit=True,
            )
        selection = result.diagnostics["hyperparameter_selection"]
        table = result.diagnostics["cv_table"]
        expected = 18 * 3**(depth - 2) if method == "CopDDE" and dimension == "fixed" else (
            (3**depth if method == "CopDDE" else 2 * 3**depth) if dimension == "csp" else 3 * 2**depth)
        if len(table) != expected or len({r["candidate_index"] for r in table}) != expected:
            raise AssertionError(f"Incomplete Cartesian grid: {len(table)} vs {expected}")
        chosen = table[selection.best_index]
        fitted_K = tuple(map(int, result.K_list))
        row.update(
            status="ok", fitted_K=json.dumps(fitted_K),
            candidate_count=len(table), best_index=selection.best_index,
            best_score=float(chosen["mean_score"]),
            selected_hyperparameters=json.dumps(selection.best_candidate, default=_json_default),
            selected_tau=float(chosen["tau"]),
            dimension_exact=int(fitted_K == K_true),
            dimension_l1=sum(abs(a-b) for a, b in zip(fitted_K, K_true)),
            iterations=result.diagnostics.get("iterations", ""),
            candidate_timed_seconds=float(sum(
                timing["total_seconds"] for timing in result.diagnostics["timing_table"])),
        )
        if dimension == "fixed":
            mse_raw, mse_thresholded = _aligned_mse(
                result, truth, float(chosen["tau"]), copula=method == "CopDDE")
            if not np.isfinite(mse_raw) or not np.isfinite(mse_thresholded):
                raise FloatingPointError("Non-finite overall parameter MSE")
            row.update(parameter_mse_raw=mse_raw,
                       parameter_rmse_raw=float(np.sqrt(mse_raw)),
                       parameter_mse_thresholded=mse_thresholded,
                       parameter_rmse_thresholded=float(np.sqrt(mse_thresholded)))
        cell_dir = args.output / "cells"
        cell_dir.mkdir(parents=True, exist_ok=True)
        diagnostic = {
            "task": task, "seed": seed, "scenario": truth["name"],
            "true_K": K_true, "cv_table": table,
            "fold_scores": result.diagnostics.get("fold_scores"),
            "timing_table": result.diagnostics.get("timing_table"),
            "best_index": selection.best_index,
        }
        temporary = cell_dir / f"{basename}.json.tmp"
        temporary.write_text(json.dumps(diagnostic, indent=2, default=_json_default))
        os.replace(temporary, cell_dir / f"{basename}.json")
        parameters = {"prop": result.prop,
                      **{f"B{index+1}": b for index, b in enumerate(result.B_list)}}
        if result.gamma is not None:
            gamma = np.asarray(result.gamma)
            if gamma.dtype == object and gamma.size == 1 and gamma.item() is None:
                pass  # Bernoulli/Poisson kernels can return a scalar None array.
            else:
                gamma = np.asarray(gamma, dtype=float)
                if not np.isfinite(gamma).all():
                    raise FloatingPointError("Non-finite fitted gamma")
                parameters["gamma"] = gamma
        np.savez_compressed(cell_dir / f"{basename}.npz", **parameters)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - started
    return row


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path, rows):
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=KERNELS)
    parser.add_argument("--depths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=N_VALUES)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--dde-iterations", type=int, default=20)
    parser.add_argument("--semi-outer-iterations", type=int, default=2)
    parser.add_argument("--semi-inner-iterations", type=int, default=20)
    parser.add_argument("--copula-iterations", type=int, default=10)
    parser.add_argument("--optimizer-iterations", type=int, default=30)
    parser.add_argument("--copula-optimizer-iterations", type=int, default=30)
    parser.add_argument("--score-mc-samples", type=int, default=4096)
    parser.add_argument("--pmse-repeats", type=int, default=2)
    parser.add_argument("--pmse-folds", type=int, default=3)
    parser.add_argument("--pmse-trees", type=int, default=30)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key != "output"}
    config["output"] = str(args.output)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    path = args.output / "fits.csv"
    rows = _read(path) if path.exists() else []
    completed = {(r["kernel"], int(r["D"]), int(r["N"]), int(r["replicate"]))
                 for r in rows if r["status"] == "ok" or not args.retry_errors}
    tasks = [(k, d, n, rep) for k in args.kernels for d in args.depths
             for n in args.sample_sizes for rep in range(args.replicates)
             if (k, d, n, rep) not in completed]
    # Launch the longest Copula cells first.  This changes only scheduling,
    # never candidate grids or numerical settings, and avoids a multi-hour
    # tail after the smaller grids have finished.
    copula_weight = {("fixed", 3): 5, ("csp", 3): 3,
                     ("fixed", 2): 2, ("csp", 2): 1}
    tasks.sort(key=lambda task: (
        0 if not task[0].startswith("copula") else 1,
        -task[2] * copula_weight[(task[0].split("_")[-1], task[1])]
        if task[0].startswith("copula") else 0,
    ))
    print(f"Pending {len(tasks)} of {len(args.kernels)*len(args.depths)*len(args.sample_sizes)*args.replicates}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, task, args): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"kernel": task[0], "D": task[1], "N": task[2],
                       "replicate": task[3], "status": "error", "error": f"Worker {exc}"}
            rows = [r for r in rows if (r["kernel"], int(r["D"]), int(r["N"]), int(r["replicate"])) != task]
            rows.append(row)
            rows.sort(key=lambda r: (r["kernel"], int(r["D"]), int(r["N"]), int(r["replicate"])))
            _write(path, rows)
            print(f"[{index}/{len(tasks)}] {task}: {row['status']} ({float(row.get('seconds',0)):.1f}s) {row.get('error','')}", flush=True)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
