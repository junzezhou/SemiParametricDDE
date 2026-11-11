"""Old (RMSE / absolute L2) versus relative ``tol`` for DDE and CopDDE fixed fits.

The package code under test is the one on ``PYTHONPATH``; ``--code`` only
labels the rows.  Every (design, N, replicate) uses the same data and the
same global NumPy seed before the fit in every arm, so arms are paired.

DDE fixed (``--part dde``): Normal ``build_true_params`` designs D2
(K = 6, 2; J = 18) and D3 (K = 18, 6, 2; J = 54), and the Bernoulli and
Poisson ``Sim_sid`` B_s design (K = 6, 2; J = 18).  ``pen_list = N**(1/4)``,
``tau = 0.3``, ``C = 1``, ``max_it = 50``, oracle-labelled initialization
(so layers need no alignment), package defaults otherwise.

CopDDE fixed (``--part copdde``): the CopDDE paper truths ``d2_j100``
(K = 10, 3) and ``d3_j108`` (K = 18, 6, 2) from the MATLAB baseline
settings, ``pen_list = N**(1/4)``, ``tau = max(0.3, 3 N**-0.3)``, ``C = 1``,
``max_it = 50``.  Estimates are aligned to the truth per layer (Hungarian
assignment on the summed coefficients over each true column's positive
support, then a column's 0/1 coding is flipped when that sum is negative);
B1 is standardized by the fitted latent Gaussian (``rescale_B1_D``) as in
``evaluate_copdde_fixed_graph.py``.

    PYTHONPATH=<worktree>/src python scripts/compare_fixed_tol_metric.py \
        --part dde --code new --tols 0.01 0.005 0 --workers 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "DDE-Copula-m" / "Results" / "copdde_baseline" / "settings"
DDE_DESIGNS = ("normal_D2", "normal_D3", "bernoulli_D2", "poisson_D2")
COPDDE_DESIGNS = ("d2_j100", "d3_j108")


def bs_layer(K):
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


def dde_truth_and_data(design, N, seed):
    family, depth = design.split("_")
    if family == "normal":
        from dde.identifiability import build_true_params
        from dde.simulation import generate_X_normal

        K = [6, 2] if depth == "D2" else [18, 6, 2]
        prop, B_list, G_list, gamma = build_true_params(K, 3 * K[0], 4.0)
        prop = np.asarray(prop, float)
        B_list = [np.asarray(B, float) for B in B_list]
        gamma = np.asarray(gamma, float)
        np.random.seed(seed)
        X, _ = generate_X_normal(int(N), prop, B_list, gamma)
        return family, K, prop, B_list, gamma, X
    K = [6, 2]
    rng = np.random.default_rng(seed)
    B_list = [bs_layer(6), bs_layer(2)]
    prop = np.full(2, 0.5)
    A2 = rng.binomial(1, 0.5, size=(N, 2))
    A1 = rng.binomial(1, expit(B_list[1][:, 0] + A2 @ B_list[1][:, 1:].T))
    eta = B_list[0][:, 0] + A1 @ B_list[0][:, 1:].T
    X = rng.binomial(1, expit(eta)) if family == "bernoulli" else rng.poisson(np.exp(eta))
    return family, K, prop, B_list, None, X


def run_dde(task):
    design, N, rep, tol, code = task
    from dde import fit

    seed = 20261109 + 1_000_000 * DDE_DESIGNS.index(design) + 1000 * int(N) + int(rep)
    row = {"part": "dde", "code": code, "design": design, "N": int(N), "rep": int(rep),
           "tol": float(tol), "seed": seed, "status": "ok", "error": ""}
    try:
        family, K, prop_t, B_t, gamma_t, X = dde_truth_and_data(design, N, seed)
        G_t = [np.asarray(B)[:, 1:] != 0 for B in B_t]
        np.random.seed((seed + 1) % 2**32)
        started = time.perf_counter()
        result = fit(X, model="DDE", family=family, D=len(K), dimension_method="fixed", K_list=K,
                     pen_list=[float(N) ** 0.25] * len(K), tau=0.3, C=1, tol=float(tol), max_it=50,
                     initialize="oracle", B_list_true=B_t, G_list_true=G_t, keep_trace=True,
                     verbose=False)
        seconds = time.perf_counter() - started
        trace = result.diagnostics.get("trace") or []
        warm = result.diagnostics.get("warm_start") or {}
        row.update(
            iterations=int(result.diagnostics["iterations"]),
            stage1_iterations=warm.get("stage1_iterations"),
            stage2_iterations=warm.get("stage2_iterations"),
            final_error=float(trace[-1]["error"]) if trace else None,
            layer_rmse=[float(np.sqrt(np.mean((np.asarray(B) - Bt) ** 2))) for B, Bt in zip(result.B_list, B_t)],
            layer_graph_accuracy=[float(np.mean((np.asarray(B)[:, 1:] != 0) == G)) for B, G in zip(result.B_list, G_t)],
            prop_rmse=float(np.sqrt(np.mean((np.asarray(result.prop) - prop_t) ** 2))),
            gamma_rmse=None if gamma_t is None else float(np.sqrt(np.mean((np.asarray(result.gamma) - gamma_t) ** 2))),
            seconds=seconds,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def align_columns(B_list, prop, T_list):
    """Permute and recode each layer's latent columns to the truth (see module docstring)."""
    B = [np.asarray(b, dtype=float).copy() for b in B_list]
    prop = np.asarray(prop, dtype=float).copy()
    D = len(B)
    for d in range(D):
        K = B[d].shape[1] - 1
        positive = [T_list[d][:, k + 1] > 0 for k in range(K)]
        cost = np.array([[-abs(B[d][positive[k], c + 1].sum()) for c in range(K)] for k in range(K)])
        _, perm = linear_sum_assignment(cost)
        B[d] = np.column_stack((B[d][:, 0], B[d][:, 1:][:, perm]))
        if d + 1 < D:
            B[d + 1] = B[d + 1][perm, :]
        else:
            prop = prop[perm]
        for k in range(K):
            if B[d][positive[k], k + 1].sum() < 0:
                B[d][:, 0] += B[d][:, k + 1]
                B[d][:, k + 1] *= -1.0
                if d + 1 < D:
                    B[d + 1][k, :] *= -1.0
                else:
                    prop[k] = 1.0 - prop[k]
    return B, prop


def run_copdde(task):
    design, N, rep, tol, code = task
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copdde_paper_metrics import load_truth
    from dde import fit
    from dde.copula import rescale_B1_D
    from dde.simulation.generate_copula_D import generate_X_Cop_marg_emp_D

    seed = 7_000_000 + 1000 * int(N) + int(rep)
    row = {"part": "copdde", "code": code, "design": design, "N": int(N), "rep": int(rep),
           "tol": float(tol), "seed": seed, "status": "ok", "error": ""}
    try:
        truth = load_truth(SETTINGS / f"{design}.mat")
        D = truth["D"]
        gen = generate_X_Cop_marg_emp_D(
            N, truth["prop_true"], truth["lambdas"], truth["gen_gamma"], truth["gen_B"], random_state=seed
        )
        if D == 2:
            T_list = [truth["gen_B"][0], truth["gen_B"][1]]
        else:
            B1_scaled = rescale_B1_D(truth["prop_true"], truth["gen_B"], truth["gen_gamma"], emp=True, Z=gen["Z"])[0][0]
            T_list = [B1_scaled] + [B.copy() for B in truth["gen_B"][1:]]
        K = [int(B.shape[1] - 1) for B in truth["gen_B"]]
        np.random.seed(seed % 2**32)
        started = time.perf_counter()
        result = fit(gen["X"], model="CopDDE", dimension_method="fixed", K_list=K,
                     pen_list=[float(N) ** 0.25] * D, tau=max(0.3, 3.0 * float(N) ** -0.3),
                     C=1, tol=float(tol), max_it=50, random_state=seed, verbose=False)
        seconds = time.perf_counter() - started
        B_aligned, prop_aligned = align_columns(result.B_list, result.prop, T_list)
        B_scaled = rescale_B1_D(prop_aligned, B_aligned, result.gamma, emp=True, Z=result.transformed)[0]
        support = [np.asarray(T)[:, 1:] != 0 for T in T_list]
        row.update(
            iterations=int(result.diagnostics["iterations"]),
            final_error=float(result.diagnostics["err"]),
            layer_mse=[float(np.mean((B_scaled[d] - T_list[d]) ** 2)) for d in range(D)],
            layer_graph_accuracy=[float(np.mean((B_scaled[d][:, 1:] != 0) == support[d])) for d in range(D)],
            prop_abs_error=float(np.max(np.abs(prop_aligned - truth["prop_true"]))),
            seconds=seconds,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=["dde", "copdde"], required=True)
    parser.add_argument("--code", required=True, help="label of the package version on PYTHONPATH")
    parser.add_argument("--designs", nargs="+")
    parser.add_argument("--N", nargs="+", type=int)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--tols", nargs="+", type=float, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "Results" / "paper_f3_reltol")
    args = parser.parse_args()
    import dde

    designs = args.designs or list(DDE_DESIGNS if args.part == "dde" else COPDDE_DESIGNS)
    sizes = args.N or ([500, 2000, 8000] if args.part == "dde" else [500, 2000])
    reps = args.replicates or (20 if args.part == "dde" else 10)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.part}_{args.code}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["design"], r["N"], r["rep"], r["tol"]))
    tasks = [(d, n, r, t, args.code) for d in designs for n in sizes for r in range(reps) for t in args.tols
             if (d, n, r, float(t)) not in done]
    (args.out / f"{args.part}_{args.code}_meta.json").write_text(json.dumps(
        {"dde_file": dde.__file__, "args": {k: str(v) for k, v in vars(args).items()},
         "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}, indent=2))
    print(f"{len(tasks)} fits; package {dde.__file__}", flush=True)
    worker = run_dde if args.part == "dde" else run_copdde
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(path, "a") as out:
        futures = [pool.submit(worker, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 20 == 0 or row["status"] != "ok":
                print(i, row["design"], row["N"], row["tol"], row["status"], row["error"][:120], flush=True)


if __name__ == "__main__":
    main()
