"""CopDDE fixed widths: annealing temperature 0.7 / 0.9 (select_hyperparameters) versus 1 (fit).

The CopDDE paper truths d2_j100 (K = 10, 3) and d3_j108 (K = 18, 6, 2),
pen_list = N**(1/4), tau = max(0.3, 3 N**-0.3), C = 1, tol = 0.01 (paper),
max_it = 50, the true widths; metrics as in compare_fixed_tol_metric.py
(B1 standardized by the fitted latent Gaussian, columns aligned to the truth).
The fixed kernel holds temp constant.

    PYTHONPATH=<worktree>/src python scripts/compare_copdde_fixed_temp.py --out DIR
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

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "DDE-Copula-m" / "Results" / "copdde_baseline" / "settings"


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
    design, N, rep, temp, code = task
    tol = 0.01
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copdde_paper_metrics import load_truth
    from dde import fit
    from dde.copula import rescale_B1_D
    from dde.simulation.generate_copula_D import generate_X_Cop_marg_emp_D

    seed = 7_000_000 + 1000 * int(N) + int(rep)
    row = {"part": "copdde", "code": code, "design": design, "N": int(N), "rep": int(rep),
           "temp": float(temp), "seed": seed, "status": "ok", "error": ""}
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
                     C=1, tol=float(tol), temp=float(temp), max_it=50, random_state=seed, verbose=False)
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--designs", nargs="+", default=["d2_j100", "d3_j108"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 2000])
    parser.add_argument("--temps", nargs="+", type=float, default=[0.7, 0.9, 1.0])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["design"], r["N"], r["rep"], r["temp"]))
    tasks = [(d, N, rep, t, "temp") for d in args.designs for N in args.sizes for rep in range(args.reps)
             for t in args.temps if (d, N, rep, t) not in done]
    tasks.sort(key=lambda t: (-t[1], t[0] != "d3_j108"))
    print(f"{len(tasks)} fits", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(path, "a") as fh:
        for fut in as_completed([ex.submit(run_copdde, t) for t in tasks]):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row["status"] != "ok":
                print("failed", row, flush=True)


if __name__ == "__main__":
    main()
