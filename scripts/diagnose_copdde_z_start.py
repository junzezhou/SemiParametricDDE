"""B^(1) error of CopDDE fits on the true graph: paper Z start vs the true Z.

The rank-likelihood step resamples each Z_ij between the values of the
neighbouring rank groups, so the group boundaries of Z (and with them its
marginal shape) move little from the start, and less as N grows.  This
script refits replicates of ``evaluate_copdde_fixed_graph.py`` with the
package defaults twice: from the paper's Z start (``simulate_gaussian_mixture``,
Algorithm 3) and from the data-generating Z, and reports the paper metric.

    python scripts/diagnose_copdde_z_start.py --n-jobs 2

Writes ``z_start_rows.csv`` under ``Results/copdde_fixed_graph_v1``.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_copdde_fixed_graph import DEFAULT_OUT, SETTINGS, align_to_truth, padded, seed_for  # noqa: E402


def run_one(task):
    scenario, N, rep, start = task
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from copdde_paper_metrics import load_truth
    from dde import fit
    from dde.copula import rescale_B1_D
    from dde.simulation.generate_copula_D import generate_X_Cop_marg_emp_D

    truth = load_truth(SETTINGS / f"{scenario}.mat")
    seed = seed_for(N, rep)
    gen = generate_X_Cop_marg_emp_D(
        N, truth["prop_true"], truth["lambdas"], truth["gen_gamma"], truth["gen_B"], random_state=seed
    )
    G_list = [B[:, 1:] != 0 for B in truth["gen_B"]]
    T_list = [truth["gen_B"][0], truth["gen_B"][1]]
    J, K_max = gen["X"].shape[1], truth["K_max"]
    shapes = [(J, K_max[0] + 1), (K_max[0], K_max[1] + 1)]
    np.random.seed(seed % (2**32))
    extra = {"Z": gen["Z"].copy()} if start == "true_Z" else {}
    result = fit(gen["X"], method="CopDDE", G_list=[G.astype(int) for G in G_list],
                 random_state=seed, verbose=False, **extra)
    B, prop, _ = align_to_truth(result.B_list, result.prop, T_list, G_list)
    B = rescale_B1_D(prop, B, result.gamma, emp=True, Z=result.transformed)[0]
    on = T_list[0][:, 1:] != 0
    return {
        "scenario": scenario, "N": N, "rep": rep, "z_start": start,
        "mse_B1": float(np.mean((padded(B[0], shapes[0]) - padded(T_list[0], shapes[0])) ** 2)),
        "mse_B2": float(np.mean((padded(B[1], shapes[1]) - padded(T_list[1], shapes[1])) ** 2)),
        "slope_ratio_median": float(np.median(B[0][:, 1:][on] / T_list[0][:, 1:][on])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="d2_j50")
    parser.add_argument("--N", nargs="+", type=int, default=[500, 2000, 8000])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    tasks = [(args.scenario, N, rep, start) for N in args.N for rep in range(1, args.reps + 1)
             for start in ("paper_Z", "true_Z")]
    with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(run_one, tasks))
    with open(DEFAULT_OUT / "z_start_rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for N in args.N:
        for start in ("paper_Z", "true_Z"):
            cell = [r for r in rows if r["N"] == N and r["z_start"] == start]
            print(N, start, "mean B1 MSE", round(np.mean([r["mse_B1"] for r in cell]), 5),
                  "mean B2 MSE", round(np.mean([r["mse_B2"] for r in cell]), 5),
                  "median slope ratio", round(np.median([r["slope_ratio_median"] for r in cell]), 3))


if __name__ == "__main__":
    main()
