"""Full Python CopDDE CSP pipeline on the MATLAB baseline designs.

Truth (B, lambdas, gamma, prop) is read from the MATLAB baseline's
settings/<scenario>.mat, data are generated in Python (generate_X_Cop_marg_emp_D,
the same construction as generate_X_cop_D.m / generate_X_Cop_marg_emp.m), Z is
initialized by the Python mixture initializer, parameters by Normal_init_cop, and
the model is fitted by get_SAEM_RL_CSP_D with the MATLAB hyperparameters
(C=1, 50 iterations, t_spike, temp0, tau = max(.3, 3 N^-.3)).  Every fit is scored
with the paper's graph accuracy (copdde_paper_metrics.paper_graph_accuracy, which
reproduces the MATLAB numbers exactly) and its final widths.

Variants
  declared   observed_scale='declared' (package default)
  matlab     observed_scale='matlab'   (gamma_j / temp, as the MATLAB code computes)

Usage: python scripts/copdde_python_sim.py --settings <copdde_baseline/settings> --out <dir>
           [--scenarios d3_j108 d2_j50] [--N 500 1000 2000 4000 8000] [--reps 30] [--variants matlab] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

VARIANTS = {"declared": "declared", "matlab": "matlab"}


def run_one(task):
    settings_dir, scenario, N, rep, variant = task
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copdde_paper_metrics import load_truth, paper_graph_accuracy
    from dde.algorithms.copula import normal_initialize_copula_dde, prepare_rank_likelihood_inputs
    from dde.algorithms.get_SAEM_RL_CSP_D import get_SAEM_RL_CSP_D
    from dde.copula import simulate_gaussian_mixture
    from dde.simulation.generate_copula_D import generate_X_Cop_marg_emp_D

    truth = load_truth(Path(settings_dir) / f"{scenario}.mat")
    seed = 7_000_000 + 1000 * int(N) + int(rep)
    gen = generate_X_Cop_marg_emp_D(N, truth["prop_true"], truth["lambdas"], truth["gen_gamma"], truth["gen_B"],
                                    random_state=seed)
    X = gen["X"]
    J = X.shape[1]
    _, R, Rlevels = prepare_rank_likelihood_inputs(X)
    np.random.seed(seed % (2 ** 32))
    Z = simulate_gaussian_mixture(X, J, random_state=seed)
    init = normal_initialize_copula_dde(X, truth["K_max"].tolist(), Z=Z)
    np.random.seed(seed % (2 ** 32))
    tau = max(0.3, 3.0 * N ** (-0.3))
    traj = []
    started = time.perf_counter()
    raw = get_SAEM_RL_CSP_D(
        X, Z, R, Rlevels, init["prop"], init["gamma"], init["B_list"], init["A_list"], 1, 50,
        truth["t_spike"], truth["temp0"], tau, optimizer_maxiter=15, verbose=False,
        observed_scale=VARIANTS[variant],
        trace_callback=lambda rec: traj.append(np.asarray(rec["num_active"]).astype(int).tolist()),
    )
    return {
        "scenario": scenario, "N": int(N), "rep": int(rep), "variant": variant, "seed": seed,
        "num_act": np.asarray(raw["num_act"]).astype(int).tolist(), "it": int(raw["it_num"]),
        "accG": paper_graph_accuracy(truth, raw["B_list"]), "traj": traj,
        "seconds": time.perf_counter() - started,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scenarios", nargs="*", default=["d3_j108", "d2_j50"])
    ap.add_argument("--N", nargs="*", type=int, default=[500, 1000, 2000, 4000, 8000])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--variants", nargs="*", default=["matlab", "declared"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "rows.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["scenario"], r["N"], r["rep"], r["variant"]))
    tasks = [(str(args.settings), s, N, rep, v) for N in args.N for s in args.scenarios
             for rep in range(1, args.reps + 1) for v in args.variants if (s, N, rep, v) not in done]
    print(f"{len(done)} cached, {len(tasks)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out_file.open("a") as fh:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as exc:
                print("FAILED", futures[fut][1:], repr(exc), flush=True)
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{row['scenario']} N={row['N']} rep={row['rep']} {row['variant']}: K={row['num_act']} it={row['it']} "
                  f"accG={np.round(row['accG'], 3).tolist()} {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
