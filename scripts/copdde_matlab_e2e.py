"""End-to-end comparison of the Python CopDDE CSP port with the MATLAB baseline.

For every MATLAB fit that saved its data (X, true latent states, Z_init and the
initial parameters), refit with the Python port under one or more variants and
record final widths, the width trajectory and aligned per-layer latent accuracy
for both implementations on the same dataset.

Variants (name -> settings)
  declared_minit   observed_scale='declared', MATLAB init      (current package)
  matlab_minit     observed_scale='matlab',   MATLAB init      (MATLAB objective)
  matlab_pyinit    observed_scale='matlab',   Python init      (full Python pipeline)

Usage: python scripts/copdde_matlab_e2e.py --root <copdde_baseline dir> --out <dir>
           [--scenarios d3_j108 d2_j50] [--N 500 1000] [--variants ...] [--seeds 1 2] [--workers 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

VARIANTS = {
    "declared_minit": dict(observed_scale="declared", init="matlab"),
    "matlab_minit": dict(observed_scale="matlab", init="matlab"),
    "matlab_pyinit": dict(observed_scale="matlab", init="python"),
    "declared_pyinit": dict(observed_scale="declared", init="python"),
    "matlab_pyinit_mZ": dict(observed_scale="matlab", init="python_on_matlab_Z"),
    "matlab_minit_pyZ": dict(observed_scale="matlab", init="matlab_init_python_Z"),
}


def _cells(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [np.asarray(v, dtype=float) for v in x]
    return [np.asarray(x, dtype=float)]


def _2d(a, N):
    a = np.asarray(a, dtype=float)
    return a.reshape(N, -1) if a.ndim == 1 else a


def aligned_latent_accuracy(A_fit, active, A_true):
    """Mean over true columns of the best (flip-invariant) agreement with a distinct active fitted column."""
    A_fit = A_fit[:, active]
    Kt = A_true.shape[1]
    if A_fit.shape[1] == 0:
        return 0.5, 0
    agree = (A_fit[:, :, None] == A_true[:, None, :]).mean(axis=0)
    score = np.maximum(agree, 1.0 - agree)
    rows, cols = linear_sum_assignment(-score)
    total = 0.5 * Kt
    matched = 0
    for r, c in zip(rows, cols):
        total += score[r, c] - 0.5
        matched += int(score[r, c] >= 0.9)
    return float(total / Kt), matched


def load_task_inputs(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    fit = m["fit"]
    meta = fit.meta
    N = int(meta.N)
    data = fit.data
    init = data.init
    res = fit.result
    D = int(meta.D)
    K_max = np.atleast_1d(np.asarray(meta.K_max, dtype=float)).astype(int)
    z_cusp = [np.atleast_1d(z).astype(int) for z in _cells(res.z_cusp)]
    iters = np.atleast_1d(fit.trace.iter)
    return {
        "scenario": str(meta.scenario), "N": N, "rep": int(meta.rep), "D": D, "K_max": K_max.tolist(),
        "K_true": np.atleast_1d(np.asarray(meta.K_true, dtype=float)).astype(int).tolist(),
        "t_spike": np.atleast_1d(np.asarray(meta.t_spike, dtype=float)).tolist(),
        "temp0": float(meta.temp0), "tau": float(meta.tau), "max_iter": int(meta.max_iter),
        "X": np.asarray(data.X, dtype=float),
        "A_true": [_2d(a, N) for a in _cells(data.A_true)],
        "Z_init": np.asarray(data.Z_init, dtype=float),
        "init_prop": np.atleast_1d(np.asarray(init.prop, dtype=float)),
        "init_gamma": np.atleast_1d(np.asarray(init.gamma, dtype=float)),
        "init_B": [np.atleast_2d(np.asarray(b, dtype=float)) for b in _cells(init.B)],
        "init_A": [_2d(a, N) for a in _cells(init.A)],
        "matlab_num_act": np.atleast_1d(np.asarray(res.num_act, dtype=float)).astype(int).tolist(),
        "matlab_it": int(res.it_num),
        "matlab_A": [_2d(a, N) for a in _cells(res.A)],
        "matlab_active": [z > np.arange(1, z.size + 1) for z in z_cusp],
        "matlab_traj": [np.atleast_1d(np.asarray(r.num_act, dtype=float)).astype(int).tolist() for r in iters],
    }


def run_task(task):
    path, variant, seed = task
    from dde.algorithms.copula import normal_initialize_copula_dde, prepare_rank_likelihood_inputs
    from dde.algorithms.get_SAEM_RL_CSP_D import get_SAEM_RL_CSP_D
    from dde.copula import simulate_gaussian_mixture

    inp = load_task_inputs(path)
    cfg = VARIANTS[variant]
    X = inp["X"]
    _, R, Rlevels = prepare_rank_likelihood_inputs(X)
    np.random.seed(seed)
    if cfg["init"] == "matlab":
        Z = inp["Z_init"].copy()
        prop, gamma, B_list, A_list = inp["init_prop"], inp["init_gamma"], inp["init_B"], inp["init_A"]
    elif cfg["init"] == "python_on_matlab_Z":
        Z = inp["Z_init"].copy()
        init = normal_initialize_copula_dde(X, inp["K_max"], Z=Z.copy())
        prop, gamma, B_list, A_list = init["prop"], init["gamma"], init["B_list"], init["A_list"]
        np.random.seed(seed)
    elif cfg["init"] == "matlab_init_python_Z":
        # MATLAB's initial parameters, but SAEM starts from a Python mixture Z.
        Z = simulate_gaussian_mixture(X, X.shape[1], random_state=seed)
        prop, gamma, B_list, A_list = inp["init_prop"], inp["init_gamma"], inp["init_B"], inp["init_A"]
        np.random.seed(seed)
    else:
        Z = simulate_gaussian_mixture(X, X.shape[1], random_state=seed)
        init = normal_initialize_copula_dde(X, inp["K_max"], Z=Z)
        prop, gamma, B_list, A_list = init["prop"], init["gamma"], init["B_list"], init["A_list"]
        np.random.seed(seed)
    traj = []
    started = time.perf_counter()
    raw = get_SAEM_RL_CSP_D(
        X, Z, R, Rlevels, prop, gamma, B_list, A_list, 1, inp["max_iter"], inp["t_spike"], inp["temp0"], inp["tau"],
        optimizer_maxiter=15, verbose=False, observed_scale=cfg["observed_scale"],
        trace_callback=lambda rec: traj.append(np.asarray(rec["num_active"]).astype(int).tolist()),
    )
    seconds = time.perf_counter() - started
    active = [np.asarray(z) > np.arange(1, len(z) + 1) for z in raw["z_cusp"]]
    py_acc = [aligned_latent_accuracy(np.asarray(raw["A_list"][d]), active[d], inp["A_true"][d]) for d in range(inp["D"])]
    ml_acc = [aligned_latent_accuracy(inp["matlab_A"][d], inp["matlab_active"][d], inp["A_true"][d]) for d in range(inp["D"])]
    return {
        "scenario": inp["scenario"], "N": inp["N"], "rep": inp["rep"], "variant": variant, "seed": seed,
        "K_true": inp["K_true"], "py_num_act": np.asarray(raw["num_act"]).astype(int).tolist(), "py_it": int(raw["it_num"]),
        "matlab_num_act": inp["matlab_num_act"], "matlab_it": inp["matlab_it"],
        "py_latent_acc": [a for a, _ in py_acc], "py_matched": [m for _, m in py_acc],
        "matlab_latent_acc": [a for a, _ in ml_acc], "matlab_matched": [m for _, m in ml_acc],
        "py_traj": traj, "matlab_traj": inp["matlab_traj"], "seconds": seconds,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scenarios", nargs="*", default=["d3_j108", "d2_j50"])
    ap.add_argument("--N", nargs="*", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=["declared_minit", "matlab_minit", "matlab_pyinit"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "e2e_rows.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["scenario"], r["N"], r["rep"], r["variant"], r["seed"]))
    tasks = []
    for scenario in args.scenarios:
        for path in sorted(glob.glob(str(args.root / "fits" / scenario / "csp_D" / "N*" / "rep*.mat"))):
            N = int(Path(path).parts[-2][1:])
            rep = int(Path(path).stem[3:])
            if args.N and N not in args.N:
                continue
            m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["fit"])
            if "data" not in m["fit"]._fieldnames:
                continue
            for variant in args.variants:
                for seed in args.seeds:
                    if (scenario, N, rep, variant, seed) not in done:
                        tasks.append((path, variant, seed))
    tasks.sort(key=lambda t: int(Path(t[0]).parts[-2][1:]))
    print(f"{len(done)} cached, {len(tasks)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out_file.open("a") as fh:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as exc:  # keep going; report the failure
                print("FAILED", futures[fut], repr(exc), flush=True)
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{row['scenario']} N={row['N']} rep={row['rep']} {row['variant']} s{row['seed']}: py {row['py_num_act']} "
                  f"(it {row['py_it']}, {row['seconds']:.0f}s) | matlab {row['matlab_num_act']} (it {row['matlab_it']}) | "
                  f"acc py {np.round(row['py_latent_acc'], 3).tolist()} ml {np.round(row['matlab_latent_acc'], 3).tolist()}",
                  flush=True)


if __name__ == "__main__":
    main()
