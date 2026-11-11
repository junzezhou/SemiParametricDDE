"""Old versus relative ``tol`` for SemiDDE fixed, the exact EM on a graph and the CSP kernels.

The package code under test is the one on ``PYTHONPATH``; ``--code`` only
labels the rows.  Every (design, N, replicate) uses the same data and the
same seeds in every arm, so arms are paired.  ``tol = None`` in a row means
the routine's default.

SemiDDE fixed (``--part semi``): Normal ``build_true_params`` designs D2
(K = 6, 2; J = 18) and D3 (K = 18, 6, 2; J = 54) observed as ``Y = exp(X)``,
``pen_list = N**(1/4)``, ``tau = 0.3``, ``C = 1``, 50 inner iterations,
oracle-labelled initialization, package defaults otherwise.  ``tol`` 0.01
(manuscript), 0.005 (the new default) and 1.0 (``K_D / 2``, which
``select_hyperparameters`` used to impose).

Exact EM on a graph (``--part em``): the ``Sim_sid`` B_s designs with their
true graph as ``G_list``, Normal / Bernoulli / Poisson.  s18 (K = 6, 2) is
fitted by the exact EM itself (10 starts); s54 (K = 18, 6) by the SAEM,
whose layer-wise warm start fits the deep layer by the exact EM with the
same ``tol``.  ``tol`` default and 0.01.

CSP (``--part csp``): the ``csp_bench.py`` designs (CopDDE paper latent
structure ``d2_j50`` and ``d3_j108``), kernels CopDDE CSP and DDE Normal CSP
on the Normal data, SemiDDE CSP on ``exp`` of it, each with its defaults and
with ``tol = 0.01``; DDE Bernoulli CSP (default ``tol = 0``, no early stop)
with ``tol = 0.01`` only.  Scores as in ``csp_bench.py``.

    PYTHONPATH=<worktree>/src python scripts/compare_tol_metric_extended.py --part semi --code new
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SETTINGS = ROOT / "DDE-Copula-m" / "Results" / "copdde_baseline" / "settings"
FAMILIES = ("normal", "bernoulli", "poisson")
SEMI_DESIGNS = {"semi_D2": [6, 2], "semi_D3": [18, 6, 2]}
EM_DESIGNS = {"s18": (6, 2), "s54": (18, 6)}
CSP_KERNELS = (("copdde", "normal"), ("dde_csp", "normal"), ("semi_csp", "semidde"), ("dde_csp", "bernoulli"))


def bs_layer(K):
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


def run_semi(task):
    design, N, rep, tol, code = task
    warnings.simplefilter("ignore")
    from dde import fit
    from dde.identifiability import build_true_params
    from dde.simulation import generate_X_normal

    K = SEMI_DESIGNS[design]
    seed = 20261110 + 1_000_000 * list(SEMI_DESIGNS).index(design) + 1000 * int(N) + int(rep)
    row = {"part": "semi", "code": code, "design": design, "N": int(N), "rep": int(rep), "tol": tol,
           "seed": seed, "status": "ok", "error": ""}
    try:
        prop, B_t, G_t, gamma = build_true_params(K, 3 * K[0], 4.0)
        B_t = [np.asarray(B, float) for B in B_t]
        G_t = [np.asarray(G, bool) for G in G_t]
        np.random.seed(seed)
        X, _ = generate_X_normal(int(N), np.asarray(prop, float), B_t, np.asarray(gamma, float))
        Y = np.exp(X)
        np.random.seed((seed + 1) % 2**32)
        started = time.perf_counter()
        res = fit(Y, model="SemiDDE", D=len(K), K_list=K, dimension_method="fixed", initialize="oracle",
                  B_list_true=B_t, G_list_true=G_t, pen_list=[float(N) ** 0.25] * len(K), tau=0.3,
                  tol=float(tol), C=1, max_it=50, verbose=False)
        d = res.diagnostics
        warm = d.get("warm_start") or {}
        row.update(
            iterations=int(d["iterations"]), g_iterations=int(d["g_iterations"]), g_converged=bool(d["g_converged"]),
            transform_inner=int(sum(d["inner_iterations_by_outer"])), final_iterations=d["final_iterations"],
            warm_iterations=[warm.get("stage1_iterations"), warm.get("stage2_iterations")],
            layer_rmse=[float(np.sqrt(np.mean((np.asarray(B) - Bt) ** 2))) for B, Bt in zip(res.B_list, B_t)],
            layer_graph_accuracy=[float(np.mean((np.asarray(B)[:, 1:] != 0) == G)) for B, G in zip(res.B_list, G_t)],
            transform_rmse=float(np.sqrt(np.mean((np.asarray(res.transformed) - X) ** 2))),
            seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def run_em(task):
    (design, family), N, rep, tol, code = task
    warnings.simplefilter("ignore")
    from dde import fit

    K1, K2 = EM_DESIGNS[design]
    seed = int(10_000_000 * (1 + FAMILIES.index(family)) + 1_000_000 * (1 + list(EM_DESIGNS).index(design))
               + 100 * N + rep)
    row = {"part": "em", "code": code, "design": f"{family}_{design}", "N": int(N), "rep": int(rep), "tol": tol,
           "seed": seed, "status": "ok", "error": ""}
    try:
        rng = np.random.default_rng(seed)
        B_t = [bs_layer(K1), bs_layer(K2)]
        A2 = rng.binomial(1, 0.5, size=(N, K2))
        A1 = rng.binomial(1, expit(B_t[1][:, 0] + A2 @ B_t[1][:, 1:].T))
        eta = B_t[0][:, 0] + A1 @ B_t[0][:, 1:].T
        if family == "bernoulli":
            X = rng.binomial(1, expit(eta))
        elif family == "poisson":
            X = rng.poisson(np.exp(eta))
        else:
            X = eta + rng.normal(size=eta.shape)
        G = [(B[:, 1:] != 0).astype(int) for B in B_t]
        kwargs = dict(model="DDE", family=family, D=2, dimension_method="fixed", G_list=G, verbose=False)
        if tol is not None:
            kwargs["tol"] = float(tol)
        if design == "s18":
            kwargs["random_state"] = seed
        np.random.seed(seed % 2**32)
        started = time.perf_counter()
        res = fit(X, **kwargs)
        d = res.diagnostics
        warm = d.get("warm_start") or {}
        row.update(
            graph_algorithm=d.get("graph_algorithm"), iterations=int(d["iterations"]),
            loglik=d.get("loglik"), best_start=d.get("best_start"),
            warm_iterations=[warm.get("stage1_iterations"), warm.get("stage2_iterations")],
            layer_rmse=[float(np.sqrt(np.mean((np.asarray(B) - Bt) ** 2))) for B, Bt in zip(res.B_list, B_t)],
            latent_accuracy=[float(np.mean(np.asarray(A) == At)) for A, At in zip(res.A_list, [A1, A2])],
            seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def run_csp(task):
    (scenario, kernel, family), N, rep, tol, code = task
    warnings.simplefilter("ignore")
    sys.path.insert(0, str(HERE))
    from copdde_paper_metrics import load_truth, paper_graph_accuracy
    from csp_bench import fitted_full_B, latent_recovery, simulate
    from dde import fit

    truth = load_truth(SETTINGS / f"{scenario}.mat")
    seed = 9_000_000 + 1_000 * int(N) + int(rep) + {"normal": 0, "bernoulli": 100, "poisson": 200, "semidde": 300}[family]
    row = {"part": "csp", "code": code, "design": f"{scenario}_{kernel}_{family}", "N": int(N), "rep": int(rep),
           "tol": tol, "seed": seed, "status": "ok", "error": "", "K_true": [int(k) for k in truth["K_true"]]}
    try:
        Y, A_true = simulate(truth, family, N, seed)
        D = truth["D"]
        common = dict(dimension_method="csp", t_spike=truth["t_spike"].tolist(), temp=truth["temp0"],
                      tau=max(0.3, 3.0 * N ** (-0.3)), C=1, random_state=seed, verbose=False)
        if tol is not None:
            common["tol"] = float(tol)
        started = time.perf_counter()
        if kernel == "copdde":
            np.random.seed(seed % 2**32)
            res = fit(Y, model="CopDDE", K_list=truth["K_max"].tolist(), max_it=50, **common)
        elif kernel == "dde_csp":
            res = fit(Y, model="DDE", family=family.capitalize(), D=D, **common)
        else:
            res = fit(Y, model="SemiDDE", D=D, max_it=50, **common)
        row.update(
            widths=[int(k) for k in res.K_list], iterations=int(res.diagnostics.get("iterations", -1)),
            accG=paper_graph_accuracy(truth, fitted_full_B(res, truth)),
            latent_corr=latent_recovery(A_true, res.A_list), seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def tasks_for(part, code):
    if part == "semi":
        return [(d, n, r, t, code) for d in SEMI_DESIGNS for n in (500, 2000, 8000) for r in range(20)
                for t in (0.01, 0.005, 1.0)]
    if part == "em":
        return [((d, f), n, r, t, code) for d in EM_DESIGNS for f in FAMILIES for n in (500, 2000, 8000)
                for r in range(10) for t in (None, 0.01)]
    out = []
    for scenario in ("d2_j50", "d3_j108"):
        for kernel, family in CSP_KERNELS:
            tols = (0.01,) if family == "bernoulli" else (None, 0.01)
            out += [((scenario, kernel, family), n, r, t, code) for n in (1000, 4000) for r in range(1, 11) for t in tols]
    return out


def key(row):
    return (row["design"], row["N"], row["rep"], row["tol"])


def task_key(part, task):
    design, N, rep, tol, _ = task
    if part == "em":
        design = f"{design[1]}_{design[0]}"
    elif part == "csp":
        design = "_".join(design)
    return (design, N, rep, tol)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=["semi", "em", "csp"], required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "Results" / "paper_f3_reltol" / "extended")
    args = parser.parse_args()
    import dde

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.part}_{args.code}.jsonl"
    done = set()
    if path.exists():
        done = {key(json.loads(line)) for line in path.read_text().splitlines() if json.loads(line)["status"] == "ok"}
    tasks = [t for t in tasks_for(args.part, args.code) if task_key(args.part, t) not in done]
    (args.out / f"{args.part}_{args.code}_meta.json").write_text(json.dumps(
        {"dde_file": dde.__file__, "args": {k: str(v) for k, v in vars(args).items()},
         "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}, indent=2))
    print(f"{args.part}/{args.code}: {len(tasks)} fits; package {dde.__file__}", flush=True)
    worker = {"semi": run_semi, "em": run_em, "csp": run_csp}[args.part]
    # Longest fits first.
    tasks.sort(key=lambda t: -t[1])
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(path, "a") as out:
        futures = [pool.submit(worker, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 40 == 0 or row["status"] != "ok":
                print(i, row["design"], row["N"], row["tol"], row["status"], row["error"][:160], flush=True)


if __name__ == "__main__":
    main()
