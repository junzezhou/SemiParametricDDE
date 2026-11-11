"""SemiDDE fixed widths: rank-normal versus rank-spectral starting pseudo-observations.

Manuscript Section 4.1 starts SemiDDE, like CopDDE, from rank-preserving
spectral clustering (``dde.copula.simulate_gaussian_mixture``); the package
starts from the normal scores of the ranks.  ``pseudo_init`` selects the
start; everything else is the package default for fixed widths.

Designs: ``build_true_params`` D2 (K = 6, 2; J = 18) and D3 (K = 18, 6, 2;
J = 54), observed as ``Y = exp(X)``; ``pen_list = N**(1/4)``, ``tau = 0.3``,
``C = 1``, default ``tol`` (0.005).  Starts: ``data`` (the package data
initialization; the fitted latent columns are aligned to the truth layer by
layer by Hungarian assignment on ``|corr|`` of the latent states, and a
column whose correlation is negative is recoded, which maps its
coefficients exactly) and ``oracle`` (true labels for the spectral start).
The data seed is ``150 + replicate`` (as in evaluate_semidde_fixed_recovery);
both arms of a replicate use the same data and the same fit seed.

    PYTHONPATH=<worktree>/src python scripts/compare_semidde_pseudo_init.py \
        --out Results/paper_f9_semi_init --workers 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

DESIGNS = {"d2": ((6, 2), 18), "d3": ((18, 6, 2), 54)}
PSEUDO_INITS = ("rank_normal", "rank_spectral")


def truth(design):
    from dde.identifiability import build_true_params

    K, J = DESIGNS[design]
    prop, B_list, G_list, gamma = build_true_params(list(K), J, 4)
    return (np.asarray(prop, float), [np.asarray(B, float) for B in B_list],
            [np.asarray(G, bool) for G in G_list], np.asarray(gamma, float))


def align(B_list, prop, A_list, A_true):
    """Permute and recode the fitted latent columns to the true ones.

    Recoding column k of layer d (A -> 1 - A) maps the coefficients exactly:
    in B_d the intercept gains the slope and the slope changes sign; row k
    of B_{d+1} (or the top prevalence) is negated (p -> 1 - p).
    """
    B = [np.asarray(b, float).copy() for b in B_list]
    prop = np.asarray(prop, float).copy()
    min_abs_corr = []
    for d, (A, At) in enumerate(zip(A_list, A_true)):
        A = np.asarray(A, float)
        At = np.asarray(At, float)
        K = At.shape[1]
        corr = np.zeros((K, A.shape[1]))
        for i in range(K):
            for j in range(A.shape[1]):
                a, b = At[:, i], A[:, j]
                if a.std() > 0 and b.std() > 0:
                    corr[i, j] = np.corrcoef(a, b)[0, 1]
        rows, cols = linear_sum_assignment(-np.abs(corr))
        perm = cols[np.argsort(rows)]
        sign = np.sign(corr[np.arange(K), perm])
        sign[sign == 0] = 1.0
        min_abs_corr.append(float(np.min(np.abs(corr[np.arange(K), perm]))))
        Bd = B[d]
        Bd = np.column_stack([Bd[:, 0], Bd[:, 1:][:, perm]])
        for k in np.flatnonzero(sign < 0):
            Bd[:, 0] += Bd[:, k + 1]
            Bd[:, k + 1] *= -1.0
        B[d] = Bd
        if d + 1 < len(B):
            above = B[d + 1][perm]
            above[sign < 0] *= -1.0
            B[d + 1] = above
        else:
            prop = prop[perm]
            prop[sign < 0] = 1.0 - prop[sign < 0]
    return B, prop, min_abs_corr


def run(task):
    design, N, rep, pseudo_init, start = task
    from threadpoolctl import threadpool_limits

    from dde import fit
    from dde.algorithms.semi import _initial_pseudo_observations
    from dde.simulation import generate_X_normal
    from dde.utils import thres

    K, J = DESIGNS[design]
    prop_t, B_t, G_t, gamma_t = truth(design)
    data_seed = 150 + rep
    fit_seed = 10_000 + 1000 * rep + N % 997
    row = dict(design=design, N=int(N), rep=int(rep), pseudo_init=pseudo_init, start=start,
               status="ok", error="")
    try:
        np.random.seed(data_seed)
        X_true, A_true = generate_X_normal(int(N), prop_t, B_t, gamma_t)
        Y = np.exp(X_true)
        with threadpool_limits(limits=1):
            np.random.seed(fit_seed)
            X0 = _initial_pseudo_observations(Y, pseudo_init)
            kwargs = dict(model="SemiDDE", D=len(K), K_list=list(K), dimension_method="fixed",
                          pen_list=[float(N) ** 0.25] * len(K), tau=0.3, C=1, pseudo_init=pseudo_init,
                          verbose=False)
            if start == "oracle":
                kwargs.update(initialize="oracle", B_list_true=B_t, G_list_true=G_t)
            np.random.seed(fit_seed)
            t0 = time.perf_counter()
            res = fit(Y, **kwargs)
            seconds = time.perf_counter() - t0
        B_fit = []
        for b in res.B_list:
            b = np.asarray(b, float).copy()
            b[:, 1:] = thres(b[:, 1:], 0.3)
            B_fit.append(b)
        prop_fit = np.asarray(res.prop, float)
        if start == "data":
            B_fit, prop_fit, min_corr = align(B_fit, prop_fit, res.A_list, A_true)
        else:
            min_corr = None
        sq = [float(np.sum((b - bt) ** 2)) for b, bt in zip(B_fit, B_t)]
        n = [int(bt.size) for bt in B_t]
        A_fit = res.A_list
        if start == "data":
            # latent accuracy after alignment: recompute from correlations' signs
            acc = []
            for A, At in zip(res.A_list, A_true):
                A = np.asarray(A, float)
                At = np.asarray(At, float)
                agree = (At[:, :, None] == A[:, None, :]).mean(axis=0)
                score = np.maximum(agree, 1 - agree)
                r, c = linear_sum_assignment(-score)
                acc.append(float(score[r, c].mean()))
        else:
            acc = [float(np.mean(np.asarray(A) == At)) for A, At in zip(A_fit, A_true)]
        row.update(
            layer_rmse=[float(np.sqrt(s / m)) for s, m in zip(sq, n)],
            pooled_rmse=float(np.sqrt(sum(sq) / sum(n))),
            layer_graph_accuracy=[float(np.mean((b[:, 1:] != 0) == g)) for b, g in zip(B_fit, G_t)],
            latent_accuracy=acc,
            prop_rmse=float(np.sqrt(np.mean((prop_fit - prop_t) ** 2))),
            transform_rmse=float(np.sqrt(np.mean((np.asarray(res.transformed) - X_true) ** 2))),
            start_transform_rmse=float(np.sqrt(np.mean((X0 - X_true) ** 2))),
            start_rank_corr=float(np.mean([np.corrcoef(X0[:, j], X_true[:, j])[0, 1] for j in range(J)])),
            min_alignment_corr=min_corr,
            g_iterations=int(res.diagnostics.get("g_iterations", -1)),
            iterations=int(res.diagnostics["iterations"]),
            initialization=res.diagnostics.get("initialization"),
            seconds=seconds,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--designs", nargs="+", default=list(DESIGNS))
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 1000, 2000, 4000, 8000])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--starts", nargs="+", default=["data", "oracle"])
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["design"], r["N"], r["rep"], r["pseudo_init"], r["start"]))
    tasks = [(d, N, rep, p, s) for d in args.designs for N in args.sizes for rep in range(1, args.reps + 1)
             for p in PSEUDO_INITS for s in args.starts if (d, N, rep, p, s) not in done]
    tasks.sort(key=lambda t: -t[1])  # large N first
    print(f"{len(tasks)} fits to run", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(path, "a") as fh:
        futures = [ex.submit(run, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 50 == 0 or row["status"] != "ok":
                print(i, row["design"], row["N"], row["pseudo_init"], row["start"], row["status"], row["error"],
                      flush=True)


if __name__ == "__main__":
    main()
