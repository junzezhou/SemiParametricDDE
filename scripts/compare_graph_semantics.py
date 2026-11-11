"""Old (fixed, unpenalized graph) versus new (allowable-edge) ``G_list`` fits.

The package code under test is the one on ``PYTHONPATH``; ``--code`` labels
the rows and picks the arm:

* ``old``: the graph is a fixed support fitted without penalty or threshold,
  with the old defaults (exact EM when every layer is at most 10 wide,
  otherwise the unpenalized SAEM; CopDDE: the unpenalized rank-likelihood
  SAEM);
* ``new``: the graph marks allowable connections, fitted by the penalized
  SAEM with ``pen_list = N**(1/4)`` and ``tau = 0.3`` (CopDDE:
  ``max(0.3, 3 N**-0.3)``);
* ``new_em``: new code, the exact EM reference implementation
  ``fit_fixed_graph_reference`` (must reproduce ``old`` on the designs where
  ``old`` uses the exact EM);
* ``new_unpen``: new code, penalized SAEM path with ``pen_list = 0`` and
  ``tau = 1e-8`` (isolates the penalty from the SAEM-path changes);
* ``new_full``: as ``new`` with ``tol = 0`` (all 50 sweeps; isolates the
  early stop).

Graphs: ``true`` is the generating support; ``super`` also allows extra
edges (DDE B_s designs: every connection of the two-parent rows; CopDDE:
one random extra parent per row), so a fit should shrink the extra edges.

Designs: DDE ``Sim_sid`` B_s s18 (J = 18, K = 6, 2) and s54 (J = 54,
K = 18, 6) for the Normal, Bernoulli and Poisson families; CopDDE
``d2_j100`` (J = 100, K = 10, 3).  Metrics per layer: coefficient RMSE (DDE;
the graph labels the columns) or the CopDDE paper's B MSE (estimates mapped
to the truth's coding and interchangeable labels as in
``evaluate_copdde_fixed_graph.py``, B1 standardized by the fitted latent
Gaussian), the share of true edges estimated as zero (``fn``) and the share
of allowed false edges estimated as nonzero (``fp``).
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
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "DDE-Copula-m" / "Results" / "copdde_baseline" / "settings"
DDE_DESIGNS = {"s18": (18, 6, 2), "s54": (54, 18, 6)}
FAMILIES = ("normal", "bernoulli", "poisson")


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


def dde_super_graph(G_true):
    """Allow every connection of the two-parent rows (the last third of B_s)."""
    out = []
    for G in G_true:
        G = G.copy()
        G[2 * G.shape[1]:, :] = True
        out.append(G)
    return out


def edge_rates(B_list, G_true, G_fit):
    fn, fp = [], []
    for B, Gt, Gf in zip(B_list, G_true, G_fit):
        nonzero = np.asarray(B)[:, 1:] != 0
        fn.append(float(np.mean(~nonzero[Gt])))
        extra = Gf & ~Gt
        fp.append(float(np.mean(nonzero[extra])) if extra.any() else None)
    return fn, fp


def run_dde(task):
    family, design, graph, N, rep, code = task
    from dde import fit

    J, K1, K2 = DDE_DESIGNS[design]
    seed = int(10_000_000 * (1 + FAMILIES.index(family)) + 1_000_000 * (1 + list(DDE_DESIGNS).index(design))
               + 100 * N + rep)
    row = {"part": "dde", "code": code, "family": family, "design": design, "graph": graph,
           "N": int(N), "rep": int(rep), "seed": seed, "status": "ok", "error": ""}
    try:
        rng = np.random.default_rng(seed)
        B_true = [bs_layer(K1), bs_layer(K2)]
        A2 = rng.binomial(1, 0.5, size=(N, K2))
        A1 = rng.binomial(1, expit(B_true[1][:, 0] + A2 @ B_true[1][:, 1:].T))
        eta = B_true[0][:, 0] + A1 @ B_true[0][:, 1:].T
        if family == "bernoulli":
            X = rng.binomial(1, expit(eta))
        elif family == "poisson":
            X = rng.poisson(np.exp(eta))
        else:
            X = eta + rng.normal(size=eta.shape)
        G_true = [B[:, 1:] != 0 for B in B_true]
        G_fit = G_true if graph == "true" else dde_super_graph(G_true)
        kwargs = dict(model="DDE", family=family, D=2, dimension_method="fixed",
                      G_list=[G.astype(int) for G in G_fit], verbose=False)
        if code == "new":
            kwargs.update(pen_list=[float(N) ** 0.25] * 2, tau=0.3)
        elif code == "new_unpen":
            kwargs.update(pen_list=[0.0, 0.0], tau=1e-8)
        elif code == "new_full":
            kwargs.update(pen_list=[float(N) ** 0.25] * 2, tau=0.3, tol=0.0, max_it=50)
        np.random.seed(seed % 2**32)
        started = time.perf_counter()
        if code == "new_em":
            from dde.algorithms.dde import fit_fixed_graph_reference

            result = fit_fixed_graph_reference(X, family=family, G_list=kwargs["G_list"], verbose=False)
        else:
            result = fit(X, **kwargs)
        seconds = time.perf_counter() - started
        fn, fp = edge_rates(result.B_list, G_true, G_fit)
        row.update(
            graph_algorithm=result.diagnostics.get("graph_algorithm"),
            layer_rmse=[float(np.sqrt(np.mean((np.asarray(B) - Bt) ** 2))) for B, Bt in zip(result.B_list, B_true)],
            fn=fn, fp=fp,
            prop_rmse=float(np.sqrt(np.mean((np.asarray(result.prop) - 0.5) ** 2))),
            gamma_rmse=float(np.sqrt(np.mean((np.asarray(result.gamma) - 1.0) ** 2))) if family == "normal" else None,
            latent_accuracy=[float(np.mean(np.asarray(A) == At)) for A, At in zip(result.A_list, [A1, A2])],
            iterations=int(result.diagnostics["iterations"]),
            seconds=seconds,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def copdde_super_graph(G_true, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for G in G_true:
        G = G.copy()
        for j in range(G.shape[0]):
            free = np.flatnonzero(~G[j])
            if free.size:
                G[j, rng.choice(free)] = True
        out.append(G)
    return out


def run_copdde(task):
    _, design, graph, N, rep, code = task
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copdde_paper_metrics import load_truth
    from evaluate_copdde_fixed_graph import align_to_truth
    from dde import fit
    from dde.copula import rescale_B1_D
    from dde.simulation.generate_copula_D import generate_X_Cop_marg_emp_D

    seed = 7_000_000 + 1000 * int(N) + int(rep)
    row = {"part": "copdde", "code": code, "family": "copula", "design": design, "graph": graph,
           "N": int(N), "rep": int(rep), "seed": seed, "status": "ok", "error": ""}
    try:
        truth = load_truth(SETTINGS / f"{design}.mat")
        gen = generate_X_Cop_marg_emp_D(
            N, truth["prop_true"], truth["lambdas"], truth["gen_gamma"], truth["gen_B"], random_state=seed
        )
        T_list = [truth["gen_B"][0], truth["gen_B"][1]]
        G_true = [B[:, 1:] != 0 for B in truth["gen_B"]]
        G_fit = G_true if graph == "true" else copdde_super_graph(G_true)
        kwargs = dict(model="CopDDE", dimension_method="fixed", G_list=[G.astype(int) for G in G_fit],
                      random_state=seed, verbose=False)
        if code == "new":
            kwargs.update(pen_list=[float(N) ** 0.25] * 2, tau=max(0.3, 3.0 * float(N) ** -0.3))
        np.random.seed(seed % 2**32)
        started = time.perf_counter()
        result = fit(gen["X"], **kwargs)
        seconds = time.perf_counter() - started
        B_aligned, prop_aligned, _ = align_to_truth(result.B_list, result.prop, T_list, G_true)
        B_scaled = rescale_B1_D(prop_aligned, B_aligned, result.gamma, emp=True, Z=result.transformed)[0]
        fn, fp = edge_rates(result.B_list, G_true, G_fit)
        row.update(
            layer_mse=[float(np.mean((B_scaled[d] - T_list[d]) ** 2)) for d in range(2)],
            fn=fn, fp=fp,
            prop_abs_error=float(np.max(np.abs(prop_aligned - truth["prop_true"]))),
            iterations=int(result.diagnostics["iterations"]),
            seconds=seconds,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per fit
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=["dde", "copdde"], required=True)
    parser.add_argument("--code", choices=["old", "new", "new_em", "new_unpen", "new_full"], required=True)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--designs", nargs="+")
    parser.add_argument("--graphs", nargs="+", default=["true", "super"])
    parser.add_argument("--N", nargs="+", type=int)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "Results" / "paper_f6_glist")
    args = parser.parse_args()
    import dde

    if args.part == "dde":
        designs = args.designs or list(DDE_DESIGNS)
        families = args.families
    else:
        designs = args.designs or ["d2_j100"]
        families = ["copula"]
    sizes = args.N or ([500, 2000, 8000] if args.part == "dde" else [500, 2000])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.part}_{args.code}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["family"], r["design"], r["graph"], r["N"], r["rep"]))
    tasks = [(f, d, g, n, r, args.code) for f in families for d in designs for g in args.graphs
             for n in sizes for r in range(args.replicates) if (f, d, g, n, r) not in done]
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
                print(i, row["family"], row["design"], row["graph"], row["N"], row["status"],
                      row["error"][:120], flush=True)


if __name__ == "__main__":
    main()
