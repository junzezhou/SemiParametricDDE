"""Common CSP benchmark: DDE CSP kernels vs the (MATLAB-aligned) Python CopDDE CSP.

Every family shares the latent structure of the CopDDE paper designs (truth read
from the MATLAB baseline's settings/<scenario>.mat: K, B_2..B_D, prop, and the
layer-1 graph G_1 with its signs).  Only the observed layer differs:

  normal     X = Z, the copula's latent Gaussian layer (B_1 and gamma of the design)
  semidde    Y = exp(standardized Z), a monotone transform of the Normal data
  bernoulli  Y_j ~ Bernoulli(logistic(-2 + 4 * [G_1 positive] - 2 * [G_1 negative]))
  poisson    Y_j ~ Poisson(exp(0 + 1.5 * [G_1 positive] - 0.75 * [G_1 negative]))

Kernels (all with the CopDDE hyperparameters of the design: t_spike, temp0,
tau = max(.3, 3 N^-.3), 50 iterations, C = 1, maximal widths J/3, J/9, J/27):

  copdde     fit(method='CopDDE', dimension_method='csp')  (aligned Python CopDDE)
  dde_csp    fit(method='DDE', family=..., dimension_method='csp') with the kernel's
             own defaults otherwise (the copula kernel for every family)
  dde_csp_cop  Bernoulli/Poisson on the copula-structured engine (csp_backend='copula')
  semi_csp   fit(model='SemiDDE', dimension_method='csp') with the kernel's defaults
  semi_csp_cop SemiDDE on the copula-structured engine with a g step every iteration
  semi_csp_cop100 the same with 100 iterations and no early stop

  bench_v1 rows labelled dde_csp / semi_csp were fitted before the default switch
  (MCEM backend for Bernoulli, Poisson and SemiDDE); the MCEM backends were removed
  afterwards (git tag csp-mcem-last).

Scores: final widths and the paper's graph accuracy (copdde_paper_metrics), which
needs only the fitted support and the alignment, so it applies to every family.

Usage: python scripts/csp_bench.py --settings <copdde_baseline/settings> --out <dir>
           [--families normal bernoulli poisson semidde] [--scenarios d2_j50 d3_j108]
           [--N 500 1000 2000 4000 8000] [--reps 20] [--kernels copdde dde_csp] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FAMILY_KERNELS = {
    "normal": ("copdde", "dde_csp", "cop_norm100"),
    "bernoulli": ("copdde", "dde_csp", "dde_csp_cop", "cop_mf", "cop_mf_p1", "cop_mf_p0", "cop_mf_l1", "cop_mf_l1p1", "cop_gibbs_merge", "cop_gibbs_merge_top", "cop_seq", "cop_seq_thr", "cop_gibbs_thr", "cop_seq_thr_s05", "cop_seq_thr_s05top", "cop_gibbs_thr_s05", "cop_seqL", "cop_seqP", "cop_gibbsP", "cop_seqP100", "cop_seqPA100", "cop_seqPD100", "cop_seqPDA100", "dde_csp_sb10", "dde_csp_sb20", "dde_csp_sb20L1", "dde_csp_sb20t10", "dde_csp_sb20t0"),
    "poisson": ("copdde", "dde_csp", "dde_csp_cop", "cop_mf", "cop_mf_p1", "cop_mf_p0", "cop_mf_l1", "cop_mf_l1p1", "cop_gibbs_merge", "cop_gibbs_merge_top", "cop_seq", "cop_seq_thr", "cop_gibbs_thr", "cop_seq_thr_s05", "cop_seq_thr_s05top", "cop_gibbs_thr_s05", "cop_seqL", "cop_seqP", "cop_gibbsP", "cop_seqP100", "cop_seqPA100", "cop_seqPD100", "cop_seqPDA100", "dde_csp_sb10", "dde_csp_sb20", "dde_csp_sb20L1", "dde_csp_sb20t10", "dde_csp_sb20t0"),
    "semidde": ("copdde", "semi_csp", "semi_csp_cop", "semi_csp_cop100", "semi_csp_seqPD", "semi_csp_seqPDall", "semi_csp_seqPDall05", "semi_csp_seqPD100"),
}


# Copula-engine settings for the discrete families (Normal CSP style unless noted).
_MF = dict(csp_latent_transition="synchronous_mean_field", csp_shrinkage_burn=0)
ENGINE_VARIANTS = {
    "cop_mf": dict(_MF, csp_penalty_scaling_power=0.5),
    "cop_mf_p1": dict(_MF, csp_penalty_scaling_power=1.0),
    "cop_mf_p0": dict(_MF, csp_penalty_scaling_power=0.0),
    # N-scaling on the observed layer only (callables get D)
    "cop_mf_l1": lambda D: dict(_MF, csp_penalty_scaling_power=[0.5] + [0.0] * (D - 1), csp_penalty_scaling_layers="all"),
    "cop_mf_l1p1": lambda D: dict(_MF, csp_penalty_scaling_power=[1.0] + [0.0] * (D - 1), csp_penalty_scaling_layers="all"),
    # family default transition (Gibbs) with merging of near-parallel split columns
    "cop_gibbs_merge": dict(csp_merge_cosine=0.9, csp_merge_layers="logistic", csp_merge_after=10),
    "cop_gibbs_merge_top": dict(csp_merge_cosine=0.9, csp_merge_layers="top", csp_merge_after=10),
    # mean-field draws taken layer by layer from the data upward (N-scaling on the observed layer only)
    "cop_seq": lambda D: dict(_MF, csp_latent_transition="sequential_mean_field",
                              csp_penalty_scaling_power=[0.5] + [0.0] * (D - 1), csp_penalty_scaling_layers="all"),
    "cop_seq_thr": lambda D: dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                                  csp_penalty_scaling_power=[0.5] + [0.0] * (D - 1), csp_penalty_scaling_layers="all"),
    # family defaults with the deepest layer thresholded too
    "cop_gibbs_thr": dict(csp_threshold_rule="all"),
    # ... plus the sqrt(N)-scaled penalty extended to the deep layers
    "cop_seq_thr_s05": dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                            csp_penalty_scaling_power=0.5, csp_penalty_scaling_layers="all"),
    "cop_seq_thr_s05top": lambda D: dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                                         csp_penalty_scaling_power=[0.5] + [0.0] * (D - 2) + [0.5],
                                         csp_penalty_scaling_layers="all"),
    "cop_gibbs_thr_s05": dict(csp_threshold_rule="all", csp_penalty_scaling_power=0.5, csp_penalty_scaling_layers="all"),
    # longer budget, stopping only once the active widths are stable for 20 iterations ("max_it" is a bench key)
    "cop_seqL": dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                     csp_penalty_scaling_power=0.5, csp_penalty_scaling_layers="all",
                     csp_convergence_width_window=20, max_it=100),
    # ... plus the marginal-likelihood (BIC) deletion test for the deepest layer
    "cop_seqP": dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                     csp_penalty_scaling_power=0.5, csp_penalty_scaling_layers="all",
                     csp_convergence_width_window=20, csp_top_prune="bic", max_it=100),
    "cop_gibbsP": dict(csp_threshold_rule="all", csp_convergence_width_window=20, csp_top_prune="bic", max_it=150),
    # no early stop (tol=0: always max_it iterations); "anchor_init" (bench key) starts from the family's anchor initializer
    "cop_seqP100": dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                        csp_penalty_scaling_power=0.5, csp_penalty_scaling_layers="all",
                        csp_top_prune="bic", tol=0.0, max_it=100),
    "cop_seqPA100": dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                         csp_penalty_scaling_power=0.5, csp_penalty_scaling_layers="all",
                         csp_top_prune="bic", tol=0.0, max_it=100, anchor_init=True),
    # dense BIC deletion test (the reduced refit is adopted)
    "cop_seqPD100": dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                         csp_penalty_scaling_power=0.5, csp_penalty_scaling_layers="all",
                         csp_top_prune="bic_dense", tol=0.0, max_it=100),
    # Normal default engine with 100 iterations and no early stop
    "cop_norm100": dict(tol=0.0, max_it=100),
    "cop_seqPDA100": dict(_MF, csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                          csp_penalty_scaling_power=0.5, csp_penalty_scaling_layers="all",
                          csp_top_prune="bic_dense", tol=0.0, max_it=100, anchor_init=True),
}


def _variant(kernel, D):
    spec = ENGINE_VARIANTS[kernel]
    return spec(D) if callable(spec) else spec


def family_layer1(truth, family):
    B1 = np.asarray(truth["gen_B"][0], dtype=float)
    pos = B1[:, 1:] > 0
    neg = B1[:, 1:] < 0
    if family == "bernoulli":
        icpt, a, b = -2.0, 4.0, -2.0
    elif family == "poisson":
        icpt, a, b = 0.0, 1.5, -0.75
    else:
        return B1
    out = np.zeros_like(B1)
    out[:, 0] = icpt
    out[:, 1:] = a * pos + b * neg
    return out


def simulate(truth, family, N, seed):
    from scipy.special import expit
    from dde.simulation.generate_copula_D import generate_X_Cop_marg_emp_D, simulate_latent_layers_D

    rng = np.random.default_rng(seed)
    if family in ("normal", "semidde"):
        gen = generate_X_Cop_marg_emp_D(N, truth["prop_true"], truth["lambdas"], truth["gen_gamma"], truth["gen_B"],
                                        random_state=rng)
        Z = gen["Z"]
        if family == "normal":
            return Z, gen["A_list"]
        Zs = (Z - Z.mean(0)) / Z.std(0)
        return np.exp(Zs), gen["A_list"]
    B_list = [family_layer1(truth, family)] + [np.asarray(b, dtype=float) for b in truth["gen_B"][1:]]
    A_list, _ = simulate_latent_layers_D(N, truth["prop_true"], B_list, random_state=rng)
    eta = np.column_stack([np.ones(N), A_list[0]]) @ B_list[0].T
    if family == "bernoulli":
        Y = rng.binomial(1, expit(eta)).astype(float)
    else:
        Y = rng.poisson(np.exp(eta)).astype(float)
    return Y, A_list


def run_kernel(kernel, family, Y, truth, N, seed):
    from dde import fit

    D = truth["D"]
    tau = max(0.3, 3.0 * N ** (-0.3))
    common = dict(dimension_method="csp", t_spike=truth["t_spike"].tolist(), temp=truth["temp0"], tau=tau, C=1)
    if kernel == "copdde":
        np.random.seed(seed % (2 ** 32))
        res = fit(Y, method="CopDDE", K_list=truth["K_max"].tolist(), it=50, random_state=seed, verbose=False, **common)
    elif kernel == "dde_csp":
        # the kernel's own defaults, iteration budget included (Bernoulli/Poisson: 100, no early stop)
        res = fit(Y, method="DDE", family=family.capitalize(), D=D, random_state=seed, verbose=False, **common)
    elif kernel.startswith("dde_csp_sb"):
        # the kernel's defaults with a slab-start shrinkage burn-in of that many
        # iterations (suffix L1: layer 1 only; tB: deepest layer B)
        spec = kernel[len("dde_csp_sb"):]
        if spec.endswith("L1"):
            burn = [int(spec[:-2])] + [0] * (D - 1)
        elif "t" in spec:
            lower, top = spec.split("t")
            burn = [int(lower)] * (D - 1) + [int(top)]
        else:
            burn = int(spec)
        res = fit(Y, method="DDE", family=family.capitalize(), D=D, random_state=seed, verbose=False,
                  csp_shrinkage_burn=burn, **common)
    elif kernel == "dde_csp_cop":
        res = fit(Y, method="DDE", family=family.capitalize(), D=D, max_it=50, random_state=seed, verbose=False,
                  csp_backend="copula", **common)
    elif kernel in ENGINE_VARIANTS:
        variant = dict(_variant(kernel, D))
        max_it = variant.pop("max_it", 50)
        if variant.pop("anchor_init", False):
            variant["fixed_init_binarize"] = "anchor_lca" if family == "bernoulli" else "anchor_mixture"
        res = fit(Y, method="DDE", family=family.capitalize(), D=D, max_it=max_it, random_state=seed, verbose=False,
                  csp_backend="copula", **variant, **common)
    elif kernel == "semi_csp_cop100":
        res = fit(Y, model="SemiDDE", D=D, max_it=100, random_state=seed, verbose=False, csp_backend="copula",
                  tol=0.0, **common)
    elif kernel == "semi_csp_cop":
        res = fit(Y, model="SemiDDE", D=D, max_it=50, random_state=seed, verbose=False, csp_backend="copula", **common)
    elif kernel.startswith("semi_csp_seqPD"):
        # the discrete-family top-layer settings: sequential transition, every
        # layer thresholded, dense deletion test for the deepest layer;
        # suffix all / all05: N (or sqrt(N)) penalty scaling in every layer
        extra = {"semi_csp_seqPD": {},
                 "semi_csp_seqPD100": dict(tol=0.0, max_it=100),
                 "semi_csp_seqPDall": dict(csp_penalty_scaling_layers="all"),
                 "semi_csp_seqPDall05": dict(csp_penalty_scaling_layers="all", csp_penalty_scaling_power=0.5)}[kernel]
        extra = {"max_it": 50, **extra}
        res = fit(Y, model="SemiDDE", D=D, random_state=seed, verbose=False,
                  csp_latent_transition="sequential_mean_field", csp_threshold_rule="all",
                  csp_top_prune="bic_dense", **extra, **common)
    elif kernel == "semi_csp":
        res = fit(Y, model="SemiDDE", D=D, max_it=50, random_state=seed, verbose=False, **common)
    else:
        raise ValueError(kernel)
    return res


def fitted_full_B(res, truth):
    """B matrices at the maximal widths (zeros for removed columns) for the paper metric."""
    B_list = [np.asarray(b, dtype=float) for b in res.B_list]
    K_max = truth["K_max"]
    if all(b.shape[1] - 1 == k for b, k in zip(B_list, K_max)):
        return B_list
    # compressed result: pad columns (and rows of the next layer) back to the maximal widths
    out = []
    rows = truth["J"]
    for d, (b, k) in enumerate(zip(B_list, K_max)):
        full = np.zeros((rows, k + 1))
        full[: b.shape[0], : b.shape[1]] = b
        out.append(full)
        rows = k
    return out


def latent_recovery(A_true, A_fit):
    """Per layer: Hungarian-matched |corr| between true and fitted latent columns, averaged over the true columns."""
    from scipy.optimize import linear_sum_assignment

    out = []
    for At, Af in zip(A_true, A_fit):
        At = np.asarray(At, dtype=float)
        Af = np.asarray(Af, dtype=float)
        Af = Af[:, Af.std(axis=0) > 0]
        if Af.shape[1] == 0:
            out.append(0.0)
            continue
        corr = np.abs(np.corrcoef(At.T, Af.T)[: At.shape[1], At.shape[1]:])
        rows, cols = linear_sum_assignment(-corr)
        out.append(float(corr[rows, cols].sum() / At.shape[1]))
    return out


def run_task(task):
    settings_dir, scenario, family, kernel, N, rep = task
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copdde_paper_metrics import load_truth, paper_graph_accuracy

    truth = load_truth(Path(settings_dir) / f"{scenario}.mat")
    seed = 9_000_000 + 1_000 * int(N) + int(rep) + {"normal": 0, "bernoulli": 100, "poisson": 200, "semidde": 300}[family]
    Y, A_true = simulate(truth, family, N, seed)
    started = time.perf_counter()
    row = {"scenario": scenario, "family": family, "kernel": kernel, "N": int(N), "rep": int(rep), "seed": seed}
    try:
        res = run_kernel(kernel, family, Y, truth, N, seed)
        widths = [int(k) for k in res.K_list]
        B_full = fitted_full_B(res, truth)
        A_fit = getattr(res, "A_list", None)
        row.update(status="ok", widths=widths, accG=paper_graph_accuracy(truth, B_full),
                   latent_corr=None if A_fit is None else latent_recovery(A_true, A_fit),
                   iterations=int(res.diagnostics.get("iterations", -1)) if isinstance(res.diagnostics, dict) else -1)
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
    row["seconds"] = time.perf_counter() - started
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--families", nargs="*", default=list(FAMILY_KERNELS))
    ap.add_argument("--scenarios", nargs="*", default=["d2_j50", "d3_j108"])
    ap.add_argument("--N", nargs="*", type=int, default=[500, 1000, 2000, 4000, 8000])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--kernels", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "rows.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add((r["scenario"], r["family"], r["kernel"], r["N"], r["rep"]))
    tasks = []
    for N in args.N:
        for scenario in args.scenarios:
            for family in args.families:
                kernels = FAMILY_KERNELS[family] if args.kernels is None else [k for k in args.kernels if k in FAMILY_KERNELS[family]]
                for kernel in kernels:
                    for rep in range(1, args.reps + 1):
                        if (scenario, family, kernel, N, rep) not in done:
                            tasks.append((str(args.settings), scenario, family, kernel, N, rep))
    print(f"{len(done)} cached, {len(tasks)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, out_file.open("a") as fh:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row["status"] == "ok":
                print(f"{row['scenario']} {row['family']:9s} {row['kernel']:8s} N={row['N']} rep={row['rep']}: "
                      f"K={row['widths']} accG={np.round(row['accG'], 3).tolist()} "
                      f"latent={None if row['latent_corr'] is None else np.round(row['latent_corr'], 2).tolist()} "
                      f"{row['seconds']:.0f}s", flush=True)
            else:
                print(f"ERROR {row['scenario']} {row['family']} {row['kernel']} N={row['N']} rep={row['rep']}: {row['error']}",
                      flush=True)


if __name__ == "__main__":
    main()
