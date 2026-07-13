"""Targeted trace replay for suspicious D=2 CopDDE kernel replicates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
SCRIPT_DIR = PACKAGE_ROOT / "scripts"
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_csp_comparison import initialize_state, make_true_model
from semidde.algorithms.copula_D import fit_copula_dde_D
from semidde.algorithms.copula_csp import get_SAEM_RL_CSP
from semidde.copula import format_estimates_D
from semidde.results import DDEResult


def parse_ints(value):
    return [int(x) for x in str(value).split(",") if x]


def parse_t_spike(value):
    out = [float(x) for x in str(value).split(",") if x]
    if len(out) != 2:
        raise ValueError("D=2 trace needs exactly two t-spike values.")
    return out


def parse_backends(value):
    backends = [x.strip() for x in str(value).split(",") if x.strip()]
    valid = {"legacy", "D"}
    bad = [x for x in backends if x not in valid]
    if bad:
        raise ValueError(f"Unknown backend(s): {bad}. Use comma-separated values from {sorted(valid)}.")
    return backends


def copy_init(init):
    return {
        "prop": np.asarray(init["prop"], dtype=float).copy(),
        "gamma": np.asarray(init["gamma"], dtype=float).copy(),
        "B_list": [np.asarray(B, dtype=float).copy() for B in init["B_list"]],
        "A_list": [np.asarray(A, dtype=float).copy() for A in init["A_list"]],
    }


def metrics_from_params(state, B_list, gamma, prop, Z):
    aligned = format_estimates_D(
        state["true_scaled"],
        B_list,
        gamma,
        prop,
        scale=True,
        Z=Z,
        emp=True,
    )
    row = {}
    for d, (B_hat, B_ref) in enumerate(zip(aligned["B_list"], state["true_scaled"]), start=1):
        row[f"MSE_B{d}"] = float(np.mean((B_hat - B_ref) ** 2))
        row[f"ACC_G{d}"] = float(np.mean((B_hat != 0) == (B_ref != 0)))
        row[f"B{d}_nnz"] = int(np.count_nonzero(B_hat[:, 1:]))
        row[f"B{d}_slope_fro"] = float(np.linalg.norm(B_hat[:, 1:], ord="fro"))
        row[f"B{d}_slope_maxabs"] = float(np.max(np.abs(B_hat[:, 1:])))
    row["MSE_gamma"] = float(np.mean((aligned["gamma"] - state["gamma_scaled"]) ** 2))
    prop_true = np.asarray(state["true"]["prop_true"], dtype=float).reshape(-1)
    prop_est = np.asarray(aligned["prop"], dtype=float).reshape(-1)
    prop_compare = np.full_like(prop_true, np.nan, dtype=float)
    n = min(prop_true.size, prop_est.size)
    prop_compare[:n] = prop_est[:n]
    row["MSE_prop"] = float(np.nanmean((prop_compare - prop_true) ** 2))
    row["assignments"] = ";".join(
        ",".join(str(int(x)) for x in np.asarray(a).reshape(-1))
        for a in aligned["assignments"]
    )
    return row, aligned


def make_trace_callback(backend, state, rows, N, rep_one_based, fit_seed):
    def callback(info):
        metrics, _ = metrics_from_params(
            state,
            info["B_list"],
            info["gamma"],
            info["prop"],
            info["Z"],
        )
        num_active = np.asarray(info["num_active"], dtype=float).reshape(-1)
        rel = np.asarray(info["rel"], dtype=float).reshape(-1)
        row = {
            "backend": backend,
            "N": int(N),
            "rep": int(rep_one_based),
            "fit_seed": int(fit_seed),
            "iteration": int(info["iteration"]),
            "temp": float(info["temp"]),
            "will_stop": bool(info["will_stop"]),
        }
        for d in range(2):
            row[f"num_active_{d + 1}"] = float(num_active[d]) if d < num_active.size else np.nan
            row[f"rel_{d + 1}"] = float(rel[d]) if d < rel.size else np.nan
            z = np.asarray(info["z_cusp"][d]).reshape(-1)
            row[f"z_cusp_{d + 1}"] = ",".join(str(int(x)) for x in z)
        row.update(metrics)
        rows.append(row)

    return callback


def result_from_legacy(raw):
    (
        num_act1,
        num_act2,
        theta,
        theta2,
        gamma,
        p,
        q,
        A1_sample_long,
        A2_sample_long,
        Z,
        prop,
        B1,
        B2,
        A1,
        A2,
        it_num,
    ) = raw
    return DDEResult(
        model="copula",
        prop=prop,
        B_list=[B1, B2],
        gamma=gamma,
        A_list=[A1, A2],
        transformed=Z,
        diagnostics={"iterations": it_num, "num_active": [num_act1, num_act2]},
        raw=raw,
    )


def fit_legacy_with_trace(state, args, callback):
    init = copy_init(state["init"])
    raw = get_SAEM_RL_CSP(
        X=state["X"],
        Z=state["Z_init"].copy(),
        R=state["R"],
        Rlevels=state["Rlevels"],
        prop_in=init["prop"],
        gamma_in=init["gamma"],
        B1_in=init["B_list"][0],
        B2_in=init["B_list"][1],
        A1_in=init["A_list"][0],
        A2_in=init["A_list"][1],
        C=args.C,
        it=args.it,
        t1=args.t_spike[0],
        t2=args.t_spike[1],
        temp=args.temp,
        tau=args.tau,
        optimizer_maxiter=args.optimizer_maxiter,
        n_jobs=1,
        use_analytic_objectives=True,
        verbose=False,
        trace_callback=callback,
    )
    return result_from_legacy(raw)


def fit_d_with_trace(state, scenario, args, callback):
    init = copy_init(state["init"])
    return fit_copula_dde_D(
        state["X"],
        K_list=scenario["K_fit"],
        dimension_method="csp",
        Z=state["Z_init"].copy(),
        R=state["R"],
        Rlevels=state["Rlevels"],
        prop_in=init["prop"],
        gamma_in=init["gamma"],
        B_list_in=init["B_list"],
        A_list_in=init["A_list"],
        C=args.C,
        it=args.it,
        t_spike=np.asarray(args.t_spike, dtype=float),
        temp=args.temp,
        tau=args.tau,
        optimizer_maxiter=args.optimizer_maxiter,
        n_jobs=1,
        verbose=False,
        trace_callback=callback,
    )


def final_summary_row(backend, result, state, N, rep_one_based, fit_seed, elapsed):
    metrics, aligned = metrics_from_params(state, result.B_list, result.gamma, result.prop, result.transformed)
    num_active = np.asarray(result.diagnostics.get("num_active", np.nan), dtype=float).reshape(-1)
    row = {
        "backend": backend,
        "N": int(N),
        "rep": int(rep_one_based),
        "seed": int(state["seed"]),
        "fit_seed": int(fit_seed),
        "elapsed_sec": float(elapsed),
        "iterations": int(result.diagnostics.get("iterations", -1)),
        "num_active_1": float(num_active[0]) if num_active.size > 0 else np.nan,
        "num_active_2": float(num_active[1]) if num_active.size > 1 else np.nan,
    }
    row.update(metrics)
    return row, aligned


def run(args):
    scenario = {
        "scenario": f"D2_J{args.J}",
        "D": 2,
        "J": args.J,
        "K_true": [args.K1, args.K2],
        "K_fit": [args.K1, args.K2],
    }
    make_true_model(scenario)
    out_dir = PROJECT_ROOT / "Results" / "copdde_kernel_pair_comparison" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = []
    summary_rows = []
    matrices = {}
    reps = parse_ints(args.reps)
    fit_seeds = parse_ints(args.fit_seeds) if args.fit_seeds else None
    backends = parse_backends(args.backends)

    for N in args.N:
        for rep_one_based in reps:
            rep_index = int(rep_one_based) - 1
            state = initialize_state(scenario, int(N), rep_index, args.seed_base)
            prefix = f"N{int(N)}_rep{int(rep_one_based)}"

            seeds_for_rep = fit_seeds if fit_seeds is not None else [args.fit_seed_base + rep_index]
            for fit_seed in seeds_for_rep:
                for backend in backends:
                    callback = make_trace_callback(backend, state, trace_rows, int(N), int(rep_one_based), fit_seed)
                    np.random.seed(int(fit_seed))
                    start = time.perf_counter()
                    if backend == "legacy":
                        result = fit_legacy_with_trace(state, args, callback)
                    else:
                        result = fit_d_with_trace(state, scenario, args, callback)
                    elapsed = time.perf_counter() - start
                    summary, aligned = final_summary_row(
                        backend,
                        result,
                        state,
                        int(N),
                        int(rep_one_based),
                        int(fit_seed),
                        elapsed,
                    )
                    summary_rows.append(summary)

                    key = f"{prefix}_fitseed{int(fit_seed)}_{backend}"
                    matrices[f"{key}_B1_raw"] = np.asarray(result.B_list[0])
                    matrices[f"{key}_B2_raw"] = np.asarray(result.B_list[1])
                    matrices[f"{key}_B1_aligned"] = np.asarray(aligned["B_list"][0])
                    matrices[f"{key}_B2_aligned"] = np.asarray(aligned["B_list"][1])
                    matrices[f"{key}_gamma"] = np.asarray(result.gamma)
                    matrices[f"{key}_prop"] = np.asarray(result.prop)

            pd.DataFrame(trace_rows).to_csv(out_dir / "targeted_trace_partial.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(out_dir / "targeted_summary_partial.csv", index=False)

    trace = pd.DataFrame(trace_rows)
    summary = pd.DataFrame(summary_rows)
    trace.to_csv(out_dir / "targeted_trace.csv", index=False)
    summary.to_csv(out_dir / "targeted_summary.csv", index=False)
    np.savez_compressed(out_dir / "targeted_matrices.npz", **matrices)

    metadata = vars(args).copy()
    metadata["scenario"] = scenario
    (out_dir / "targeted_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    print(f"Output: {out_dir}")
    print(summary.to_string(index=False))
    if not trace.empty:
        print("\nLast trace rows:")
        keep = [
            "backend",
            "N",
            "rep",
            "fit_seed",
            "iteration",
            "MSE_B1",
            "MSE_B2",
            "ACC_G2",
            "num_active_1",
            "num_active_2",
            "rel_1",
            "rel_2",
            "assignments",
        ]
        print(trace[keep].groupby(["backend", "N", "rep", "fit_seed"], group_keys=False).tail(3).to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="d2_j100_targeted_trace")
    parser.add_argument("--J", type=int, default=100)
    parser.add_argument("--K1", type=int, default=10)
    parser.add_argument("--K2", type=int, default=3)
    parser.add_argument("--N", type=parse_ints, default=parse_ints("500"))
    parser.add_argument("--reps", default="7,48", help="1-based replicate numbers, comma separated.")
    parser.add_argument("--seed-base", type=int, default=700)
    parser.add_argument("--fit-seed-base", type=int, default=1100)
    parser.add_argument(
        "--fit-seeds",
        default=None,
        help="Exact fit RNG seeds to replay for each requested replicate. Overrides --fit-seed-base.",
    )
    parser.add_argument("--backends", default="legacy,D", help="Comma-separated subset of legacy,D.")
    parser.add_argument("--it", type=int, default=50)
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.90)
    parser.add_argument("--optimizer-maxiter", type=int, default=20)
    parser.add_argument("--t-spike", type=parse_t_spike, default=parse_t_spike("0.005,0.03"))
    parser.add_argument("--tau", type=float, default=0.6)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
