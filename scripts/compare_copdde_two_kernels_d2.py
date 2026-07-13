"""Paired D=2 comparison for the legacy and D-layer CopDDE CSP kernels.

Each replicate uses one simulated dataset, one rank-likelihood initialization,
and one copied parameter initialization.  The legacy two-layer kernel
(`copula_csp.get_SAEM_RL_CSP`) and the package API kernel
(`copula_D.fit_copula_dde_D`) are then run from the same starting point.
"""

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
    vals = [float(x) for x in str(value).split(",") if x]
    if len(vals) != 2:
        raise ValueError("--t-spike must contain two comma-separated values for D=2.")
    return vals


def copy_init(init):
    return {
        "prop": np.asarray(init["prop"], dtype=float).copy(),
        "gamma": np.asarray(init["gamma"], dtype=float).copy(),
        "B_list": [np.asarray(B, dtype=float).copy() for B in init["B_list"]],
        "A_list": [np.asarray(A, dtype=float).copy() for A in init["A_list"]],
    }


def fit_legacy(state, scenario, args):
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
    )
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
        diagnostics={
            "iterations": it_num,
            "num_active": [num_act1, num_act2],
            "theta": theta,
            "theta2": theta2,
            "p": p,
            "q": q,
            "A1_sample_long": A1_sample_long,
            "A2_sample_long": A2_sample_long,
            "K_list": tuple(scenario["K_fit"]),
            "backend": "legacy_copula_csp",
        },
        raw=raw,
    )


def fit_d_layer(state, scenario, args):
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
    )


def aligned_metrics(result, state):
    aligned = format_estimates_D(
        state["true_scaled"],
        result.B_list,
        result.gamma,
        result.prop,
        scale=True,
        Z=result.transformed,
        emp=True,
    )
    row = {}
    for d, (B_hat, B_ref) in enumerate(zip(aligned["B_list"], state["true_scaled"]), start=1):
        row[f"MSE_B{d}"] = float(np.mean((B_hat - B_ref) ** 2))
        row[f"ACC_G{d}"] = float(np.mean((B_hat != 0) == (B_ref != 0)))
    row["MSE_gamma"] = float(np.mean((aligned["gamma"] - state["gamma_scaled"]) ** 2))
    prop_true = np.asarray(state["true"]["prop_true"], dtype=float).reshape(-1)
    prop_est = np.asarray(aligned["prop"], dtype=float).reshape(-1)
    prop_compare = np.full_like(prop_true, np.nan, dtype=float)
    n = min(prop_true.size, prop_est.size)
    prop_compare[:n] = prop_est[:n]
    row["MSE_prop"] = float(np.nanmean((prop_compare - prop_true) ** 2))
    return row, aligned


def summarize_pair(legacy, d_layer, state, scenario, N, rep, elapsed_legacy, elapsed_d):
    legacy_metrics, legacy_aligned = aligned_metrics(legacy, state)
    d_metrics, d_aligned = aligned_metrics(d_layer, state)

    row = {
        "scenario": scenario["scenario"],
        "D": scenario["D"],
        "J": scenario["J"],
        "N": int(N),
        "rep": int(rep + 1),
        "seed": int(state["seed"]),
        "K_fit": tuple(scenario["K_fit"]),
        "elapsed_legacy_sec": float(elapsed_legacy),
        "elapsed_D_sec": float(elapsed_d),
        "iterations_legacy": int(legacy.diagnostics["iterations"]),
        "iterations_D": int(d_layer.diagnostics["iterations"]),
    }
    num_legacy = np.asarray(legacy.diagnostics.get("num_active", np.nan), dtype=float).reshape(-1)
    num_d = np.asarray(d_layer.diagnostics.get("num_active", np.nan), dtype=float).reshape(-1)
    for d in range(2):
        row[f"num_active_legacy_{d + 1}"] = float(num_legacy[d]) if d < num_legacy.size else np.nan
        row[f"num_active_D_{d + 1}"] = float(num_d[d]) if d < num_d.size else np.nan
        row[f"num_active_absdiff_{d + 1}"] = abs(row[f"num_active_legacy_{d + 1}"] - row[f"num_active_D_{d + 1}"])

    for key, value in legacy_metrics.items():
        row[f"legacy_{key}"] = value
    for key, value in d_metrics.items():
        row[f"D_{key}"] = value
        row[f"absdiff_{key}"] = abs(value - legacy_metrics[key])

    for d in range(2):
        B_old = legacy_aligned["B_list"][d]
        B_new = d_aligned["B_list"][d]
        row[f"aligned_B{d + 1}_fro_diff"] = float(np.linalg.norm(B_old - B_new, ord="fro"))
        row[f"aligned_B{d + 1}_max_abs_diff"] = float(np.max(np.abs(B_old - B_new)))
    row["aligned_gamma_fro_diff"] = float(np.linalg.norm(legacy_aligned["gamma"] - d_aligned["gamma"]))
    row["aligned_prop_fro_diff"] = float(np.linalg.norm(legacy_aligned["prop"] - d_aligned["prop"]))
    return row


def run(args):
    scenario = {
        "scenario": f"D2_J{args.J}",
        "D": 2,
        "J": args.J,
        "K_true": [args.K1, args.K2],
        "K_fit": [args.K1, args.K2],
    }
    # Validate that the requested scenario can be constructed before spending time.
    make_true_model(scenario)

    out_dir = PROJECT_ROOT / "Results" / "copdde_kernel_pair_comparison" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for N in args.N:
        for rep in range(args.reps):
            state = initialize_state(scenario, int(N), rep, args.seed_base)

            np.random.seed(args.fit_seed_base + rep)
            start = time.perf_counter()
            legacy = fit_legacy(state, scenario, args)
            elapsed_legacy = time.perf_counter() - start

            np.random.seed(args.fit_seed_base + rep)
            start = time.perf_counter()
            d_layer = fit_d_layer(state, scenario, args)
            elapsed_d = time.perf_counter() - start

            rows.append(summarize_pair(legacy, d_layer, state, scenario, N, rep, elapsed_legacy, elapsed_d))
            pd.DataFrame(rows).to_csv(out_dir / "paired_raw_partial.csv", index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "paired_raw.csv", index=False)

    metric_cols = [
        c
        for c in raw.columns
        if c.startswith("absdiff_")
        or c.endswith("_fro_diff")
        or c.endswith("_max_abs_diff")
        or c.startswith("num_active_absdiff_")
    ]
    summary = raw.groupby(["scenario", "N"], as_index=False).agg(
        reps=("rep", "count"),
        **{f"{c}_mean": (c, "mean") for c in metric_cols},
        **{f"{c}_max": (c, "max") for c in metric_cols},
    )
    summary.to_csv(out_dir / "paired_summary.csv", index=False)

    metadata = vars(args).copy()
    metadata["scenario"] = scenario
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    print(f"Output: {out_dir}")
    print(summary.to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--J", type=int, default=50)
    parser.add_argument("--K1", type=int, default=10)
    parser.add_argument("--K2", type=int, default=3)
    parser.add_argument("--N", type=parse_ints, default=parse_ints("200"))
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=700)
    parser.add_argument("--fit-seed-base", type=int, default=1100)
    parser.add_argument("--it", type=int, default=10)
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.90)
    parser.add_argument("--optimizer-maxiter", type=int, default=10)
    parser.add_argument("--t-spike", type=parse_t_spike, default=parse_t_spike("0.005,0.03"))
    parser.add_argument("--tau", type=float, default=0.6)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
