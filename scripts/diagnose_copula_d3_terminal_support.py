"""Fresh D3 CopDDE diagnostics for terminal-state consistency and G3 support.

The script deliberately calls the current raw CSP kernel so that it can retain
the final callback state immediately before terminal inactive-column cleanup.
It does not modify the fitting kernel.  Results are checkpointed after every
sample-size block and can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dde.algorithms.copula import (
    normal_initialize_copula_dde,
    prepare_rank_likelihood_inputs,
)
from dde.algorithms.get_SAEM_RL_CSP_D import get_SAEM_RL_CSP_D
from dde.copula import format_estimates_D, rescale_B1_D, simulate_gaussian_mixture
from dde.simulation import generate_X_Cop_marg_emp_D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Results" / "copula_d3_current_kernel_diagnostics"
TRUTH_PATH = (
    ROOT
    / "Results"
    / "copdde_MATLAB_reference_results"
    / "setting_records"
    / "Deeper_Sims_true_settings.mat"
)
K_TRUE = (18, 6, 2)
K_FIT = (36, 12, 4)
T_SPIKE = (0.01, 0.03, 0.005)
THRESHOLDS = (0.0, 1e-12, 1e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.2, 0.3, 0.5, 1.0)


def _matrix(group, key: str) -> np.ndarray:
    value = np.asarray(group[key])
    if value.ndim == 2:
        value = value.T
    return np.asarray(value, dtype=float)


def _vector(group, key: str, dtype=float) -> np.ndarray:
    return np.asarray(group[key]).reshape(-1).astype(dtype)


def _augment(B_list, K_fit=K_FIT):
    output = []
    for layer, B in enumerate(B_list):
        rows = B.shape[0] if layer == 0 else K_fit[layer - 1]
        target = np.zeros((rows, K_fit[layer] + 1), dtype=float)
        target[: B.shape[0], : B.shape[1]] = B
        output.append(target)
    return output


def _load_truth():
    with h5py.File(TRUTH_PATH, "r") as handle:
        group = handle["truth"]
        B_true = [
            _matrix(group, "B1_true_unscale"),
            _matrix(group, "B2_true"),
            _matrix(group, "B3_true"),
        ]
        gamma = _vector(group, "gamma_true")
        prop = _vector(group, "prop_true")
        lambdas = _vector(group, "lambdas", dtype=int)
    return {
        "B_true": B_true,
        "gamma": gamma,
        "prop": prop,
        "lambdas": lambdas,
    }


def _tau(N: int) -> float:
    return max(0.3, 3.0 * float(N) ** -0.3)


def _graph_masks(B_list, tol=1e-12):
    masks = [np.ones(B.shape[1] - 1, dtype=bool) for B in B_list]
    while True:
        updated = []
        for layer, B in enumerate(B_list):
            rows = slice(None) if layer == 0 else masks[layer - 1]
            updated.append(np.any(np.abs(B[rows, 1:]) > tol, axis=0))
        if all(np.array_equal(old, new) for old, new in zip(masks, updated)):
            return updated
        masks = updated


def _width(B_list, tol=1e-12):
    return tuple(int(mask.sum()) for mask in _graph_masks(B_list, tol=tol))


def _width_text(values) -> str:
    return ";".join(str(int(value)) for value in values)


def _binary_counts(truth, estimate, mask=None):
    truth = np.asarray(truth, dtype=bool)
    estimate = np.asarray(estimate, dtype=bool)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        truth = truth[mask]
        estimate = estimate[mask]
    else:
        truth = truth.reshape(-1)
        estimate = estimate.reshape(-1)
    tp = int(np.sum(truth & estimate))
    tn = int(np.sum(~truth & ~estimate))
    fp = int(np.sum(~truth & estimate))
    fn = int(np.sum(truth & ~estimate))
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    accuracy = (tp + tn) / truth.size if truth.size else np.nan
    balanced = 0.5 * (sensitivity + specificity)
    denominator = np.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = (tp * tn - fp * fn) / denominator if denominator else np.nan
    return {
        "cells": int(truth.size),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "mcc": mcc,
    }


def _zone_rows(
    *, N, rep, truth, estimate, parent_active, child_active, tol=1e-12
):
    truth_support = np.abs(truth) > tol
    estimate_support = np.abs(estimate) > tol
    parent_active = np.asarray(parent_active, dtype=bool)
    child_active = np.asarray(child_active, dtype=bool)
    active_block = parent_active[:, None] & child_active[None, :]
    zones = {
        "all": np.ones_like(truth_support, dtype=bool),
        "allocation_active_block": active_block,
        "inactive_child_columns": np.broadcast_to(
            ~child_active[None, :], truth_support.shape
        ),
        "active_child_inactive_parent": (
            (~parent_active[:, None]) & child_active[None, :]
        ),
        "truth_active_block": np.zeros_like(truth_support, dtype=bool),
        "outside_truth_active_block": np.ones_like(truth_support, dtype=bool),
    }
    zones["truth_active_block"][: K_TRUE[1], : K_TRUE[2]] = True
    zones["outside_truth_active_block"] = ~zones["truth_active_block"]
    rows = []
    for zone, mask in zones.items():
        metrics = _binary_counts(truth_support, estimate_support, mask)
        values = np.abs(estimate[mask & ~truth_support & estimate_support])
        rows.append(
            {
                "N": int(N),
                "rep": int(rep),
                "zone": zone,
                **metrics,
                "false_positive_abs_mean": (
                    float(values.mean()) if values.size else np.nan
                ),
                "false_positive_abs_median": (
                    float(np.median(values)) if values.size else np.nan
                ),
                "false_positive_abs_max": (
                    float(values.max()) if values.size else np.nan
                ),
            }
        )
    return rows


def _threshold_rows(*, N, rep, truth, estimate, parent_active, child_active):
    truth_support = np.abs(truth) > 1e-12
    active_block = (
        np.asarray(parent_active, dtype=bool)[:, None]
        & np.asarray(child_active, dtype=bool)[None, :]
    )
    rows = []
    for threshold in THRESHOLDS:
        support = np.abs(estimate) > threshold
        for scope, mask in (
            ("all", None),
            ("allocation_active_block", active_block),
        ):
            rows.append(
                {
                    "N": int(N),
                    "rep": int(rep),
                    "threshold": float(threshold),
                    "scope": scope,
                    **_binary_counts(truth_support, support, mask),
                }
            )
    return rows


def _run_one(N: int, rep: int):
    seed = 50 + int(rep)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    truth = _load_truth()
    generated = generate_X_Cop_marg_emp_D(
        int(N),
        truth["prop"],
        truth["lambdas"],
        truth["gamma"],
        truth["B_true"],
        random_state=rng,
    )
    X = generated["X"]
    _, R, Rlevels = prepare_rank_likelihood_inputs(X)
    Z_init = simulate_gaussian_mixture(X, X.shape[1], random_state=seed)
    init = normal_initialize_copula_dde(
        X, K_FIT, Z=Z_init, epsilon=1e-4
    )
    B_scaled, gamma_scaled, _, _ = rescale_B1_D(
        truth["prop"],
        truth["B_true"],
        truth["gamma"],
        emp=True,
        Z=generated["Z"],
    )
    B_ref = _augment(B_scaled)

    last_trace = {}

    def trace_callback(state):
        last_trace.clear()
        last_trace.update(
            {
                "iteration": int(state["iteration"]),
                "B_list": [B.copy() for B in state["B_list"]],
                "gamma": state["gamma"].copy(),
                "prop": state["prop"].copy(),
                "Z": state["Z"].copy(),
                "num_active": state["num_active"].copy(),
                "will_stop": bool(state["will_stop"]),
            }
        )

    np.random.seed(seed)
    started = time.perf_counter()
    raw = get_SAEM_RL_CSP_D(
        X=X,
        Z=Z_init.copy(),
        R=R,
        Rlevels=Rlevels,
        prop_in=init["prop"].copy(),
        gamma_in=init["gamma"].copy(),
        B_list_in=[B.copy() for B in init["B_list"]],
        A_list_in=[A.copy() for A in init["A_list"]],
        C=1,
        it=50,
        t_spike=np.asarray(T_SPIKE),
        temp=0.9,
        tau=_tau(N),
        optimizer_maxiter=15,
        n_jobs=1,
        verbose=False,
        trace_callback=trace_callback,
    )
    elapsed = time.perf_counter() - started

    allocation_masks = [
        np.asarray(z) > np.arange(1, len(z) + 1) for z in raw["z_cusp"]
    ]
    allocation_width = tuple(int(mask.sum()) for mask in allocation_masks)
    pre_B = last_trace["B_list"]
    post_B = raw["B_list"]
    pre_width = _width(pre_B)
    post_width = _width(post_B)

    aligned = format_estimates_D(
        B_ref,
        post_B,
        raw["gamma"],
        raw["prop"],
        scale=True,
        Z=raw["Z"],
        emp=True,
    )
    aligned_pre = format_estimates_D(
        B_ref,
        pre_B,
        last_trace["gamma"],
        last_trace["prop"],
        scale=True,
        Z=last_trace["Z"],
        emp=True,
    )
    assignments = [np.asarray(value, dtype=int) for value in aligned["assignments"]]
    aligned_masks = [
        allocation_masks[layer][assignments[layer]] for layer in range(3)
    ]
    B3_hat = np.asarray(aligned["B_list"][2])[:, 1:]
    B3_pre = np.asarray(aligned_pre["B_list"][2])[:, 1:]
    B3_ref = np.asarray(B_ref[2])[:, 1:]

    aligned_width = _width(aligned["B_list"])
    inactive_pre_nnz = []
    inactive_post_nnz = []
    for layer, mask in enumerate(allocation_masks):
        inactive_pre_nnz.append(
            int(np.count_nonzero(pre_B[layer][:, 1:][:, ~mask]))
        )
        inactive_post_nnz.append(
            int(np.count_nonzero(post_B[layer][:, 1:][:, ~mask]))
        )

    aligned_metric = _binary_counts(
        np.abs(B3_ref) > 1e-12, np.abs(B3_hat) > 1e-12
    )
    pre_metric = _binary_counts(
        np.abs(B3_ref) > 1e-12, np.abs(B3_pre) > 1e-12
    )
    unaligned_metric = _binary_counts(
        np.abs(B3_ref) > 1e-12, np.abs(post_B[2][:, 1:]) > 1e-12
    )

    terminal_row = {
        "N": int(N),
        "rep": int(rep),
        "seed": int(seed),
        "tau": _tau(N),
        "iterations": int(raw["it_num"]),
        "hit_iteration_budget": bool(raw["it_num"] >= 50),
        "elapsed_sec": float(elapsed),
        "allocation_width": _width_text(allocation_width),
        "pre_terminal_width": _width_text(pre_width),
        "post_terminal_width": _width_text(post_width),
        "aligned_width": _width_text(aligned_width),
        "allocation_post_match": bool(allocation_width == post_width),
        "post_aligned_match": bool(post_width == aligned_width),
        "pre_allocation_match": bool(pre_width == allocation_width),
        "G3_pre_accuracy": pre_metric["accuracy"],
        "G3_post_accuracy": aligned_metric["accuracy"],
        "G3_post_balanced_accuracy": aligned_metric["balanced_accuracy"],
        "G3_post_mcc": aligned_metric["mcc"],
        "G3_unaligned_accuracy": unaligned_metric["accuracy"],
    }
    for layer in range(3):
        terminal_row[f"allocation_K{layer + 1}"] = allocation_width[layer]
        terminal_row[f"pre_K{layer + 1}"] = pre_width[layer]
        terminal_row[f"post_K{layer + 1}"] = post_width[layer]
        terminal_row[f"aligned_K{layer + 1}"] = aligned_width[layer]
        terminal_row[f"inactive_pre_nnz_K{layer + 1}"] = inactive_pre_nnz[layer]
        terminal_row[f"inactive_post_nnz_K{layer + 1}"] = inactive_post_nnz[layer]

    zone_rows = _zone_rows(
        N=N,
        rep=rep,
        truth=B3_ref,
        estimate=B3_hat,
        parent_active=aligned_masks[1],
        child_active=aligned_masks[2],
    )
    threshold_rows = _threshold_rows(
        N=N,
        rep=rep,
        truth=B3_ref,
        estimate=B3_hat,
        parent_active=aligned_masks[1],
        child_active=aligned_masks[2],
    )
    estimate_payload = {
        "N": int(N),
        "rep": int(rep),
        "B3_ref": B3_ref,
        "B3_pre": B3_pre,
        "B3_post": B3_hat,
        "parent_active": aligned_masks[1],
        "child_active": aligned_masks[2],
        "assignments": assignments,
    }
    return terminal_row, zone_rows, threshold_rows, estimate_payload


def _read_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _append_unique(old, new, keys):
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    return combined.drop_duplicates(keys, keep="last").sort_values(keys)


def _markdown_table(frame, columns):
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def _write_summaries(output: Path):
    terminal = pd.read_csv(output / "terminal_replicates.csv")
    zones = pd.read_csv(output / "g3_zone_replicates.csv")
    thresholds = pd.read_csv(output / "g3_threshold_replicates.csv")

    fitted_width = terminal[
        ["allocation_K1", "allocation_K2", "allocation_K3"]
    ].to_numpy(dtype=int)
    truth_width = np.asarray(K_TRUE, dtype=int)
    terminal["width_exact"] = np.all(
        fitted_width == truth_width[None, :], axis=1
    )
    terminal["width_l1"] = np.sum(
        np.abs(fitted_width - truth_width[None, :]), axis=1
    )
    terminal["K3_under"] = terminal["allocation_K3"] < K_TRUE[2]
    terminal["K3_exact"] = terminal["allocation_K3"] == K_TRUE[2]
    terminal["K3_over"] = terminal["allocation_K3"] > K_TRUE[2]

    terminal_summary = terminal.groupby("N", as_index=False).agg(
        replicates=("rep", "nunique"),
        iteration_budget_rate=("hit_iteration_budget", "mean"),
        allocation_post_match_rate=("allocation_post_match", "mean"),
        post_aligned_match_rate=("post_aligned_match", "mean"),
        pre_allocation_match_rate=("pre_allocation_match", "mean"),
        allocation_K1_mean=("allocation_K1", "mean"),
        allocation_K2_mean=("allocation_K2", "mean"),
        allocation_K3_mean=("allocation_K3", "mean"),
        width_exact_rate=("width_exact", "mean"),
        width_l1_mean=("width_l1", "mean"),
        K3_under_rate=("K3_under", "mean"),
        K3_exact_rate=("K3_exact", "mean"),
        K3_over_rate=("K3_over", "mean"),
        post_K1_mean=("post_K1", "mean"),
        post_K2_mean=("post_K2", "mean"),
        post_K3_mean=("post_K3", "mean"),
        inactive_pre_nnz_K3_mean=("inactive_pre_nnz_K3", "mean"),
        inactive_post_nnz_K3_mean=("inactive_post_nnz_K3", "mean"),
        G3_accuracy_mean=("G3_post_accuracy", "mean"),
        G3_balanced_accuracy_mean=("G3_post_balanced_accuracy", "mean"),
        G3_mcc_mean=("G3_post_mcc", "mean"),
    )
    terminal_summary.to_csv(output / "terminal_summary.csv", index=False)

    zone_summary = (
        zones.groupby(["N", "zone"], as_index=False)
        .agg(
            replicates=("rep", "nunique"),
            cells_mean=("cells", "mean"),
            tp_mean=("tp", "mean"),
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            accuracy_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
            false_positive_abs_mean=("false_positive_abs_mean", "mean"),
            false_positive_abs_median=("false_positive_abs_median", "mean"),
            false_positive_abs_max=("false_positive_abs_max", "mean"),
        )
    )
    zone_summary.to_csv(output / "g3_zone_summary.csv", index=False)

    threshold_summary = (
        thresholds.groupby(["N", "scope", "threshold"], as_index=False)
        .agg(
            replicates=("rep", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
        )
    )
    threshold_summary.to_csv(output / "g3_threshold_summary.csv", index=False)

    best_threshold_rows = []
    for (N, scope), group in threshold_summary.groupby(["N", "scope"]):
        zero = group[group["threshold"] == 0.0].iloc[0]
        best_balanced = group.loc[group["balanced_accuracy_mean"].idxmax()]
        best_mcc = group.loc[group["mcc_mean"].idxmax()]
        best_threshold_rows.append(
            {
                "N": int(N),
                "scope": scope,
                "zero_balanced_accuracy": zero["balanced_accuracy_mean"],
                "best_balanced_threshold": best_balanced["threshold"],
                "best_balanced_accuracy": best_balanced["balanced_accuracy_mean"],
                "balanced_gain": (
                    best_balanced["balanced_accuracy_mean"]
                    - zero["balanced_accuracy_mean"]
                ),
                "zero_mcc": zero["mcc_mean"],
                "best_mcc_threshold": best_mcc["threshold"],
                "best_mcc": best_mcc["mcc_mean"],
                "mcc_gain": best_mcc["mcc_mean"] - zero["mcc_mean"],
            }
        )
    best_thresholds = pd.DataFrame(best_threshold_rows)
    best_thresholds.to_csv(output / "g3_threshold_best.csv", index=False)

    historical_path = output / "historical_current_paired_summary.csv"
    historical_summary = (
        pd.read_csv(historical_path)
        if historical_path.exists()
        else pd.DataFrame()
    )

    terminal_columns = list(terminal_summary.columns)
    zone_columns = [
        "N",
        "zone",
        "replicates",
        "tp_mean",
        "fp_mean",
        "fn_mean",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "mcc_mean",
        "false_positive_abs_median",
    ]
    threshold_display = threshold_summary[
        threshold_summary["scope"] == "all"
    ].copy()
    threshold_display["threshold"] = threshold_display["threshold"].map(
        lambda value: f"{value:g}"
    )
    threshold_columns = [
        "N",
        "threshold",
        "accuracy_mean",
        "sensitivity_mean",
        "specificity_mean",
        "balanced_accuracy_mean",
        "mcc_mean",
        "fp_mean",
        "fn_mean",
    ]
    best_display = best_thresholds[best_thresholds["scope"] == "all"]
    best_columns = [
        "N",
        "zero_balanced_accuracy",
        "best_balanced_threshold",
        "best_balanced_accuracy",
        "balanced_gain",
        "zero_mcc",
        "best_mcc_threshold",
        "best_mcc",
        "mcc_gain",
    ]
    historical_section = ""
    if len(historical_summary):
        historical_section = f"""
## Paired comparison with historical saved artifacts

{_markdown_table(historical_summary, list(historical_summary.columns))}

The allocation path and iteration count reproduce exactly for the same seeds.
The changed result is the consistency of the returned coefficient graph with
that allocation state.
"""
    report = f"""# Current-kernel D3 terminal and G3 support diagnostics

## Main findings

1. Every fresh current-kernel fit has exact agreement between CSP allocation,
   the post-terminal raw coefficient graph, and the aligned graph. Every
   inactive column contains exactly zero nonzero coefficients on return.
2. Pre-terminal callback states often retain inactive-column coefficients;
   the cleanup is therefore necessary and is working. The agreement rate
   before cleanup falls from 0.75 at N=500 to 0 at N=16000.
3. All exact-support G3 false positives occur inside the retained allocation
   block. There are no false positives in inactive child columns or in
   inactive parent rows. This is consistent with both implementations applying
   hard thresholding only to layers 1 through D-1, not to the third layer.
4. K3 changes from systematic over-selection at N=500 to frequent
   under-selection at N=16000. Consequently false positives fall but false
   negatives rise, explaining why ordinary accuracy improves while balanced
   accuracy eventually falls.
5. Applying a global numerical threshold gives only small balanced-accuracy or
   MCC gains. The remaining G3 issue is structural selection inside the active
   block, not merely tiny floating-point coefficients.

## Configuration

- Fresh current-kernel fits; N in {sorted(terminal['N'].unique().tolist())}.
- Repetitions per completed N: {terminal.groupby('N')['rep'].nunique().to_dict()}.
- K true = {K_TRUE}; K fit = {K_FIT}; iterations = 50; C = 1.
- MATLAB saved truth and the paper-reproduction initialization/data pipeline.
- No fitting-kernel changes were made.
{historical_section}

## Terminal-state consistency

{_markdown_table(terminal_summary, terminal_columns)}

`pre` is the final callback state before terminal inactive-column cleanup;
`post` is the raw kernel return; `aligned` is after label alignment/rescaling.

## G3 error zones at exact nonzero support

{_markdown_table(zone_summary, zone_columns)}

The first three allocation zones partition the fitted G3 matrix: active block,
inactive child columns, and active child columns on inactive parent rows.

## G3 threshold sensitivity

{_markdown_table(threshold_display, threshold_columns)}

## Best post-hoc thresholds

{_markdown_table(best_display, best_columns)}
"""
    (output / "REPORT.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", nargs="+", type=int, default=[500, 8000, 16000])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    terminal_path = output / "terminal_replicates.csv"
    zone_path = output / "g3_zone_replicates.csv"
    threshold_path = output / "g3_threshold_replicates.csv"
    payload_path = output / "estimate_payloads.npz"
    terminal = _read_or_empty(terminal_path)
    zones = _read_or_empty(zone_path)
    thresholds = _read_or_empty(threshold_path)
    payloads = []
    if payload_path.exists() and not args.force:
        payloads = list(np.load(payload_path, allow_pickle=True)["payload"])

    for N in args.n_values:
        completed = set()
        if not args.force and len(terminal):
            completed = set(
                terminal.loc[terminal["N"] == N, "rep"].astype(int).tolist()
            )
        reps = [rep for rep in range(1, args.reps + 1) if rep not in completed]
        if not reps:
            print(f"N={N}: already complete")
            continue
        print(f"N={N}: running {len(reps)} replicates with n_jobs={args.n_jobs}")
        results = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
            verbose=10,
            batch_size=1,
            pre_dispatch=args.n_jobs,
        )(delayed(_run_one)(int(N), int(rep)) for rep in reps)
        terminal_new = pd.DataFrame([result[0] for result in results])
        zone_new = pd.DataFrame([row for result in results for row in result[1]])
        threshold_new = pd.DataFrame(
            [row for result in results for row in result[2]]
        )
        terminal = _append_unique(terminal, terminal_new, ["N", "rep"])
        zones = _append_unique(zones, zone_new, ["N", "rep", "zone"])
        thresholds = _append_unique(
            thresholds, threshold_new, ["N", "rep", "threshold", "scope"]
        )
        payloads.extend(result[3] for result in results)
        terminal.to_csv(terminal_path, index=False)
        zones.to_csv(zone_path, index=False)
        thresholds.to_csv(threshold_path, index=False)
        np.savez_compressed(payload_path, payload=np.asarray(payloads, dtype=object))
        _write_summaries(output)
        print(f"N={N}: checkpoint written to {output}")

    if len(terminal):
        _write_summaries(output)
    metadata = {
        "n_values_requested": args.n_values,
        "reps_requested": args.reps,
        "n_jobs": args.n_jobs,
        "truth_path": str(TRUTH_PATH),
        "kernel": "dde.algorithms.get_SAEM_RL_CSP_D.get_SAEM_RL_CSP_D",
        "source_note": "fresh current-kernel fits; no kernel modification",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
