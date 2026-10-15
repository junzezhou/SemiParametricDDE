"""Summarize the checkpointed full-grid validation and paper comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_hp_selection_validation import DEFAULT_OUTPUT, KERNELS, N_VALUES


COPULA_PAPER_WIDTHS = {
    2: {500: (9.650, 2.340), 1000: (10.060, 2.380),
        2000: (10.130, 2.470), 4000: (10.190, 2.520),
        8000: (10.250, 2.690)},
    3: {500: (17.660, 9.820, 3.270), 1000: (17.890, 8.930, 3.420),
        2000: (17.990, 7.750, 3.260), 4000: (18.080, 6.660, 2.740),
        8000: (18.040, 6.220, 2.260)},
}

DDE_PAPER_RMSE_D2 = {
    "dde_normal_fixed": (0.189, 0.156, 0.147, 0.137, 0.133),
    "dde_bernoulli_fixed": (0.498, 0.431, 0.386, 0.359, 0.341),
    "dde_poisson_fixed": (0.281, 0.258, 0.245, 0.243, 0.242),
}
DDE_PAPER_PEM_RMSE_D2 = {
    "dde_normal_fixed": (0.170, 0.128, 0.080, 0.063, 0.054),
    "dde_bernoulli_fixed": (0.404, 0.304, 0.231, 0.203, 0.184),
    "dde_poisson_fixed": (0.219, 0.158, 0.108, 0.080, 0.061),
}


def _paper_comparison(rows: pd.DataFrame, root: Path) -> pd.DataFrame:
    semi = pd.read_csv(root / "Results/paper_results/simulation_semidde_summary.csv")
    entries = []
    for _, row in rows.iterrows():
        if row["status"] != "ok":
            continue
        kernel, depth, N = row["kernel"], int(row["D"]), int(row["N"])
        entry = {"kernel": kernel, "D": depth, "N": N,
                 "replicate": int(row["replicate"]), "comparison_type": "none",
                 "selected_metric": np.nan, "paper_metric": np.nan,
                 "paper_pem_metric": np.nan, "selected_minus_paper": np.nan, "paper_source": "",
                 "setting_note": "No paper value for this exact endpoint and design."}
        if kernel in DDE_PAPER_RMSE_D2 and depth == 2:
            table = {"dde_bernoulli_fixed": "S.9", "dde_poisson_fixed": "S.10",
                     "dde_normal_fixed": "S.11"}[kernel]
            entry.update(comparison_type="parameter_rmse",
                         selected_metric=row["parameter_rmse_thresholded"],
                         paper_metric=DDE_PAPER_RMSE_D2[kernel][list(N_VALUES).index(N)],
                         paper_pem_metric=DDE_PAPER_PEM_RMSE_D2[kernel][list(N_VALUES).index(N)],
                         paper_source=f"DDE paper Eq. S.36 / Table {table}, SAEM and PEM B_s",
                         setting_note="Same sparse B_s design and overall parameter RMSE; paper reports SAEM and the stronger PEM algorithm over 100 replicates.")
        elif kernel == "semidde_fixed" and depth == 2:
            entry.update(comparison_type="parameter_rmse",
                         selected_metric=row["parameter_rmse_thresholded"],
                         paper_metric=float(semi.loc[semi.N == N, "theta_rmse"].iloc[0]),
                         paper_source="SemiparametricDDE paper simulation table",
                         setting_note="Same log-normal D=2 design and default iteration budget; selected fit uses data initialization, paper summary uses oracle initialization and 100 replicates.")
        elif kernel == "copula_csp":
            fitted = np.asarray(json.loads(row["fitted_K"]), dtype=float)
            paper = np.asarray(COPULA_PAPER_WIDTHS[depth][N])
            entry.update(comparison_type="mean_layer_width_reference",
                         selected_metric=json.dumps(fitted.tolist()),
                         paper_metric=json.dumps(paper.tolist()),
                         paper_source="Current SemiDDE manuscript simulation table (D=2 J=100) or CopDDE paper Table S.3 (D=3 J=108)",
                         setting_note="Same stored MATLAB true setting; individual selected widths versus published 100-replicate mean widths. Paper does not report all-layer exact recovery.")
        if entry["comparison_type"] == "parameter_rmse":
            entry["selected_minus_paper"] = float(entry["selected_metric"] - entry["paper_metric"])
        entries.append(entry)
    return pd.DataFrame(entries)


def _plot_fixed(rows: pd.DataFrame, paper: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for i, kernel in enumerate([k for k in KERNELS if k.endswith("fixed")]):
        ax = axes.flat[i]
        subset = rows[(rows.kernel == kernel) & (rows.status == "ok")]
        for depth, color in ((2, "#2374AB"), (3, "#D05A47")):
            selected = (subset[subset.D == depth]
                        .groupby("N", as_index=False).parameter_mse_thresholded.mean()
                        .sort_values("N"))
            if not selected.empty:
                ax.plot(selected.N, np.sqrt(selected.parameter_mse_thresholded),
                        marker="o", color=color, label=f"D={depth}")
            reference = paper[(paper.kernel == kernel) & (paper.D == depth)
                              & (paper.comparison_type == "parameter_rmse")].sort_values("N").drop_duplicates("N")
            if not reference.empty:
                ax.plot(reference.N, pd.to_numeric(reference.paper_metric),
                        linestyle="--", color=color, alpha=.75, label=f"paper SAEM D={depth}" if kernel.startswith("dde_") else f"paper D={depth}")
                if kernel.startswith("dde_"):
                    ax.plot(reference.N, pd.to_numeric(reference.paper_pem_metric),
                            linestyle=":", color=color, alpha=.9, label="paper PEM D=2")
        ax.set_title(kernel.replace("_", " "))
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
        ax.grid(alpha=.25)
        ax.set_ylabel("Overall parameter RMSE")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Selected fixed-width parameters; paper lines where comparable")
    fig.tight_layout()
    fig.savefig(output / "fixed_parameter_rmse.png", dpi=180)
    plt.close(fig)


def _plot_csp(rows: pd.DataFrame, output: Path):
    kernels = [k for k in KERNELS if k.endswith("csp")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, depth in zip(axes, (2, 3)):
        values = np.full((len(kernels), len(N_VALUES)), np.nan)
        for i, kernel in enumerate(kernels):
            for j, n in enumerate(N_VALUES):
                match = rows[(rows.kernel == kernel) & (rows.D == depth)
                             & (rows.N == n) & (rows.status == "ok")]
                if not match.empty:
                    values[i, j] = float(match.dimension_exact.mean())
        image = ax.imshow(values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        for i in range(len(kernels)):
            for j in range(len(N_VALUES)):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i,j]:.0%}", ha="center", va="center", fontsize=9)
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=11)
        ax.set_title(f"D={depth}")
        ax.set_xticks(range(len(N_VALUES)), N_VALUES)
        ax.set_yticks(range(len(kernels)), [k.replace("_", " ") for k in kernels])
        ax.set_xlabel("N")
    fig.colorbar(image, ax=axes, fraction=.025, pad=.02, label="Exact in sampled run (one per cell)")
    fig.subplots_adjust(left=.17, right=.9, bottom=.13, top=.88, wspace=.16)
    fig.savefig(output / "csp_exact_recovery.png", dpi=180)
    plt.close(fig)


def _plot_speed(rows: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(KERNELS)))
    for ax, depth in zip(axes, (2, 3)):
        for kernel, color in zip(KERNELS, colors):
            selected = (rows[(rows.kernel == kernel) & (rows.D == depth)
                             & (rows.status == "ok")]
                        .groupby("N", as_index=False).candidate_timed_seconds.mean()
                        .sort_values("N"))
            if not selected.empty:
                ax.plot(selected.N, selected.candidate_timed_seconds, marker="o", label=kernel, color=color)
        ax.set_title(f"D={depth}")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
        ax.grid(alpha=.25)
        ax.set_xlabel("N")
    axes[0].set_ylabel("Full grid wall time (seconds, log scale)")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "runtime_scaling.png", dpi=180)
    plt.close(fig)


def _plot_copula_widths(rows: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, depth in zip(axes, (2, 3)):
        subset = rows[(rows.kernel == "copula_csp") & (rows.D == depth)
                      & (rows.status == "ok")]
        for layer in range(depth):
            color = plt.cm.tab10(layer)
            selected = []
            for N in N_VALUES:
                at_n = subset[subset.N == N]
                selected.append(float(np.mean([
                    json.loads(value)[layer] for value in at_n.fitted_K
                ])) if len(at_n) else np.nan)
            paper = [COPULA_PAPER_WIDTHS[depth][N][layer] for N in N_VALUES]
            ax.plot(N_VALUES, selected, marker="o", color=color,
                    label=f"Layer {layer + 1}: selected")
            ax.plot(N_VALUES, paper, linestyle="--", color=color, alpha=.75,
                    label=f"Layer {layer + 1}: paper mean")
        truth = (10, 3) if depth == 2 else (18, 6, 2)
        ax.set_title(f"Copula CSP, D={depth}, true widths {truth}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES, [str(N) for N in N_VALUES])
        ax.set_xlabel("N")
        ax.set_ylabel("Active width")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("One selected run per setting versus published 100-run means")
    fig.tight_layout()
    fig.savefig(output / "copula_paper_widths.png", dpi=180)
    plt.close(fig)


def summarize(output: Path):
    root = Path(__file__).resolve().parents[2]
    rows = pd.read_csv(output / "fits.csv")
    # Older batches include data-generation time in ``seconds``. Candidate
    # timings isolate the public selector work for a fair speed assessment.
    timed, fitted, scored = [], [], []
    for _, row in rows.iterrows():
        if row["status"] != "ok":
            timed.append(np.nan)
            fitted.append(np.nan)
            scored.append(np.nan)
            continue
        basename = f"{row['kernel']}_d{int(row['D'])}_n{int(row['N'])}_r{int(row['replicate'])}"
        diagnostic = json.loads((output / "cells" / f"{basename}.json").read_text())
        parts = diagnostic["timing_table"]
        timed.append(float(sum(part["total_seconds"] for part in parts)))
        fitted.append(float(sum(part["fit_seconds"] for part in parts)))
        scored.append(float(sum(part["score_seconds"] for part in parts)))
    rows["candidate_timed_seconds"] = timed
    rows["candidate_fit_seconds"] = fitted
    rows["candidate_score_seconds"] = scored
    paper = _paper_comparison(rows, root)
    paper.to_csv(output / "paper_comparison.csv", index=False)
    direct = paper[paper.comparison_type == "parameter_rmse"].copy()
    if not direct.empty:
        direct["selected_metric"] = pd.to_numeric(direct.selected_metric)
        comparison = (direct.groupby(["kernel", "D", "N"], as_index=False)
                      .agg(replicates=("selected_metric", "size"),
                           selected_rmse_mean=("selected_metric", "mean"),
                           selected_rmse_root_mean_mse=("selected_metric", lambda values: float(np.sqrt(np.mean(np.square(values))))),
                           paper_rmse=("paper_metric", "first"),
                           paper_pem_rmse=("paper_pem_metric", "first"),
                           paper_source=("paper_source", "first"),
                           setting_note=("setting_note", "first")))
        intervals = {}
        for key, subset in direct.groupby(["kernel", "D", "N"]):
            values = np.square(subset.selected_metric.to_numpy(dtype=float))
            if len(values) == 1:
                ci = (float(np.sqrt(values[0])), float(np.sqrt(values[0])))
            else:
                seed = 20261014 + int(key[1]) * 100000 + int(key[2])
                rng = np.random.default_rng(seed)
                draws = rng.choice(values, size=(5000, len(values)), replace=True)
                estimates = np.sqrt(draws.mean(axis=1))
                ci = tuple(np.percentile(estimates, [2.5, 97.5]).tolist())
            intervals[key] = ci
        comparison["selected_ci_low"] = [
            intervals[(row.kernel, row.D, row.N)][0] for row in comparison.itertuples()]
        comparison["selected_ci_high"] = [
            intervals[(row.kernel, row.D, row.N)][1] for row in comparison.itertuples()]
        comparison["delta"] = comparison.selected_rmse_root_mean_mse - comparison.paper_rmse.astype(float)
        comparison["ratio"] = comparison.selected_rmse_root_mean_mse / comparison.paper_rmse.astype(float)
        comparison["ratio_to_pem"] = comparison.selected_rmse_root_mean_mse / comparison.paper_pem_rmse.astype(float)
    else:
        comparison = pd.DataFrame()
    comparison.to_csv(output / "paper_comparison_summary.csv", index=False)
    groups = []
    for (kernel, depth, N), subset in rows.groupby(["kernel", "D", "N"]):
        good = subset[subset.status == "ok"]
        entry = {"kernel": kernel, "D": depth, "N": N,
                 "completed": len(good), "attempted": len(subset),
                 "errors": len(subset)-len(good)}
        if not good.empty:
            entry["mean_seconds"] = float(good.candidate_timed_seconds.mean())
            entry["mean_fit_seconds"] = float(good.candidate_fit_seconds.mean())
            entry["mean_score_seconds"] = float(good.candidate_score_seconds.mean())
            entry["candidate_count"] = int(good.candidate_count.iloc[0])
            if kernel.endswith("fixed"):
                entry["parameter_rmse"] = float(np.sqrt(good.parameter_mse_thresholded.mean()))
            else:
                entry["exact_recovery_rate"] = float(good.dimension_exact.mean())
                entry["mean_dimension_l1"] = float(good.dimension_l1.mean())
        groups.append(entry)
    grouped = pd.DataFrame(groups)
    grouped.to_csv(output / "summary.csv", index=False)
    _plot_fixed(rows, paper, output)
    _plot_csp(rows, output)
    _plot_speed(rows, output)
    _plot_copula_widths(rows, output)
    complete = int((rows.status == "ok").sum())
    unique_complete = rows[rows.status == "ok"].groupby(["kernel", "D", "N"]).ngroups
    worse = comparison[comparison.delta > 0] if not comparison.empty else comparison
    better = comparison[comparison.delta < 0] if not comparison.empty else comparison
    pem_comparable = comparison[comparison.paper_pem_rmse.notna()] if not comparison.empty else comparison
    better_pem = pem_comparable[pem_comparable.ratio_to_pem < 1] if not pem_comparable.empty else pem_comparable
    csp = rows[(rows.dimension_method == "csp") & (rows.status == "ok")]
    lines = ["# Hyperparameter selection validation", "",
             f"Completed unique full-grid settings: {unique_complete}/100; successful fits: {complete}; error rows: {len(rows)-complete}.",
             "", "## Metrics", "",
             "Fixed: Hungarian-aligned all-parameter RMSE, including intercepts and gamma where applicable; coefficients are thresholded at the selected tau. Copula B1/gamma are standardized on the latent Gaussian scale.",
             "CSP: all-layer exact width recovery, plus L1 width error.",
             "", "## Paper comparisons", "",
             f"Direct RMSE reference settings: {len(comparison)}; selected pooled RMSE lower than paper SAEM/original comparator in {len(better)}, higher in {len(worse)}. For the {len(pem_comparable)} DDE settings with a PEM reference, selected RMSE is lower in {len(better_pem)}. See paper_comparison_summary.csv for replicate counts and ratios; ten replicates remain less precise than the papers' 100-replicate summaries.",
             "The DDE paper's PEM rows use a different, stronger estimation algorithm. The SemiDDE paper uses oracle initialization while this validation initializes from the data; numerical differences are not a controlled algorithm-only comparison.",
             "Copula CSP uses the exact stored MATLAB true setting, but the paper reports mean layer widths over 100 replicates rather than all-layer exact recovery. See paper_comparison.csv.",
             "D=3 fixed paths have no published overall-parameter RMSE for the exact deeper setting; CSP paths outside Copula also lack published all-layer exact recovery for these designs.",
             "", "## Scope and limits", "",
             "Every successful cell evaluated the entire default Cartesian grid through select_hyperparameters(). All non-Copula settings use default iteration and optimizer limits; D=2 fixed paper comparisons use ten replicates and the other settings use one. SemiDDE D=3 has mc_samples configured as 4096 instead of the package default 50000, but the N=500 full-grid audit found all 27 scores identical: this design uses exact conditional enumeration, so that setting is inactive. Copula full grids use one short-budget run per setting, with default-budget N=500 pilots. Per-batch settings are recorded in each batch's config.json and validation_manifest.json. Runtime plots and summary.csv use summed candidate timing from selector diagnostics, excluding truth/data generation; fits.csv seconds includes the whole cell. Several cells ran concurrently, so isolated-call wall time may differ. Single-replicate CSP rates remain descriptive, and the Copula full grids do not establish production-budget convergence.",
             "", f"CSP exact cells: {int(csp.dimension_exact.sum())}/{len(csp)} successful cells.",
             "", "## D=2 fixed: selected versus paper RMSE", "",
             "| Kernel | N | Selected pooled RMSE | Paper SAEM / SemiDDE | Paper PEM | Ratio to SAEM / SemiDDE | Ratio to PEM | 95% bootstrap interval |",
             "|---|---:|---:|---:|---:|---:|---:|---:|",
             ]
    for item in comparison.itertuples():
        pem = f"{item.paper_pem_rmse:.3f}" if np.isfinite(item.paper_pem_rmse) else "—"
        pem_ratio = f"{item.ratio_to_pem:.2f}" if np.isfinite(item.ratio_to_pem) else "—"
        lines.append(
            f"| {item.kernel} | {item.N} | {item.selected_rmse_root_mean_mse:.3f} | "
            f"{float(item.paper_rmse):.3f} | {pem} | {item.ratio:.2f} | {pem_ratio} | "
            f"[{item.selected_ci_low:.3f}, {item.selected_ci_high:.3f}] |"
        )
    if len(pem_comparable):
        high_n = pem_comparable[pem_comparable.N == max(N_VALUES)]
        ratios = ", ".join(f"{item.kernel.replace('dde_', '').replace('_fixed', '')} {item.ratio_to_pem:.2f}×"
                           for item in high_n.sort_values("kernel").itertuples())
        lines.extend(["", f"At N={max(N_VALUES)}, selected DDE RMSE relative to paper PEM: {ratios}."])
    lines.extend(["", "## CSP all-layer exact recovery", "",
                  "| Kernel | D=2 | D=3 |", "|---|---:|---:|"])
    for kernel in [name for name in KERNELS if name.endswith("csp")]:
        values = []
        for depth in (2, 3):
            subset = csp[(csp.kernel == kernel) & (csp.D == depth)]
            values.append(f"{int(subset.dimension_exact.sum())}/{len(subset)}")
        lines.append(f"| {kernel} | {values[0]} | {values[1]} |")
    zero_width = csp[csp.fitted_K.map(lambda value: 0 in json.loads(value))]
    if len(zero_width):
        lines.extend(["", f"Zero-width CSP layers occurred in {len(zero_width)} completed cells:", ""])
        for item in zero_width.sort_values(["kernel", "D", "N"]).itertuples():
            lines.append(f"- {item.kernel}, D={item.D}, N={item.N}: {item.fitted_K}")
    copula = csp[csp.kernel == "copula_csp"].sort_values(["D", "N"])
    lines.extend(["", "## Copula selected widths versus paper means", "",
                  "True widths are D=2 [10, 3] and D=3 [18, 6, 2]. The selected width is one run; the paper value is a mean over 100 runs, so these are descriptive comparisons.",
                  "", "| D | N | Selected widths | Paper mean widths | All layers exact |",
                  "|---:|---:|---|---|---:|"])
    for item in copula.itertuples():
        paper_widths = COPULA_PAPER_WIDTHS[int(item.D)][int(item.N)]
        lines.append(f"| {item.D} | {item.N} | {item.fitted_K} | "
                     f"{[round(v, 2) for v in paper_widths]} | {int(item.dimension_exact)} |")
    audit_path = output / "candidate_dimension_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        lines.extend(["", "## Candidate width audit", "",
                      "Representative screening-budget cells were refit one candidate at a time; scores matched the original full-grid scores. This audit does not establish default-budget candidate widths.",
                      "", "| Kernel | D | N | Exact candidates | Evaluated | Best fitted K |",
                      "|---|---:|---:|---:|---:|---|"])
        for (kernel, depth, N), subset in audit.groupby(["kernel", "D", "N"]):
            good = subset[subset.status == "ok"]
            best = good.loc[good.score.idxmax()] if len(good) else None
            lines.append(f"| {kernel} | {depth} | {N} | "
                         f"{int(good.dimension_exact.sum())} | {len(good)} | "
                         f"{best.fitted_K if best is not None else '—'} |")
    copula_audit_path = output / "copula_candidate_width_audit.csv"
    if copula_audit_path.exists():
        audit = pd.read_csv(copula_audit_path)
        lines.extend(["", "## Copula D=3 candidate audit", "",
                      "The five lowest-pMSE candidates at N=500 were refit with default budgets to inspect their active widths.",
                      "", "| Candidate index | pMSE | Fitted widths | Exact |",
                      "|---:|---:|---|---:|"])
        for item in audit.sort_values("mean_pmse").itertuples():
            exact = str(int(item.dimension_exact)) if item.status == "ok" else "—"
            lines.append(f"| {item.candidate_index} | {item.mean_pmse:.6f} | "
                         f"{item.fitted_K if item.status == 'ok' else item.error} | {exact} |")
    pilot_path = output / "default_budget_pilot" / "fits.csv"
    if pilot_path.exists():
        pilot = pd.read_csv(pilot_path)
        lines.extend(["", "## Default iteration budget pilot", "",
                      "Full grids at N=500 with DDE 50 iterations, SemiDDE 10 outer × 50 inner, and optimizer maxiter 100; SemiDDE mc_samples is configured as 4096, but does not affect the examined D=3 hierarchy.",
                      "", "| Kernel | D | Selected widths | Exact | Selector time |",
                      "|---|---:|---|---:|---:|"])
        for item in pilot.sort_values(["kernel", "D"]).itertuples():
            lines.append(f"| {item.kernel} | {item.D} | {item.fitted_K} | "
                         f"{int(item.dimension_exact)} | {float(item.seconds):.1f} s |")
    copula_pilot_paths = [output / name / "fits.csv" for name in
                          ("copula_default_budget_pilot", "copula_d3_default_pilot")]
    copula_pilots = [pd.read_csv(path) for path in copula_pilot_paths if path.exists()]
    if copula_pilots:
        pilot = pd.concat(copula_pilots, ignore_index=True)
        lines.extend(["", "## Copula default budget pilots", "",
                      "N=500, with 50 kernel iterations, optimizer maxiter 100, and pMSE 10 repeats × 5 folds × 100 trees. D=2 uses J=100; D=3 uses J=108.",
                      "", "| Kernel | D | Short widths | Default widths | Short RMSE | Default RMSE | Short time | Default time |",
                      "|---|---:|---|---|---:|---:|---:|---:|"])
        for item in pilot.sort_values(["D", "kernel"]).itertuples():
            short_rows = rows[(rows.kernel == item.kernel) & (rows.D == item.D)
                              & (rows.N == item.N) & (rows.replicate == 0)]
            short = short_rows.iloc[0] if len(short_rows) else None
            short_K = short.fitted_K if short is not None and short.status == "ok" else "—"
            short_rmse = (f"{float(short.parameter_rmse_thresholded):.3f}"
                          if short is not None and short.status == "ok"
                          and item.kernel.endswith("fixed") else "—")
            default_rmse = (f"{float(item.parameter_rmse_thresholded):.3f}"
                            if item.kernel.endswith("fixed") and item.status == "ok" else "—")
            short_time = f"{float(short.seconds) / 60:.1f} min" if short is not None else "—"
            lines.append(f"| {item.kernel} | {item.D} | {short_K} | {item.fitted_K} | "
                         f"{short_rmse} | {default_rmse} | {short_time} | "
                         f"{float(item.seconds) / 60:.1f} min |")
    fixed_pilot_path = output / "fixed_default_budget_pilot" / "fits.csv"
    if fixed_pilot_path.exists():
        pilot = pd.read_csv(fixed_pilot_path)
        short_path = output / "paper_fixed_replicates" / "fits.csv"
        short_rows = pd.read_csv(short_path) if short_path.exists() else rows
        lines.extend(["", "## Fixed-width default budget pilot", "",
                      "D=2 at N=500 and 8000, with DDE 50 iterations, SemiDDE 10 outer × 50 inner, and optimizer maxiter 100. Short-budget reference uses the same replicate-zero seed.",
                      "", "| Kernel | N | Short-budget RMSE | Default-budget RMSE | Default wall time |",
                      "|---|---:|---:|---:|---:|"])
        for item in pilot.sort_values(["kernel", "N"]).itertuples():
            reference = short_rows[(short_rows.kernel == item.kernel) & (short_rows.D == item.D)
                                   & (short_rows.N == item.N) & (short_rows.replicate == 0)]
            short = (f"{float(reference.parameter_rmse_thresholded.iloc[0]):.3f}"
                     if len(reference) and reference.status.iloc[0] == "ok" else "—")
            default = (f"{float(item.parameter_rmse_thresholded):.3f}"
                       if item.status == "ok" else "—")
            lines.append(f"| {item.kernel} | {item.N} | {short} | {default} | "
                         f"{float(item.seconds):.1f} s |")
    semi_full_mc_path = output / "semidde_full_mc_pilot" / "fits.csv"
    if semi_full_mc_path.exists():
        pilot = pd.read_csv(semi_full_mc_path)
        lines.extend(["", "## SemiDDE full-Monte-Carlo pilot", "",
                      "D=3, N=500, same full CSP grid and iteration limits, with 50000 Monte Carlo samples instead of 4096.",
                      "", "| Budget | Selected widths | Exact | Wall time |",
                      "|---|---|---:|---:|"])
        reference = rows[(rows.kernel == "semidde_csp") & (rows.D == 3)
                         & (rows.N == 500) & (rows.replicate == 0)]
        if len(reference):
            item = reference.iloc[0]
            lines.append(f"| MC 4096 | {item.fitted_K} | {int(item.dimension_exact)} | "
                         f"{float(item.seconds) / 60:.1f} min |")
        for item in pilot.itertuples():
            lines.append(f"| MC 50000 | {item.fitted_K} | {int(item.dimension_exact)} | "
                         f"{float(item.seconds) / 60:.1f} min |")
        equivalence_path = output / "semidde_mc_equivalence.json"
        if equivalence_path.exists():
            equivalence = json.loads(equivalence_path.read_text())
            lines.extend(["", f"Maximum absolute difference across all 27 candidate scores: {equivalence['max_absolute_candidate_score_difference']:.3g}."])
    slowest = grouped[grouped.completed > 0].nlargest(5, "mean_seconds")
    lines.extend(["", "## Slowest complete grids", "",
                  "| Kernel | D | N | Candidates | Selector time |",
                  "|---|---:|---:|---:|---:|"])
    for item in slowest.itertuples():
        lines.append(f"| {item.kernel} | {item.D} | {item.N} | "
                     f"{int(item.candidate_count)} | {item.mean_seconds / 60:.1f} min |")
    lines.extend(["", "## Files", "",
             "fits.csv: per-fit status, selected hyperparameters, accuracy and wall time.",
             "summary.csv: per-setting metrics; paper_comparison.csv: per-fit references; paper_comparison_summary.csv: pooled comparisons.",
             "cells/: complete candidate scores and timings (JSON), plus selected parameters (NPZ).",
             "integrity_check.json: row, grid, score, width, and parameter-file checks; validation_manifest.json, environment.json, and semidde_mc_equivalence.json: provenance and MC-budget check.",
             "fixed_parameter_rmse.png, csp_exact_recovery.png, copula_paper_widths.png, runtime_scaling.png: visual summaries.", ""])
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summarize(args.output)
