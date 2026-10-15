"""Summarize mean recovered layer widths in the selected-HP CSP simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulate_selected_csp_hp_rep100 import DEFAULT_OUTPUT, DEPTHS, KERNELS, SAMPLE_SIZES
from summarize_hp_selection_validation import COPULA_PAPER_WIDTHS


LABELS = ("DDE Normal", "DDE Bernoulli", "DDE Poisson", "SemiDDE", "CopulaDDE")
COLORS = ("#2867a0", "#db7a37", "#36916c", "#8556a6", "#b84b63")


def summarize(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits = pd.read_csv(output / "fits.csv")
    existing = pd.read_csv(output / "summary.csv").set_index(["kernel", "D", "N"])
    config = json.loads((output / "config.json").read_text())
    rep_count = int(config["replicates"])
    expected = len(KERNELS) * len(DEPTHS) * len(SAMPLE_SIZES) * rep_count
    if len(fits) != expected or not fits.status.eq("ok").all():
        raise ValueError(f"Expected {expected} successful fits, found {len(fits)} rows")
    if fits.duplicated(["kernel", "D", "N", "replicate"]).any():
        raise ValueError("Duplicate replicate keys")

    rows = []
    comparison = []
    rng = np.random.default_rng(20261015)
    for kernel in KERNELS:
        for depth in DEPTHS:
            for sample_size in SAMPLE_SIZES:
                key = (kernel, depth, sample_size)
                group = fits[(fits.kernel == kernel) & (fits.D == depth) & (fits.N == sample_size)]
                if sorted(group.replicate.tolist()) != list(range(rep_count)):
                    raise ValueError(f"Missing replicate at {key}")
                true_values = [json.loads(value) for value in group.true_K]
                recovered = np.asarray([json.loads(value) for value in group.fitted_K], dtype=float)
                if len({tuple(value) for value in true_values}) != 1 or recovered.shape != (rep_count, depth):
                    raise ValueError(f"Inconsistent widths at {key}")
                truth = true_values[0]
                if json.loads(existing.loc[key, "true_K"]) != truth:
                    raise ValueError(f"Truth differs from original summary at {key}")
                for layer in range(depth):
                    values = recovered[:, layer]
                    mean = float(values.mean())
                    if not np.isclose(mean, existing.loc[key, f"layer{layer+1}_mean_width"], atol=1e-12):
                        raise ValueError(f"Mean differs from original summary at {key}, layer {layer+1}")
                    bootstrap = rng.choice(values, size=(2000, rep_count), replace=True).mean(axis=1)
                    row = {
                        "kernel": kernel, "D": depth, "N": sample_size, "layer": layer + 1,
                        "replicates": rep_count, "true_width": truth[layer], "mean_recovered_width": mean,
                        "mean_bias": mean - truth[layer], "sd_recovered_width": float(values.std(ddof=1)),
                        "p10_recovered_width": float(np.quantile(values, .1, method="nearest")),
                        "median_recovered_width": float(np.median(values)),
                        "p90_recovered_width": float(np.quantile(values, .9, method="nearest")),
                        "mean_ci95_low": float(np.quantile(bootstrap, .025)),
                        "mean_ci95_high": float(np.quantile(bootstrap, .975)),
                        "mean_absolute_width_error": float(np.abs(values - truth[layer]).mean()),
                    }
                    rows.append(row)
                    if kernel == "copula_csp":
                        paper = COPULA_PAPER_WIDTHS[depth][sample_size][layer]
                        comparison.append({**row, "paper_mean_width": paper,
                                           "ours_minus_paper": mean - paper})
    return pd.DataFrame(rows), pd.DataFrame(comparison)


def plot_all(summary: pd.DataFrame, depth: int, output: Path) -> None:
    fig, axes = plt.subplots(depth, len(KERNELS), figsize=(18.5, 3.2 * depth),
                             layout="constrained", squeeze=False)
    for column, (kernel, label, color) in enumerate(zip(KERNELS, LABELS, COLORS)):
        for layer in range(1, depth + 1):
            ax = axes[layer - 1, column]
            cell = summary[(summary.kernel == kernel) & (summary.D == depth)
                           & (summary.layer == layer)].sort_values("N")
            x = cell.N.to_numpy()
            mean = cell.mean_recovered_width.to_numpy()
            ax.fill_between(x, cell.p10_recovered_width, cell.p90_recovered_width,
                            color=color, alpha=.18, label="Replicate 10–90% range")
            ax.plot(x, mean, color=color, marker="o", linewidth=2, markersize=4,
                    label="Mean recovered width")
            ax.axhline(cell.true_width.iloc[0], color="#222222", linestyle="--",
                       linewidth=1.3, label="True width")
            low = min(float(cell.p10_recovered_width.min()), float(cell.true_width.iloc[0]))
            high = max(float(cell.p90_recovered_width.max()), float(cell.true_width.iloc[0]))
            span = max(high - low, 1.0)
            center = (high + low) / 2
            ax.set_ylim(max(0, center - .58 * span), center + .58 * span)
            ax.set_xscale("log", base=2)
            ax.set_xticks(SAMPLE_SIZES, [str(n) for n in SAMPLE_SIZES], rotation=45)
            ax.grid(alpha=.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if layer == 1:
                ax.set_title(label, fontsize=10.5, weight="semibold")
            if layer == depth:
                ax.set_xlabel("Sample size N", fontsize=9)
            if column == 0:
                ax.set_ylabel(f"Layer {layer} width", fontsize=9)
            if layer == 1 and column == 0:
                ax.legend(fontsize=7.5, loc="best", frameon=False)
    fig.suptitle(f"CSP: mean recovered widths by N (D={depth}, 100 replicates per setting)", fontsize=14)
    fig.savefig(output / f"mean_recovered_widths_D{depth}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_copula(comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), layout="constrained")
    for row, depth in enumerate(DEPTHS):
        for layer in range(1, 4):
            ax = axes[row, layer - 1]
            if layer > depth:
                ax.axis("off")
                continue
            cell = comparison[(comparison.D == depth) & (comparison.layer == layer)].sort_values("N")
            x = cell.N.to_numpy()
            ax.plot(x, cell.mean_recovered_width, marker="o", color="#b84b63",
                    linewidth=2, label="This simulation")
            ax.plot(x, cell.paper_mean_width, marker="s", color="#2867a0",
                    linewidth=1.6, label="Paper")
            ax.axhline(cell.true_width.iloc[0], linestyle="--", color="#222222",
                       linewidth=1.1, label="True width")
            ax.set_xscale("log", base=2)
            ax.set_xticks(SAMPLE_SIZES, [str(n) for n in SAMPLE_SIZES], rotation=45)
            ax.set_title(f"D={depth}, layer {layer}", fontsize=10)
            ax.grid(alpha=.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if row == 1:
                ax.set_xlabel("Sample size N")
            if layer == 1:
                ax.set_ylabel("Mean recovered width")
            if row == 0 and layer == 1:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle("CopulaDDE CSP: 100-replicate mean widths vs paper", fontsize=13)
    fig.savefig(output / "copula_mean_recovered_widths_vs_paper.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, comparison: pd.DataFrame, output: Path) -> None:
    exact = pd.read_csv(output / "summary.csv")

    def vector(frame: pd.DataFrame, field: str) -> str:
        return "[" + ", ".join(f"{v:.2f}" for v in frame.sort_values("layer")[field]) + "]"

    copula_2 = comparison[(comparison.D == 2) & (comparison.N == 8000)]
    copula_3 = comparison[(comparison.D == 3) & (comparison.N == 8000)]
    copula_500_exact = int(exact[(exact.kernel == "copula_csp") & (exact.D == 2)
                                 & (exact.N == 500)].exact_count.iloc[0])
    lines = ["# CSP mean recovered layer widths under selected hyperparameters", "",
             "Each cell summarizes 100 independently generated datasets fitted with the fixed hyperparameter combination selected for that kernel, D, and N. Values below are layer-width vectors in model order.",
             "", "Shaded bands in the figures show the replicate 10th–90th percentile range, not a confidence interval for the mean. The CSV also includes sample standard deviations and bootstrap 95% intervals for each mean."]
    for depth in DEPTHS:
        lines.extend(["", f"## D={depth}", "",
                      "| Kernel | Truth | " + " | ".join(f"N={n}" for n in SAMPLE_SIZES) + " |",
                      "|---|---|" + "---|" * len(SAMPLE_SIZES)])
        for kernel, label in zip(KERNELS, LABELS):
            cells = []
            for n in SAMPLE_SIZES:
                cell = summary[(summary.kernel == kernel) & (summary.D == depth)
                               & (summary.N == n)].sort_values("layer")
                if n == SAMPLE_SIZES[0]:
                    truth = "[" + ", ".join(str(int(v)) for v in cell.true_width) + "]"
                cells.append("[" + ", ".join(f"{v:.2f}" for v in cell.mean_recovered_width) + "]")
            lines.append("| " + " | ".join([label, truth, *cells]) + " |")
    lines.extend(["", "## CopulaDDE comparison with reported paper mean widths", "",
                  f"At N=8000, the D=2 mean is {vector(copula_2, 'mean_recovered_width')} versus paper {vector(copula_2, 'paper_mean_width')} (truth [10, 3]). The D=3 mean is {vector(copula_3, 'mean_recovered_width')} versus paper {vector(copula_3, 'paper_mean_width')} (truth [18, 6, 2]). All N and layers are in `copula_mean_recovered_widths_vs_paper.csv`.",
                  "", "The CopulaDDE simulation used the recorded screening budget of 10 iterations, so the gap does not isolate hyperparameter choice from optimizer convergence.",
                  "", "## Interpretation", "",
                  f"Mean widths alone can hide run-to-run errors that cancel. For example, CopulaDDE D=2 at N=500 averaged {vector(comparison[(comparison.D == 2) & (comparison.N == 500)], 'mean_recovered_width')}, close to [10, 3], but all-layer exact recovery was only {copula_500_exact}/100. Compare the shaded ranges or original `summary.csv` exact rates alongside the means.",
                  "", "PoissonDDE means are close to truth at both depths. SemiDDE D=2 recovers the truth in every run, but SemiDDE D=3 systematically estimates the final width as 1 instead of 2. BernoulliDDE D=2 systematically underestimates the second width. NormalDDE and CopulaDDE D=3 overestimate deeper widths.",
                  "", "## Files", "",
                  "- `mean_recovered_widths.csv`: 125 setting-layer rows with mean, bias, standard deviation, quantiles, bootstrap interval, and mean absolute width error.",
                  "- `mean_recovered_widths_D2.png` and `mean_recovered_widths_D3.png`: N curves for all five kernels.",
                  "- `copula_mean_recovered_widths_vs_paper.csv` and `.png`: descriptive paper comparison.",
                  "- `fits.csv`: 5,000 replicate-level fitted-width vectors.",
                  "- `dde/scripts/summarize_selected_csp_mean_widths.py`: reproducible summary and figure generation."])
    (output / "MEAN_WIDTH_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    summary, comparison = summarize(output)
    if len(summary) != 125 or len(comparison) != 25:
        raise ValueError("Unexpected number of layer summaries")
    summary.to_csv(output / "mean_recovered_widths.csv", index=False)
    comparison.to_csv(output / "copula_mean_recovered_widths_vs_paper.csv", index=False)
    for depth in DEPTHS:
        plot_all(summary, depth, output)
    plot_copula(comparison, output)
    write_report(summary, comparison, output)
    print(f"Saved {len(summary)} layer summaries and {len(comparison)} Copula/paper comparisons to {output}")


if __name__ == "__main__":
    main()
