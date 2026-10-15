"""Plot every validated kernel's selected-parameter metric against sample size.

The plots use the completed validation results in Results/hp_selection_validation.
Fixed-width RMSE pools squared parameter errors over replicates; CSP exactness is
the fraction of runs recovering every layer width (one run in each current cell).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Results" / "hp_selection_validation"
N_VALUES = (500, 1000, 2000, 4000, 8000)
METHODS = ("dde_normal", "dde_bernoulli", "dde_poisson", "semidde", "copula")
LABELS = ("DDE Normal", "DDE Bernoulli", "DDE Poisson", "SemiDDE", "CopulaDDE")
COLORS = {2: "#1877a5", 3: "#d2673f"}


def load_plot_data(output: Path) -> pd.DataFrame:
    """Check completeness and independently verify the aggregated plot values."""
    summary = pd.read_csv(output / "summary.csv")
    fits = pd.read_csv(output / "fits.csv")
    expected = {(f"{method}_{mode}", depth, n)
                for method in METHODS for mode in ("fixed", "csp")
                for depth in (2, 3) for n in N_VALUES}
    actual = set(map(tuple, summary[["kernel", "D", "N"]].itertuples(index=False, name=None)))
    if len(summary) != len(expected) or actual != expected:
        raise ValueError(f"Incomplete summary grid: missing={expected - actual}, extra={actual - expected}")
    if not (summary.errors.eq(0) & summary.completed.gt(0)).all():
        raise ValueError("Some validation cells failed or have no completed runs")
    if not fits.status.eq("ok").all():
        raise ValueError("fits.csv contains unsuccessful runs")

    records = []
    for row in summary.itertuples(index=False):
        selected = fits[(fits.kernel == row.kernel) & (fits.D == row.D) & (fits.N == row.N)]
        if len(selected) != row.completed:
            raise ValueError(f"Replicate count differs for {row.kernel}, D={row.D}, N={row.N}")
        mode = "fixed" if row.kernel.endswith("_fixed") else "csp"
        if mode == "fixed":
            expected_value = float(np.sqrt(selected.parameter_mse_thresholded.mean()))
            value = float(row.parameter_rmse)
            metric = "overall_parameter_rmse"
        else:
            expected_value = float(selected.dimension_exact.mean())
            value = float(row.exact_recovery_rate)
            metric = "all_layer_exact_recovery"
        if not np.isfinite(value) or not np.isclose(value, expected_value, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Metric differs from fits.csv for {row.kernel}, D={row.D}, N={row.N}")
        records.append(dict(kernel=row.kernel, mode=mode, D=int(row.D), N=int(row.N),
                            metric=metric, value=value, completed=int(row.completed)))
    return pd.DataFrame(records).sort_values(["mode", "kernel", "D", "N"]).reset_index(drop=True)


def _axes_style(ax: plt.Axes, *, csp: bool) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.grid(axis="both", color="#d5dce2", linewidth=.65, alpha=.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if csp:
        ax.set_ylim(-.13, 1.13)
        ax.set_yticks([0, 1])


def _line(ax: plt.Axes, subset: pd.DataFrame, depth: int, *, csp: bool) -> None:
    subset = subset.sort_values("N")
    color = COLORS[depth]
    ax.plot(subset.N, subset.value, color=color, linewidth=1.8,
            marker="o" if depth == 2 else "s", markersize=5.2,
            label=f"D={depth}")
    if csp:
        for row in subset.itertuples(index=False):
            ax.annotate(str(int(row.value)), (row.N, row.value),
                        xytext=(0, 7 if row.value == 0 else -13),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=color)


def plot_overview(data: pd.DataFrame, mode: str, output: Path) -> Path:
    csp = mode == "csp"
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), layout="constrained")
    for col, (method, label) in enumerate(zip(METHODS, LABELS)):
        kernel = f"{method}_{mode}"
        both = data[data.kernel == kernel]
        upper = max(1e-6, float(both.value.max()) * 1.18)
        for row, depth in enumerate((2, 3)):
            ax = axes[row, col]
            subset = both[both.D == depth]
            _line(ax, subset, depth, csp=csp)
            _axes_style(ax, csp=csp)
            if not csp:
                ax.set_ylim(0, upper)
            if row == 0:
                ax.set_title(label, fontsize=12, weight="semibold")
            if row == 1:
                ax.set_xlabel("Sample size N", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"D={depth}\n" + ("All-layer exact (0/1)" if csp else "Overall parameter RMSE"),
                              fontsize=9)
    metric_label = "all-layer exact recovery" if csp else "overall parameter RMSE"
    subtitle = "one completed run per setting; 1 means every layer width is correct" if csp else "pooled RMSE across completed replicates"
    fig.suptitle(f"Hyperparameter-selected kernels: {metric_label} versus N\n{subtitle}", fontsize=15)
    path = output / f"kernel_{mode}_{'exact' if csp else 'rmse'}_vs_n.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_individual(data: pd.DataFrame, method: str, label: str, mode: str, output: Path) -> Path:
    csp = mode == "csp"
    kernel = f"{method}_{mode}"
    subset = data[data.kernel == kernel]
    if csp:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
        for ax, depth in zip(axes, (2, 3)):
            _line(ax, subset[subset.D == depth], depth, csp=True)
            _axes_style(ax, csp=True)
            ax.set_ylabel(f"D={depth}: exact (0/1)")
            ax.set_xlabel("Sample size N")
        fig.suptitle(f"{label} CSP: all-layer exact recovery versus N\nOne completed run per setting", fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for depth in (2, 3):
            _line(ax, subset[subset.D == depth], depth, csp=False)
        _axes_style(ax, csp=False)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Sample size N")
        ax.set_ylabel("Overall parameter RMSE")
        ax.legend(frameon=False)
        fig.suptitle(f"{label} fixed: overall parameter RMSE versus N", fontsize=13)
    path = output / f"{kernel}_vs_n.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    data = load_plot_data(output)
    data.to_csv(output / "kernel_metric_plot_data.csv", index=False)
    individual = output / "by_kernel"
    individual.mkdir(exist_ok=True)
    paths = [plot_overview(data, mode, output) for mode in ("fixed", "csp")]
    paths.extend(plot_individual(data, method, label, mode, individual)
                 for mode in ("fixed", "csp")
                 for method, label in zip(METHODS, LABELS))
    print(f"Verified {len(data)} metric points against fits.csv; saved {len(paths)} figures")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
