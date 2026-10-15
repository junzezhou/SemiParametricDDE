"""Validate and summarize 100-replicate selected-CSP-hyperparameter simulations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from simulate_selected_csp_hp_rep100 import DEFAULT_OUTPUT, DEPTHS, KERNELS, SAMPLE_SIZES


LABELS = ("DDE Normal", "DDE Bernoulli", "DDE Poisson", "SemiDDE", "CopulaDDE")
COLORS = {2: "#1877a5", 3: "#d2673f"}


def wilson_interval(successes: int, total: int, alpha: float = .05) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(1 - alpha / 2))
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return max(0., float(center-radius)), min(1., float(center+radius))


def aggregate(output: Path, *, allow_partial: bool = False) -> pd.DataFrame:
    fits = pd.read_csv(output / "fits.csv")
    config = json.loads((output / "config.json").read_text())
    selected = json.loads((output / "selected_hp.json").read_text())
    selected_map = {(row["kernel"], row["D"], row["N"]): row["hp"] for row in selected}
    expected = {(k, d, n, rep) for k in KERNELS for d in DEPTHS
                for n in SAMPLE_SIZES for rep in range(config["replicates"])}
    keys = list(map(tuple, fits[["kernel", "D", "N", "replicate"]].itertuples(index=False, name=None)))
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("Duplicate or unexpected replicate keys")
    if not allow_partial and set(keys) != expected:
        raise ValueError(f"Missing {len(expected-set(keys))} replicate keys")
    if fits.seed.duplicated().any():
        raise ValueError("Replicate seeds are not unique")
    if not fits.status.isin(["ok", "error"]).all():
        raise ValueError("Unknown fit status")
    for row in fits.itertuples(index=False):
        key = (row.kernel, int(row.D), int(row.N))
        if json.loads(row.hp) != selected_map[key]:
            raise ValueError(f"HP changed at {key}")
        if row.status == "ok":
            true_K, fitted_K = json.loads(row.true_K), json.loads(row.fitted_K)
            delta = np.asarray(fitted_K) - np.asarray(true_K)
            if (int(row.exact) != int(not np.any(delta))
                    or int(row.dimension_l1) != int(np.abs(delta).sum())
                    or int(row.under_width) != int(np.maximum(-delta, 0).sum())
                    or int(row.over_width) != int(np.maximum(delta, 0).sum())):
                raise ValueError(f"Metric mismatch at {key}, replicate {row.replicate}")
    rows = []
    for kernel in KERNELS:
        for depth in DEPTHS:
            for n in SAMPLE_SIZES:
                subset = fits[(fits.kernel == kernel) & (fits.D == depth) & (fits.N == n)]
                good = subset[subset.status == "ok"]
                exact = int(good.exact.sum())
                denominator = len(subset)
                low, high = wilson_interval(exact, denominator)
                layer_exact = []
                fitted = [tuple(json.loads(row.fitted_K)) for row in good.itertuples(index=False)]
                for layer in range(depth):
                    layer_exact.append(float(np.mean([
                        widths[layer] == json.loads(row.true_K)[layer]
                        for widths, row in zip(fitted, good.itertuples(index=False))
                    ])) if fitted else float("nan"))
                mean_widths = np.mean(np.asarray(fitted), axis=0).tolist() if fitted else [float("nan")] * depth
                width_counts = {str(list(widths)): count for widths, count in Counter(fitted).most_common()}
                rows.append({
                    "kernel": kernel, "D": depth, "N": n,
                    "true_K": subset.true_K.dropna().iloc[0] if subset.true_K.notna().any() else "",
                    "requested": int(config["replicates"]), "completed": denominator,
                    "successful": len(good), "failed": denominator-len(good),
                    "exact_count": exact,
                    "exact_rate_all": exact/denominator if denominator else float("nan"),
                    "exact_rate_success": exact/len(good) if len(good) else float("nan"),
                    "exact_ci_low": low, "exact_ci_high": high,
                    "mean_dimension_l1": good.dimension_l1.mean(),
                    "median_dimension_l1": good.dimension_l1.median(),
                    "mean_under_width": good.under_width.mean(),
                    "mean_over_width": good.over_width.mean(),
                    "mean_seconds": subset.seconds.mean(),
                    "median_seconds": subset.seconds.median(),
                    "p95_seconds": subset.seconds.quantile(.95),
                    "layer1_exact": layer_exact[0],
                    "layer2_exact": layer_exact[1],
                    "layer3_exact": layer_exact[2] if depth == 3 else float("nan"),
                    "layer1_mean_width": mean_widths[0],
                    "layer2_mean_width": mean_widths[1],
                    "layer3_mean_width": mean_widths[2] if depth == 3 else float("nan"),
                    "fitted_width_counts": json.dumps(width_counts),
                    "selected_hp": json.dumps(selected_map[kernel, depth, n], sort_keys=True),
                })
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, field: str, title: str, ylabel: str,
                output: Path) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), layout="constrained")
    for col, (kernel, label) in enumerate(zip(KERNELS, LABELS)):
        for row, depth in enumerate(DEPTHS):
            ax = axes[row, col]
            subset = summary[(summary.kernel == kernel) & (summary.D == depth)].sort_values("N")
            x = subset.N.to_numpy()
            y = subset[field].to_numpy(dtype=float)
            ax.plot(x, y, marker="o" if depth == 2 else "s", linewidth=1.8,
                    color=COLORS[depth], markersize=5)
            if field == "exact_rate_all":
                ax.fill_between(x, subset.exact_ci_low.to_numpy(dtype=float),
                                subset.exact_ci_high.to_numpy(dtype=float),
                                color=COLORS[depth], alpha=.17)
                ax.set_ylim(-.04, 1.04)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(SAMPLE_SIZES, [str(n) for n in SAMPLE_SIZES], rotation=45)
            ax.tick_params(labelsize=8)
            ax.grid(color="#d5dce2", alpha=.8, linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=11, weight="semibold")
            if row == 1:
                ax.set_xlabel("Sample size N", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"D={depth}\n{ylabel}", fontsize=9)
    fig.suptitle(title, fontsize=14)
    path = output / f"{field}_vs_n.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def report(summary: pd.DataFrame, output: Path) -> None:
    fits = pd.read_csv(output / "fits.csv")
    config = json.loads((output / "config.json").read_text())
    original = pd.read_csv(output.parent / "fits.csv")
    original = original[original.kernel.isin(KERNELS)]
    joined = summary.merge(original[["kernel", "D", "N", "dimension_exact"]],
                           on=["kernel", "D", "N"], how="left", validate="one_to_one")
    per_depth_total = len(SAMPLE_SIZES) * int(config["replicates"])

    def exact_total(kernel: str, depth: int) -> int:
        return int(joined[(joined.kernel == kernel) & (joined.D == depth)].exact_count.sum())

    def original_total(kernel: str, depth: int) -> int:
        return int(joined[(joined.kernel == kernel) & (joined.D == depth)].dimension_exact.sum())

    def exact_at(kernel: str, depth: int, n: int) -> int:
        return int(joined[(joined.kernel == kernel) & (joined.D == depth) & (joined.N == n)].exact_count.iloc[0])

    semi_deepest_mean = float(joined[(joined.kernel == "semidde_csp") & (joined.D == 3)].layer3_mean_width.mean())
    rows = ["# Selected CSP hyperparameters: 100-replicate simulation", "",
            f"Completed {len(fits)} / {len(KERNELS)*len(DEPTHS)*len(SAMPLE_SIZES)*config['replicates']} requested fits; successful {(fits.status=='ok').sum()}; errors {(fits.status!='ok').sum()}.",
            "", "Each setting uses the hyperparameter combination selected in the original grid search. New random seeds regenerate independent data under the same scenario. Each replicate fits the kernel once without reselection.",
            "", "The primary endpoint is all-layer exact width recovery. Failed fits count as nonrecovery in the primary rate. Wilson 95% intervals quantify replicate uncertainty; they do not cover simulation-design or hyperparameter-selection uncertainty.",
            "", "## Across the five N values (equally weighted)", "",
            f"| Kernel | D | Exact / {per_depth_total} | Mean width L1 | Original single-run exact / 5 |",
            "|---|---:|---:|---:|---:|"]
    for (kernel, depth), subset in joined.groupby(["kernel", "D"], sort=False):
        rows.append(f"| {kernel} | {depth} | {int(subset.exact_count.sum())}/{int(subset.completed.sum())} | "
                    f"{subset.mean_dimension_l1.mean():.2f} | {int(subset.dimension_exact.sum())}/5 |")
    rows.extend([
            "", "## Main findings", "",
            f"- SemiDDE D=2 recovered every layer in {exact_total('semidde_csp', 2)}/{per_depth_total} new fits, while D=3 recovered all layers in {exact_total('semidde_csp', 3)}/{per_depth_total}; the mean deepest D=3 width was {semi_deepest_mean:.2f} instead of 2.",
            f"- CopulaDDE D=3 recovered all layers in {exact_total('copula_csp', 3)}/{per_depth_total} new fits. At D=2, the originally selected candidate was exact in {original_total('copula_csp', 2)}/5 selection datasets but in only {exact_total('copula_csp', 2)}/{per_depth_total} independent datasets.",
            f"- NormalDDE D=2 was exact in {original_total('dde_normal_csp', 2)}/5 selection datasets but in {exact_total('dde_normal_csp', 2)}/{per_depth_total} independent datasets. BernoulliDDE was exact in {exact_total('dde_bernoulli_csp', 2)}/{per_depth_total} D=2 and {exact_total('dde_bernoulli_csp', 3)}/{per_depth_total} D=3 datasets.",
            f"- PoissonDDE reached {exact_total('dde_poisson_csp', 2)}/{per_depth_total} exact recovery at D=2 and {exact_total('dde_poisson_csp', 3)}/{per_depth_total} at D=3. Its recovery remained imperfect even at N=8000.",
            f"- CopulaDDE D=2 exact recovery changed from {exact_at('copula_csp', 2, 4000)}/{config['replicates']} at N=4000 to {exact_at('copula_csp', 2, 8000)}/{config['replicates']} at N=8000 under separately selected HP. This is evidence of nonmonotone robustness, not proof of which candidate is best.",
            "", "## Fit budgets", "",
            f"DDE: {config['dde_iterations']} iterations; SemiDDE: {config['semi_outer_iterations']} outer × {config['semi_inner_iterations']} inner iterations; CopulaDDE: {config['copula_iterations']} iterations; optimizer cap: {config['optimizer_iterations']}. The CopulaDDE budget matches the screening validation and does not establish production-budget convergence.",
            "", "## Recovery by setting", "",
            "| Kernel | D | N | Exact / 100 | Exact rate [95% CI] | Mean width L1 | Mean seconds | Original single run |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for item in joined.itertuples(index=False):
        original_exact = int(item.dimension_exact) if pd.notna(item.dimension_exact) else "—"
        rows.append(f"| {item.kernel} | {item.D} | {item.N} | {int(item.exact_count)}/{int(item.completed)} | "
                    f"{item.exact_rate_all:.2f} [{item.exact_ci_low:.2f}, {item.exact_ci_high:.2f}] | "
                    f"{item.mean_dimension_l1:.2f} | {item.mean_seconds:.1f} | {original_exact} |")
    rows.extend(["", "## Interpretation", "",
                 "The original 0/1 result is one selected-data fit. The new rates show how the fixed selected candidate generalizes to 100 fresh datasets at the same N. High rates support robust width recovery; low rates show that the selected candidate does not consistently recover all layer widths in this scenario.",
                 "", "These results evaluate the chosen candidates, but cannot establish that they are optimal among the entire grid: that would require paired 100-replicate evaluation of alternative candidates.",
                 "", "## Reproducibility", "",
                 "- `selected_hp.json`: frozen selected candidates and source selection seeds.",
                 "- `config.json`: simulation budgets, seed scheme, and source hashes.",
                 "- `cells/*.json`: one atomic checkpoint per fit.",
                 "- `fits.csv`: all replicate-level results.",
                 "- `summary.csv`: setting-level rates, width errors, runtime, and intervals.",
                 "- `integrity_check.json`: all 5,000 checkpoint, metric, seed, HP, code hash, and figure checks.",
                 "- `dde/scripts/simulate_selected_csp_hp_rep100.py`: simulation runner.",
                 "- `dde/scripts/summarize_selected_csp_hp_rep100.py`: verification, summary, and plotting.",
                 "- `dde/scripts/check_selected_csp_hp_rep100.py`: independent file-level integrity check."])
    (output / "REPORT.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    summary = aggregate(output, allow_partial=args.allow_partial)
    summary.to_csv(output / "summary.csv", index=False)
    if not args.allow_partial:
        plot_metric(summary, "exact_rate_all", "All-layer exact recovery under selected CSP hyperparameters (100 replicates)",
                    "Recovery rate (95% CI)", output)
        plot_metric(summary, "mean_dimension_l1", "Width error under selected CSP hyperparameters (100 replicates)",
                    "Mean width L1 error", output)
        plot_metric(summary, "median_seconds", "Per-fit runtime under selected CSP hyperparameters (100 replicates)",
                    "Median seconds", output)
        report(summary, output)
    print(f"Verified {int(summary.completed.sum())} fits across 50 settings; "
          f"{int(summary.failed.sum())} failures")


if __name__ == "__main__":
    main()
