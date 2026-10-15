"""Validate and summarize the 100-cell hyperparameter-selection benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path


SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000)
DEPTHS = (2, 3)
KERNELS = (
    "dde_normal_fixed", "dde_normal_csp",
    "dde_bernoulli_fixed", "dde_bernoulli_csp",
    "dde_poisson_fixed", "dde_poisson_csp",
    "semidde_fixed", "semidde_csp",
    "copula_fixed", "copula_csp",
)
DISPLAY = {
    "dde_normal_fixed": "Normal fixed",
    "dde_normal_csp": "Normal CSP",
    "dde_bernoulli_fixed": "Bernoulli fixed",
    "dde_bernoulli_csp": "Bernoulli CSP",
    "dde_poisson_fixed": "Poisson fixed",
    "dde_poisson_csp": "Poisson CSP",
    "semidde_fixed": "SemiDDE fixed",
    "semidde_csp": "SemiDDE CSP",
    "copula_fixed": "CopulaDDE fixed",
    "copula_csp": "CopulaDDE CSP",
}


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row):
    return row["kernel"], int(row["D"]), int(row["N"]), int(row["replicate"])


def _expected_candidates(kernel: str, depth: int) -> int:
    if kernel.endswith("csp"):
        return 9 if depth == 2 else 27
    if kernel.startswith("copula"):
        return 18 if depth == 2 else 54
    return 12 if depth == 2 else 24


def _duration(seconds: float, timeout: bool = False) -> str:
    prefix = ">" if timeout else ""
    if seconds >= 60:
        return f"{prefix}{seconds / 60:.1f}m"
    return f"{prefix}{seconds:.2f}s"


def _json_list(value: str):
    return tuple(int(v) for v in json.loads(value)) if value else ()


def validate(rows):
    expected = {
        (kernel, depth, N, 0)
        for kernel in KERNELS for depth in DEPTHS for N in SAMPLE_SIZES
    }
    keys = [_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise AssertionError(f"duplicate benchmark cells: {duplicates}")
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(keys) - expected)
    if missing or unexpected:
        raise AssertionError(f"coverage mismatch; missing={missing}, unexpected={unexpected}")
    errors = [row for row in rows if row["status"] == "error"]
    if errors:
        raise AssertionError(f"benchmark contains error rows: {[(_key(row), row['error']) for row in errors]}")
    for row in rows:
        if row["status"] not in {"ok", "timeout"}:
            raise AssertionError(f"unsupported status in {_key(row)}: {row['status']}")
        if row["status"] == "timeout":
            if float(row["search_seconds"]) < 1799:
                raise AssertionError(f"timeout lower bound is too short in {_key(row)}")
            continue
        depth = int(row["D"])
        expected_candidates = _expected_candidates(row["kernel"], depth)
        if int(row["candidate_count"]) != expected_candidates:
            raise AssertionError(
                f"incomplete grid in {_key(row)}: {row['candidate_count']} != {expected_candidates}"
            )
        if (
            row["finite"] != "1"
            or not math.isfinite(float(row["best_score"]))
            or not math.isfinite(float(row["best_score_sd"]))
            or float(row["search_seconds"]) <= 0
        ):
            raise AssertionError(f"non-finite successful result in {_key(row)}")
        if row["predictive_metric"] and not math.isfinite(float(row["predictive_metric"])):
            raise AssertionError(f"non-finite predictive metric in {_key(row)}")
        if row["dimension_method"] == "csp":
            if not row["selected_t_spike"] or row["selected_pen_list"]:
                raise AssertionError(f"CSP hyperparameters are inconsistent in {_key(row)}")
        elif not row["selected_pen_list"] or row["selected_t_spike"]:
            raise AssertionError(f"fixed-path hyperparameters are inconsistent in {_key(row)}")
        if len(_json_list(row["fitted_K"])) != depth:
            raise AssertionError(f"wrong fitted depth in {_key(row)}")
        if row["dimension_method"] == "fixed" and row["dimension_exact"] != "1":
            raise AssertionError(f"fixed-width result changed the requested width in {_key(row)}")


def _runtime_table(index, depth: int):
    lines = [
        f"### D={depth}", "",
        "| 内核 | N=500 | N=1000 | N=2000 | N=4000 | N=8000 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kernel in KERNELS:
        cells = []
        for N in SAMPLE_SIZES:
            row = index[(kernel, depth, N, 0)]
            cells.append(_duration(float(row["search_seconds"]), row["status"] == "timeout"))
        lines.append(f"| {DISPLAY[kernel]} | " + " | ".join(cells) + " |")
    return lines


def _width_table(index, depth: int):
    lines = [
        f"### D={depth}", "",
        "| CSP 内核 | N=500 | N=1000 | N=2000 | N=4000 | N=8000 |",
        "|---|---|---|---|---|---|",
    ]
    for kernel in [value for value in KERNELS if value.endswith("csp")]:
        cells = []
        for N in SAMPLE_SIZES:
            row = index[(kernel, depth, N, 0)]
            if row["status"] == "timeout":
                cells.append("timeout")
            else:
                mark = "✓" if row["dimension_exact"] == "1" else f"Δ{row['dimension_l1']}"
                cells.append(f"{row['fitted_K']} {mark}")
        lines.append(f"| {DISPLAY[kernel]} | " + " | ".join(cells) + " |")
    return lines


def _metric_tables(index):
    lines = []
    families = (
        ("Normal", "dde_normal", "conditional RMSE"),
        ("Bernoulli", "dde_bernoulli", "conditional log loss"),
        ("Poisson", "dde_poisson", "conditional NLL"),
        ("CopulaDDE", "copula", "pMSE"),
    )
    for title, prefix, metric_name in families:
        lines.extend([
            f"### {title}: {metric_name}（越低越好）", "",
            "| 路径 / D | N=500 | N=1000 | N=2000 | N=4000 | N=8000 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for path in ("fixed", "csp"):
            kernel = f"{prefix}_{path}"
            for depth in DEPTHS:
                cells = []
                for N in SAMPLE_SIZES:
                    row = index[(kernel, depth, N, 0)]
                    if row["status"] == "timeout" or not row["predictive_metric"]:
                        cells.append("—")
                    else:
                        cells.append(f"{float(row['predictive_metric']):.4f}")
                lines.append(f"| {path} / {depth} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _write_summary_csv(path: Path, rows):
    fields = (
        "kernel", "D", "successful_cells", "timeout_cells", "exact_cells",
        "mean_dimension_l1", "mean_success_seconds", "max_success_seconds",
        "n500_seconds", "n8000_seconds_or_bound", "mean_predictive_metric",
    )
    output = []
    for kernel in KERNELS:
        for depth in DEPTHS:
            group = [row for row in rows if row["kernel"] == kernel and int(row["D"]) == depth]
            successful = [row for row in group if row["status"] == "ok"]
            metrics = [float(row["predictive_metric"]) for row in successful if row["predictive_metric"]]
            output.append({
                "kernel": kernel,
                "D": depth,
                "successful_cells": len(successful),
                "timeout_cells": sum(row["status"] == "timeout" for row in group),
                "exact_cells": sum(row.get("dimension_exact") == "1" for row in successful),
                "mean_dimension_l1": (
                    sum(float(row["dimension_l1"]) for row in successful) / len(successful)
                    if successful else ""
                ),
                "mean_success_seconds": (
                    sum(float(row["search_seconds"]) for row in successful) / len(successful)
                    if successful else ""
                ),
                "max_success_seconds": max((float(row["search_seconds"]) for row in successful), default=""),
                "n500_seconds": next(float(row["search_seconds"]) for row in group if int(row["N"]) == 500),
                "n8000_seconds_or_bound": next(float(row["search_seconds"]) for row in group if int(row["N"]) == 8000),
                "mean_predictive_metric": sum(metrics) / len(metrics) if metrics else "",
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def _plot_runtime(path: Path, index):
    cache = Path(tempfile.gettempdir()) / "dde-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "dde_normal": "#1f77b4", "dde_bernoulli": "#ff7f0e",
        "dde_poisson": "#2ca02c", "semidde": "#9467bd", "copula": "#d62728",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, depth in zip(axes, DEPTHS):
        for kernel in KERNELS:
            family = next(prefix for prefix in colors if kernel.startswith(prefix))
            rows = [index[(kernel, depth, N, 0)] for N in SAMPLE_SIZES]
            seconds = [float(row["search_seconds"]) for row in rows]
            linestyle = "--" if kernel.endswith("csp") else "-"
            axis.plot(
                SAMPLE_SIZES, seconds, color=colors[family], linestyle=linestyle,
                marker="o", linewidth=1.6, markersize=4, label=DISPLAY[kernel],
            )
            timeout_x = [N for N, row in zip(SAMPLE_SIZES, rows) if row["status"] == "timeout"]
            if timeout_x:
                axis.scatter(timeout_x, [1800] * len(timeout_x), color=colors[family], marker="x", s=48, zorder=4)
        axis.axhline(1800, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(f"D={depth}")
        axis.set_xlabel("N")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("full-grid selection wall time (seconds, log scale)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Hyperparameter selection scaling (x = 30-minute timeout)")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, rows, index):
    status = Counter(row["status"] for row in rows)
    csp_summary = []
    for kernel in [value for value in KERNELS if value.endswith("csp")]:
        for depth in DEPTHS:
            group = [
                row for row in rows
                if row["kernel"] == kernel and int(row["D"]) == depth and row["status"] == "ok"
            ]
            exact = sum(row["dimension_exact"] == "1" for row in group)
            csp_summary.append((kernel, depth, exact, len(group)))

    copula_success = [row for row in rows if row["kernel"].startswith("copula") and row["status"] == "ok"]
    fit_share = sum(float(row["candidate_fit_seconds"]) for row in copula_success) / sum(
        float(row["search_seconds"]) for row in copula_success
    )
    lines = [
        "# 全内核超参数选择规模测试", "",
        "## 结论", "",
        f"- 共覆盖 100 个唯一组合：{status['ok']} 个完整结束，{status['timeout']} 个在 30 分钟完整网格上限处超时，0 个代码或数值错误。",
        "- 除 CopulaDDE 外，N=8000、D=3 的所有路径均在本机 4-worker 条件下完成；最慢的是 Poisson CSP（251.6 秒）和 Bernoulli fixed（238.1 秒）。",
        "- CopulaDDE 是明确瓶颈。fixed 在 D=2 可完整运行到 N=4000（50.7 分钟实测），D=3 仅 N=500 完成（52.6 分钟）；CSP 在 D=2 可在 30 分钟内运行到 N=4000（24.1 分钟），D=3 仅 N=500 完成（18.3 分钟）。",
        f"- 已完成 CopulaDDE 点中，候选拟合占总时间 {fit_share * 100:.1f}%，pMSE 评分只占很小部分；优化重点应放在重复拟合、初始化复用或候选级并行。",
        "- CSP 维度恢复并不均匀：SemiDDE D=2 为 5/5 精确，Poisson D=3 为 4/5；Bernoulli D=2/3 和 Normal D=3 均未精确恢复。预测指标仍较稳定，因此不能用预测拟合质量代替维度恢复检查。",
        "", "## 测试设置", "",
        "- N = [500, 1000, 2000, 4000, 8000]，D = 2/3，每个组合 1 个固定随机种子重复。",
        "- 每个任务直接调用当前 `select_hyperparameters()`，所有候选均为笛卡尔积完整 grid search。",
        "- 候选数：普通 fixed 为 D2=12/D3=24，普通 CSP 为 9/27；Copula fixed 为 18/54，Copula CSP 为 9/27。",
        "- 迭代预算：DDE=20；SemiDDE=2 个 outer × 20 个 inner；CopulaDDE=10；优化器 maxiter=30。",
        "- Copula pMSE：2 repeats、3 folds、30 trees。主批次使用 4 个进程并行，每进程 BLAS 单线程；最后的 D=3、N=8000 fixed 点另以单 worker 复核。",
        "- 初始无上限批次保留了两个超过 30 分钟但最终完成的 Copula 点；确认小时级成本后，剩余任务采用 1800 秒可用性上限。timeout 不保留部分网格的最优值。",
        "", "## 运行时间", "",
    ]
    for depth in DEPTHS:
        lines.extend(_runtime_table(index, depth))
        lines.append("")
    lines.extend(["![runtime scaling](runtime_scaling.png)", "", "## CSP 选择出的层宽", "", "标记 ✓ 表示与真值相同；Δ 为层宽 L1 误差。", ""])
    for depth in DEPTHS:
        lines.extend(_width_table(index, depth))
        lines.append("")
    lines.extend(["## 评估指标", "", "这些指标是在每个生成数据集上对最终选中模型计算；不同分布间不可横向比较。", ""])
    lines.extend(_metric_tables(index))
    lines.extend([
        "## CSP 精确恢复汇总", "",
        "| 内核 | D | 完整结束 | 精确恢复 |", "|---|---:|---:|---:|",
    ])
    for kernel, depth, exact, completed in csp_summary:
        lines.append(f"| {DISPLAY[kernel]} | {depth} | {completed}/5 | {exact}/{completed} |")
    lines.extend([
        "", "## 数据文件", "",
        "- `fits.csv`：100 个组合的逐项状态、最优超参、层宽、分数、指标和分项计时。",
        "- `summary.csv`：按内核和 D 聚合的成功数、timeout 数、维度误差、耗时及平均指标。",
        "- `runtime_scaling.png`：对数时间曲线；x 标记表示 30 分钟 timeout。",
        "", "## 解释限制", "",
        "- 本轮目标是覆盖所有路径与规模点，因此每格只运行 1 个重复；它能评估接口完整性与运行成本，不能估计恢复率的置信区间。",
        "- 为使 100 格测试可执行，迭代次数低于部分生产默认值；数值指标用于路径间一致性和规模趋势检查。生产收敛研究应在选定 N/D 上增加迭代与重复。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/hyperparameter_selection_scale"))
    args = parser.parse_args()
    rows = _read(args.results / "fits.csv")
    validate(rows)
    index = {_key(row): row for row in rows}
    _write_summary_csv(args.results / "summary.csv", rows)
    _plot_runtime(args.results / "runtime_scaling.png", index)
    _write_report(args.results / "REPORT.md", rows, index)
    print(f"validated {len(rows)} cells and wrote {args.results / 'REPORT.md'}")


if __name__ == "__main__":
    main()
