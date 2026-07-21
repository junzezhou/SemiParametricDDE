"""Unified visualization report generation for fitted DDE models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from semidde.results import DDEResult


MODEL_CONFIG = {
    "semi": {
        "label": "SemiDDE",
        "prefix": "semi",
        "header_color": "#e8f2df",
        "latent_binary_colors": ("#f7fcf5", "#238b45"),
        "latent_cmap": "Greens",
        "cluster_default": "when_no_alignment",
        "cluster_title": "SemiDDE item clusters by B1 loading pattern",
        "top_loading_title": "Largest loading patterns",
    },
    "copula": {
        "label": "CopDDE",
        "prefix": "copula",
        "header_color": "#d9ecf7",
        "latent_binary_colors": ("#f7fbff", "#2171b5"),
        "latent_cmap": "Blues",
        "cluster_default": "always",
        "cluster_title": "CopDDE item clusters by B1 loading pattern",
        "top_loading_title": "Largest active-column loadings",
    },
}


def _deps():
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "semidde-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "semidde-cache"))
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "SemiDDE visualization requires the optional dependencies "
            "`matplotlib` and `pandas`. Install with `pip install semidde[viz]` "
            "or install those packages directly."
        ) from exc
    return plt, pd


def _model_key(model: Optional[str], result: Optional[DDEResult] = None) -> str:
    chosen = str(model or (result.model if result is not None else "semi")).lower()
    if chosen in {"semidde"}:
        chosen = "semi"
    if chosen in {"copdde"}:
        chosen = "copula"
    if chosen not in MODEL_CONFIG:
        raise ValueError("model must be 'semi' or 'copula'.")
    return chosen


def _ensure_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save(fig, output_dir: str | Path, filename: str) -> str:
    out = _ensure_dir(output_dir) / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return str(out)


def _as_cv_table(result: DDEResult, cv_table=None):
    _, pd = _deps()
    if cv_table is None:
        cv_table = result.diagnostics.get("cv_table") if result.diagnostics else None
    if cv_table is None:
        return None
    return pd.DataFrame(cv_table)


def _best_row_from_cv(cv):
    if cv is None or cv.empty or "mean_score" not in cv:
        return {}
    return cv.loc[cv["mean_score"].astype(float).idxmax()].to_dict()


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=object).reshape(-1)
        if arr.size > 6:
            return "[" + ", ".join(map(str, arr[:6])) + ", ...]"
        return "[" + ", ".join(map(str, arr)) + "]"
    return str(value)


def _active_columns(B: np.ndarray, max_columns: int = 12) -> list[int]:
    slopes = np.asarray(B, dtype=float)[:, 1:]
    norms = np.linalg.norm(slopes, axis=0)
    active = np.where(norms > 0)[0]
    if active.size == 0:
        active = np.argsort(norms)[-min(max_columns, norms.size) :]
    else:
        active = active[np.argsort(norms[active])[::-1]]
    return [int(i) for i in active[:max_columns]]


def _latent_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{j + 1}" for j in range(n)]


def _item_labels(n: int, feature_names: Optional[Sequence[str]]) -> list[str]:
    if feature_names is None:
        return [f"item_{i + 1}" for i in range(n)]
    labels = list(map(str, feature_names))
    if len(labels) < n:
        labels.extend(f"item_{i + 1}" for i in range(len(labels), n))
    return labels[:n]


def _top_loading_frame(B, row_names, col_prefix, top_n=10):
    _, pd = _deps()
    B = np.asarray(B, dtype=float)
    rows = []
    for col in _active_columns(B, max_columns=B.shape[1] - 1):
        slopes = B[:, col + 1]
        order = np.argsort(np.abs(slopes))[::-1][:top_n]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "latent": f"{col_prefix}_{col + 1}",
                    "rank": rank,
                    "row": row_names[idx],
                    "loading": float(slopes[idx]),
                    "abs_loading": float(abs(slopes[idx])),
                }
            )
    return pd.DataFrame(rows)


def _coerce_latent_frame(result: DDEResult, latent_frame=None):
    _, pd = _deps()
    if latent_frame is not None:
        return pd.DataFrame(latent_frame).copy()
    if not result.A_list:
        return None
    data = {}
    for layer, A in enumerate(result.A_list, start=1):
        A = np.asarray(A)
        for j in range(A.shape[1]):
            data[f"A{layer}_{j + 1}"] = A[:, j]
    return pd.DataFrame(data)


def _with_default_top_label(result: DDEResult, latent_df, label_col):
    if label_col is not None and label_col in latent_df:
        return latent_df, label_col
    if not result.A_list:
        return latent_df, label_col
    top = np.asarray(result.A_list[-1])
    if top.ndim == 1:
        top = top.reshape(-1, 1)
    if top.ndim != 2 or top.shape[0] != latent_df.shape[0] or top.shape[1] < 1:
        return latent_df, label_col
    resolved = "top_latent_state"
    while resolved in latent_df.columns:
        resolved = "_" + resolved
    layer_name = f"A{len(result.A_list)}"
    if top.shape[1] == 1:
        labels = np.where(top[:, 0] > 0.5, f"{layer_name}=1", f"{layer_name}=0")
    else:
        labels = [
            f"{layer_name}=" + "".join(str(int(value > 0.5)) for value in row)
            for row in top
        ]
    out = latent_df.copy()
    out[resolved] = labels
    return out, resolved


def _table_figure(rows, title, header_color, figsize=(8.6, 3.2)):
    plt, _ = _deps()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Field", "Value"],
        bbox=[0.0, 0.0, 1.0, 0.86],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        else:
            cell.set_edgecolor("#d0d0d0")
    fig.suptitle(title, fontsize=14, y=0.98)
    return fig


def _plot_heatmap(matrix, title, xlabels, ylabels, output_dir, filename, center_zero=True):
    plt, _ = _deps()
    M = np.asarray(matrix, dtype=float)
    fig_w = max(7.5, min(16.0, 0.28 * M.shape[1] + 4.0))
    fig_h = max(4.5, min(14.0, 0.22 * M.shape[0] + 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if center_zero:
        vmax = float(np.nanmax(np.abs(M))) if M.size else 1.0
        vmax = max(vmax, 1e-8)
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)
    step = max(1, int(np.ceil(len(ylabels) / 35)))
    ax.set_yticks(range(0, len(ylabels), step))
    ax.set_yticklabels([ylabels[i] for i in range(0, len(ylabels), step)], fontsize=8)
    ax.set_xlabel("latent column")
    ax.set_ylabel("rows")
    fig.colorbar(im, ax=ax, shrink=0.75, label="loading")
    path = _save(fig, output_dir, filename)
    plt.close(fig)
    return path


def _cluster_item_order(slopes, max_clusters=8):
    X = np.asarray(slopes, dtype=float)
    n_items = X.shape[0]
    if n_items <= 1:
        return np.arange(n_items), np.ones(n_items, dtype=int)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_scaled = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    try:
        from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
        from scipy.spatial.distance import pdist

        distances = pdist(X_scaled, metric="cosine")
        distances = np.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=1.0)
        if np.allclose(distances, 0.0):
            order = np.argsort(np.linalg.norm(X, axis=1))[::-1]
            clusters = np.ones(n_items, dtype=int)
        else:
            tree = linkage(distances, method="average", optimal_ordering=True)
            order = leaves_list(tree)
            n_clusters = min(max_clusters, max(2, int(np.sqrt(n_items))))
            clusters = fcluster(tree, t=n_clusters, criterion="maxclust")
    except Exception:
        order = np.argsort(np.linalg.norm(X, axis=1))[::-1]
        clusters = np.ones(n_items, dtype=int)
    ordered_clusters = clusters[order]
    remap = {}
    next_id = 1
    cluster_labels = []
    for cluster in ordered_clusters:
        cluster = int(cluster)
        if cluster not in remap:
            remap[cluster] = next_id
            next_id += 1
        cluster_labels.append(remap[cluster])
    return order, np.asarray(cluster_labels, dtype=int)


def _plot_item_cluster_heatmap(B1, row_names, output_dir, filename, title):
    plt, _ = _deps()
    B1 = np.asarray(B1, dtype=float)
    if B1.ndim != 2 or B1.shape[1] <= 1:
        return None
    slopes = B1[:, 1:]
    order, cluster_labels = _cluster_item_order(slopes)
    heat = slopes[order]
    labels = [row_names[i] for i in order]
    col_labels = _latent_names("A1", slopes.shape[1])
    fig_w = max(7.8, min(16.0, 0.34 * heat.shape[1] + 5.2))
    fig_h = max(5.0, min(16.0, 0.18 * heat.shape[0] + 2.8))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.nanmax(np.abs(heat))) if heat.size else 1.0
    vmax = max(vmax, 1e-8)
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    step = max(1, int(np.ceil(len(labels) / 35)))
    ax.set_yticks(range(0, len(labels), step))
    ax.set_yticklabels([f"C{cluster_labels[i]} | {labels[i]}" for i in range(0, len(labels), step)], fontsize=7)
    ax.set_xlabel("A1 latent")
    ax.set_ylabel("cluster | item")
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != cluster_labels[i - 1]:
            ax.axhline(i - 0.5, color="black", linewidth=0.8, alpha=0.65)
    fig.colorbar(im, ax=ax, shrink=0.72, label="loading")
    path = _save(fig, output_dir, filename)
    plt.close(fig)
    return path


def _plot_bar_frame(df, output_dir, filename, title, value_col="loading", label_col="row", group_col="latent"):
    plt, _ = _deps()
    if df.empty:
        return None
    show = df.sort_values("abs_loading", ascending=False).head(24).copy()
    labels = show[group_col].astype(str) + " | " + show[label_col].astype(str)
    values = show[value_col].astype(float)
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.28 * len(show) + 1.5)))
    ax.barh(range(len(show)), values, color=np.where(values >= 0, "#2b8cbe", "#de2d26"))
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(value_col)
    path = _save(fig, output_dir, filename)
    plt.close(fig)
    return path


def _summary_rows(result: DDEResult, model_key: str, best: Mapping[str, Any], B_list, A_list):
    K_list = result.diagnostics.get("K_list") if result.diagnostics else None
    if K_list is None and B_list:
        K_list = [B.shape[1] - 1 for B in B_list]
    original_K_list = result.diagnostics.get("original_K_list") if result.diagnostics else None
    rows = [
        ("model", result.model),
        ("active latent dimensions", _format_value(K_list)),
        ("original latent capacity", _format_value(original_K_list) if original_K_list else "not available"),
        ("sample size", str(A_list[0].shape[0]) if A_list else "not available"),
        ("observed variables", str(B_list[0].shape[0]) if B_list else "not available"),
        ("iterations", _format_value(result.diagnostics.get("iterations", "not available"))),
        ("criterion", _format_value(best.get("criterion", "not available"))),
        ("best score", _format_value(best.get("mean_score", "not available"))),
        ("tau", _format_value(best.get("tau", "not available"))),
    ]
    if model_key == "semi":
        rows.extend(
            [
                ("pen_list", _format_value(best.get("pen_list", "not available"))),
                ("lambda_exp", _format_value(best.get("lambda_exp", "not available"))),
                ("g update rho", _format_value(best.get("g_update_rho", "not available"))),
            ]
        )
    else:
        rows[6:6] = [
            (
                "dimension method",
                _format_value(best.get("dimension_method", result.diagnostics.get("backend", "not available"))),
            )
        ]
        rows.extend(
            [
                ("t_spike", _format_value(best.get("t_spike", "not available"))),
                ("temperature", _format_value(best.get("temp", "not available"))),
            ]
        )
    return rows


def plot_model_summary(
    result: DDEResult,
    *,
    output_dir: str | Path,
    model: Optional[str] = None,
    prefix: Optional[str] = None,
    cv_table=None,
    feature_names: Optional[Sequence[str]] = None,
    max_heatmap_rows: int = 80,
    top_loadings: int = 8,
    include_item_cluster: Optional[bool] = None,
) -> Dict[str, str]:
    """Create model-summary visualizations for a fitted DDE result."""
    plt, pd = _deps()
    model_key = _model_key(model, result)
    config = MODEL_CONFIG[model_key]
    prefix = prefix or config["prefix"]
    cv = _as_cv_table(result, cv_table)
    best = _best_row_from_cv(cv)
    B_list = [np.asarray(B, dtype=float) for B in result.B_list or []]
    A_list = [np.asarray(A) for A in result.A_list or []]
    rows = _summary_rows(result, model_key, best, B_list, A_list)
    fig = _table_figure(
        rows,
        f"{config['label']} model and selected hyperparameters",
        config["header_color"],
    )
    paths = {"summary": _save(fig, output_dir, f"{prefix}_model_summary.png")}
    plt.close(fig)

    row_names = _item_labels(B_list[0].shape[0], feature_names) if B_list else []
    if include_item_cluster is None:
        include_item_cluster = config["cluster_default"] == "always"
    if include_item_cluster and B_list:
        path = _plot_item_cluster_heatmap(
            B_list[0],
            row_names,
            output_dir,
            f"{prefix}_item_cluster_heatmap.png",
            config["cluster_title"],
        )
        if path is not None:
            paths["item_cluster_heatmap"] = path
    top_frames = []
    for layer, B in enumerate(B_list, start=1):
        slopes = B[:, 1:]
        layer_rows = row_names if layer == 1 else _latent_names(f"A{layer - 1}", B.shape[0])
        col_labels = _latent_names(f"A{layer}", slopes.shape[1])
        if slopes.shape[0] > max_heatmap_rows:
            row_order = np.argsort(np.linalg.norm(slopes, axis=1))[::-1][:max_heatmap_rows]
            heat = slopes[row_order]
            ylabels = [layer_rows[i] for i in row_order]
        else:
            heat = slopes
            ylabels = layer_rows
        paths[f"B{layer}_heatmap"] = _plot_heatmap(
            heat,
            f"{config['label']} layer {layer} loading matrix",
            col_labels,
            ylabels,
            output_dir,
            f"{prefix}_B{layer}_heatmap.png",
        )
        top_frames.append(_top_loading_frame(slopes, layer_rows, f"A{layer}", top_n=top_loadings))
    if top_frames:
        top_df = pd.concat(top_frames, ignore_index=True)
        path = _plot_bar_frame(
            top_df,
            output_dir,
            f"{prefix}_top_loadings.png",
            config["top_loading_title"],
        )
        if path is not None:
            paths["top_loadings"] = path
    return paths


def _plot_latent_profile_heatmap(
    latent_df,
    output_dir,
    filename,
    title,
    *,
    model_key,
    max_samples=300,
    label_col=None,
    average_by_label=False,
    continuous_label_bins=10,
):
    plt, pd = _deps()
    latent_cols = [c for c in latent_df.columns if str(c).startswith("A")]
    if not latent_cols:
        return None

    data = latent_df[latent_cols].astype(float).copy()
    label_values = None
    label_numeric = None
    is_continuous_label = False
    average_labels = None
    average_label_counts = {}
    if label_col is not None and label_col in latent_df:
        label_values = latent_df[label_col].copy()
        numeric = pd.to_numeric(label_values, errors="coerce")
        all_numeric = bool(numeric.notna().all())
        is_continuous_label = all_numeric and numeric.nunique(dropna=True) > 12
        if all_numeric:
            order = np.argsort(numeric.to_numpy(), kind="stable")
            label_numeric = numeric.iloc[order].reset_index(drop=True)
        else:
            order = np.argsort(label_values.astype(str).to_numpy(), kind="stable")
        data = data.iloc[order].reset_index(drop=True)
        label_values = label_values.iloc[order].reset_index(drop=True)
        if average_by_label:
            if is_continuous_label:
                n_bins = min(int(continuous_label_bins), max(1, numeric.nunique(dropna=True)))
                ordered_numeric = numeric.iloc[order].reset_index(drop=True)
                bins = pd.qcut(ordered_numeric, q=n_bins, duplicates="drop")
                average_labels = bins.astype(str).reset_index(drop=True)
                label_numeric = ordered_numeric
            else:
                average_labels = label_values.astype(str).reset_index(drop=True)
            average_label_counts = average_labels.value_counts(sort=False).to_dict()
            means = data.groupby(average_labels, sort=False)[latent_cols].transform("mean")
            data = means.astype(float)
    elif len(latent_cols) > 1:
        order = np.lexsort([data[c].to_numpy() for c in reversed(latent_cols)])
        data = data.iloc[order].reset_index(drop=True)

    if data.shape[0] > max_samples:
        idx = np.linspace(0, data.shape[0] - 1, max_samples).round().astype(int)
        data = data.iloc[idx].reset_index(drop=True)
        if label_values is not None:
            label_values = label_values.iloc[idx].reset_index(drop=True)
        if label_numeric is not None:
            label_numeric = label_numeric.iloc[idx].reset_index(drop=True)
        if average_labels is not None:
            average_labels = average_labels.iloc[idx].reset_index(drop=True)

    if is_continuous_label and label_numeric is not None:
        fig, (ax, label_ax) = plt.subplots(
            1,
            2,
            figsize=(max(8.0, 0.4 * len(latent_cols) + 4.8), 6.0),
            gridspec_kw={"width_ratios": [24, 1], "wspace": 0.08},
            sharey=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(max(7.8, 0.4 * len(latent_cols) + 5.3), 6.0))
        label_ax = None

    values = data.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    is_binary_profile = finite_values.size > 0 and np.isin(finite_values, [0.0, 1.0]).all()
    config = MODEL_CONFIG[model_key]
    if is_binary_profile and not average_by_label:
        from matplotlib.colors import BoundaryNorm, ListedColormap

        cmap = ListedColormap(list(config["latent_binary_colors"]))
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(values, aspect="auto", cmap=config["latent_cmap"], vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("latent variable")
    if average_by_label and label_values is not None:
        ax.set_ylabel("label-averaged samples sorted by " + str(label_col))
    else:
        ax.set_ylabel("samples sorted by " + str(label_col) if label_values is not None else "samples")
    ax.set_xticks(range(len(latent_cols)))
    ax.set_xticklabels(latent_cols, rotation=90, fontsize=8)
    ax.set_yticks([])

    if label_values is not None and (not is_continuous_label or average_by_label):
        labels_source = average_labels if average_labels is not None else label_values.astype(str)
        labels = labels_source.astype(str).to_numpy()
        starts = [0]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                starts.append(i)
        ends = starts[1:] + [len(labels)]
        for start, end in zip(starts, ends):
            if start > 0:
                ax.axhline(start - 0.5, color="black", linewidth=0.7, alpha=0.7)
            mid = (start + end - 1) / 2
            ax.text(
                len(latent_cols) + 0.15,
                mid,
                (
                    f"{labels[start]} (n={average_label_counts[labels[start]]})"
                    if average_by_label and labels[start] in average_label_counts
                    else f"{labels[start]} (shown={end - start})"
                ),
                va="center",
                ha="left",
                fontsize=8,
                clip_on=False,
            )
        ax.set_xlim(-0.5, len(latent_cols) + 2.4)
    elif label_ax is not None and label_numeric is not None:
        strip = label_numeric.to_numpy(dtype=float).reshape(-1, 1)
        label_im = label_ax.imshow(strip, aspect="auto", cmap="viridis")
        label_ax.set_title(str(label_col), fontsize=9)
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        fig.colorbar(label_im, ax=label_ax, fraction=0.85, pad=0.05)

    cbar_label = "modal state" if is_binary_profile and not average_by_label else "Pr(A=1)"
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, label=cbar_label)
    if is_binary_profile and not average_by_label:
        cbar.set_ticks([0, 1])
    path = _save(fig, output_dir, filename)
    plt.close(fig)
    return path


def _posterior_quality_table(latent_df, label_col=None):
    _, pd = _deps()
    latent_cols = [c for c in latent_df.columns if str(c).startswith("A")]
    if label_col is None or label_col not in latent_df or not latent_cols:
        rows = [
            {
                "latent": col,
                "state": 1,
                "n": int(latent_df[col].notna().sum()),
                "posterior_mass": float(latent_df[col].astype(float).mean()),
                "baseline_mass": float(latent_df[col].astype(float).mean()),
                "lift": 1.0,
            }
            for col in latent_cols
        ]
        return pd.DataFrame(rows)
    rows = []
    for label, sub in latent_df.groupby(label_col):
        label_n = int(sub.shape[0])
        for col in latent_cols:
            baseline = float(latent_df[col].astype(float).mean())
            mass = float(sub[col].astype(float).mean())
            rows.append(
                {
                    "label": label,
                    "latent": col,
                    "state": 1,
                    "n": label_n,
                    "posterior_mass": mass,
                    "baseline_mass": baseline,
                    "lift": mass / baseline if baseline > 0 else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["label", "lift"], ascending=[True, False])


def _plot_quality_table(df, output_dir, filename, title, header_color, max_rows=18):
    plt, _ = _deps()
    if df.empty:
        return None
    show = df.head(max_rows).copy()
    for col in ["posterior_mass", "baseline_mass", "lift"]:
        if col in show:
            show[col] = show[col].map(lambda x: "" if np.isnan(float(x)) else f"{float(x):.3f}")
    fig, ax = plt.subplots(figsize=(11.5, max(3.5, 0.38 * len(show) + 1.3)))
    ax.axis("off")
    table = ax.table(
        cellText=show.astype(str).values,
        colLabels=list(show.columns),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.3)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        else:
            cell.set_edgecolor("#d0d0d0")
    ax.set_title(title, fontsize=14, pad=12)
    path = _save(fig, output_dir, filename)
    plt.close(fig)
    return path


def _plot_alignment_heatmap(alignment_weights, output_dir, filename, title):
    plt, pd = _deps()
    if alignment_weights is None:
        return None
    df = pd.DataFrame(alignment_weights).copy()
    if df.empty:
        return None
    row_labels = df.iloc[:, 0].astype(str).tolist() if not np.issubdtype(df.iloc[:, 0].dtype, np.number) else df.index.astype(str).tolist()
    values = df.iloc[:, 1:] if len(row_labels) == df.shape[0] and df.shape[1] > 1 else df
    fig, ax = plt.subplots(figsize=(max(7.0, 0.5 * values.shape[1] + 3), max(4.0, 0.38 * values.shape[0] + 2)))
    im = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(list(values.columns), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.75, label="alignment weight")
    path = _save(fig, output_dir, filename)
    plt.close(fig)
    return path


def plot_latent_profiles(
    result: DDEResult,
    *,
    output_dir: str | Path,
    model: Optional[str] = None,
    prefix: Optional[str] = None,
    latent_frame=None,
    label_col: Optional[str] = None,
    alignment_weights=None,
    max_samples: int = 300,
    average_by_label: bool = False,
    continuous_label_bins: int = 10,
) -> Dict[str, str]:
    """Visualize latent profiles with model-specific auxiliary plots."""
    model_key = _model_key(model, result)
    config = MODEL_CONFIG[model_key]
    prefix = prefix or config["prefix"]
    latent_df = _coerce_latent_frame(result, latent_frame)
    if latent_df is None:
        return {}
    latent_df, label_col = _with_default_top_label(result, latent_df, label_col)
    paths: Dict[str, str] = {}
    latent_profile_key = "latent_profiles_prevalence" if average_by_label else "latent_profiles"
    latent_profile_name = (
        f"{prefix}_latent_profiles_prevalence.png" if average_by_label else f"{prefix}_latent_profiles.png"
    )
    latent_profile_title = (
        f"{config['label']} latent profile prevalence by label"
        if average_by_label
        else f"{config['label']} sample latent profiles"
    )
    path = _plot_latent_profile_heatmap(
        latent_df,
        output_dir,
        latent_profile_name,
        latent_profile_title,
        model_key=model_key,
        max_samples=max_samples,
        label_col=label_col,
        average_by_label=average_by_label,
        continuous_label_bins=continuous_label_bins,
    )
    if path:
        paths[latent_profile_key] = path
    if model_key == "semi":
        quality = _posterior_quality_table(latent_df, label_col=label_col)
        path = _plot_quality_table(
            quality,
            output_dir,
            f"{prefix}_posterior_quality_table.png",
            "Label posterior quality summary" if label_col else "Latent posterior mass summary",
            config["header_color"],
        )
        if path:
            paths["posterior_quality_table"] = path
        path = _plot_alignment_heatmap(
            alignment_weights,
            output_dir,
            f"{prefix}_alignment_heatmap.png",
            "Estimated latent-to-label alignment",
        )
        if path:
            paths["alignment_heatmap"] = path
    return paths


def plot_theme_table(
    comparison_table,
    *,
    output_dir: str | Path,
    prefix: str = "copula",
    title: str = "Latent theme comparison",
) -> Dict[str, str]:
    """Render a compact theme-comparison table or prediction confusion matrix."""
    plt, pd = _deps()
    df = pd.DataFrame(comparison_table)
    if df.empty:
        return {}
    if {"y_true", "y_pred"} <= set(df.columns):
        valid = df[["y_true", "y_pred"]].dropna().copy()
        if valid.empty:
            return {}
        labels = sorted(set(valid["y_true"]).union(set(valid["y_pred"])))
        matrix = pd.crosstab(valid["y_true"], valid["y_pred"]).reindex(
            index=labels,
            columns=labels,
            fill_value=0,
        )
        accuracy = float((valid["y_true"].to_numpy() == valid["y_pred"].to_numpy()).mean())
        fig, ax = plt.subplots(figsize=(max(5.5, 0.85 * len(labels) + 3.0), max(4.6, 0.7 * len(labels) + 2.4)))
        im = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", interpolation="nearest")
        ax.set_title(f"Latent-profile label prediction confusion matrix\naccuracy={accuracy:.3f}", fontsize=13)
        ax.set_xlabel("predicted label")
        ax.set_ylabel("true label")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([str(label) for label in labels], fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([str(label) for label in labels], fontsize=9)
        max_count = float(matrix.to_numpy().max()) if matrix.size else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                count = int(matrix.iat[i, j])
                color = "white" if max_count and count > 0.55 * max_count else "black"
                ax.text(j, i, str(count), ha="center", va="center", fontsize=10, color=color)
        fig.colorbar(im, ax=ax, shrink=0.75, label="count")
        path = _save(fig, output_dir, f"{prefix}_theme_table.png")
        plt.close(fig)
        return {"theme_table": path}

    fig, ax = plt.subplots(figsize=(12.0, max(3.5, 0.45 * len(df) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(MODEL_CONFIG["copula"]["header_color"])
            cell.set_text_props(weight="bold")
        else:
            cell.set_edgecolor("#d0d0d0")
    ax.set_title(title, fontsize=14, pad=12)
    path = _save(fig, output_dir, f"{prefix}_theme_table.png")
    plt.close(fig)
    return {"theme_table": path}


def visualization_report(
    result: DDEResult,
    *,
    output_dir: str | Path,
    model: Optional[str] = None,
    prefix: Optional[str] = None,
    cv_table=None,
    feature_names: Optional[Sequence[str]] = None,
    latent_frame=None,
    label_col: Optional[str] = None,
    alignment_weights=None,
    theme_table=None,
    average_by_label: bool = False,
    continuous_label_bins: int = 10,
) -> Dict[str, str]:
    """Generate the standard visualization bundle for SemiDDE or CopDDE."""
    model_key = _model_key(model, result)
    config = MODEL_CONFIG[model_key]
    prefix = prefix or config["prefix"]
    paths: Dict[str, str] = {}
    latent_df = _coerce_latent_frame(result, latent_frame)
    resolved_label_col = label_col
    if latent_df is not None:
        latent_df, resolved_label_col = _with_default_top_label(result, latent_df, label_col)
    include_item_cluster = None
    if model_key == "semi":
        include_item_cluster = alignment_weights is None
    paths.update(
        plot_model_summary(
            result,
            output_dir=output_dir,
            model=model_key,
            prefix=prefix,
            cv_table=cv_table,
            feature_names=feature_names,
            include_item_cluster=include_item_cluster,
        )
    )
    paths.update(
        plot_latent_profiles(
            result,
            output_dir=output_dir,
            model=model_key,
            prefix=prefix,
            latent_frame=latent_df if latent_df is not None else latent_frame,
            label_col=resolved_label_col,
            alignment_weights=alignment_weights,
            average_by_label=average_by_label,
            continuous_label_bins=continuous_label_bins,
        )
    )
    if resolved_label_col is not None and not average_by_label:
        paths.update(
            plot_latent_profiles(
                result,
                output_dir=output_dir,
                model=model_key,
                prefix=prefix,
                latent_frame=latent_df if latent_df is not None else latent_frame,
                label_col=resolved_label_col,
                alignment_weights=alignment_weights,
                average_by_label=True,
                continuous_label_bins=continuous_label_bins,
            )
        )
    if model_key == "copula" and theme_table is not None:
        paths.update(plot_theme_table(theme_table, output_dir=output_dir, prefix=prefix))
    return paths


__all__ = [
    "plot_latent_profiles",
    "plot_model_summary",
    "plot_theme_table",
    "visualization_report",
]
