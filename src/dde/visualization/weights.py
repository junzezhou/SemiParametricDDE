"""Primary visualization interfaces: weight-matrix heatmaps and anchor variables.

These are the two artifacts the package manuscript (Section 5.4) documents:

* :func:`plot_weights` draws the estimated layer-specific weight matrices
  ``B_list[d]`` of a fitted :class:`~dde.results.DDEResult`, from the observed
  layer upward, as heatmaps;
* :func:`get_anchors` ranks observed variables by the relative-loading score
  ``r_jk = max(min_{l != k}(b_jk - b_jl), 0)`` and returns the strongest
  anchors of every shallow-layer latent feature.

Both functions work for fixed-width and CSP fits.  CSP results store only the
active latent columns; their original column numbers (recorded in
``result.diagnostics["active_columns_original_index"]``) are used as labels,
and :func:`plot_weights` adds one blank column to the right of each matrix
that counts the columns the CSP shrank away.

Observed variables are labelled by the column names of the data frame passed
to :func:`dde.fit` or :func:`dde.select_hyperparameters` (recorded in
``result.diagnostics["variable_names"]``), otherwise ``y1, y2, ...``.
``variable_names`` replaces them: a sequence with one name per variable, or
a mapping that renames some of them (``{"BIGFIVE_1": "Name"}``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from dde.results import DDEResult


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dde-matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "dde.visualization requires matplotlib; install with "
            "`pip install dde[viz]`."
        ) from exc
    return plt


def _coefficient_matrices(result: DDEResult) -> list[np.ndarray]:
    if result.B_list is None or len(result.B_list) == 0:
        raise ValueError("result.B_list is empty; fit a model first.")
    return [np.asarray(B, dtype=float) for B in result.B_list]


def _active_index(result: DDEResult, layer: int, width: int) -> list[int]:
    """1-based original latent column numbers of layer ``layer`` (1-based)."""
    mapping = (result.diagnostics or {}).get("active_columns_original_index")
    if isinstance(mapping, Mapping):
        values = mapping.get(f"layer_{layer}")
        if values is not None and len(values) == width:
            return [int(v) for v in values]
    return list(range(1, width + 1))


def recorded_variable_names(result: DDEResult, n_rows: int) -> Optional[list[str]]:
    """Column names of the fitted data frame, when ``fit`` recorded them."""
    names = (result.diagnostics or {}).get("variable_names")
    if names is not None and len(names) == n_rows:
        return [str(v) for v in names]
    return None


def observed_variable_labels(
    result: DDEResult,
    n_rows: int,
    variable_names=None,
    *,
    default_prefix: str = "y",
    argument: str = "variable_names",
) -> list[str]:
    """Labels of the observed variables (rows of ``B(1)``).

    The default labels are the recorded column names of the data, else
    ``{default_prefix}1, {default_prefix}2, ...``.  ``variable_names`` is
    either one name per variable or a mapping from default labels to new
    names; variables missing from the mapping keep their default label.
    """
    defaults = recorded_variable_names(result, n_rows) or [
        f"{default_prefix}{j + 1}" for j in range(n_rows)
    ]
    if variable_names is None:
        return defaults
    if isinstance(variable_names, Mapping):
        renamed = {str(key): str(value) for key, value in variable_names.items()}
        unknown = [key for key in renamed if key not in set(defaults)]
        if unknown:
            shown = ", ".join(repr(key) for key in unknown[:5])
            raise ValueError(
                f"{argument} renames unknown variable(s) {shown}; the default labels "
                f"are {defaults[0]!r}, {defaults[min(1, n_rows - 1)]!r}, ..."
            )
        return [renamed.get(name, name) for name in defaults]
    names = [str(v) for v in variable_names]
    if len(names) != n_rows:
        raise ValueError(f"{argument} has {len(names)} entries; layer 1 has {n_rows} rows.")
    return names


def shrunk_column_count(result: DDEResult, layer: int, width: int) -> int:
    """Latent columns of ``layer`` (1-based) that the CSP shrank away."""
    original = (result.diagnostics or {}).get("original_K_list")
    if original is None or len(original) < layer:
        return 0
    return max(int(original[layer - 1]) - int(width), 0)


def draw_shrunk_column(ax, n_rows: int, n_cols: int, n_shrunk: int) -> None:
    """Shade the blank column at ``x = n_cols`` that stands for the shrunk columns."""
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (n_cols - 0.5, -0.5), 1.0, n_rows, facecolor="#f0f0f0", edgecolor="#b0b0b0",
            hatch="///", linewidth=0.6, zorder=2,
        )
    )
    ax.axvline(n_cols - 0.5, color="#404040", linewidth=1.0, linestyle="--", zorder=3)
    ax.text(
        n_cols, (n_rows - 1) / 2, f"{n_shrunk} column{'s' if n_shrunk != 1 else ''} shrunk",
        rotation=90, ha="center", va="center", fontsize=8, color="#404040", zorder=4,
        bbox={"facecolor": "#f0f0f0", "edgecolor": "none", "pad": 1.5},
    )
    ax.set_xlim(-0.5, n_cols + 0.5)


def _row_labels(result, B_list, layer, variable_names):
    n_rows = B_list[layer - 1].shape[0]
    if layer == 1:
        return observed_variable_labels(result, n_rows, variable_names)
    return [f"A{layer - 1}_{k}" for k in _active_index(result, layer - 1, n_rows)]


def _column_labels(result, B_list, layer):
    width = B_list[layer - 1].shape[1] - 1
    return [f"A{layer}_{k}" for k in _active_index(result, layer, width)]


def plot_weights(
    result: DDEResult,
    layer: Optional[int] = None,
    *,
    variable_names: Optional[Sequence[str]] = None,
    include_intercept: bool = False,
    output_dir: Optional[str | Path] = None,
    prefix: str = "weights",
    cmap: str = "RdBu_r",
    max_rows: Optional[int] = None,
    annotate: Optional[bool] = None,
    show_shrunk: bool = True,
):
    """Plot the estimated weight matrices of a fitted DDE as heatmaps.

    Parameters
    ----------
    result : fitted ``DDEResult`` (any model, fixed or CSP widths).
    layer : 1-based layer to plot; ``None`` plots every layer from the
        observed layer upward and returns one figure per layer.
    variable_names : labels for the observed variables (rows of layer 1):
        one name per variable, or a mapping that renames some of the default
        labels (the recorded column names of the data, else ``y1, y2, ...``),
        e.g. ``{"BIGFIVE_1": "Name"}``.
    include_intercept : also show the intercept column.
    output_dir : when given, each figure is saved as
        ``{prefix}_layer{d}.png`` in that directory.
    max_rows : keep only the ``max_rows`` rows with the largest loading norm
        (useful for very wide observed layers).
    annotate : write the values into the cells (default when the matrix has
        at most 400 cells).
    show_shrunk : for a CSP fit, add one blank column to the right of each
        matrix, labelled with the number of latent columns the CSP shrank
        away (``diagnostics["original_K_list"]`` minus the active width).

    Returns
    -------
    A matplotlib ``Figure`` when ``layer`` is given, otherwise a dict
    ``{layer: Figure}``.
    """
    plt = _matplotlib()
    B_list = _coefficient_matrices(result)
    layers = [int(layer)] if layer is not None else list(range(1, len(B_list) + 1))
    if any(d < 1 or d > len(B_list) for d in layers):
        raise ValueError(f"layer must lie in 1..{len(B_list)}.")
    figures = {}
    for d in layers:
        B = B_list[d - 1]
        matrix = B if include_intercept else B[:, 1:]
        row_labels = _row_labels(result, B_list, d, variable_names)
        col_labels = (["intercept"] if include_intercept else []) + _column_labels(result, B_list, d)
        if max_rows is not None and matrix.shape[0] > int(max_rows):
            order = np.argsort(-np.linalg.norm(B[:, 1:], axis=1))[: int(max_rows)]
            matrix = matrix[order]
            row_labels = [row_labels[i] for i in order]
        n_rows, n_cols = matrix.shape
        n_shrunk = shrunk_column_count(result, d, B.shape[1] - 1) if show_shrunk else 0
        shown_cols = n_cols + (1 if n_shrunk else 0)
        fig_w = max(4.0, min(14.0, 0.45 * shown_cols + 2.5))
        # every variable keeps its label up to 150 rows (smaller type above 60)
        fig_h = max(3.0, min(24.0, (0.22 if n_rows <= 60 else 0.16) * n_rows + 1.8))
        label_size = 8 if n_rows <= 60 else (7 if n_rows <= 110 else 6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vmax = max(float(np.max(np.abs(matrix))) if matrix.size else 1.0, 1e-8)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        tick_labels = list(col_labels)
        if n_shrunk:
            draw_shrunk_column(ax, n_rows, n_cols, n_shrunk)
            tick_labels.append(f"+{n_shrunk} shrunk")
        ax.set_xticks(range(shown_cols))
        ax.set_xticklabels(tick_labels, rotation=90 if shown_cols > 8 else 0, fontsize=8)
        step = max(1, int(np.ceil(n_rows / 150)))
        ax.set_yticks(range(0, n_rows, step))
        ax.set_yticklabels([row_labels[i] for i in range(0, n_rows, step)], fontsize=label_size)
        ax.set_xlabel(f"layer {d} latent feature")
        ax.set_ylabel("observed variable" if d == 1 else f"layer {d - 1} latent feature")
        title = f"Estimated weight matrix B({d}): {n_rows} x {n_cols}"
        if n_shrunk:
            title += f" ({n_cols + n_shrunk} before CSP shrinkage)"
        ax.set_title(title)
        if annotate is None:
            annotate = matrix.size <= 400
        if annotate:
            for i in range(n_rows):
                for j in range(n_cols):
                    value = matrix[i, j]
                    if value != 0.0:
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(value) > 0.6 * vmax else "black")
        fig.colorbar(image, ax=ax, shrink=0.8, label="weight")
        fig.tight_layout()
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{prefix}_layer{d}.png", dpi=160, bbox_inches="tight")
        figures[d] = fig
    return figures[layers[0]] if layer is not None else figures


def anchor_scores(B) -> np.ndarray:
    """Relative-loading scores ``r_jk = max(min_{l != k}(b_jk - b_jl), 0)``.

    ``B`` is a coefficient matrix with the intercept in column 0 (as stored
    in ``DDEResult.B_list``); the scores are computed on the slopes.  With a
    single latent feature the score is ``max(b_j1, 0)``.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[1] < 2:
        raise ValueError("B must be a coefficient matrix with an intercept column.")
    slopes = B[:, 1:]
    J, K = slopes.shape
    scores = np.zeros((J, K), dtype=float)
    for k in range(K):
        if K == 1:
            scores[:, k] = np.maximum(slopes[:, 0], 0.0)
            continue
        others = np.delete(slopes, k, axis=1)
        # min over l != k of (b_jk - b_jl), floored at zero
        scores[:, k] = np.maximum((slopes[:, k][:, None] - others).min(axis=1), 0.0)
    return scores


def get_anchors(
    result: DDEResult,
    n_anchors: int = 3,
    *,
    variable_names: Optional[Sequence[str]] = None,
    layer: int = 1,
    min_score: float = 0.0,
):
    """Return the strongest anchor variables of each latent feature.

    The score of observed variable ``j`` for feature ``k`` of ``layer`` is the
    relative loading ``r_jk = max(min_{l != k}(b_jk - b_jl), 0)``; anchors are
    the ``n_anchors`` variables with the largest positive scores.  The result
    is a ``pandas.DataFrame`` with columns ``latent_feature`` (original column
    number, 1-based), ``variable`` (name: ``variable_names`` as in
    :func:`plot_weights`, else the recorded column name or ``y{j}``),
    ``variable_index`` (1-based), ``rank``, ``score`` and ``loading``.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("get_anchors requires pandas; install with `pip install dde[viz]`.") from exc
    B_list = _coefficient_matrices(result)
    layer = int(layer)
    if layer < 1 or layer > len(B_list):
        raise ValueError(f"layer must lie in 1..{len(B_list)}.")
    n_anchors = int(n_anchors)
    if n_anchors < 1:
        raise ValueError("n_anchors must be a positive integer.")
    B = B_list[layer - 1]
    slopes = B[:, 1:]
    scores = anchor_scores(B)
    names = _row_labels(result, B_list, layer, variable_names)
    features = _active_index(result, layer, slopes.shape[1])
    rows = []
    for k, feature in enumerate(features):
        order = np.argsort(-scores[:, k], kind="stable")
        rank = 0
        for j in order:
            if scores[j, k] <= float(min_score):
                break
            rank += 1
            rows.append({
                "latent_feature": int(feature),
                "variable": names[j],
                "variable_index": int(j + 1),
                "rank": rank,
                "score": float(scores[j, k]),
                "loading": float(slopes[j, k]),
            })
            if rank >= n_anchors:
                break
    return pd.DataFrame(rows, columns=["latent_feature", "variable", "variable_index", "rank", "score", "loading"])


__all__ = ["anchor_scores", "get_anchors", "plot_weights"]
