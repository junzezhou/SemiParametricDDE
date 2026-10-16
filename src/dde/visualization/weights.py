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
``result.diagnostics["active_columns_original_index"]``) are used as labels.
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


def _row_labels(result, B_list, layer, variable_names):
    n_rows = B_list[layer - 1].shape[0]
    if layer == 1:
        if variable_names is not None:
            names = [str(v) for v in variable_names]
            if len(names) != n_rows:
                raise ValueError(
                    f"variable_names has {len(names)} entries; layer 1 has {n_rows} rows."
                )
            return names
        return [f"y{j + 1}" for j in range(n_rows)]
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
):
    """Plot the estimated weight matrices of a fitted DDE as heatmaps.

    Parameters
    ----------
    result : fitted ``DDEResult`` (any model, fixed or CSP widths).
    layer : 1-based layer to plot; ``None`` plots every layer from the
        observed layer upward and returns one figure per layer.
    variable_names : labels for the observed variables (rows of layer 1).
    include_intercept : also show the intercept column.
    output_dir : when given, each figure is saved as
        ``{prefix}_layer{d}.png`` in that directory.
    max_rows : keep only the ``max_rows`` rows with the largest loading norm
        (useful for very wide observed layers).
    annotate : write the values into the cells (default when the matrix has
        at most 400 cells).

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
        fig_w = max(4.0, min(14.0, 0.45 * n_cols + 2.5))
        fig_h = max(3.0, min(14.0, 0.28 * n_rows + 1.8))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vmax = max(float(np.max(np.abs(matrix))) if matrix.size else 1.0, 1e-8)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=90 if n_cols > 8 else 0, fontsize=8)
        step = max(1, int(np.ceil(n_rows / 40)))
        ax.set_yticks(range(0, n_rows, step))
        ax.set_yticklabels([row_labels[i] for i in range(0, n_rows, step)], fontsize=8)
        ax.set_xlabel(f"layer {d} latent feature")
        ax.set_ylabel("observed variable" if d == 1 else f"layer {d - 1} latent feature")
        ax.set_title(f"Estimated weight matrix B({d}): {n_rows} x {n_cols}")
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
    number, 1-based), ``variable`` (name or 1-based index),
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
