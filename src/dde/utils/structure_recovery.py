"""Per-layer graph-structure recovery metrics for fitted DDE hierarchies.

A fitted layer may be wider or narrower than the generating layer (CSP
selects the widths), so the coefficient supports cannot be compared entry by
entry.  Each latent layer is aligned to the truth on its own:

* ``"latent_state"``: Hungarian assignment maximizing the agreement of fitted
  and true latent columns, ``max(agree, 1 - agree)`` so that a flipped column
  still matches (the support of ``B`` does not change under a flip);
* ``"coefficient"`` (used when latent states are not available): Hungarian
  assignment maximizing ``|cosine|`` between fitted and true slope columns of
  ``B_d`` after its rows have been aligned by layer ``d - 1``.

The smaller side is padded with empty columns, so layer ``d`` is scored on
``max(K_true, K_fit)`` columns.  The alignment of layer ``d`` also permutes
(and pads) the rows of ``B_{d+1}``.  An entry is in the support when it is
nonzero and ``|B| >= threshold``; intercepts are excluded.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _support(B, threshold: float) -> np.ndarray:
    slopes = np.asarray(B, dtype=float)[:, 1:]
    return (slopes != 0.0) & (np.abs(slopes) >= threshold)


def _abs_correlation(fitted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """|corr| between true (rows) and fitted (columns); 0 for constant columns."""
    fitted = fitted - fitted.mean(axis=0, keepdims=True)
    truth = truth - truth.mean(axis=0, keepdims=True)
    fitted_norm = np.linalg.norm(fitted, axis=0)
    truth_norm = np.linalg.norm(truth, axis=0)
    denominator = np.outer(truth_norm, fitted_norm)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.abs(truth.T @ fitted) / denominator
    return np.where(denominator > 0, value, 0.0)


def _latent_scores(A_fit, A_true):
    """Return (assignment score, match score), both true-by-fitted."""
    A_fit = np.asarray(A_fit, dtype=float)
    A_true = np.asarray(A_true, dtype=float)
    if A_fit.ndim != 2 or A_true.ndim != 2 or A_fit.shape[0] != A_true.shape[0]:
        raise ValueError("fitted and true latent layers must share their rows.")
    agreement = (A_true.T @ A_fit + (1.0 - A_true).T @ (1.0 - A_fit)) / A_true.shape[0]
    return np.maximum(agreement, 1.0 - agreement), _abs_correlation(A_fit, A_true)


def _cosine_scores(fitted_slopes: np.ndarray, true_slopes: np.ndarray):
    fitted_norm = np.linalg.norm(fitted_slopes, axis=0)
    true_norm = np.linalg.norm(true_slopes, axis=0)
    denominator = np.outer(true_norm, fitted_norm)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.abs(true_slopes.T @ fitted_slopes) / denominator
    value = np.where(denominator > 0, value, 0.0)
    return value, value


def _place_rows(B, row_order, n_rows):
    """Put row ``i`` of ``B`` at aligned position ``row_order[i]``."""
    out = np.zeros((n_rows, B.shape[1]), dtype=B.dtype)
    out[row_order] = B
    return out


def _union_order(pairs, n_true, n_fit):
    """Aligned position of every fitted column: its true partner, else padding."""
    order = np.empty(n_fit, dtype=int)
    paired = {fit: true for true, fit in pairs}
    next_position = n_true
    for fit in range(n_fit):
        if fit in paired:
            order[fit] = paired[fit]
        else:
            order[fit] = next_position
            next_position += 1
    return order, max(n_true, n_fit)


def layer_structure_recovery_metrics(
    B_list: Sequence[np.ndarray],
    B_true_list: Sequence[np.ndarray],
    A_list: Sequence[np.ndarray] | None = None,
    A_true_list: Sequence[np.ndarray] | None = None,
    *,
    threshold: float = 0.0,
    match_threshold: float = 0.5,
    alignment: str = "auto",
) -> dict:
    """Aligned per-layer support recovery of a fitted DDE hierarchy.

    Parameters
    ----------
    B_list, B_true_list : fitted and generating coefficient matrices, each
        with the intercept in column 0; ``B_1`` has ``J`` rows and ``B_d``
        (``d >= 2``) has one row per column of layer ``d - 1``.
    A_list, A_true_list : fitted and true latent states, ``N``-by-``K_d``;
        required for ``alignment="latent_state"``.
    threshold : support threshold on ``|B|`` (``0`` keeps every nonzero).
    match_threshold : an aligned pair counts as a matched true column when
        its ``|corr|`` (latent states) or ``|cosine|`` (coefficients) reaches
        this value.
    alignment : ``"auto"`` (latent states when both are given), or
        ``"latent_state"`` / ``"coefficient"``.

    Returns
    -------
    dict with ``alignment`` and per-layer lists ``layer_graph_accuracy``,
    ``layer_sensitivity``, ``layer_specificity``, ``layer_matched_columns``,
    ``layer_graph_counts`` (``[tp, fp, fn, tn]``), ``layer_column_pairs``
    (``[true, fitted]`` index pairs of the assignment) and
    ``layer_pair_similarity`` (alignment similarity of each pair).
    """
    B_list = [np.asarray(B, dtype=float) for B in B_list]
    B_true_list = [np.asarray(B, dtype=float) for B in B_true_list]
    D = len(B_true_list)
    if len(B_list) != D:
        raise ValueError("B_list and B_true_list must have the same depth.")
    if B_list[0].shape[0] != B_true_list[0].shape[0]:
        raise ValueError("fitted and true B_1 must have the same observed rows.")
    for d in range(1, D):
        if B_true_list[d].shape[0] != B_true_list[d - 1].shape[1] - 1:
            raise ValueError(f"true B_{d + 1} rows do not match the width of layer {d}.")
    have_states = A_list is not None and A_true_list is not None
    if alignment == "auto":
        alignment = "latent_state" if have_states else "coefficient"
    if alignment not in {"latent_state", "coefficient"}:
        raise ValueError("alignment must be 'auto', 'latent_state' or 'coefficient'.")
    if alignment == "latent_state":
        if not have_states or len(A_list) != D or len(A_true_list) != D:
            raise ValueError("latent_state alignment needs D fitted and true latent layers.")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative.")

    out = {
        "alignment": alignment,
        "layer_graph_accuracy": [],
        "layer_sensitivity": [],
        "layer_specificity": [],
        "layer_matched_columns": [],
        "layer_graph_counts": [],
        "layer_column_pairs": [],
        "layer_pair_similarity": [],
    }
    # Aligned row positions of B_d: the observed rows are already aligned.
    row_order = np.arange(B_list[0].shape[0])
    n_rows = B_list[0].shape[0]
    for d in range(D):
        B_fit, B_true = B_list[d], B_true_list[d]
        n_true, n_fit = B_true.shape[1] - 1, B_fit.shape[1] - 1
        if B_fit.shape[0] != row_order.size:
            raise ValueError(f"B_{d + 1} rows do not match the width of layer {d}.")
        if alignment == "latent_state":
            A_fit, A_true = A_list[d], A_true_list[d]
            if np.shape(A_fit)[1] != n_fit or np.shape(A_true)[1] != n_true:
                raise ValueError(f"A_{d + 1} widths do not match B_{d + 1}.")
            assign_score, match_score = _latent_scores(A_fit, A_true)
        else:
            fitted_slopes = _place_rows(B_fit[:, 1:], row_order, n_rows)
            true_slopes = np.zeros((n_rows, n_true))
            true_slopes[: B_true.shape[0]] = B_true[:, 1:]
            assign_score, match_score = _cosine_scores(fitted_slopes, true_slopes)

        if n_true and n_fit:
            true_index, fit_index = linear_sum_assignment(-assign_score)
        else:
            true_index = fit_index = np.zeros(0, dtype=int)
        pairs = [[int(t), int(f)] for t, f in zip(true_index, fit_index)]
        column_order, n_columns = _union_order(pairs, n_true, n_fit)

        estimated = np.zeros((n_rows, n_columns), dtype=bool)
        estimated[np.ix_(row_order, column_order)] = _support(B_fit, threshold)
        truth = np.zeros_like(estimated)
        truth[: B_true.shape[0], :n_true] = _support(B_true, 0.0)

        tp = int(np.sum(estimated & truth))
        fp = int(np.sum(estimated & ~truth))
        fn = int(np.sum(~estimated & truth))
        tn = int(np.sum(~estimated & ~truth))
        out["layer_graph_accuracy"].append(float((tp + tn) / estimated.size) if estimated.size else float("nan"))
        out["layer_sensitivity"].append(float(tp / (tp + fn)) if tp + fn else float("nan"))
        out["layer_specificity"].append(float(tn / (tn + fp)) if tn + fp else float("nan"))
        out["layer_matched_columns"].append(
            int(np.sum(match_score[true_index, fit_index] >= match_threshold))
        )
        out["layer_graph_counts"].append([tp, fp, fn, tn])
        out["layer_column_pairs"].append(pairs)
        out["layer_pair_similarity"].append(
            [float(value) for value in assign_score[true_index, fit_index]]
        )
        # Layer d's alignment orders (and pads) the rows of B_{d+1}.
        row_order, n_rows = column_order, n_columns
    return out


__all__ = ["layer_structure_recovery_metrics"]
