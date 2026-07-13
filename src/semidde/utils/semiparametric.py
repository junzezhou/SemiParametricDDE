"""SemiDDE transformation-estimation helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def g_est_rversion(y, z, n, p, K, pro0, delta0):
    """Estimate monotone transformations for the SemiDDE outer loop.

    This is a cleaned package version of the existing ``g_est_rversion`` helper.
    ``z`` contains class labels in ``0, ..., K-1`` and ``pro0`` contains the
    corresponding class probabilities.

    The original translation materialized a ``(K, n, p)`` probability cube.
    This implementation computes the same weighted sums by sweeping through
    each column's ranks once. Unobserved classes contribute exactly the
    original clipped value ``norm.ppf(delta0)``.
    """
    y = np.asarray(y, dtype=float)
    z_idx = np.asarray(z, dtype=int).reshape(-1)
    pro0 = np.asarray(pro0, dtype=float).reshape(-1)

    if y.shape != (n, p):
        raise ValueError(f"y must have shape ({n}, {p}); got {y.shape}.")
    if z_idx.shape[0] != n:
        raise ValueError(f"z must have length {n}; got {z_idx.shape[0]}.")
    if pro0.shape[0] != K:
        raise ValueError(f"pro0 must have length {K}; got {pro0.shape[0]}.")

    one_minus_delta0 = 1.0 - delta0
    g_est = np.zeros((n, p))
    q_delta = norm.ppf(delta0)

    valid = (z_idx >= 0) & (z_idx < K)
    if np.any((z_idx >= K) & (z_idx >= 0)):
        raise ValueError("z contains class labels outside 0, ..., K-1.")

    # MARK: Original package implementation looped over observed classes and
    # searched all ranks for each class, giving O(p * n * n_observed_classes)
    # runtime when K_1 has many observed patterns.
    # observed_classes, inverse, counts = np.unique(
    #     z_idx[valid],
    #     return_inverse=True,
    #     return_counts=True,
    # )
    #
    # observed_mass = float(np.sum(pro0[observed_classes])) if observed_classes.size else 0.0
    # base_value = float(np.sum(pro0) - observed_mass) * q_delta
    #
    # class_positions = []
    # if observed_classes.size:
    #     for class_index in range(observed_classes.size):
    #         valid_positions = np.flatnonzero(valid)
    #         class_positions.append(valid_positions[inverse == class_index])
    #
    # for j in range(p):
    #     order_j = np.argsort(y[:, j], kind="mergesort")
    #     ranks_j = np.empty(n, dtype=np.int64)
    #     ranks_j[order_j] = np.arange(n, dtype=np.int64)
    #
    #     g_col = np.full(n, base_value, dtype=float)
    #
    #     for class_label, count, positions in zip(observed_classes, counts, class_positions):
    #         sorted_ranks = np.sort(ranks_j[positions])
    #         cumulative_counts = np.searchsorted(sorted_ranks, ranks_j, side="right")
    #         probs = np.clip(cumulative_counts / float(count), delta0, one_minus_delta0)
    #         g_col += pro0[class_label] * norm.ppf(probs)
    #
    #     g_col = (g_col - np.mean(g_col)) / (np.std(g_col) + 1e-4)
    #     g_est[:, j] = g_col
    #
    # return g_est
    class_counts = np.bincount(z_idx[valid], minlength=K).astype(float)
    initial_total = float(np.sum(pro0) * q_delta)

    for j in range(p):
        order_j = np.argsort(y[:, j], kind="mergesort")
        z_ordered = z_idx[order_j]
        valid_ordered = (z_ordered >= 0) & (z_ordered < K)
        delta_ordered = np.zeros(n, dtype=float)

        valid_positions = np.flatnonzero(valid_ordered)
        labels = z_ordered[valid_positions]
        if labels.size:
            label_order = np.argsort(labels, kind="mergesort")
            labels_sorted = labels[label_order]

            group_starts = np.empty(labels_sorted.size, dtype=bool)
            group_starts[0] = True
            group_starts[1:] = labels_sorted[1:] != labels_sorted[:-1]
            first_in_group = np.maximum.accumulate(
                np.where(group_starts, np.arange(labels_sorted.size), 0)
            )
            occurrences_sorted = np.arange(labels_sorted.size) - first_in_group + 1
            occurrences = np.empty_like(occurrences_sorted)
            occurrences[label_order] = occurrences_sorted

            denominators = class_counts[labels]
            prob_new = np.clip(occurrences / denominators, delta0, one_minus_delta0)
            prob_old = np.clip((occurrences - 1) / denominators, delta0, one_minus_delta0)
            delta_ordered[valid_positions] = pro0[labels] * (
                norm.ppf(prob_new) - norm.ppf(prob_old)
            )

        g_col_ordered = initial_total + np.cumsum(delta_ordered)
        g_col = np.empty(n, dtype=float)
        g_col[order_j] = g_col_ordered

        g_col = (g_col - np.mean(g_col)) / (np.std(g_col) + 1e-4)
        g_est[:, j] = g_col

    return g_est
