"""Spectral initialization for Bernoulli-observed parametric DDEs."""

from __future__ import annotations

import warnings

import numpy as np

from dde.utils import logit, thres

from .normal_init_semi import (
    _permute_columns_hungarian,
    _rotate_varimax,
    _safe_lstsq,
    _safe_proj,
    _truncated_svd,
    _varimax,
)


def _stable_varimax(loadings):
    """Use the optional rotator when possible and a finite fallback otherwise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            rotated = _rotate_varimax(loadings)
        except (ValueError, np.linalg.LinAlgError):
            rotated = _varimax(loadings, q=1000)
    if not np.all(np.isfinite(rotated)):
        rotated = _varimax(loadings, q=1000)
    return rotated


def _bernoulli_working_response(children, K_parent, epsilon):
    """Return the MATLAB double-SVD inverse-link working response."""
    children = np.asarray(children, dtype=float)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly between 0 and 0.5.")
    N, n_children = children.shape
    K_parent = int(K_parent)
    if K_parent < 1 or K_parent > n_children:
        raise ValueError(
            "K_parent must lie between 1 and the number of child variables."
        )

    minimum_dimension = min(children.shape)
    if minimum_dimension < K_parent:
        raise ValueError(
            "Bernoulli spectral initialization requires "
            "min(n_observations, n_children) >= K_parent."
        )
    requested_rank = min(max(K_parent + 2, 2), minimum_dimension)
    threshold = 1.01 * np.sqrt(N)
    while True:
        U, singular_values, Vt = _truncated_svd(
            children, requested_rank
        )
        if (
            requested_rank >= minimum_dimension
            or singular_values[-1] <= threshold
        ):
            break
        requested_rank = min(2 * requested_rank, minimum_dimension)
    rank = max(
        K_parent + 1,
        int(np.sum(singular_values > threshold)),
    )
    rank = min(rank, singular_values.size)
    low_rank = (
        U[:, :rank] * singular_values[:rank][None, :]
    ) @ Vt[:rank, :]
    np.clip(low_rank, epsilon, 1.0 - epsilon, out=low_rank)
    working = logit(low_rank)
    working -= np.mean(working, axis=0, keepdims=True)
    return working


def _initialize_binary_parent_layer(
    children,
    K_parent,
    *,
    epsilon,
    threshold_constant,
    B_reference=None,
    G_reference=None,
):
    """Initialize one binary parent layer from binary child variables."""
    children = np.asarray(children, dtype=np.uint8)
    K_parent = int(K_parent)
    working = _bernoulli_working_response(children, K_parent, epsilon)
    n_children = children.shape[1]

    _, _, Vt = _truncated_svd(working, K_parent)
    if Vt.shape[0] < K_parent:
        raise ValueError(
            "Bernoulli spectral initialization could not recover K_parent directions."
        )
    loading_threshold = 1.0 / (
        float(threshold_constant) * np.sqrt(n_children)
    )
    if K_parent == 1:
        B_estimate = thres(Vt.T[:, :1], loading_threshold)
    else:
        B_estimate = thres(
            _stable_varimax(Vt.T[:, :K_parent]),
            loading_threshold,
        )

    signs = np.where(np.mean(B_estimate, axis=0) > 0.0, 1.0, -1.0)
    B_estimate *= signs
    oracle = B_reference is not None or G_reference is not None
    if oracle:
        if B_reference is None or G_reference is None:
            raise ValueError(
                "B_reference and G_reference must be supplied together."
            )
        B_reference = np.asarray(B_reference, dtype=float)
        G_reference = np.asarray(G_reference)
        B_estimate, _ = _permute_columns_hungarian(
            B_estimate, B_reference[:, 1 : K_parent + 1]
        )

    graph_estimate = B_estimate != 0.0
    parent_nonbinary = working @ _safe_lstsq(
        B_estimate.T @ B_estimate, B_estimate.T
    ).T
    parent = (parent_nonbinary > 0.0).astype(np.uint8)
    parent_centered = parent - np.mean(parent, axis=0, keepdims=True)
    B_reestimated = _safe_proj(parent_centered, working).T / 2.0

    if oracle:
        B_reestimated, _ = _permute_columns_hungarian(
            B_reestimated, B_reference[:, 1 : K_parent + 1]
        )
        diagonal_size = min(
            K_parent, B_reestimated.shape[0], B_reestimated.shape[1]
        )
        for index in range(diagonal_size):
            B_reestimated[index, index] = max(
                B_reestimated[index, index], 2.5
            )
    else:
        B_reestimated = thres(B_reestimated * graph_estimate, 0.0)

    intercept = (
        np.mean(working, axis=0).reshape(-1, 1)
        - B_reestimated @ np.mean(parent, axis=0).reshape(-1, 1)
    )
    B = np.hstack([intercept, B_reestimated])
    graph_error = None
    if oracle:
        graph_error = float(
            np.mean(
                graph_estimate[:, :K_parent]
                == G_reference[:, :K_parent]
            )
        )
    return B, parent, working, graph_error


def Bernoulli_init(
    X,
    K_list,
    *,
    epsilon=1e-4,
    mode="data",
    B_list=None,
    G_list=None,
):
    """Initialize a D-layer Bernoulli DDE using the MATLAB double-SVD scheme."""
    X_array = np.asarray(X)
    if X_array.ndim != 2 or min(X_array.shape) < 1:
        raise ValueError("X must be a nonempty two-dimensional binary matrix.")
    try:
        valid = np.all(np.isfinite(X_array)) and np.all(
            (X_array == 0) | (X_array == 1)
        )
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("X must contain only binary 0/1 values.")
    X = X_array.astype(np.uint8, copy=False)
    K_list = tuple(int(width) for width in K_list)
    D = len(K_list)
    if D < 1 or any(width < 1 for width in K_list):
        raise ValueError("K_list must contain positive layer widths.")
    mode = str(mode).strip().lower()
    if mode not in {"data", "oracle"}:
        raise ValueError("mode must be 'data' or 'oracle'.")
    oracle = mode == "oracle"
    if oracle:
        if B_list is None or G_list is None:
            raise ValueError(
                "B_list and G_list are required when mode='oracle'."
            )
        if len(B_list) != D or len(G_list) != D:
            raise ValueError("B_list and G_list lengths must match K_list.")

    B_initial = []
    A_initial = []
    graph_errors = []
    children = X
    for layer, K_parent in enumerate(K_list):
        threshold_constant = 2.5 if not oracle or layer == 0 else 3.5
        B_layer, parent, _, graph_error = _initialize_binary_parent_layer(
            children,
            K_parent,
            epsilon=epsilon,
            threshold_constant=threshold_constant,
            B_reference=None if not oracle else B_list[layer],
            G_reference=None if not oracle else G_list[layer],
        )
        B_initial.append(B_layer)
        A_initial.append(parent)
        graph_errors.append(graph_error)
        children = parent

    prop_initial = np.mean(A_initial[-1] > 0, axis=0)
    return prop_initial, B_initial, None, A_initial, graph_errors


bernoulli_init = Bernoulli_init


__all__ = [
    "Bernoulli_init",
    "_bernoulli_working_response",
    "_initialize_binary_parent_layer",
    "_stable_varimax",
    "bernoulli_init",
]
