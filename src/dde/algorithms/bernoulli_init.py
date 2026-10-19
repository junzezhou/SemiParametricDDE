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


def _bernoulli_working_response(children, K_parent, epsilon, *, rng=None):
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
            children, requested_rank, rng=rng
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


def _two_class_latent_class_model(Y, *, max_iter=200, tol=1e-7):
    """Two-class latent class model with conditionally independent Bernoulli
    items.  Returns the responsibility of the high-prevalence-of-ones class
    and that class's prevalence."""
    Y = np.asarray(Y, dtype=float)
    m = Y.shape[1]
    lo = np.full(m, 0.15)
    hi = np.full(m, 0.85)
    pi = 0.3
    resp = np.full(Y.shape[0], pi)
    for _ in range(max_iter):
        log1 = np.log(pi) + (Y * np.log(hi) + (1.0 - Y) * np.log1p(-hi)).sum(axis=1)
        log0 = np.log1p(-pi) + (Y * np.log(lo) + (1.0 - Y) * np.log1p(-lo)).sum(axis=1)
        resp = 1.0 / (1.0 + np.exp(np.clip(log0 - log1, -50.0, 50.0)))
        pi_new = float(np.clip(resp.mean(), 1e-3, 1.0 - 1e-3))
        hi_new = np.clip((resp[:, None] * Y).sum(axis=0) / max(resp.sum(), 1e-12), 1e-3, 1.0 - 1e-3)
        lo_new = np.clip(((1.0 - resp)[:, None] * Y).sum(axis=0) / max((1.0 - resp).sum(), 1e-12), 1e-3, 1.0 - 1e-3)
        converged = abs(pi_new - pi) + np.abs(hi_new - hi).sum() + np.abs(lo_new - lo).sum() < tol
        pi, hi, lo = pi_new, hi_new, lo_new
        if converged:
            break
    if hi.mean() < lo.mean():
        resp, pi = 1.0 - resp, 1.0 - pi
    return resp, pi


def _anchor_lca_binarize(children, B_estimate):
    """Classify every latent column from its positive-loading anchor children.

    Binary children carry no single-variable mixture information, so each
    column is read off a two-class latent class model over its anchors
    (identified with three or more anchors); with fewer than two anchors the
    two largest loadings are used.  Returns the binary parent matrix and the
    per-column prevalence estimates.
    """
    children = np.asarray(children, dtype=float)
    B_estimate = np.asarray(B_estimate, dtype=float)
    N = children.shape[0]
    K = B_estimate.shape[1]
    parent = np.zeros((N, K), dtype=np.uint8)
    prevalence = np.zeros(K, dtype=float)
    for k in range(K):
        anchors = np.flatnonzero(B_estimate[:, k] > 0.0)
        if anchors.size < 2:
            anchors = np.argsort(-B_estimate[:, k])[:2]
        resp, pi = _two_class_latent_class_model(children[:, anchors])
        parent[:, k] = resp > 0.5
        prevalence[k] = pi
    return parent, prevalence


def _initialize_binary_parent_layer(
    children,
    K_parent,
    *,
    epsilon,
    threshold_constant,
    B_reference=None,
    G_reference=None,
    rng=None,
    binarize="projection",
):
    """Initialize one binary parent layer from binary child variables.

    ``binarize="projection"`` thresholds the projected centered working
    response at zero (MATLAB ``binary_init``); ``"anchor_lca"`` classifies
    each column with a two-class latent class model over its anchor children,
    which recovers the column prevalence.
    """
    binarize = str(binarize).strip().lower()
    if binarize not in {"projection", "anchor_lca"}:
        raise ValueError("binarize must be 'projection' or 'anchor_lca'.")
    children = np.asarray(children, dtype=np.uint8)
    K_parent = int(K_parent)
    working = _bernoulli_working_response(
        children, K_parent, epsilon, rng=rng
    )
    n_children = children.shape[1]

    _, _, Vt = _truncated_svd(working, K_parent, rng=rng)
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
    if binarize == "anchor_lca":
        parent, _ = _anchor_lca_binarize(children, B_estimate)
    else:
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
    random_state=None,
    binarize="projection",
):
    """Initialize a D-layer Bernoulli DDE using the MATLAB double-SVD scheme.

    ``binarize`` selects the latent-state rule of every layer (see
    :func:`_initialize_binary_parent_layer`).
    """
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
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    elif random_state is None:
        rng = None
    else:
        rng = np.random.default_rng(random_state)

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
            rng=rng,
            binarize=binarize,
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
    "_anchor_lca_binarize",
    "_bernoulli_working_response",
    "_initialize_binary_parent_layer",
    "_stable_varimax",
    "bernoulli_init",
]
