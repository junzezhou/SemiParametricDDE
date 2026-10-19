"""Spectral initialization for Poisson-observed parametric DDEs."""

from __future__ import annotations

import numpy as np

from dde.utils import thres

from .bernoulli_init import (
    _initialize_binary_parent_layer,
    _stable_varimax,
)
from .normal_init_semi import (
    _permute_columns_hungarian,
    _safe_lstsq,
    _safe_proj,
    _truncated_svd,
)


def _as_rng(random_state=None):
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return None
    return np.random.default_rng(random_state)


def _poisson_working_response(counts, K_parent, epsilon, *, rng=None):
    """Return the MATLAB double-SVD log-link working response.

    The count matrix is first denoised by a data-adaptive low-rank SVD,
    floored away from zero, transformed by the inverse Poisson mean link, and
    finally centered columnwise.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or min(counts.shape) < 1:
        raise ValueError("counts must be a nonempty two-dimensional matrix.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("counts must contain finite nonnegative values.")
    if not np.all(counts == np.floor(counts)):
        raise ValueError("counts must contain integer-valued observations.")

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    N, n_children = counts.shape
    K_parent = int(K_parent)
    if K_parent < 1 or K_parent > n_children:
        raise ValueError(
            "K_parent must lie between 1 and the number of observed variables."
        )
    minimum_dimension = min(counts.shape)
    if minimum_dimension < K_parent:
        raise ValueError(
            "Poisson spectral initialization requires "
            "min(n_observations, n_variables) >= K_parent."
        )

    requested_rank = min(max(K_parent + 2, 2), minimum_dimension)
    singular_threshold = 1.01 * np.sqrt(N)
    while True:
        U, singular_values, Vt = _truncated_svd(
            counts, requested_rank, rng=rng
        )
        if (
            requested_rank >= minimum_dimension
            or singular_values[-1] <= singular_threshold
        ):
            break
        requested_rank = min(2 * requested_rank, minimum_dimension)

    rank = max(
        K_parent + 1,
        int(np.sum(singular_values > singular_threshold)),
    )
    rank = min(rank, singular_values.size)
    low_rank = (
        U[:, :rank] * singular_values[:rank][None, :]
    ) @ Vt[:rank, :]
    np.maximum(low_rank, epsilon, out=low_rank)
    np.log(low_rank, out=low_rank)
    low_rank -= np.mean(low_rank, axis=0, keepdims=True)
    return low_rank


def _two_component_poisson_mixture(x, *, max_iter=200, tol=1e-7):
    """Fit a two-component Poisson mixture to one count column by EM.

    Returns ``(pi, lam0, lam1)`` with ``lam0 <= lam1``: the prevalence of the
    high-rate component and the two rates.
    """
    x = np.asarray(x, dtype=float)
    lam0 = max(float(np.quantile(x, 0.3)), 0.05)
    lam1 = max(float(np.quantile(x, 0.9)), 3.0 * lam0 + 0.5)
    pi = 0.3
    for _ in range(max_iter):
        log1 = np.log(pi) + x * np.log(lam1) - lam1
        log0 = np.log1p(-pi) + x * np.log(lam0) - lam0
        resp = 1.0 / (1.0 + np.exp(np.clip(log0 - log1, -50.0, 50.0)))
        pi_new = float(np.clip(resp.mean(), 1e-3, 1.0 - 1e-3))
        lam1_new = max(float((resp * x).sum() / max(resp.sum(), 1e-12)), 1e-3)
        lam0_new = max(
            float(((1.0 - resp) * x).sum() / max((1.0 - resp).sum(), 1e-12)), 1e-3
        )
        converged = (
            abs(pi_new - pi) + abs(lam1_new - lam1) + abs(lam0_new - lam0) < tol
        )
        pi, lam0, lam1 = pi_new, lam0_new, lam1_new
        if converged:
            break
    if lam1 < lam0:
        lam0, lam1, pi = lam1, lam0, 1.0 - pi
    return pi, lam0, lam1


def _anchor_mixture_binarize(counts, B_estimate):
    """Classify every latent column from its positive-loading anchor variables.

    The projection step of the double-SVD initializer thresholds centered
    scores at zero, which assigns roughly half of the observations to every
    latent column whatever its true prevalence.  For a rare latent variable
    that mislabels most observations and starts the observed-layer intercepts
    far from the truth.  Here each anchor (a child with a positive spectral
    loading) is summarized by a two-component Poisson mixture; the
    log-likelihood ratios of the anchors are summed with the mixture
    prevalence as prior, and the sign of the total classifies the
    observation.  Returns the binary parent matrix and the per-column
    prevalence estimates.
    """
    counts = np.asarray(counts, dtype=float)
    B_estimate = np.asarray(B_estimate, dtype=float)
    N = counts.shape[0]
    K = B_estimate.shape[1]
    parent = np.zeros((N, K), dtype=np.uint8)
    prevalence = np.zeros(K, dtype=float)
    for k in range(K):
        anchors = np.flatnonzero(B_estimate[:, k] > 0.0)
        if anchors.size == 0:
            anchors = np.array([int(np.argmax(np.abs(B_estimate[:, k])))])
        logit = np.zeros(N, dtype=float)
        pis = []
        for j in anchors:
            x = counts[:, j]
            pi, lam0, lam1 = _two_component_poisson_mixture(x)
            logit += x * np.log(lam1 / lam0) - (lam1 - lam0)
            pis.append(pi)
        pi = float(np.clip(np.mean(pis), 1e-3, 1.0 - 1e-3))
        logit += np.log(pi / (1.0 - pi))
        parent[:, k] = logit > 0.0
        prevalence[k] = pi
    return parent, prevalence


def _initialize_poisson_parent_layer(
    counts,
    K_parent,
    *,
    epsilon,
    B_reference=None,
    G_reference=None,
    rng=None,
    binarize="projection",
):
    """Initialize the first binary layer beneath Poisson observations.

    ``binarize`` selects how the latent states are read off the spectral
    loadings: ``"projection"`` thresholds the projected centered working
    response at zero (the MATLAB ``Poi_init`` rule); ``"anchor_mixture"``
    classifies each column with two-component Poisson mixtures on its anchor
    variables, which recovers the column prevalence.
    """
    binarize = str(binarize).strip().lower()
    if binarize not in {"projection", "anchor_mixture"}:
        raise ValueError("binarize must be 'projection' or 'anchor_mixture'.")
    working = _poisson_working_response(
        counts, K_parent, epsilon, rng=rng
    )
    K_parent = int(K_parent)
    n_children = working.shape[1]
    _, _, Vt = _truncated_svd(working, K_parent, rng=rng)
    if Vt.shape[0] < K_parent:
        raise ValueError(
            "Poisson spectral initialization could not recover K_parent directions."
        )

    loading_threshold = 1.0 / (2.5 * np.sqrt(n_children))
    if K_parent == 1:
        B_estimate = thres(Vt.T[:, :1], loading_threshold)
    else:
        B_estimate = thres(
            _stable_varimax(Vt.T[:, :K_parent]), loading_threshold
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
    if binarize == "anchor_mixture":
        parent, _ = _anchor_mixture_binarize(counts, B_estimate)
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


def Poisson_init(
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
    """Initialize an arbitrary-depth Poisson DDE.

    The observed layer follows the original Poisson double-SVD construction
    (``binarize`` chooses the latent-state rule, see
    :func:`_initialize_poisson_parent_layer`).  Every deeper layer uses the
    shared Bernoulli-child spectral initializer.
    """
    X_array = np.asarray(X)
    if X_array.ndim != 2 or min(X_array.shape) < 1:
        raise ValueError("X must be a nonempty two-dimensional count matrix.")
    try:
        valid = (
            np.all(np.isfinite(X_array))
            and np.all(X_array >= 0)
            and np.all(X_array == np.floor(X_array))
        )
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("X must contain finite nonnegative integer counts.")
    X = X_array.astype(float, copy=False)

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

    rng = _as_rng(random_state)
    B_initial = []
    A_initial = []
    graph_errors = []

    B_first, A_first, _, graph_error = _initialize_poisson_parent_layer(
        X,
        K_list[0],
        epsilon=epsilon,
        B_reference=None if not oracle else B_list[0],
        G_reference=None if not oracle else G_list[0],
        rng=rng,
        binarize=binarize,
    )
    B_initial.append(B_first)
    A_initial.append(A_first)
    graph_errors.append(graph_error)

    children = A_first
    for layer in range(1, D):
        B_layer, parent, _, graph_error = _initialize_binary_parent_layer(
            children,
            K_list[layer],
            epsilon=epsilon,
            threshold_constant=3.5,
            B_reference=None if not oracle else B_list[layer],
            G_reference=None if not oracle else G_list[layer],
            rng=rng,
        )
        B_initial.append(B_layer)
        A_initial.append(parent)
        graph_errors.append(graph_error)
        children = parent

    prop_initial = np.mean(A_initial[-1] > 0, axis=0)
    return prop_initial, B_initial, None, A_initial, graph_errors


poisson_init = Poisson_init


__all__ = [
    "Poisson_init",
    "_anchor_mixture_binarize",
    "_initialize_poisson_parent_layer",
    "_poisson_working_response",
    "poisson_init",
]
