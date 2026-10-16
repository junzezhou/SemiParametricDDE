"""CSP-MAP MCEM kernels for the discrete parametric DDE families.

This module serves the Bernoulli and Poisson CSP backends and the SemiDDE
CSP inner kernel.  The parametric Normal DDE no longer uses it: its CSP path
is the CopDDE-derived kernel in :mod:`dde.algorithms.normal_csp`.

The original simultaneous kernel follows the coordinate-ascent architecture
used by the package's CopDDE CSP path.  Two staged variants share that kernel
and differ only in whether a waiting parent is rebuilt or capped while its
eligible state is preserved.  The latent control loop is family-independent;
its adapter supplies the observed conditional likelihood and M-step.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .csp import csp_active_mask, update_csp_layer
from .normal_init_semi import initialize_binary_parent
from .parametric_families import (
    BERNOULLI_FAMILY,
    NORMAL_FAMILY,
    _largest_eigenvalue_psd,
    _poisson_optimization_exp_value,
    _poisson_optimization_rate,
    _soft_threshold,
    _solve_gaussian_weighted_l1,
    _solve_gaussian_weighted_l1_from_statistics,
    resolve_parametric_family,
)
from .semi_saem import (
    _compute_bernoulli_probabilities_parametric,
    _sample_latent_layers_parametric,
)


_LEGACY_SYNCHRONOUS = "legacy_synchronous"
_SYSTEMATIC_GIBBS = "systematic_coordinate_gibbs"


def _as_rng(random_state=None):
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_spike_scales(t_spike, D: int) -> np.ndarray:
    scales = np.asarray(t_spike, dtype=float).reshape(-1)
    if scales.size == 1:
        scales = np.repeat(scales.item(), D)
    if scales.size != D:
        raise ValueError("t_spike must be a scalar or a length-D sequence.")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("t_spike entries must be finite and positive.")
    return scales


def _canonicalize_csp_layer_order(prop, B_list, A_list):
    """Order latent coordinates by loading strength without changing the model.

    The cumulative-shrinkage prior is ordered, whereas spectral/varimax
    initialization is permutation invariant.  A simultaneous permutation of
    a layer's incoming columns, persistent states, and outgoing rows preserves
    every model probability while placing the strongest coordinates first.
    """
    prop = np.asarray(prop, dtype=float).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list]
    A_list = [np.asarray(A).copy() for A in A_list]
    permutations = []
    D = len(B_list)
    for d in range(D):
        strength = np.sum(np.abs(B_list[d][:, 1:]), axis=0)
        permutation = np.argsort(-strength, kind="stable")
        permutations.append(permutation)
        B_list[d][:, 1:] = B_list[d][:, permutation + 1]
        A_list[d] = A_list[d][:, permutation]
        if d + 1 < D:
            B_list[d + 1] = B_list[d + 1][permutation, :]
        else:
            prop = prop[permutation]
    return prop, B_list, A_list, permutations


def _sample_latent_layers(ber, C: int, rng) -> list[np.ndarray]:
    samples = []
    for probs in ber:
        N, Kd = probs.shape
        augmented = np.empty((N, Kd + 1, C), dtype=np.uint8)
        augmented[:, 0, :] = 1
        augmented[:, 1:, :] = rng.random((N, Kd, C)) < probs[:, :, None]
        samples.append(augmented)
    return samples


def _spectral_parent_width(A_prev, cap, minimum_gap):
    """Select a parent start width only when a clear covariance gap exists."""
    cap = int(cap)
    centered = np.asarray(A_prev, dtype=float)
    centered = centered - np.mean(centered, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    normalized = singular_values / np.sqrt(centered.shape[0])
    if minimum_gap is None or cap < 2 or normalized.size <= cap:
        return cap, normalized[: cap + 1], None

    gaps = normalized[:cap] / np.maximum(normalized[1 : cap + 1], 1e-12)
    gap_index = int(np.argmax(gaps))
    largest_gap = float(gaps[gap_index])
    selected = gap_index + 1 if largest_gap >= minimum_gap else cap
    return selected, normalized[: cap + 1], largest_gap


def _reinitialize_staged_parent(
    layer,
    prop,
    B_list,
    A_list,
    active_masks,
    allocation,
    penalty_scale,
    slab_scale,
    omega,
    stage_caps,
    spike_scales,
    spectral_gap_threshold,
):
    """Rebuild one waiting parent at its child-implied identifiable cap."""
    child_indices = np.flatnonzero(active_masks[layer - 1])
    cap = min(B_list[layer].shape[1] - 1, child_indices.size // 3)
    if cap < 1:
        raise ValueError(
            "staged CSP selected too few child coordinates for another layer."
        )
    child_states = A_list[layer - 1][:, child_indices]
    initial_width, singular_values, largest_gap = _spectral_parent_width(
        child_states, cap, spectral_gap_threshold
    )
    B_small, A_parent = initialize_binary_parent(child_states, initial_width)

    B_new = np.zeros_like(B_list[layer])
    B_new[child_indices, 0] = B_small[:, 0]
    B_new[np.ix_(child_indices, np.arange(1, initial_width + 1))] = B_small[
        :, 1:
    ]
    B_list[layer] = B_new
    A_new = np.zeros_like(A_list[layer])
    A_new[:, :initial_width] = A_parent
    A_list[layer] = A_new

    stage_caps[layer] = cap
    active_masks[layer] = np.arange(active_masks[layer].size) < initial_width
    allocation[layer] = np.where(
        active_masks[layer],
        active_masks[layer].size + 1,
        np.arange(1, active_masks[layer].size + 1),
    )
    slab_scale[layer] = np.ones_like(slab_scale[layer])
    omega[layer] = np.ones_like(omega[layer]) / omega[layer].size
    penalty_scale[layer] = np.broadcast_to(
        np.where(active_masks[layer], 1.0, spike_scales[layer])[None, :],
        penalty_scale[layer].shape,
    ).copy()

    if layer == len(B_list) - 1:
        prop_new = np.full_like(prop, 0.5, dtype=float)
        prop_new[:initial_width] = np.mean(A_parent, axis=0)
        prop = prop_new
    else:
        B_list[layer + 1] = np.zeros_like(B_list[layer + 1])
    return prop, {
        "layer": layer + 1,
        "mode": "reinitialize",
        "child_active": int(child_indices.size),
        "structural_cap": cap,
        "initial_width": initial_width,
        "singular_values": tuple(float(value) for value in singular_values),
        "largest_gap": largest_gap,
    }


def _apply_staged_parent_cap_only(
    layer,
    prop,
    B_list,
    A_list,
    active_masks,
    allocation,
    penalty_scale,
    slab_scale,
    omega,
    stage_caps,
    spike_scales,
    spectral_gap_threshold,
):
    """Apply the child-implied cap without rebuilding the waiting parent."""
    del slab_scale, omega, spectral_gap_threshold
    child_indices = np.flatnonzero(active_masks[layer - 1])
    cap = min(B_list[layer].shape[1] - 1, child_indices.size // 3)
    if cap < 1:
        raise ValueError(
            "cap-only staged CSP selected too few child coordinates for "
            "another layer."
        )

    width = active_masks[layer].size
    parent_mask = np.arange(width) < cap
    active_masks[layer] = parent_mask
    stage_caps[layer] = cap
    allocation[layer] = np.where(
        parent_mask,
        width + 1,
        np.arange(1, width + 1),
    )
    penalty_scale[layer][:, ~parent_mask] = spike_scales[layer]

    inactive_children = np.flatnonzero(~active_masks[layer - 1])
    if inactive_children.size:
        B_list[layer][inactive_children, :] = 0.0
    B_list[layer][:, cap + 1 :] = 0.0
    A_list[layer][:, cap:] = 0

    if layer == len(B_list) - 1:
        prop = np.asarray(prop, dtype=float).copy()
        prop[cap:] = 0.5
    else:
        B_list[layer + 1][cap:, :] = 0.0

    return prop, {
        "layer": layer + 1,
        "mode": "cap_only",
        "child_active": int(child_indices.size),
        "structural_cap": int(cap),
        "initial_width": int(cap),
        "singular_values": (),
        "largest_gap": None,
    }


def _logistic_mc_design_and_response_statistic(child_sample, parent_sample):
    """Build the MC design and cache its Bernoulli response cross-product."""
    child_sample = np.asarray(child_sample)
    parent_sample = np.asarray(parent_sample)
    if parent_sample.ndim != 3:
        raise ValueError("Bernoulli parent samples must be a 3D MC array.")
    N, parent_plus_intercept, C = parent_sample.shape
    design = (
        np.transpose(parent_sample, (0, 2, 1))
        .reshape(N * C, parent_plus_intercept)
        .astype(float, copy=False)
    )
    if child_sample.ndim == 2:
        n_children = child_sample.shape[1]
        if child_sample.shape[0] != N:
            raise ValueError(
                "Bernoulli children and parents must have the same row count."
            )
        response_statistic = np.asarray(child_sample, dtype=float).T @ np.sum(
            parent_sample, axis=2, dtype=float
        )
    elif child_sample.ndim == 3:
        n_children = child_sample.shape[1] - 1
        if child_sample.shape[0] != N or child_sample.shape[2] != C:
            raise ValueError(
                "Bernoulli child and parent MC arrays must align in rows and draws."
            )
        response_statistic = np.einsum(
            "njc,npc->jp",
            child_sample[:, 1:, :],
            parent_sample,
            dtype=float,
            optimize=True,
        )
    else:
        raise ValueError(
            "Bernoulli children must be a 2D matrix or 3D MC array."
        )
    return design, response_statistic, N, n_children, parent_plus_intercept, C


def _logistic_mc_value_and_grad(
    coefficients,
    design,
    response_statistic,
    *,
    temperature,
    mc_samples,
    row_chunk_size,
    coordinate_chunk_size,
    compute_value=True,
):
    """Evaluate a Bernoulli MC smooth objective in bounded-size batches."""
    coefficients = np.asarray(coefficients, dtype=float)
    design = np.asarray(design, dtype=float)
    response_statistic = np.asarray(response_statistic, dtype=float)
    if response_statistic.shape != coefficients.shape:
        raise ValueError(
            "Bernoulli response statistic and coefficients must have equal shapes."
        )
    if design.ndim != 2 or design.shape[1] != coefficients.shape[1]:
        raise ValueError(
            "Bernoulli design and coefficients have incompatible shapes."
        )

    row_chunk_size = max(1, int(row_chunk_size))
    coordinate_chunk_size = max(1, int(coordinate_chunk_size))
    scale = float(temperature) / int(mc_samples)
    gradient = -response_statistic.copy()
    value = (
        -float(np.sum(response_statistic * coefficients))
        if compute_value
        else None
    )

    for row_start in range(0, design.shape[0], row_chunk_size):
        row_stop = min(row_start + row_chunk_size, design.shape[0])
        design_chunk = design[row_start:row_stop]
        for coordinate_start in range(
            0, coefficients.shape[0], coordinate_chunk_size
        ):
            coordinate_stop = min(
                coordinate_start + coordinate_chunk_size,
                coefficients.shape[0],
            )
            logits = (
                design_chunk @ coefficients[coordinate_start:coordinate_stop].T
            )
            if compute_value:
                value += float(np.sum(np.logaddexp(0.0, logits)))
            expit(logits, out=logits)
            gradient[coordinate_start:coordinate_stop] += (
                logits.T @ design_chunk
            )

    return (
        None if value is None else scale * value,
        scale * gradient,
    )


def _solve_logistic_weighted_l1(
    child_sample,
    parent_sample,
    B_start,
    penalty_scale,
    temperature,
    *,
    max_iter=100,
    tol=1e-6,
    bound=10.0,
    bounds=None,
    row_chunk_size=None,
    coordinate_chunk_size=None,
):
    """Chunked batched FISTA update for one conditional Bernoulli layer.

    The response-dependent linear statistic is computed once, so repeated
    FISTA evaluations neither expand a fixed observed response over all Monte
    Carlo draws nor recompute ``response.T @ design``.  Row and response
    chunks bound the temporary logits while preserving the weighted-L1 target.
    """
    B = np.asarray(B_start, dtype=float).copy()
    penalty_scale = np.asarray(penalty_scale, dtype=float)
    (
        design,
        response_statistic,
        N,
        n_children,
        parent_plus_intercept,
        C,
    ) = _logistic_mc_design_and_response_statistic(child_sample, parent_sample)
    expected_shape = (n_children, parent_plus_intercept)
    if B.shape != expected_shape:
        raise ValueError(
            f"Bernoulli coefficients must have shape {expected_shape}; got {B.shape}."
        )
    if penalty_scale.shape != (n_children, parent_plus_intercept - 1):
        raise ValueError(
            "Bernoulli penalty_scale must have one value per response and slope."
        )

    if row_chunk_size is None:
        flat_row_chunk_size = N * C
    else:
        flat_row_chunk_size = max(1, int(row_chunk_size)) * C
    if coordinate_chunk_size is None:
        coordinate_chunk_size = n_children
    coordinate_chunk_size = max(1, int(coordinate_chunk_size))

    if bounds is None:
        lower = np.full(parent_plus_intercept, -float(bound))
        upper = np.full(parent_plus_intercept, float(bound))
        lower[0] = -5.0
    else:
        bounds = list(bounds)
        if len(bounds) != parent_plus_intercept:
            raise ValueError(
                "Bernoulli coefficient bounds have an incompatible width."
            )
        lower = np.asarray([pair[0] for pair in bounds], dtype=float)
        upper = np.asarray([pair[1] for pair in bounds], dtype=float)
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ValueError(
                "Bernoulli coefficient bounds must be finite and ordered."
            )

    gram = design.T @ design
    lipschitz = float(temperature) / C * 0.25 * _largest_eigenvalue_psd(gram)
    if lipschitz <= 1e-14:
        B[:, 0] = 0.0
        B[:, 1:] = 0.0
        return B

    step = 1.0 / lipschitz
    Y = B.copy()
    momentum = 1.0
    for _ in range(int(max_iter)):
        _, gradient = _logistic_mc_value_and_grad(
            Y,
            design,
            response_statistic,
            temperature=temperature,
            mc_samples=C,
            row_chunk_size=flat_row_chunk_size,
            coordinate_chunk_size=coordinate_chunk_size,
            compute_value=False,
        )
        candidate = Y - step * gradient
        candidate[:, 0] = np.clip(candidate[:, 0], lower[0], upper[0])
        candidate[:, 1:] = _soft_threshold(
            candidate[:, 1:], step / penalty_scale
        )
        candidate[:, 1:] = np.clip(
            candidate[:, 1:], lower[None, 1:], upper[None, 1:]
        )

        relative = np.linalg.norm(candidate - B) / (np.linalg.norm(B) + 1e-12)
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        Y = candidate + ((momentum - 1.0) / next_momentum) * (candidate - B)
        B = candidate
        momentum = next_momentum
        if relative <= tol:
            break
    return B


def _poisson_mc_design_and_response_statistic(X, parent_sample):
    """Build the repeated Poisson MC design and its linear statistic."""
    X = np.asarray(X, dtype=float)
    parent_sample = np.asarray(parent_sample)
    if X.ndim != 2:
        raise ValueError("Poisson responses must be a two-dimensional matrix.")
    if parent_sample.ndim != 3:
        raise ValueError("Poisson parent samples must be a 3D MC array.")
    N, parent_plus_intercept, C = parent_sample.shape
    if X.shape[0] != N:
        raise ValueError(
            "Poisson responses and parents must have the same row count."
        )
    design = (
        np.transpose(parent_sample, (0, 2, 1))
        .reshape(N * C, parent_plus_intercept)
        .astype(float, copy=False)
    )
    response_statistic = X.T @ np.sum(parent_sample, axis=2, dtype=float)
    return design, response_statistic, N, X.shape[1], parent_plus_intercept, C


def _poisson_mc_value_and_grad(
    coefficients,
    design,
    response_statistic,
    *,
    temperature,
    mc_samples,
    row_chunk_size,
    coordinate_chunk_size,
    compute_value=True,
    compute_gradient=True,
):
    """Evaluate the tempered Poisson MC smooth objective in chunks."""
    coefficients = np.asarray(coefficients, dtype=float)
    design = np.asarray(design, dtype=float)
    response_statistic = np.asarray(response_statistic, dtype=float)
    if response_statistic.shape != coefficients.shape:
        raise ValueError(
            "Poisson response statistic and coefficients must have equal shapes."
        )
    if design.ndim != 2 or design.shape[1] != coefficients.shape[1]:
        raise ValueError(
            "Poisson design and coefficients have incompatible shapes."
        )

    row_chunk_size = max(1, int(row_chunk_size))
    coordinate_chunk_size = max(1, int(coordinate_chunk_size))
    scale = float(temperature) / int(mc_samples)
    gradient = -response_statistic.copy() if compute_gradient else None
    value = (
        -float(np.sum(response_statistic * coefficients))
        if compute_value
        else None
    )

    for row_start in range(0, design.shape[0], row_chunk_size):
        row_stop = min(row_start + row_chunk_size, design.shape[0])
        design_chunk = design[row_start:row_stop]
        for coordinate_start in range(
            0, coefficients.shape[0], coordinate_chunk_size
        ):
            coordinate_stop = min(
                coordinate_start + coordinate_chunk_size,
                coefficients.shape[0],
            )
            eta = (
                design_chunk @ coefficients[coordinate_start:coordinate_stop].T
            )
            if compute_value:
                value += float(np.sum(_poisson_optimization_exp_value(eta)))
            if compute_gradient:
                rate = _poisson_optimization_rate(eta)
                gradient[coordinate_start:coordinate_stop] += (
                    rate.T @ design_chunk
                )

    return (
        None if value is None else scale * value,
        None if gradient is None else scale * gradient,
    )


def _solve_poisson_weighted_l1(
    X,
    parent_sample,
    B_start,
    penalty_scale,
    temperature,
    *,
    max_iter=100,
    tol=1e-6,
    bound=10.0,
    bounds=None,
    row_chunk_size=None,
    coordinate_chunk_size=None,
):
    """Fit the observed Poisson layer with bounded backtracking FISTA.

    The optimized target is the tempered current-draw Monte Carlo negative
    log-likelihood plus ``sum(abs(B[:, 1:]) / penalty_scale)``. A local
    backtracking majorizer is required because, unlike the Bernoulli loss,
    the Poisson log-partition has no finite global curvature bound.
    """
    B = np.asarray(B_start, dtype=float).copy()
    penalty_scale = np.asarray(penalty_scale, dtype=float)
    (
        design,
        response_statistic,
        N,
        n_responses,
        parent_plus_intercept,
        C,
    ) = _poisson_mc_design_and_response_statistic(X, parent_sample)
    expected_shape = (n_responses, parent_plus_intercept)
    if B.shape != expected_shape:
        raise ValueError(
            f"Poisson coefficients must have shape {expected_shape}; got {B.shape}."
        )
    expected_penalty_shape = (n_responses, parent_plus_intercept - 1)
    if penalty_scale.shape != expected_penalty_shape:
        raise ValueError(
            "Poisson penalty_scale must have one value per response and slope."
        )
    if not np.all(np.isfinite(penalty_scale)) or np.any(penalty_scale <= 0.0):
        raise ValueError(
            "Poisson penalty_scale entries must be finite and positive."
        )

    if row_chunk_size is None:
        flat_row_chunk_size = N * C
    else:
        flat_row_chunk_size = max(1, int(row_chunk_size)) * C
    if coordinate_chunk_size is None:
        coordinate_chunk_size = n_responses
    coordinate_chunk_size = max(1, int(coordinate_chunk_size))

    if bounds is None:
        upper_value = min(5.0, float(bound))
        lower = np.full(parent_plus_intercept, -min(2.5, float(bound)))
        upper = np.full(parent_plus_intercept, upper_value)
        lower[0] = -upper_value
    else:
        bounds = list(bounds)
        if len(bounds) != parent_plus_intercept:
            raise ValueError(
                "Poisson coefficient bounds have an incompatible width."
            )
        lower = np.asarray([pair[0] for pair in bounds], dtype=float)
        upper = np.asarray([pair[1] for pair in bounds], dtype=float)
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
        or lower[0] > 0.0
        or upper[0] < 0.0
        or np.any(lower[1:] > 0.0)
        or np.any(upper[1:] < 0.0)
    ):
        raise ValueError(
            "Poisson coefficient bounds must be finite, ordered, and contain zero."
        )

    B = np.clip(B, lower[None, :], upper[None, :])
    gram_largest = _largest_eigenvalue_psd(design.T @ design)
    if gram_largest <= 1e-14:
        B[:, 0] = 0.0
        B[:, 1:] = 0.0
        return B

    initial_rate_max = 1.0
    for row_start in range(0, design.shape[0], flat_row_chunk_size):
        row_stop = min(row_start + flat_row_chunk_size, design.shape[0])
        design_chunk = design[row_start:row_stop]
        for coordinate_start in range(0, n_responses, coordinate_chunk_size):
            coordinate_stop = min(
                coordinate_start + coordinate_chunk_size, n_responses
            )
            eta = design_chunk @ B[coordinate_start:coordinate_stop].T
            initial_rate_max = max(
                initial_rate_max,
                float(np.max(_poisson_optimization_rate(eta))),
            )
    lipschitz = max(
        float(temperature) / C * initial_rate_max * gram_largest,
        1e-12,
    )
    Y = B.copy()
    momentum = 1.0
    for _ in range(int(max_iter)):
        Y = np.clip(Y, lower[None, :], upper[None, :])
        value_at_y, gradient = _poisson_mc_value_and_grad(
            Y,
            design,
            response_statistic,
            temperature=temperature,
            mc_samples=C,
            row_chunk_size=flat_row_chunk_size,
            coordinate_chunk_size=coordinate_chunk_size,
        )

        trial_lipschitz = max(0.5 * lipschitz, 1e-12)
        while True:
            step = 1.0 / trial_lipschitz
            candidate = Y - step * gradient
            candidate[:, 0] = np.clip(candidate[:, 0], lower[0], upper[0])
            candidate[:, 1:] = _soft_threshold(
                candidate[:, 1:], step / penalty_scale
            )
            candidate[:, 1:] = np.clip(
                candidate[:, 1:], lower[None, 1:], upper[None, 1:]
            )
            candidate_value, _ = _poisson_mc_value_and_grad(
                candidate,
                design,
                response_statistic,
                temperature=temperature,
                mc_samples=C,
                row_chunk_size=flat_row_chunk_size,
                coordinate_chunk_size=coordinate_chunk_size,
                compute_value=True,
                compute_gradient=False,
            )
            difference = candidate - Y
            quadratic_bound = (
                value_at_y
                + float(np.sum(gradient * difference))
                + 0.5 * trial_lipschitz * float(np.sum(difference**2))
            )
            comparison_slack = 1e-12 * max(
                1.0, abs(candidate_value), abs(quadratic_bound)
            )
            if candidate_value <= quadratic_bound + comparison_slack:
                break
            trial_lipschitz *= 2.0
            if not np.isfinite(trial_lipschitz):
                raise FloatingPointError(
                    "Poisson CSP backtracking failed to find a finite step."
                )

        relative = np.linalg.norm(candidate - B) / (np.linalg.norm(B) + 1e-12)
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        Y = candidate + ((momentum - 1.0) / next_momentum) * (candidate - B)
        B = candidate
        momentum = next_momentum
        lipschitz = trial_lipschitz
        if relative <= tol:
            break
    return B


def fit_parametric_csp_mcem(
    X,
    prop_in,
    B_list_in,
    dispersion_in,
    A_list_in,
    *,
    C: int,
    max_iter: int,
    t_spike,
    temperature: float,
    threshold: float,
    convergence_tol: float = 0.005,
    convergence_mean_tol: float = 0.01,
    convergence_window: int = 7,
    stagewise: bool = False,
    stage_min_iter: int = 20,
    stage_width_window: int = 10,
    stage_freeze_deeper: bool = True,
    penalty_strength=1.0,
    shrinkage_burn: int = 0,
    reorder_after_shrinkage_burn: bool = True,
    stage_transition_mode: str = "none",
    stage_spectral_gap_threshold=None,
    burn: int = 5,
    optimizer_maxiter: int = 100,
    optimizer_tol: float = 1e-6,
    bound_val: float = 10.0,
    estep_row_chunk_size: int = 2048,
    estep_coordinate_chunk_size: int = 1,
    random_state=None,
    keep_trace: bool = False,
    verbose: bool = True,
    latent_transition=None,
    canonicalize_initial_order=None,
    initial_csp_state=None,
    family,
):
    """Fit a maximal-width parametric DDE under layer-wise CSP priors.

    ``stagewise`` activates CSP from the observed-side layer toward the
    deepest layer and advances only after the current layer's width has been
    stable for a full window.  ``stage_transition_mode`` selects whether a
    waiting parent is rebuilt, only capped while preserving its state, or
    left untouched. Discrete observed families default to persistent
    systematic Gibbs; Normal keeps its established synchronous
    approximate-MCEM transition.
    """
    family = resolve_parametric_family(family)
    X = np.asarray(X, dtype=float)
    prop = np.asarray(prop_in, dtype=float).reshape(-1).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list_in]
    dispersion = family.copy_dispersion(dispersion_in)
    A_list = [np.asarray(A, dtype=np.uint8).copy() for A in A_list_in]
    if canonicalize_initial_order is None:
        canonicalize_initial_order = family.key in {"bernoulli", "poisson"}
    canonicalize_initial_order = bool(canonicalize_initial_order)
    if canonicalize_initial_order:
        prop, B_list, A_list, initial_permutations = (
            _canonicalize_csp_layer_order(prop, B_list, A_list)
        )
    else:
        initial_permutations = [
            np.arange(B.shape[1] - 1, dtype=int) for B in B_list
        ]
    D = len(B_list)
    K = [B.shape[1] - 1 for B in B_list]
    J = X.shape[1]
    response_cache = family.make_response_cache(X)

    if latent_transition is None:
        latent_transition = (
            _SYSTEMATIC_GIBBS
            if family.key in {"bernoulli", "poisson"}
            else _LEGACY_SYNCHRONOUS
        )
    latent_transition = str(latent_transition).strip().lower()
    if latent_transition not in {
        _LEGACY_SYNCHRONOUS,
        _SYSTEMATIC_GIBBS,
    }:
        raise ValueError(
            "latent_transition must be 'legacy_synchronous' or "
            "'systematic_coordinate_gibbs'."
        )

    C = int(C)
    max_iter = int(max_iter)
    convergence_window = int(convergence_window)
    stage_min_iter = int(stage_min_iter)
    stage_width_window = int(stage_width_window)
    shrinkage_burn = int(shrinkage_burn)
    reorder_after_shrinkage_burn = bool(reorder_after_shrinkage_burn)
    burn = int(burn)
    optimizer_maxiter = int(optimizer_maxiter)
    temperature = float(temperature)
    threshold = float(threshold)
    convergence_tol = float(convergence_tol)
    convergence_mean_tol = float(convergence_mean_tol)
    stage_transition_mode = str(stage_transition_mode).strip().lower()
    if stage_transition_mode not in {"none", "reinitialize", "cap_only"}:
        raise ValueError(
            "stage_transition_mode must be 'none', 'reinitialize', or "
            "'cap_only'."
        )
    if not stagewise and stage_transition_mode != "none":
        raise ValueError("stage_transition_mode requires stagewise=True.")
    if stagewise and shrinkage_burn:
        raise ValueError(
            "shrinkage_burn is only supported by simultaneous MCEM."
        )
    if (
        stage_transition_mode != "reinitialize"
        and stage_spectral_gap_threshold is not None
    ):
        raise ValueError(
            "stage_spectral_gap_threshold is only valid for the "
            "reinitialize transition mode."
        )
    if stage_spectral_gap_threshold is not None:
        stage_spectral_gap_threshold = float(stage_spectral_gap_threshold)
    if C < 1 or max_iter < 1:
        raise ValueError("C and max_iter must be positive integers.")
    if (
        convergence_window < 1
        or stage_min_iter < 1
        or stage_width_window < 1
        or burn < 0
        or shrinkage_burn < 0
        or optimizer_maxiter < 1
    ):
        raise ValueError(
            "convergence_window, stage_min_iter, stage_width_window, and "
            "optimizer_maxiter must be positive, and burn and shrinkage_burn "
            "nonnegative."
        )
    if shrinkage_burn >= max_iter:
        raise ValueError("shrinkage_burn must be smaller than max_iter.")
    if not np.isfinite(temperature) or not 0.0 < temperature <= 1.0:
        raise ValueError("temperature must lie in (0, 1].")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive.")
    if not np.isfinite(convergence_tol) or convergence_tol < 0:
        raise ValueError("convergence_tol must be finite and nonnegative.")
    if not np.isfinite(convergence_mean_tol) or convergence_mean_tol < 0:
        raise ValueError(
            "convergence_mean_tol must be finite and nonnegative."
        )
    if stage_spectral_gap_threshold is not None and (
        not np.isfinite(stage_spectral_gap_threshold)
        or stage_spectral_gap_threshold <= 1.0
    ):
        raise ValueError(
            "stage_spectral_gap_threshold must be finite and greater than 1."
        )

    spike_scales = _validate_spike_scales(t_spike, D)
    penalty_strengths = np.asarray(penalty_strength, dtype=float).reshape(-1)
    if penalty_strengths.size == 1:
        penalty_strengths = np.repeat(penalty_strengths.item(), D)
    if penalty_strengths.size != D:
        raise ValueError(
            "penalty_strength must be a scalar or a length-D sequence."
        )
    if not np.all(np.isfinite(penalty_strengths)) or np.any(
        penalty_strengths <= 0
    ):
        raise ValueError(
            "penalty_strength entries must be finite and positive."
        )
    rng = _as_rng(random_state)
    if initial_csp_state is None:
        slab_scale = [np.ones(Kd, dtype=float) for Kd in K]
        omega = [np.ones(Kd + 1, dtype=float) / (Kd + 1) for Kd in K]
        if stagewise or shrinkage_burn:
            allocation = [np.full(Kd, Kd + 1, dtype=int) for Kd in K]
            penalty_scale = [
                np.ones((J if d == 0 else K[d - 1], K[d]), dtype=float)
                for d in range(D)
            ]
        else:
            allocation = [np.full(Kd, np.floor(Kd / 2), dtype=int) for Kd in K]
            penalty_scale = [
                np.full((J if d == 0 else K[d - 1], K[d]), spike_scales[d])
                for d in range(D)
            ]
        active_masks = [np.ones(Kd, dtype=bool) for Kd in K]
        num_active = np.asarray(K, dtype=int)
        current_temperature = temperature
    else:
        if stagewise or shrinkage_burn:
            raise ValueError(
                "initial_csp_state only supports simultaneous CSP without "
                "shrinkage_burn."
            )
        slab_scale = [
            np.asarray(value, dtype=float).copy()
            for value in initial_csp_state["slab_scale"]
        ]
        omega = [
            np.asarray(value, dtype=float).copy()
            for value in initial_csp_state["omega"]
        ]
        allocation = [
            np.asarray(value, dtype=int).copy()
            for value in initial_csp_state["allocation"]
        ]
        active_masks = [csp_active_mask(value) for value in allocation]
        num_active = np.asarray(
            [np.sum(value) for value in active_masks], dtype=int
        )
        penalty_scale = [
            np.broadcast_to(
                np.where(mask, slab, spike)[None, :],
                (J if d == 0 else K[d - 1], K[d]),
            ).copy()
            for d, (mask, slab, spike) in enumerate(
                zip(active_masks, slab_scale, spike_scales)
            )
        ]
        current_temperature = float(
            initial_csp_state.get("temperature", temperature)
        )
        expected_shapes = [
            (Kd, Kd + 1, (J if d == 0 else K[d - 1], Kd))
            for d, Kd in enumerate(K)
        ]
        for d, (Kd, omega_size, penalty_shape) in enumerate(expected_shapes):
            if slab_scale[d].shape != (Kd,):
                raise ValueError("initial_csp_state slab_scale shape mismatch.")
            if omega[d].shape != (omega_size,):
                raise ValueError("initial_csp_state omega shape mismatch.")
            if allocation[d].shape != (Kd,):
                raise ValueError("initial_csp_state allocation shape mismatch.")
            if penalty_scale[d].shape != penalty_shape:
                raise ValueError("initial_csp_state penalty_scale shape mismatch.")
    relative_history = [[] for _ in range(D)]
    trace = [] if keep_trace else None
    active_stage = 0 if stagewise else D - 1
    stage_iteration = 0
    stage_width_history = []
    stage_transitions = []
    stage_transition_events = []
    stage_caps = np.asarray(K, dtype=int)
    stopped_by_convergence = False
    warmup_reordered = False

    iterations = 0
    while iterations < max_iter:
        shrinkage_active = iterations >= shrinkage_burn
        if (
            shrinkage_active
            and shrinkage_burn
            and reorder_after_shrinkage_burn
            and not warmup_reordered
        ):
            prop, B_list, A_list, warmup_permutations = (
                _canonicalize_csp_layer_order(prop, B_list, A_list)
            )
            initial_permutations = [
                np.asarray(initial, dtype=int)[np.asarray(warmup, dtype=int)]
                for initial, warmup in zip(
                    initial_permutations, warmup_permutations
                )
            ]
            warmup_reordered = True
        iteration_stage = active_stage
        if latent_transition == _SYSTEMATIC_GIBBS:
            A_samples, top_probability = _sample_latent_layers_parametric(
                X,
                prop,
                B_list,
                dispersion,
                A_list,
                C,
                family=family,
                row_chunk_size=estep_row_chunk_size,
                rng=rng,
                eligible_widths=stage_caps,
                temperature=current_temperature,
            )
        else:
            probabilities = _compute_bernoulli_probabilities_parametric(
                X,
                prop,
                B_list,
                dispersion,
                A_list,
                family=family,
                row_chunk_size=estep_row_chunk_size,
                coordinate_chunk_size=estep_coordinate_chunk_size,
                eligible_widths=stage_caps,
                temperature=current_temperature,
            )
            A_samples = _sample_latent_layers(probabilities, C, rng)
            for d in range(D):
                if stage_caps[d] < K[d]:
                    probabilities[d][:, stage_caps[d] :] = 0.0
                    A_samples[d][:, stage_caps[d] + 1 :, :] = 0
            top_probability = probabilities[-1]

        for d in range(iteration_stage, -1, -1):
            if shrinkage_active:
                num_rows = J if d == 0 else int(num_active[d - 1])
                (
                    allocation[d],
                    penalty_scale[d],
                    slab_scale[d],
                    omega[d],
                ) = update_csp_layer(
                    B_list[d],
                    omega[d],
                    spike_scales[d],
                    slab_scale[d],
                    num_rows,
                    rng=rng,
                )
            else:
                allocation[d].fill(K[d] + 1)
                penalty_scale[d][:] = slab_scale[d][None, :]
            penalty_scale[d] /= penalty_strengths[d]
            if stage_caps[d] < K[d]:
                ineligible = np.arange(K[d]) >= stage_caps[d]
                allocation[d][ineligible] = np.arange(1, K[d] + 1)[ineligible]
                penalty_scale[d][:, ineligible] = spike_scales[d]

        if stagewise and stage_freeze_deeper and iteration_stage < D - 1:
            prop_update = prop.copy()
        else:
            prop_update = np.mean(top_probability, axis=0)
        B_update = [None] * D
        observed_bounds = family.csp_coefficient_bounds(
            B_list[0].shape[1], bound_val, observed=True
        )
        B_update[0], dispersion_update = family.csp_mstep(
            X,
            A_samples[0],
            B_list[0],
            dispersion,
            penalty_scale[0],
            current_temperature,
            response_cache=response_cache,
            max_iter=optimizer_maxiter,
            tol=optimizer_tol,
            bound=bound_val,
            bounds=observed_bounds,
            row_chunk_size=estep_row_chunk_size,
            coordinate_chunk_size=None,
        )
        for d in range(1, D):
            if stagewise and stage_freeze_deeper and d > iteration_stage:
                B_update[d] = B_list[d].copy()
            else:
                B_update[d] = _solve_logistic_weighted_l1(
                    A_samples[d - 1],
                    A_samples[d],
                    B_list[d],
                    penalty_scale[d],
                    current_temperature,
                    max_iter=optimizer_maxiter,
                    tol=optimizer_tol,
                    bound=bound_val,
                    bounds=family.csp_coefficient_bounds(
                        B_list[d].shape[1], bound_val, observed=False
                    ),
                    row_chunk_size=estep_row_chunk_size,
                    coordinate_chunk_size=None,
                )

        iterations += 1
        stage_iteration += 1
        inactive_indices = []
        for d in range(D):
            if not stagewise or d <= active_stage:
                active_masks[d] = csp_active_mask(allocation[d])
                active_masks[d][stage_caps[d] :] = False
            else:
                active_masks[d] = np.ones(K[d], dtype=bool)
            inactive = np.flatnonzero(~active_masks[d])
            inactive_indices.append(inactive)
            num_active[d] = int(np.sum(active_masks[d]))

        thresholded_layers = min(2, D)
        if stagewise:
            thresholded_layers = min(thresholded_layers, iteration_stage + 1)
        if shrinkage_active:
            for d in range(thresholded_layers):
                slopes = B_update[d][:, 1:]
                slopes[np.abs(slopes) < threshold] = 0.0

        for d in range(D):
            if stage_caps[d] < K[d]:
                B_update[d][:, stage_caps[d] + 1 :] = 0.0

        for d in range(D - 1):
            if inactive_indices[d].size:
                B_update[d + 1][inactive_indices[d], :] = 0.0

        if iterations > burn:
            current_temperature = min(
                1.0,
                current_temperature + 0.01 * ((iterations - burn) - 1),
            )

        stop_now = not stagewise or active_stage == D - 1
        if not shrinkage_active:
            stop_now = False
        current_relative = []
        for d in range(D):
            relative = np.linalg.norm(
                B_update[d][:, 1:] - B_list[d][:, 1:], ord="fro"
            ) / (np.linalg.norm(B_list[d][:, 1:], ord="fro") + 1e-6)
            relative_history[d].append(float(relative))
            current_relative.append(float(relative))
            if (
                len(relative_history[d]) < convergence_window
                or np.mean(relative_history[d][-convergence_window:])
                >= convergence_mean_tol
                or np.std(relative_history[d][-convergence_window:])
                >= convergence_tol
            ):
                stop_now = False

        stage_advanced = False
        if stagewise:
            stage_width_history.append(int(num_active[active_stage]))
            stable_current_width = (
                len(stage_width_history) >= stage_width_window
                and len(set(stage_width_history[-stage_width_window:])) == 1
            )
            if (
                active_stage < D - 1
                and stage_iteration >= stage_min_iter
                and stable_current_width
            ):
                previous_stage = active_stage
                active_stage += 1
                stage_iteration = 0
                stage_width_history = []
                stage_advanced = True
                stage_transitions.append(
                    {
                        "iteration": iterations,
                        "from_layer": previous_stage + 1,
                        "to_layer": active_stage + 1,
                        "widths": tuple(int(value) for value in num_active),
                    }
                )
            if active_stage < D - 1:
                stop_now = False

        if keep_trace:
            trace.append(
                {
                    "iteration": iterations,
                    "active_stage": active_stage + 1,
                    "stage_iteration": stage_iteration,
                    "stage_advanced": stage_advanced,
                    "num_active": num_active.copy(),
                    "allocation": [value.copy() for value in allocation],
                    "omega": [value.copy() for value in omega],
                    "allocation_input_column_l1": [
                        np.sum(np.abs(B[:, 1:]), axis=0) for B in B_list
                    ],
                    "updated_column_l1": [
                        np.sum(np.abs(B[:, 1:]), axis=0) for B in B_update
                    ],
                    "relative_change": tuple(current_relative),
                    "temperature": float(current_temperature),
                    "shrinkage_active": bool(shrinkage_active),
                }
            )

        if latent_transition == _SYSTEMATIC_GIBBS:
            A_list = [sample[:, 1:, -1].copy() for sample in A_samples]
        else:
            A_list = [
                (np.mean(sample[:, 1:, :], axis=2) > 0.5).astype(np.uint8)
                for sample in A_samples
            ]
        B_list = [B.copy() for B in B_update]
        dispersion = family.copy_dispersion(dispersion_update)
        prop = prop_update.copy()
        if stage_advanced and stage_transition_mode != "none":
            transition = (
                _reinitialize_staged_parent
                if stage_transition_mode == "reinitialize"
                else _apply_staged_parent_cap_only
            )
            prop, transition_event = transition(
                active_stage,
                prop,
                B_list,
                A_list,
                active_masks,
                allocation,
                penalty_scale,
                slab_scale,
                omega,
                stage_caps,
                spike_scales,
                stage_spectral_gap_threshold,
            )
            stage_transition_events.append(transition_event)
            num_active[active_stage] = int(np.sum(active_masks[active_stage]))

        if verbose:
            print(
                f"CSP-MCEM iteration {iterations}, stage={active_stage + 1}, "
                f"active={num_active.tolist()}, "
                f"max relative change={max(current_relative):.4g}"
            )
        if stop_now:
            stopped_by_convergence = iterations < max_iter
            break

    for d in range(D):
        inactive = np.flatnonzero(~active_masks[d])
        if inactive.size:
            B_list[d][:, inactive + 1] = 0.0

    return {
        "prop": prop,
        "B_list": B_list,
        "gamma": dispersion,
        "A_list": A_list,
        "allocation": [value.copy() for value in allocation],
        "omega": [value.copy() for value in omega],
        "slab_scale": [value.copy() for value in slab_scale],
        "penalty_scale": [value.copy() for value in penalty_scale],
        "active_masks": [value.copy() for value in active_masks],
        "num_active": num_active.copy(),
        "iterations": iterations,
        "stagewise": bool(stagewise),
        "stage_transitions": tuple(stage_transitions),
        "final_stage": active_stage + 1,
        "stage_freeze_deeper": bool(stage_freeze_deeper),
        "penalty_strength": tuple(float(value) for value in penalty_strengths),
        "shrinkage_burn": shrinkage_burn,
        "reorder_after_shrinkage_burn": reorder_after_shrinkage_burn,
        "warmup_reordered": warmup_reordered,
        "stage_transition_mode": stage_transition_mode,
        "stage_spectral_gap_threshold": stage_spectral_gap_threshold,
        "stage_transition_events": tuple(stage_transition_events),
        "stage_caps": tuple(int(value) for value in stage_caps),
        "stopped_by_convergence": bool(stopped_by_convergence),
        "temperature": float(current_temperature),
        "canonicalized_initial_order": canonicalize_initial_order,
        "initial_permutations": tuple(
            tuple(int(index) for index in permutation)
            for permutation in initial_permutations
        ),
        "latent_transition": latent_transition,
        "mc_state_update": (
            "last_draw"
            if latent_transition == _SYSTEMATIC_GIBBS
            else "draw_majority"
        ),
        "relative_history": [tuple(values) for values in relative_history],
        "trace": trace,
    }


def fit_bernoulli_csp_mcem(
    X,
    prop_in,
    B_list_in,
    A_list_in,
    **kwargs,
):
    """Compatibility wrapper for the Bernoulli CSP-MCEM kernel."""
    return fit_parametric_csp_mcem(
        X,
        prop_in,
        B_list_in,
        None,
        A_list_in,
        family=BERNOULLI_FAMILY,
        **kwargs,
    )


__all__ = [
    "_solve_gaussian_weighted_l1",
    "_solve_gaussian_weighted_l1_from_statistics",
    "_solve_logistic_weighted_l1",
    "_poisson_mc_design_and_response_statistic",
    "_poisson_mc_value_and_grad",
    "_solve_poisson_weighted_l1",
    "_canonicalize_csp_layer_order",
    "_apply_staged_parent_cap_only",
    "_reinitialize_staged_parent",
    "fit_bernoulli_csp_mcem",
    "fit_parametric_csp_mcem",
]
