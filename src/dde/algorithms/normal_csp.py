"""CSP-MAP MCEM kernels for a Normal-observed DDE.

The original simultaneous kernel follows the coordinate-ascent architecture
used by the package's CopDDE CSP path.  Two staged variants share that kernel
and differ only in whether a waiting parent is rebuilt or capped while its
eligible state is preserved.  All variants work directly on Normal data,
without rank-likelihood augmentation, and use batched proximal-gradient
weighted-lasso updates.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .csp import csp_active_mask, update_csp_layer
from .normal_init_semi import initialize_binary_parent
from .semi_saem import (
    _compute_bernoulli_probabilities_normal,
    _normal_sample_statistics,
)


def _as_rng(random_state=None):
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _soft_threshold(x, threshold):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


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


def _largest_eigenvalue_psd(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0.0
    return max(float(np.linalg.eigvalsh(matrix)[-1]), 0.0)


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
    B_new[np.ix_(child_indices, np.arange(1, initial_width + 1))] = B_small[:, 1:]
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


def _solve_gaussian_weighted_l1_from_statistics(
    X,
    Q,
    q_all,
    B_start,
    gamma,
    penalty_scale,
    temperature,
    *,
    mc_samples=1,
    x_sq_sum=None,
    max_iter=100,
    tol=1e-6,
    bound=10.0,
):
    """Batched Normal-layer update from MC or SA sufficient statistics."""
    X = np.asarray(X, dtype=float)
    Q = np.asarray(Q, dtype=float)
    q_all = np.asarray(q_all, dtype=float)
    B = np.asarray(B_start, dtype=float).copy()
    gamma = np.maximum(np.asarray(gamma, dtype=float).reshape(-1), 1e-10)
    penalty_scale = np.asarray(penalty_scale, dtype=float)
    C = int(mc_samples)

    largest = _largest_eigenvalue_psd(Q)
    if largest <= 1e-14:
        B[:, 0] = np.clip(np.mean(X, axis=0), -bound, bound)
        B[:, 1:] = 0.0
    else:
        row_step = gamma / (float(temperature) * largest)
        row_step = np.maximum(row_step, 1e-14)
        Y = B.copy()
        momentum = 1.0
        for _ in range(int(max_iter)):
            gradient = (float(temperature) / gamma[:, None]) * (Y @ Q - q_all)
            candidate = Y - row_step[:, None] * gradient
            candidate[:, 0] = np.clip(candidate[:, 0], -bound, bound)
            candidate[:, 1:] = _soft_threshold(
                candidate[:, 1:],
                row_step[:, None] / penalty_scale,
            )
            candidate[:, 1:] = np.clip(candidate[:, 1:], -bound, bound)

            relative = np.linalg.norm(candidate - B) / (np.linalg.norm(B) + 1e-12)
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
            Y = candidate + ((momentum - 1.0) / next_momentum) * (candidate - B)
            B = candidate
            momentum = next_momentum
            if relative <= tol:
                break

    quadratic = np.einsum("jp,pq,jq->j", B, Q, B, optimize=True)
    cross = np.einsum("jp,jp->j", B, q_all, optimize=True)
    if x_sq_sum is None:
        x_sq_sum = np.sum(X**2, axis=0)
    else:
        x_sq_sum = np.asarray(x_sq_sum, dtype=float).reshape(-1)
        if x_sq_sum.shape != (X.shape[1],):
            raise ValueError("x_sq_sum must have one entry per observed column.")
    sse_average = np.maximum(x_sq_sum - 2.0 * cross + quadratic, 0.0)
    sse = C * sse_average
    N = X.shape[0]
    gamma_update = (1.0 + 0.5 * sse) / (1.0 + 0.5 * N * C + 1.0)
    return B, np.maximum(gamma_update, 1e-10)


def _solve_gaussian_weighted_l1(
    X,
    A_sample,
    B_start,
    gamma,
    penalty_scale,
    temperature,
    *,
    x_sq_sum=None,
    max_iter=100,
    tol=1e-6,
    bound=10.0,
):
    """Batched FISTA update for one current Normal Monte Carlo sample."""
    Q, q_all = _normal_sample_statistics(X, A_sample)
    return _solve_gaussian_weighted_l1_from_statistics(
        X,
        Q,
        q_all,
        B_start,
        gamma,
        penalty_scale,
        temperature,
        mc_samples=A_sample.shape[2],
        x_sq_sum=x_sq_sum,
        max_iter=max_iter,
        tol=tol,
        bound=bound,
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
):
    """Batched FISTA update for one conditional Bernoulli layer."""
    child_sample = np.asarray(child_sample)
    parent_sample = np.asarray(parent_sample)
    B = np.asarray(B_start, dtype=float).copy()
    penalty_scale = np.asarray(penalty_scale, dtype=float)
    N, parent_plus_intercept, C = parent_sample.shape
    n_children = child_sample.shape[1] - 1

    design = (
        np.transpose(parent_sample, (0, 2, 1))
        .reshape(N * C, parent_plus_intercept)
        .astype(float, copy=False)
    )
    response = (
        np.transpose(child_sample[:, 1:, :], (0, 2, 1))
        .reshape(N * C, n_children)
        .astype(float, copy=False)
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
        logits = design @ Y.T
        gradient = (float(temperature) / C) * ((expit(logits) - response).T @ design)
        candidate = Y - step * gradient
        candidate[:, 0] = np.clip(candidate[:, 0], -5.0, bound)
        candidate[:, 1:] = _soft_threshold(candidate[:, 1:], step / penalty_scale)
        candidate[:, 1:] = np.clip(candidate[:, 1:], -bound, bound)

        relative = np.linalg.norm(candidate - B) / (np.linalg.norm(B) + 1e-12)
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        Y = candidate + ((momentum - 1.0) / next_momentum) * (candidate - B)
        B = candidate
        momentum = next_momentum
        if relative <= tol:
            break
    return B


def fit_normal_csp_mcem(
    X,
    prop_in,
    B_list_in,
    gamma_in,
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
):
    """Fit maximal-width Normal DDE parameters under layer-wise CSP priors.

    ``stagewise`` activates CSP from the observed-side layer toward the
    deepest layer and advances only after the current layer's width has been
    stable for a full window.  ``stage_transition_mode`` selects whether a
    waiting parent is rebuilt, only capped while preserving its state, or
    left untouched.
    """
    X = np.asarray(X, dtype=float)
    prop = np.asarray(prop_in, dtype=float).reshape(-1).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list_in]
    gamma = np.asarray(gamma_in, dtype=float).reshape(-1).copy()
    A_list = [np.asarray(A, dtype=np.uint8).copy() for A in A_list_in]
    D = len(B_list)
    K = [B.shape[1] - 1 for B in B_list]
    J = X.shape[1]
    x_sq_sum = np.sum(X**2, axis=0)

    C = int(C)
    max_iter = int(max_iter)
    convergence_window = int(convergence_window)
    stage_min_iter = int(stage_min_iter)
    stage_width_window = int(stage_width_window)
    burn = int(burn)
    optimizer_maxiter = int(optimizer_maxiter)
    temperature = float(temperature)
    threshold = float(threshold)
    convergence_tol = float(convergence_tol)
    convergence_mean_tol = float(convergence_mean_tol)
    stage_transition_mode = str(stage_transition_mode).strip().lower()
    if stage_transition_mode not in {"none", "reinitialize", "cap_only"}:
        raise ValueError(
            "stage_transition_mode must be 'none', 'reinitialize', or " "'cap_only'."
        )
    if not stagewise and stage_transition_mode != "none":
        raise ValueError("stage_transition_mode requires stagewise=True.")
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
        or optimizer_maxiter < 1
    ):
        raise ValueError(
            "convergence_window, stage_min_iter, stage_width_window, and "
            "optimizer_maxiter must be positive, and burn nonnegative."
        )
    if not np.isfinite(temperature) or not 0.0 < temperature <= 1.0:
        raise ValueError("temperature must lie in (0, 1].")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive.")
    if not np.isfinite(convergence_tol) or convergence_tol < 0:
        raise ValueError("convergence_tol must be finite and nonnegative.")
    if not np.isfinite(convergence_mean_tol) or convergence_mean_tol < 0:
        raise ValueError("convergence_mean_tol must be finite and nonnegative.")
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
        raise ValueError("penalty_strength must be a scalar or a length-D sequence.")
    if not np.all(np.isfinite(penalty_strengths)) or np.any(penalty_strengths <= 0):
        raise ValueError("penalty_strength entries must be finite and positive.")
    rng = _as_rng(random_state)
    slab_scale = [np.ones(Kd, dtype=float) for Kd in K]
    omega = [np.ones(Kd + 1, dtype=float) / (Kd + 1) for Kd in K]
    if stagewise:
        allocation = [np.full(Kd, Kd + 1, dtype=int) for Kd in K]
        penalty_scale = [
            np.ones((J if d == 0 else K[d - 1], K[d]), dtype=float) for d in range(D)
        ]
    else:
        allocation = [np.full(Kd, np.floor(Kd / 2), dtype=int) for Kd in K]
        penalty_scale = [
            np.full((J if d == 0 else K[d - 1], K[d]), spike_scales[d])
            for d in range(D)
        ]
    active_masks = [np.ones(Kd, dtype=bool) for Kd in K]
    num_active = np.asarray(K, dtype=int)
    relative_history = [[] for _ in range(D)]
    trace = [] if keep_trace else None
    current_temperature = temperature
    active_stage = 0 if stagewise else D - 1
    stage_iteration = 0
    stage_width_history = []
    stage_transitions = []
    stage_transition_events = []
    stage_caps = np.asarray(K, dtype=int)
    stopped_by_convergence = False

    iterations = 0
    while iterations < max_iter:
        iteration_stage = active_stage
        probabilities = _compute_bernoulli_probabilities_normal(
            X,
            prop,
            B_list,
            gamma,
            A_list,
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

        for d in range(iteration_stage, -1, -1):
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
            penalty_scale[d] /= penalty_strengths[d]
            if stage_caps[d] < K[d]:
                ineligible = np.arange(K[d]) >= stage_caps[d]
                allocation[d][ineligible] = np.arange(1, K[d] + 1)[ineligible]
                penalty_scale[d][:, ineligible] = spike_scales[d]

        if stagewise and stage_freeze_deeper and iteration_stage < D - 1:
            prop_update = prop.copy()
        else:
            prop_update = np.mean(probabilities[-1], axis=0)
        B_update = [None] * D
        B_update[0], gamma_update = _solve_gaussian_weighted_l1(
            X,
            A_samples[0],
            B_list[0],
            gamma,
            penalty_scale[0],
            current_temperature,
            x_sq_sum=x_sq_sum,
            max_iter=optimizer_maxiter,
            tol=optimizer_tol,
            bound=bound_val,
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
                or np.std(relative_history[d][-convergence_window:]) >= convergence_tol
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
                }
            )

        A_list = [
            (np.mean(sample[:, 1:, :], axis=2) > 0.5).astype(np.uint8)
            for sample in A_samples
        ]
        B_list = [B.copy() for B in B_update]
        gamma = gamma_update.copy()
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
        "gamma": gamma,
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
        "stage_transition_mode": stage_transition_mode,
        "stage_spectral_gap_threshold": stage_spectral_gap_threshold,
        "stage_transition_events": tuple(stage_transition_events),
        "stage_caps": tuple(int(value) for value in stage_caps),
        "stopped_by_convergence": bool(stopped_by_convergence),
        "temperature": float(current_temperature),
        "relative_history": [tuple(values) for values in relative_history],
        "trace": trace,
    }


__all__ = [
    "_solve_gaussian_weighted_l1",
    "_solve_gaussian_weighted_l1_from_statistics",
    "_solve_logistic_weighted_l1",
    "_canonicalize_csp_layer_order",
    "_apply_staged_parent_cap_only",
    "_reinitialize_staged_parent",
    "fit_normal_csp_mcem",
]
