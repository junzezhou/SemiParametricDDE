"""CSP-MAP approximate MCEM for a Normal-observed DDE.

This estimator follows the coordinate-ascent MCEM architecture used by the
package's CopDDE CSP path, without rank-likelihood augmentation.  It reuses the
vectorized Normal-DDE latent-probability sweep and solves the conditional
weighted-lasso M-steps with batched proximal gradient updates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from scipy.special import expit

from .csp import csp_active_mask, update_csp_layer
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


def _solve_gaussian_weighted_l1(
    X,
    A_sample,
    B_start,
    gamma,
    penalty_scale,
    temperature,
    *,
    max_iter=100,
    tol=1e-6,
    bound=10.0,
):
    """Batched FISTA update for the Normal observed layer and its variances."""
    X = np.asarray(X, dtype=float)
    B = np.asarray(B_start, dtype=float).copy()
    gamma = np.maximum(np.asarray(gamma, dtype=float).reshape(-1), 1e-10)
    penalty_scale = np.asarray(penalty_scale, dtype=float)
    C = int(A_sample.shape[2])
    Q, q_all = _normal_sample_statistics(X, A_sample)

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
            gradient = (
                float(temperature) / gamma[:, None]
            ) * (Y @ Q - q_all)
            candidate = Y - row_step[:, None] * gradient
            candidate[:, 0] = np.clip(candidate[:, 0], -bound, bound)
            candidate[:, 1:] = _soft_threshold(
                candidate[:, 1:],
                row_step[:, None] / penalty_scale,
            )
            candidate[:, 1:] = np.clip(candidate[:, 1:], -bound, bound)

            relative = np.linalg.norm(candidate - B) / (
                np.linalg.norm(B) + 1e-12
            )
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
            Y = candidate + ((momentum - 1.0) / next_momentum) * (
                candidate - B
            )
            B = candidate
            momentum = next_momentum
            if relative <= tol:
                break

    quadratic = np.einsum("jp,pq,jq->j", B, Q, B, optimize=True)
    cross = np.einsum("jp,jp->j", B, q_all, optimize=True)
    sse_average = np.maximum(np.sum(X**2, axis=0) - 2.0 * cross + quadratic, 0.0)
    sse = C * sse_average
    N = X.shape[0]
    gamma_update = (1.0 + 0.5 * sse) / (1.0 + 0.5 * N * C + 1.0)
    return B, np.maximum(gamma_update, 1e-10)


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

    design = np.transpose(parent_sample, (0, 2, 1)).reshape(
        N * C, parent_plus_intercept
    ).astype(float, copy=False)
    response = np.transpose(child_sample[:, 1:, :], (0, 2, 1)).reshape(
        N * C, n_children
    ).astype(float, copy=False)

    gram = design.T @ design
    lipschitz = (
        float(temperature) / C * 0.25 * _largest_eigenvalue_psd(gram)
    )
    if lipschitz <= 1e-14:
        B[:, 0] = 0.0
        B[:, 1:] = 0.0
        return B

    step = 1.0 / lipschitz
    Y = B.copy()
    momentum = 1.0
    for _ in range(int(max_iter)):
        logits = design @ Y.T
        gradient = (
            float(temperature) / C
        ) * ((expit(logits) - response).T @ design)
        candidate = Y - step * gradient
        candidate[:, 0] = np.clip(candidate[:, 0], -5.0, bound)
        candidate[:, 1:] = _soft_threshold(
            candidate[:, 1:], step / penalty_scale
        )
        candidate[:, 1:] = np.clip(candidate[:, 1:], -bound, bound)

        relative = np.linalg.norm(candidate - B) / (
            np.linalg.norm(B) + 1e-12
        )
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        Y = candidate + ((momentum - 1.0) / next_momentum) * (
            candidate - B
        )
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
    convergence_window: int = 7,
    burn: int = 5,
    optimizer_maxiter: int = 100,
    optimizer_tol: float = 1e-6,
    bound_val: float = 10.0,
    estep_row_chunk_size: int = 2048,
    random_state=None,
    keep_trace: bool = False,
    verbose: bool = True,
):
    """Fit maximal-width Normal DDE parameters under layer-wise CSP priors."""
    X = np.asarray(X, dtype=float)
    prop = np.asarray(prop_in, dtype=float).reshape(-1).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list_in]
    gamma = np.asarray(gamma_in, dtype=float).reshape(-1).copy()
    A_list = [np.asarray(A, dtype=np.uint8).copy() for A in A_list_in]
    D = len(B_list)
    K = [B.shape[1] - 1 for B in B_list]
    J = X.shape[1]

    C = int(C)
    max_iter = int(max_iter)
    convergence_window = int(convergence_window)
    burn = int(burn)
    optimizer_maxiter = int(optimizer_maxiter)
    temperature = float(temperature)
    threshold = float(threshold)
    if C < 1 or max_iter < 1:
        raise ValueError("C and max_iter must be positive integers.")
    if convergence_window < 1 or burn < 0 or optimizer_maxiter < 1:
        raise ValueError(
            "convergence_window and optimizer_maxiter must be positive, and burn nonnegative."
        )
    if not np.isfinite(temperature) or not 0.0 < temperature <= 1.0:
        raise ValueError("temperature must lie in (0, 1].")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive.")
    if not np.isfinite(convergence_tol) or convergence_tol < 0:
        raise ValueError("convergence_tol must be finite and nonnegative.")

    spike_scales = _validate_spike_scales(t_spike, D)
    rng = _as_rng(random_state)
    slab_scale = [np.ones(Kd, dtype=float) for Kd in K]
    omega = [np.ones(Kd + 1, dtype=float) / (Kd + 1) for Kd in K]
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

    iterations = 0
    while iterations < max_iter:
        probabilities = _compute_bernoulli_probabilities_normal(
            X,
            prop,
            B_list,
            gamma,
            A_list,
            row_chunk_size=estep_row_chunk_size,
            temperature=current_temperature,
        )
        A_samples = _sample_latent_layers(probabilities, C, rng)

        for d in range(D - 1, -1, -1):
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

        prop_update = np.mean(probabilities[-1], axis=0)
        B_update = [None] * D
        B_update[0], gamma_update = _solve_gaussian_weighted_l1(
            X,
            A_samples[0],
            B_list[0],
            gamma,
            penalty_scale[0],
            current_temperature,
            max_iter=optimizer_maxiter,
            tol=optimizer_tol,
            bound=bound_val,
        )
        for d in range(1, D):
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
        inactive_indices = []
        for d in range(D):
            active_masks[d] = csp_active_mask(allocation[d])
            inactive = np.flatnonzero(~active_masks[d])
            inactive_indices.append(inactive)
            num_active[d] = int(np.sum(active_masks[d]))

        for d in range(min(2, D)):
            slopes = B_update[d][:, 1:]
            slopes[np.abs(slopes) < threshold] = 0.0

        for d in range(D - 1):
            if inactive_indices[d].size:
                B_update[d + 1][inactive_indices[d], :] = 0.0

        if iterations > burn:
            current_temperature = min(
                1.0,
                current_temperature + 0.01 * ((iterations - burn) - 1),
            )

        stop_now = True
        current_relative = []
        for d in range(D):
            relative = np.linalg.norm(
                B_update[d][:, 1:] - B_list[d][:, 1:], ord="fro"
            ) / (np.linalg.norm(B_list[d][:, 1:], ord="fro") + 1e-6)
            relative_history[d].append(float(relative))
            current_relative.append(float(relative))
            if (
                len(relative_history[d]) < convergence_window
                or np.std(relative_history[d][-convergence_window:])
                >= convergence_tol
            ):
                stop_now = False

        if keep_trace:
            trace.append(
                {
                    "iteration": iterations,
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

        if verbose:
            print(
                f"CSP-MCEM iteration {iterations}, active={num_active.tolist()}, "
                f"max relative change={max(current_relative):.4g}"
            )
        if stop_now:
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
        "temperature": float(current_temperature),
        "relative_history": [tuple(values) for values in relative_history],
        "trace": trace,
    }


__all__ = [
    "_solve_gaussian_weighted_l1",
    "_solve_logistic_weighted_l1",
    "fit_normal_csp_mcem",
]
