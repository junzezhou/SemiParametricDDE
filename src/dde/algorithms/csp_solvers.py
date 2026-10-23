"""Weighted-L1 M-step solvers for the logistic and Poisson CSP layers.

Shared by the copula-structured CSP engine (:mod:`dde.algorithms.normal_csp`),
the MCEM CSP kernels (:mod:`dde.algorithms.parametric_csp_mcem`) and the
Bernoulli and Poisson family adapters, so that none of them depends on the
MCEM module for these solves.  The Monte Carlo draws enter through a design
cross-product cache; the logistic solver is the tempered weighted-L1 problem of
a binary child layer, the Poisson solver the log-link counterpart with a
backtracking step.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .parametric_families import (
    _largest_eigenvalue_psd,
    _poisson_optimization_exp_value,
    _poisson_optimization_rate,
    _soft_threshold,
)


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


__all__ = [
    "_logistic_mc_design_and_response_statistic",
    "_logistic_mc_value_and_grad",
    "_poisson_mc_design_and_response_statistic",
    "_poisson_mc_value_and_grad",
    "_solve_logistic_weighted_l1",
    "_solve_poisson_weighted_l1",
]
