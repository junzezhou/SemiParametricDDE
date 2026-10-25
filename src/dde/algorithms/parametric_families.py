"""Observed-family adapters for parametric DDE estimators.

The latent hierarchy is common to every parametric DDE family.  Adapters in
this module own only observed-layer behavior: data validation and
initialization, the observed contribution to latent conditional log-odds,
observed-layer M-steps, dispersion semantics, scoring, and simulation.

Normal, Bernoulli, and Poisson adapters share the fixed-width SAEM and the
copula CSP control paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from dde.utils import TLP, thres

from .bernoulli_init import Bernoulli_init, _bernoulli_working_response
from .normal_init_semi import Normal_init_semi
from .poisson_init import Poisson_init, _poisson_working_response


def _tlp_penalty(x, pen, tau):
    """Return the paper/MATLAB TLP penalty ``pen * sum(min(abs(x), tau))``."""
    return float(pen) * TLP(x, tau)


def _tlp_penalty_value_grad(x, pen, tau):
    """Return the TLP value and a valid gradient away from its knots."""
    x = np.asarray(x, dtype=float)
    value = _tlp_penalty(x, pen, tau)
    grad = np.zeros_like(x)
    active = (np.abs(x) < tau) & (x != 0.0)
    grad[active] = float(pen) * np.sign(x[active])
    return value, grad


def _solve_bernoulli_tlp_rowwise(
    accumulator,
    B_start,
    penalty,
    tau,
    bounds,
    *,
    max_iter=10,
):
    """Reference implementation of the original per-response L-BFGS-B."""
    B_start = np.asarray(B_start, dtype=float)
    B_update = np.zeros_like(B_start)
    for response_index in range(B_start.shape[0]):

        def objective(x, __index=response_index):
            value, gradient = accumulator.value_and_grad(__index, x)
            penalty_value, penalty_gradient = _tlp_penalty_value_grad(
                x[1:], penalty, tau
            )
            gradient[1:] += penalty_gradient
            return value + penalty_value, gradient

        result = minimize(
            objective,
            B_start[response_index],
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )
        B_update[response_index] = result.x
    return B_update


def _solve_bernoulli_tlp_batched(
    accumulator,
    B_start,
    penalty,
    tau,
    bounds,
    *,
    max_iter=10,
    row_chunk_size=2048,
):
    """Joint L-BFGS-B evaluation of separable Bernoulli TLP objectives.

    This uses the same likelihood, TLP parameterization, coefficient bounds,
    and iteration cap as ``_solve_bernoulli_tlp_rowwise``.  Combining the
    independent response objectives allows their logits and gradients to be
    evaluated by chunked matrix multiplications and removes one optimizer
    invocation per response.
    """
    B_start = np.asarray(B_start, dtype=float)
    if B_start.ndim != 2:
        raise ValueError("Bernoulli coefficient start must be a matrix.")
    bounds = list(bounds)
    if len(bounds) != B_start.shape[1]:
        raise ValueError(
            "Bernoulli coefficient bounds have an incompatible width."
        )
    shape = B_start.shape

    def objective(flattened):
        coefficients = np.asarray(flattened, dtype=float).reshape(shape)
        value, gradient = accumulator.batched_value_and_grad(
            coefficients, row_chunk_size=row_chunk_size
        )
        penalty_value, penalty_gradient = _tlp_penalty_value_grad(
            coefficients[:, 1:], penalty, tau
        )
        gradient[:, 1:] += penalty_gradient
        return value + penalty_value, gradient.ravel()

    result = minimize(
        objective,
        B_start.ravel(),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds * B_start.shape[0],
        options={"maxiter": int(max_iter)},
    )
    return result.x.reshape(shape)


def _solve_poisson_tlp_rowwise(
    accumulator,
    B_start,
    penalty,
    tau,
    bounds,
    *,
    max_iter=10,
):
    """Fit independent Poisson responses with the MATLAB TLP objective."""
    B_start = np.asarray(B_start, dtype=float)
    B_update = np.zeros_like(B_start)
    for response_index in range(B_start.shape[0]):

        def objective(x, __index=response_index):
            value, gradient = accumulator.value_and_grad(__index, x)
            penalty_value, penalty_gradient = _tlp_penalty_value_grad(
                x[1:], penalty, tau
            )
            gradient[1:] += penalty_gradient
            return value + penalty_value, gradient

        result = minimize(
            objective,
            B_start[response_index],
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )
        B_update[response_index] = result.x
    return B_update


def _solve_poisson_tlp_active_set(
    accumulator,
    B_start,
    penalty,
    tau,
    bounds,
    *,
    max_rounds=3,
    max_iter=50,
    max_support=12,
    max_added=4,
    newton_screen=True,
):
    """Row-wise TLP Poisson M-step restricted to each row's active support.

    Same objective and penalty as ``_solve_poisson_tlp_rowwise``.  For row
    ``j`` the coefficients outside the current support ``S_j`` are held at
    zero and the problem over ``(intercept, x_S)`` is solved exactly with
    L-BFGS-B on the sub-pattern weights of ``S_j`` (``2**|S_j|`` terms
    instead of one term per stored pattern).  The full gradient at the
    solution is then evaluated from the cross table; a coordinate at zero
    whose gradient exceeds the TLP slope ``penalty`` violates the optimality
    condition.  Because the sweep ends with the hard threshold at ``tau``, a
    violating coordinate is added to the support (the ``max_added`` largest)
    only if its one-step Newton estimate ``(|g_k| - penalty) / H_kk`` would
    exceed ``tau``; the row is then re-solved.  Each round costs one pass
    over the patterns, against one pass per optimizer evaluation in the dense
    solver.
    """
    B_start = np.asarray(B_start, dtype=float)
    J, n_features = B_start.shape
    K = n_features - 1
    penalty = float(penalty)
    bounds = list(bounds)
    B_update = np.zeros_like(B_start)
    for j in range(J):
        row = B_start[j].copy()
        support = np.flatnonzero(row[1:] != 0.0)
        if support.size > max_support:
            support = support[np.argsort(-np.abs(row[1:][support]))[:max_support]]
        x_full = row.copy()
        for _ in range(int(max_rounds)):
            support = np.sort(support)
            weights, cross = accumulator.support_tables(support)
            active = np.flatnonzero(weights > 0.0)
            w_active = weights[active]
            bits = ((active[:, None] >> np.arange(support.size)[None, :]) & 1).astype(float)
            design_s = np.column_stack([np.ones(active.size), bits])
            q_full = accumulator.q_all[j]
            q_s = np.concatenate([[q_full[0]], q_full[1:][support]])
            sub_bounds = [bounds[0]] + [bounds[k + 1] for k in support]

            def objective(z):
                eta = design_s @ z
                rate = _poisson_optimization_rate(eta)
                excess = np.maximum(eta - _POISSON_OPTIMIZATION_EXP_LIMIT, 0.0)
                value = float(w_active @ (rate * (1.0 + excess)) - q_s @ z)
                gradient = design_s.T @ (w_active * rate) - q_s
                pen_value, pen_grad = _tlp_penalty_value_grad(z[1:], penalty, tau)
                gradient[1:] += pen_grad
                return value + pen_value, gradient

            z0 = np.concatenate([[x_full[0]], x_full[1:][support]])
            z = minimize(
                objective,
                z0,
                method="L-BFGS-B",
                jac=True,
                bounds=sub_bounds,
                options={"maxiter": int(max_iter)},
            ).x
            x_full = np.zeros(n_features, dtype=float)
            x_full[0] = z[0]
            x_full[1:][support] = z[1:]
            # full gradient and diagonal curvature of the data term at
            # x_full, from the cross table (d_pk^2 = d_pk for binary designs)
            rate_s = _poisson_optimization_rate(design_s @ z)
            curvature_full = (w_active * rate_s) @ cross[active]
            gradient_full = curvature_full - q_full
            inactive = np.ones(K, dtype=bool)
            inactive[support] = False
            slope_grad = gradient_full[1:]
            excess = np.abs(slope_grad) - penalty
            newton_step = excess / np.maximum(curvature_full[1:], 1e-12)
            if newton_screen:
                violation = np.where(excess > 0.0, newton_step - float(tau), -np.inf)
            else:
                violation = excess.copy()
            violation[~inactive] = -np.inf
            # a zero coordinate must also be able to move inside its bounds
            lower = np.array([b[0] for b in bounds[1:]])
            upper = np.array([b[1] for b in bounds[1:]])
            violation[(slope_grad > 0) & (lower >= 0.0)] = -np.inf
            violation[(slope_grad < 0) & (upper <= 0.0)] = -np.inf
            candidates = np.flatnonzero(violation > 0.0)
            if candidates.size == 0 or support.size >= max_support:
                break
            candidates = candidates[np.argsort(-violation[candidates])[:max_added]]
            room = max_support - support.size
            support = np.concatenate([support, candidates[:room]])
        B_update[j] = x_full
    return B_update


def _matlab_debias(x, tau):
    """Match the deep-Normal MATLAB ``debias.m`` elementwise (reference only; the kernel hard-thresholds)."""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    out[x > tau] = x[x > tau] + tau
    out[x < tau] = x[x < tau] - tau
    out[np.abs(x) < tau] = 0.0
    return out


class _ObservedNormalAccumulator:
    """Layer-shared running quadratic statistics for Normal responses."""

    def __init__(self, n_responses, n_features):
        self.Q = np.zeros((n_features, n_features), dtype=float)
        self.q_all = np.zeros((n_responses, n_features), dtype=float)

    def update(self, X, A_samples, t):
        alpha = 1.0 / t
        Q_new, q_new = _normal_sample_statistics(X, A_samples)
        self.Q = (1.0 - alpha) * self.Q + alpha * Q_new
        self.q_all = (1.0 - alpha) * self.q_all + alpha * q_new
        return Q_new, q_new

    def value(self, response_index, x):
        q = self.q_all[response_index]
        return float(x @ self.Q @ x - 2.0 * q @ x)

    def value_and_grad(self, response_index, x):
        q = self.q_all[response_index]
        value = float(x @ self.Q @ x - 2.0 * q @ x)
        grad = 2.0 * (self.Q @ x - q)
        return value, grad


_POISSON_OPTIMIZATION_EXP_LIMIT = 500.0


def _poisson_optimization_rate(eta):
    """Exponentiate natural parameters with a finite optimization guard."""
    eta = np.asarray(eta, dtype=float)
    return np.exp(np.minimum(eta, _POISSON_OPTIMIZATION_EXP_LIMIT))


def _poisson_optimization_exp_value(eta):
    """Return a finite convex continuation of ``exp`` above the guard."""
    eta = np.asarray(eta, dtype=float)
    rate = _poisson_optimization_rate(eta)
    excess = np.maximum(eta - _POISSON_OPTIMIZATION_EXP_LIMIT, 0.0)
    return rate * (1.0 + excess)


class _ObservedPoissonAccumulator:
    """SA statistics for conditionally independent Poisson responses.

    Binary Monte Carlo designs are aggregated by unique parent pattern.  The
    response-dependent linear term is retained as a conventional running
    sufficient statistic, so its storage does not grow with the number of
    observed variables or SAEM iterations.
    """

    def __init__(self, n_responses, n_features):
        self.n_responses = int(n_responses)
        self.n_features = int(n_features)
        self.parent_width = self.n_features - 1
        self.packed_width = (self.parent_width + 7) // 8
        self._packed_patterns = np.empty(
            (0, self.packed_width), dtype=np.uint8
        )
        self.counts = np.empty(0, dtype=float)
        self.q_all = np.zeros((self.n_responses, self.n_features), dtype=float)
        self._design = None
        self._bits = None
        self._bits_float = None

    @property
    def design(self):
        if self._design is None:
            patterns = np.unpackbits(
                self._packed_patterns,
                axis=1,
                count=self.parent_width,
                bitorder="little",
            )
            self._design = np.column_stack(
                [np.ones(self.pattern_count, dtype=float), patterns]
            )
        return self._design

    @property
    def patterns(self):
        """Return unpacked patterns for compatibility and diagnostics."""
        return np.unpackbits(
            self._packed_patterns,
            axis=1,
            count=self.parent_width,
            bitorder="little",
        )

    @property
    def pattern_count(self):
        return int(self._packed_patterns.shape[0])

    @property
    def storage_nbytes(self):
        return int(
            self._packed_patterns.nbytes
            + self.counts.nbytes
            + self.q_all.nbytes
            + (0 if self._design is None else self._design.nbytes)
        )

    def update(self, X, A_samples, t):
        X = np.asarray(X, dtype=float)
        A_samples = np.asarray(A_samples, dtype=np.uint8)
        if X.ndim != 2 or X.shape[1] != self.n_responses:
            raise ValueError("Poisson responses have an incompatible shape.")
        if (
            A_samples.ndim != 3
            or A_samples.shape[0] != X.shape[0]
            or A_samples.shape[1] != self.n_features
        ):
            raise ValueError(
                "Poisson Monte Carlo designs have an incompatible shape."
            )
        t = int(t)
        if t < 1:
            raise ValueError("Poisson accumulator iteration must be positive.")
        alpha = 1.0 / t
        C = A_samples.shape[2]
        parents = np.moveaxis(A_samples[:, 1:, :], 1, 2).reshape(
            -1, self.parent_width
        )
        packed = np.packbits(
            np.ascontiguousarray(parents, dtype=np.uint8),
            axis=1,
            bitorder="little",
        )
        new_patterns, inverse = np.unique(packed, axis=0, return_inverse=True)
        new_counts = np.bincount(inverse).astype(float) / C

        if self._packed_patterns.size and alpha < 1.0:
            stacked = np.vstack([self._packed_patterns, new_patterns])
            all_patterns, positions = np.unique(
                stacked, axis=0, return_inverse=True
            )
            old_positions = positions[: self.pattern_count]
            new_positions = positions[self.pattern_count :]
            merged_counts = np.zeros(all_patterns.shape[0], dtype=float)
            merged_counts[old_positions] = (1.0 - alpha) * self.counts
            merged_counts[new_positions] += alpha * new_counts
            self._packed_patterns = all_patterns.astype(np.uint8, copy=False)
            self.counts = merged_counts
        else:
            self._packed_patterns = new_patterns.astype(np.uint8, copy=False)
            self.counts = alpha * new_counts

        q_new = X.T @ np.mean(A_samples, axis=2)
        self.q_all = (1.0 - alpha) * self.q_all + alpha * q_new
        self._design = None
        self._bits = None
        self._bits_float = None

    @property
    def bits(self):
        """Unpacked parent patterns (uint8, patterns x parent_width), cached."""
        if self._bits is None:
            self._bits = np.unpackbits(
                self._packed_patterns,
                axis=1,
                count=self.parent_width,
                bitorder="little",
            )
        return self._bits

    def support_tables(self, support, *, row_chunk_size=65536):
        """Aggregate the pattern weights by the sub-pattern on ``support``.

        Returns ``(weights, cross)``: ``weights[s]`` is the total weight of
        the patterns whose restriction to the latent coordinates in
        ``support`` has code ``s`` (bit ``i`` = coordinate ``support[i]``),
        and ``cross[s, :]`` is the weighted sum of the full design rows
        (intercept column first) of those patterns.  For a coefficient
        vector supported on ``support`` the objective depends on the patterns
        only through ``weights``, and the full gradient only through
        ``cross``, so one pass over the patterns replaces one objective
        evaluation per optimizer step.
        """
        from scipy.sparse import csr_matrix

        support = np.asarray(support, dtype=np.intp).reshape(-1)
        n_sub = 1 << support.size
        bits = self.bits
        n_patterns = bits.shape[0]
        code_weights = (np.int64(1) << np.arange(support.size, dtype=np.int64))
        codes = bits[:, support].astype(np.int64) @ code_weights
        weights = np.bincount(codes, weights=self.counts, minlength=n_sub)
        cross = np.zeros((n_sub, self.n_features), dtype=float)
        cross[:, 0] = weights
        bits_float = self.bits_float
        for start in range(0, n_patterns, int(row_chunk_size)):
            stop = min(start + int(row_chunk_size), n_patterns)
            grouping = csr_matrix(
                (self.counts[start:stop], (codes[start:stop], np.arange(stop - start))),
                shape=(n_sub, stop - start),
            )
            cross[:, 1:] += grouping @ bits_float[start:stop]
        return weights, cross

    @property
    def bits_float(self):
        """Float copy of :attr:`bits`, cached per update for the sparse products."""
        if self._bits_float is None:
            self._bits_float = self.bits.astype(float)
        return self._bits_float

    def value(self, response_index, x):
        x = np.asarray(x, dtype=float)
        eta = self.design @ x
        exp_value = _poisson_optimization_exp_value(eta)
        return float(
            self.counts @ exp_value - self.q_all[int(response_index)] @ x
        )

    def value_and_grad(self, response_index, x):
        response_index = int(response_index)
        x = np.asarray(x, dtype=float)
        eta = self.design @ x
        rate = _poisson_optimization_rate(eta)
        excess = np.maximum(eta - _POISSON_OPTIMIZATION_EXP_LIMIT, 0.0)
        exp_value = rate * (1.0 + excess)
        q = self.q_all[response_index]
        value = float(self.counts @ exp_value - q @ x)
        gradient = self.design.T @ (self.counts * rate) - q
        return value, gradient


def _normal_sample_statistics(X, A_samples):
    """Return current-MC Normal design cross-products, averaged over draws."""
    X = np.asarray(X, dtype=float)
    A_samples = np.asarray(A_samples, dtype=float)
    C = A_samples.shape[2]
    Q = np.einsum("nkc,nlc->kl", A_samples, A_samples, optimize=True) / C
    q_all = X.T @ np.mean(A_samples, axis=2)
    return Q, q_all


def _update_deep_normal_observed(X, Q, q_all, tau, *, x_sq_sum=None):
    """MATLAB D>=3 closed-form observed-layer update and variances."""
    X = np.asarray(X, dtype=float)
    try:
        coefficients_t = np.linalg.solve(Q, q_all.T)
    except np.linalg.LinAlgError:
        coefficients_t = np.linalg.lstsq(Q, q_all.T, rcond=1e-10)[0]
    B_update = coefficients_t.T
    B_update[:, 1:] = thres(B_update[:, 1:], tau)

    quadratic = np.einsum("jp,pq,jq->j", B_update, Q, B_update, optimize=True)
    cross = np.einsum("jp,jp->j", B_update, q_all, optimize=True)
    if x_sq_sum is None:
        x_sq_sum = np.sum(X**2, axis=0)
    residual_sum_squares = x_sq_sum - 2.0 * cross + quadratic
    gamma_update = residual_sum_squares / X.shape[0]
    return B_update, gamma_update


def _soft_threshold(x, threshold):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _binary_child_conditional_logit(
    children,
    B,
    parents,
    *,
    parent_width,
    child_width=None,
    row_chunk_size=2048,
    coordinate_chunk_size=1,
    sparse=False,
):
    """Return Bernoulli-child likelihood odds for each parent coordinate.

    The sparse path evaluates each coordinate only on children whose slope is
    nonzero.  It is algebraically identical to the dense flip-state formula,
    but reuses cached ``children @ slope`` terms and avoids allocating dense
    shifted predictors for zero-loading children.
    """
    children = np.asarray(children)
    parents = np.asarray(parents)
    B = np.asarray(B, dtype=float)
    N = children.shape[0]
    parent_width = int(parent_width)
    if child_width is None:
        child_width = children.shape[1]
    child_width = int(child_width)
    if row_chunk_size is None:
        row_chunk_size = N
    row_chunk_size = max(1, int(row_chunk_size))
    coordinate_chunk_size = max(1, int(coordinate_chunk_size))
    B_active = B[:child_width, : parent_width + 1]
    like_logit = np.empty((N, parent_width), dtype=float)

    if sparse == "auto":
        slopes = B_active[:, 1:]
        sparse = np.count_nonzero(slopes) <= slopes.size / 2
    elif not isinstance(sparse, (bool, np.bool_)):
        raise ValueError("sparse must be a boolean or 'auto'.")

    if sparse:
        active_children = [
            np.flatnonzero(B_active[:, coordinate + 1] != 0.0)
            for coordinate in range(parent_width)
        ]
        child_linear = np.zeros((N, parent_width), dtype=float)
        for coordinate, indices in enumerate(active_children):
            if indices.size:
                child_linear[:, coordinate] = (
                    children[:, indices] @ B_active[indices, coordinate + 1]
                )

    for start in range(0, N, row_chunk_size):
        stop = min(start + row_chunk_size, N)
        parent_chunk = parents[start:stop, :parent_width]
        base_eta = B_active[:, 0][None, :] + parent_chunk @ B_active[:, 1:].T
        if sparse:
            for block_start in range(0, parent_width, coordinate_chunk_size):
                block_stop = min(
                    block_start + coordinate_chunk_size, parent_width
                )
                for coordinate in range(block_start, block_stop):
                    indices = active_children[coordinate]
                    if not indices.size:
                        like_logit[start:stop, coordinate] = 0.0
                        continue
                    slope = B_active[indices, coordinate + 1]
                    eta_zero = (
                        base_eta[:, indices]
                        - parent_chunk[:, coordinate, None] * slope[None, :]
                    )
                    normalizer_delta = np.logaddexp(
                        0.0, eta_zero + slope[None, :]
                    ) - np.logaddexp(0.0, eta_zero)
                    like_logit[start:stop, coordinate] = child_linear[
                        start:stop, coordinate
                    ] - np.sum(normalizer_delta, axis=1)
            continue

        base_softplus = np.logaddexp(0.0, base_eta)
        child_linear = children[start:stop, :child_width] @ B_active[:, 1:]
        for block_start in range(0, parent_width, coordinate_chunk_size):
            block_stop = min(block_start + coordinate_chunk_size, parent_width)
            slopes_block = B_active[:, block_start + 1 : block_stop + 1].T
            direction = 1.0 - 2.0 * parent_chunk[
                :, block_start:block_stop
            ].astype(float)
            if block_stop - block_start == 1:
                shifted_eta = direction[:, :1] * slopes_block[0][None, :]
                shifted_eta += base_eta
            else:
                shifted_eta = direction[:, :, None] * slopes_block[None, :, :]
                shifted_eta += base_eta[:, None, :]
            np.logaddexp(0.0, shifted_eta, out=shifted_eta)
            if block_stop - block_start == 1:
                shifted_eta -= base_softplus
                log_normalizer_delta = direction[:, 0] * np.sum(
                    shifted_eta, axis=1
                )
            else:
                shifted_eta -= base_softplus[:, None, :]
                log_normalizer_delta = direction * np.sum(shifted_eta, axis=2)
            like_logit[start:stop, block_start:block_stop] = child_linear[
                :, block_start:block_stop
            ] - np.asarray(log_normalizer_delta).reshape(
                stop - start, block_stop - block_start
            )
    return like_logit


def _largest_eigenvalue_psd(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0.0
    return max(float(np.linalg.eigvalsh(matrix)[-1]), 0.0)


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
    if x_sq_sum is None:
        x_sq_sum = np.sum(X**2, axis=0)
    else:
        x_sq_sum = np.asarray(x_sq_sum, dtype=float).reshape(-1)
        if x_sq_sum.shape != (X.shape[1],):
            raise ValueError(
                "x_sq_sum must have one entry per observed column."
            )
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


@dataclass(frozen=True)
class NormalFamilyAdapter:
    """Observed-layer operations for a conditionally Normal DDE."""

    name: str = "Normal"
    key: str = "normal"
    aliases: tuple[str, ...] = ("normal", "gaussian")
    supported_dimension_methods: tuple[str, ...] = ("fixed", "csp")
    has_dispersion: bool = True
    dispersion_name: str = "conditional_variance"

    def inverse_mean_link(self, mean, *, epsilon=None):
        del epsilon
        return np.asarray(mean, dtype=float)

    def initial_working_response(self, Y, *, epsilon, K1=None) -> np.ndarray:
        """Return the response used by the shared spectral initializer.

        Normal uses the identity mean link.  Discrete-family adapters can
        instead return a stabilized inverse-mean-link response while keeping
        the latent spectral initialization outside the fitting runners.
        """
        del epsilon, K1
        return self.validate_y(Y).copy()

    def validate_y(self, Y) -> np.ndarray:
        X = np.asarray(Y, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"Y must be a two-dimensional matrix; got shape {X.shape}."
            )
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError(
                "Y must contain at least one observation and one variable."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("Normal DDE requires finite values in Y.")
        return X

    def validate_dispersion(self, dispersion, n_responses: int) -> np.ndarray:
        values = np.asarray(dispersion, dtype=float).reshape(-1)
        if values.shape != (int(n_responses),):
            raise ValueError(
                f"gamma shape {values.shape} must be ({int(n_responses)},)."
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("gamma entries must be finite and positive.")
        return values

    def initialize(
        self,
        X,
        K_list,
        *,
        epsilon,
        mode,
        B_list_true=None,
        G_list_true=None,
        random_state=None,
        binarize="projection",
    ):
        del random_state
        working_response = self.initial_working_response(X, epsilon=epsilon)
        if mode == "data":
            return Normal_init_semi(
                working_response,
                K_list[0],
                K_list,
                epsilon=epsilon,
                mode="data",
                binarize=binarize,
            )
        return Normal_init_semi(
            working_response,
            K_list[0],
            K_list,
            B_list_true,
            G_list_true,
            epsilon,
            mode="oracle",
            binarize=binarize,
        )

    def observed_conditional_logit(
        self,
        X,
        B1,
        dispersion,
        A1,
        *,
        width,
        row_chunk_size,
        coordinate_chunk_size=1,
    ) -> np.ndarray:
        """Return Normal observed-layer log-likelihood odds for ``A_1``."""
        del coordinate_chunk_size
        X = np.asarray(X, dtype=float)
        gamma = np.asarray(dispersion, dtype=float)
        B1 = np.asarray(B1, dtype=float)
        A1 = np.asarray(A1)
        N = X.shape[0]
        slopes = B1[:, 1 : width + 1]
        weighted_slopes = slopes / gamma[:, None]
        slope_gram = slopes.T @ weighted_slopes
        slope_norms = np.diag(slope_gram)
        like_logit = np.empty((N, width), dtype=float)
        for start in range(0, N, row_chunk_size):
            stop = min(start + row_chunk_size, N)
            A_chunk = A1[start:stop, :width]
            data_linear = (
                (X[start:stop] - B1[:, 0][None, :]) / gamma[None, :]
            ) @ slopes
            like_logit[start:stop] = (
                data_linear
                - A_chunk @ slope_gram
                + (A_chunk - 0.5) * slope_norms[None, :]
            )
        return like_logit

    def observed_coordinate_loglik_delta(self, X, eta_zero, slope, dispersion):
        """Return Normal log-likelihood odds for one latent coordinate."""
        X = np.asarray(X, dtype=float)
        eta_zero = np.asarray(eta_zero, dtype=float)
        slope = np.asarray(slope, dtype=float)
        gamma = np.asarray(dispersion, dtype=float)
        return np.sum(
            ((X - eta_zero) * slope[None, :] - 0.5 * slope[None, :] ** 2)
            / gamma[None, :],
            axis=1,
        )

    def make_fixed_accumulator(self, n_responses, n_features):
        return _ObservedNormalAccumulator(n_responses, n_features)

    def make_response_cache(self, X):
        """Cache response-only quantities shared by observed M-steps."""
        return np.sum(np.asarray(X, dtype=float) ** 2, axis=0)

    def copy_dispersion(self, dispersion):
        return np.asarray(dispersion, dtype=float).copy()

    def fixed_mstep(
        self,
        X,
        A_samples,
        B_start,
        dispersion,
        accumulator,
        *,
        iteration,
        depth,
        penalty,
        tau,
        bound_val,
        response_cache,
        observed_solver=None,
    ):
        """Perform the existing fixed-width Normal observed-layer update."""
        N, J = X.shape
        K1 = B_start.shape[1] - 1
        C = A_samples.shape[2]
        Q_current, q_current = accumulator.update(X, A_samples, iteration)
        if depth >= 3:
            return _update_deep_normal_observed(
                X,
                Q_current,
                q_current,
                tau,
                x_sq_sum=response_cache,
            )

        B_update = np.zeros_like(B_start)
        for j in range(J):

            def obj(x, __j=j, __pen=penalty):
                value, grad = accumulator.value_and_grad(__j, x)
                pen_value, pen_grad = _tlp_penalty_value_grad(
                    x[1:], __pen, tau
                )
                grad[1:] += pen_grad
                return value + pen_value, grad

            bounds = [(-bound_val, bound_val)] * (K1 + 1)
            res = minimize(
                obj,
                B_start[j, :],
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 10},
            )
            B_update[j, :] = res.x
            # MARK: Original variance update formed each row's (N, C) mean
            # inside the loop; the residuals are now computed once for all rows.
            # eta = np.tensordot(A_samples, res.x, axes=([1], [0]))
            # gamma_update[j] = np.sum((Xj[:, None] - eta) ** 2) / (N * C)
        eta = np.tensordot(A_samples, B_update, axes=([1], [1]))
        gamma_update = np.sum((X[:, None, :] - eta) ** 2, axis=(0, 1)) / (N * C)
        return B_update, gamma_update

    def csp_mstep(
        self,
        X,
        A_samples,
        B_start,
        dispersion,
        penalty_scale,
        temperature,
        *,
        response_cache,
        max_iter,
        tol,
        bound,
        bounds,
        row_chunk_size,
        coordinate_chunk_size,
    ):
        del bounds, row_chunk_size, coordinate_chunk_size
        return _solve_gaussian_weighted_l1(
            X,
            A_samples,
            B_start,
            dispersion,
            penalty_scale,
            temperature,
            x_sq_sum=response_cache,
            max_iter=max_iter,
            tol=tol,
            bound=bound,
        )

    def dispersion_distance_sq(self, old, new) -> float:
        return float(np.sum((np.asarray(old) - np.asarray(new)) ** 2))

    def fixed_latent_bounds(self, n_parameters, bound_val):
        return [(-bound_val, bound_val)] * int(n_parameters)

    def csp_coefficient_bounds(self, n_parameters, bound_val, *, observed):
        """Preserve the existing Normal CSP bounds policy."""
        if observed:
            return [(-bound_val, bound_val)] * int(n_parameters)
        return [(-5.0, bound_val)] + [(-bound_val, bound_val)] * (
            int(n_parameters) - 1
        )

    def postprocess_fixed_coefficients(self, B_updates, *, tau, depth):
        # Hard threshold at every depth.  The deep-Normal MATLAB script also
        # applies ``debias.m`` (``_matlab_debias``) for D >= 3, but TLP leaves
        # coefficients above ``tau`` unshrunk, so adding ``tau`` back biases
        # every surviving coefficient by ``tau`` regardless of N.
        del depth
        for B_update in B_updates:
            B_update[:, 1:] = thres(B_update[:, 1:], tau)
        return B_updates

    def conditional_log_prob(self, Y, eta, dispersion):
        Y = np.asarray(Y, dtype=float)
        eta = np.asarray(eta, dtype=float)
        gamma = np.asarray(dispersion, dtype=float).reshape(1, -1)
        return -0.5 * (np.log(2.0 * np.pi * gamma) + ((Y - eta) ** 2) / gamma)

    def sample_observed(self, eta, dispersion, rng):
        eta = np.asarray(eta, dtype=float)
        scale = np.sqrt(np.asarray(dispersion, dtype=float)).reshape(1, -1)
        return rng.normal(loc=eta, scale=scale, size=eta.shape)


@dataclass(frozen=True)
class PoissonFamilyAdapter:
    """Observed-layer operations for a log-link Poisson DDE."""

    name: str = "Poisson"
    key: str = "poisson"
    aliases: tuple[str, ...] = ("poisson",)
    supported_dimension_methods: tuple[str, ...] = ("fixed", "csp")
    has_dispersion: bool = False
    dispersion_name: Optional[str] = None

    def inverse_mean_link(self, mean, *, epsilon=1e-4):
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        return np.log(np.maximum(np.asarray(mean, dtype=float), epsilon))

    def initial_working_response(self, Y, *, epsilon, K1=None):
        if K1 is None:
            raise ValueError("Poisson initial_working_response requires K1.")
        return _poisson_working_response(
            self.validate_y(Y), int(K1), float(epsilon)
        )

    def validate_y(self, Y) -> np.ndarray:
        X = np.asarray(Y)
        if X.ndim != 2:
            raise ValueError(
                f"Y must be a two-dimensional matrix; got shape {X.shape}."
            )
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError(
                "Y must contain at least one observation and one variable."
            )
        try:
            valid = (
                np.all(np.isfinite(X))
                and np.all(X >= 0)
                and np.all(X == np.floor(X))
            )
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(
                "Poisson DDE requires finite nonnegative integer counts in Y."
            )
        return X.astype(float, copy=False)

    def validate_dispersion(self, dispersion, n_responses):
        del n_responses
        if dispersion is not None:
            raise ValueError(
                "Poisson has no dispersion parameter; gamma_in must be omitted."
            )
        return None

    def initialize(
        self,
        X,
        K_list,
        *,
        epsilon,
        mode,
        B_list_true=None,
        G_list_true=None,
        random_state=None,
        binarize="projection",
    ):
        return Poisson_init(
            X,
            K_list,
            epsilon=epsilon,
            mode=mode,
            B_list=B_list_true,
            G_list=G_list_true,
            random_state=random_state,
            binarize=binarize,
        )

    def observed_conditional_logit(
        self,
        X,
        B1,
        dispersion,
        A1,
        *,
        width,
        row_chunk_size,
        coordinate_chunk_size=1,
    ):
        """Return Poisson observed-layer log-likelihood odds for ``A_1``."""
        del dispersion
        X = np.asarray(X, dtype=float)
        B1 = np.asarray(B1, dtype=float)
        A1 = np.asarray(A1, dtype=float)
        N = X.shape[0]
        width = int(width)
        if row_chunk_size is None:
            row_chunk_size = N
        row_chunk_size = max(1, int(row_chunk_size))
        coordinate_chunk_size = max(1, int(coordinate_chunk_size))
        slopes = B1[:, 1 : width + 1]
        like_logit = np.empty((N, width), dtype=float)
        for start in range(0, N, row_chunk_size):
            stop = min(start + row_chunk_size, N)
            A_chunk = A1[start:stop, :width]
            eta = B1[:, 0][None, :] + A_chunk @ slopes.T
            data_linear = X[start:stop] @ slopes
            for block_start in range(0, width, coordinate_chunk_size):
                block_stop = min(block_start + coordinate_chunk_size, width)
                for coordinate in range(block_start, block_stop):
                    slope = slopes[:, coordinate]
                    eta_zero = (
                        eta - A_chunk[:, coordinate, None] * slope[None, :]
                    )
                    rate_delta = _poisson_optimization_rate(eta_zero)
                    rate_delta *= np.expm1(slope)[None, :]
                    like_logit[start:stop, coordinate] = data_linear[
                        :, coordinate
                    ] - np.sum(rate_delta, axis=1)
        return like_logit

    def observed_coordinate_loglik_delta(self, X, eta_zero, slope, dispersion):
        """Return Poisson log-likelihood odds for one latent coordinate."""
        del dispersion
        X = np.asarray(X, dtype=float)
        eta_zero = np.asarray(eta_zero, dtype=float)
        slope = np.asarray(slope, dtype=float)
        rate_delta = _poisson_optimization_rate(eta_zero)
        rate_delta *= np.expm1(slope)[None, :]
        return X @ slope - np.sum(rate_delta, axis=1)

    def make_fixed_accumulator(self, n_responses, n_features):
        return _ObservedPoissonAccumulator(n_responses, n_features)

    def make_response_cache(self, X):
        del X
        return None

    def copy_dispersion(self, dispersion):
        return self.validate_dispersion(dispersion, 0)

    def fixed_mstep(
        self,
        X,
        A_samples,
        B_start,
        dispersion,
        accumulator,
        *,
        iteration,
        depth,
        penalty,
        tau,
        bound_val,
        response_cache,
        observed_solver=None,
    ):
        del dispersion, depth, response_cache
        accumulator.update(X, A_samples, iteration)
        bounds = self.fixed_latent_bounds(B_start.shape[1], bound_val)
        solver = str(observed_solver or "rowwise").strip().lower()
        if solver == "hybrid":
            # dense row-wise steps while the step size is one (burn-in,
            # single-draw objective), active set once the averaging starts
            solver = "rowwise" if int(iteration) <= 1 else "active_set"
        if solver == "active_set":
            B_update = _solve_poisson_tlp_active_set(
                accumulator, B_start, penalty, tau, bounds
            )
        elif solver == "active_set_wide":
            B_update = _solve_poisson_tlp_active_set(
                accumulator, B_start, penalty, tau, bounds,
                max_support=16, max_added=8, newton_screen=False,
            )
        elif solver == "rowwise":
            B_update = _solve_poisson_tlp_rowwise(
                accumulator, B_start, penalty, tau, bounds, max_iter=10
            )
        else:
            raise ValueError("Poisson observed_solver must be 'active_set', 'rowwise', 'hybrid' or 'active_set_wide'.")
        return B_update, None

    def csp_mstep(
        self,
        X,
        A_samples,
        B_start,
        dispersion,
        penalty_scale,
        temperature,
        *,
        response_cache,
        max_iter,
        tol,
        bound,
        bounds,
        row_chunk_size,
        coordinate_chunk_size,
    ):
        del dispersion, response_cache
        from .csp_solvers import _solve_poisson_weighted_l1

        return (
            _solve_poisson_weighted_l1(
                X,
                A_samples,
                B_start,
                penalty_scale,
                temperature,
                max_iter=max_iter,
                tol=tol,
                bound=bound,
                bounds=bounds,
                row_chunk_size=row_chunk_size,
                coordinate_chunk_size=coordinate_chunk_size,
            ),
            None,
        )

    def dispersion_distance_sq(self, old, new) -> float:
        if old is not None or new is not None:
            raise ValueError("Poisson has no dispersion parameter.")
        return 0.0

    def fixed_latent_bounds(self, n_parameters, bound_val):
        bound_val = float(bound_val)
        upper = min(5.0, bound_val)
        intercept_lower = -upper
        slope_lower = -min(2.5, bound_val)
        return [(intercept_lower, upper)] + [(slope_lower, upper)] * (
            int(n_parameters) - 1
        )

    def csp_coefficient_bounds(self, n_parameters, bound_val, *, observed):
        """Use the fixed-path validated bounds throughout Poisson CSP."""
        del observed
        return self.fixed_latent_bounds(n_parameters, bound_val)

    def postprocess_fixed_coefficients(self, B_updates, *, tau, depth):
        del depth
        for B_update in B_updates:
            B_update[:, 1:] = thres(B_update[:, 1:], tau)
        return B_updates

    def conditional_log_prob(self, Y, eta, dispersion):
        del dispersion
        Y = np.asarray(Y, dtype=float)
        eta = np.asarray(eta, dtype=float)
        with np.errstate(over="ignore"):
            rate = np.exp(eta)
        return Y * eta - rate - gammaln(Y + 1.0)

    def sample_observed(self, eta, dispersion, rng):
        del dispersion
        eta = np.asarray(eta, dtype=float)
        with np.errstate(over="ignore"):
            rate = np.exp(eta)
        if not np.all(np.isfinite(rate)) or np.any(rate > 1.0e18):
            raise ValueError(
                "Poisson natural parameters produce rates too large to sample."
            )
        return rng.poisson(rate, size=rate.shape)


@dataclass(frozen=True)
class BernoulliFamilyAdapter:
    """Observed-layer operations for a conditionally Bernoulli DDE."""

    name: str = "Bernoulli"
    key: str = "bernoulli"
    aliases: tuple[str, ...] = ("bernoulli", "binary")
    supported_dimension_methods: tuple[str, ...] = ("fixed", "csp")
    has_dispersion: bool = False
    dispersion_name: Optional[str] = None

    def inverse_mean_link(self, mean, *, epsilon=1e-4):
        clipped = np.clip(
            np.asarray(mean, dtype=float), epsilon, 1.0 - epsilon
        )
        return np.log(clipped / (1.0 - clipped))

    def initial_working_response(self, Y, *, epsilon, K1=None):
        if K1 is None:
            raise ValueError("Bernoulli initial_working_response requires K1.")
        return _bernoulli_working_response(
            self.validate_y(Y), int(K1), float(epsilon)
        )

    def validate_y(self, Y) -> np.ndarray:
        X = np.asarray(Y)
        if X.ndim != 2:
            raise ValueError(
                f"Y must be a two-dimensional matrix; got shape {X.shape}."
            )
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError(
                "Y must contain at least one observation and one variable."
            )
        try:
            finite = np.isfinite(X)
        except TypeError as exc:
            raise ValueError(
                "Bernoulli DDE requires numeric binary 0/1 values in Y."
            ) from exc
        if not np.all(finite) or not np.all((X == 0) | (X == 1)):
            raise ValueError("Bernoulli DDE requires binary 0/1 values in Y.")
        return X.astype(float, copy=False)

    def validate_dispersion(self, dispersion, n_responses):
        del n_responses
        if dispersion is not None:
            raise ValueError(
                "Bernoulli has no dispersion parameter; gamma_in must be omitted."
            )
        return None

    def initialize(
        self,
        X,
        K_list,
        *,
        epsilon,
        mode,
        B_list_true=None,
        G_list_true=None,
        random_state=None,
        binarize="projection",
    ):
        return Bernoulli_init(
            X,
            K_list,
            epsilon=epsilon,
            mode=mode,
            B_list=B_list_true,
            G_list=G_list_true,
            random_state=random_state,
        
            binarize=binarize,
        )

    def observed_conditional_logit(
        self,
        X,
        B1,
        dispersion,
        A1,
        *,
        width,
        row_chunk_size,
        coordinate_chunk_size=1,
    ):
        del dispersion
        return _binary_child_conditional_logit(
            X,
            B1,
            A1,
            parent_width=width,
            child_width=np.asarray(X).shape[1],
            row_chunk_size=row_chunk_size,
            coordinate_chunk_size=coordinate_chunk_size,
            sparse="auto",
        )

    def observed_coordinate_loglik_delta(self, X, eta_zero, slope, dispersion):
        """Return Bernoulli log-likelihood odds for one latent coordinate."""
        del dispersion
        X = np.asarray(X, dtype=float)
        eta_zero = np.asarray(eta_zero, dtype=float)
        slope = np.asarray(slope, dtype=float)
        eta_one = eta_zero + slope[None, :]
        return X @ slope - np.sum(
            np.logaddexp(0.0, eta_one) - np.logaddexp(0.0, eta_zero),
            axis=1,
        )

    def make_fixed_accumulator(self, n_responses, n_features):
        # Imported lazily because the shared accumulator is defined by the
        # SAEM module that imports this adapter module.
        from .semi_saem import _make_bernoulli_layer_accumulator

        return _make_bernoulli_layer_accumulator(
            int(n_features) - 1, int(n_responses)
        )

    def make_response_cache(self, X):
        del X
        return None

    def copy_dispersion(self, dispersion):
        return self.validate_dispersion(dispersion, 0)

    def fixed_mstep(
        self,
        X,
        A_samples,
        B_start,
        dispersion,
        accumulator,
        *,
        iteration,
        depth,
        penalty,
        tau,
        bound_val,
        response_cache,
        observed_solver=None,
    ):
        del dispersion, depth, response_cache
        accumulator.update(X, A_samples, iteration)
        bounds = self.fixed_latent_bounds(B_start.shape[1], bound_val)
        return (
            _solve_bernoulli_tlp_batched(
                accumulator,
                B_start,
                penalty,
                tau,
                bounds,
                max_iter=10,
            ),
            None,
        )

    def csp_mstep(
        self,
        X,
        A_samples,
        B_start,
        dispersion,
        penalty_scale,
        temperature,
        *,
        response_cache,
        max_iter,
        tol,
        bound,
        bounds,
        row_chunk_size,
        coordinate_chunk_size,
    ):
        del dispersion, response_cache
        from .csp_solvers import _solve_logistic_weighted_l1

        return (
            _solve_logistic_weighted_l1(
                X,
                A_samples,
                B_start,
                penalty_scale,
                temperature,
                max_iter=max_iter,
                tol=tol,
                bound=bound,
                bounds=bounds,
                row_chunk_size=row_chunk_size,
                coordinate_chunk_size=coordinate_chunk_size,
            ),
            None,
        )

    def dispersion_distance_sq(self, old, new) -> float:
        if old is not None or new is not None:
            raise ValueError("Bernoulli has no dispersion parameter.")
        return 0.0

    def fixed_latent_bounds(self, n_parameters, bound_val):
        bound_val = float(bound_val)
        upper = min(5.0, bound_val)
        intercept_lower = -upper
        slope_lower = -min(2.5, bound_val)
        return [(intercept_lower, upper)] + [(slope_lower, upper)] * (
            int(n_parameters) - 1
        )

    def csp_coefficient_bounds(self, n_parameters, bound_val, *, observed):
        """Use the validated Bernoulli bounds for observed and latent logits."""
        del observed
        return self.fixed_latent_bounds(n_parameters, bound_val)

    def postprocess_fixed_coefficients(self, B_updates, *, tau, depth):
        del depth
        for B_update in B_updates:
            B_update[:, 1:] = thres(B_update[:, 1:], tau)
        return B_updates

    def conditional_log_prob(self, Y, eta, dispersion):
        del dispersion
        Y = np.asarray(Y, dtype=float)
        eta = np.asarray(eta, dtype=float)
        return np.where(
            Y == 1.0,
            -np.logaddexp(0.0, -eta),
            -np.logaddexp(0.0, eta),
        )

    def sample_observed(self, eta, dispersion, rng):
        del dispersion
        probabilities = expit(np.asarray(eta, dtype=float))
        return rng.binomial(1, probabilities, size=probabilities.shape)


NORMAL_FAMILY = NormalFamilyAdapter()
BERNOULLI_FAMILY = BernoulliFamilyAdapter()
POISSON_FAMILY = PoissonFamilyAdapter()

_FAMILY_REGISTRY = {alias: NORMAL_FAMILY for alias in NORMAL_FAMILY.aliases}
_FAMILY_REGISTRY.update(
    {alias: BERNOULLI_FAMILY for alias in BERNOULLI_FAMILY.aliases}
)
_FAMILY_REGISTRY.update(
    {alias: POISSON_FAMILY for alias in POISSON_FAMILY.aliases}
)


def register_parametric_family(adapter):
    """Register one adapter under all of its aliases."""
    aliases = tuple(str(alias).strip().lower() for alias in adapter.aliases)
    if not aliases or any(not alias for alias in aliases):
        raise ValueError("A parametric family must define at least one alias.")
    for alias in aliases:
        existing = _FAMILY_REGISTRY.get(alias)
        if existing is not None and existing is not adapter:
            raise ValueError(
                f"Parametric family alias {alias!r} is already registered."
            )
    for alias in aliases:
        _FAMILY_REGISTRY[alias] = adapter
    return adapter


def resolve_parametric_family(value):
    """Return a registered family adapter from a name or adapter instance."""
    if not isinstance(value, (str, bytes)):
        required = (
            "aliases",
            "name",
            "key",
            "supported_dimension_methods",
            "has_dispersion",
            "validate_y",
            "validate_dispersion",
            "inverse_mean_link",
            "initial_working_response",
            "initialize",
            "observed_conditional_logit",
            "observed_coordinate_loglik_delta",
            "make_fixed_accumulator",
            "make_response_cache",
            "copy_dispersion",
            "fixed_mstep",
            "csp_mstep",
            "dispersion_distance_sq",
            "fixed_latent_bounds",
            "csp_coefficient_bounds",
            "postprocess_fixed_coefficients",
            "conditional_log_prob",
            "sample_observed",
        )
        if all(hasattr(value, attribute) for attribute in required):
            return value
        raise TypeError("family must be a registered name or family adapter.")
    normalized = str(value).strip().lower()
    if normalized in _FAMILY_REGISTRY:
        return _FAMILY_REGISTRY[normalized]
    raise NotImplementedError(
        "The current DDE interface implements Normal, Bernoulli, and Poisson "
        "families only."
    )


__all__ = [
    "BERNOULLI_FAMILY",
    "BernoulliFamilyAdapter",
    "NORMAL_FAMILY",
    "NormalFamilyAdapter",
    "_ObservedNormalAccumulator",
    "POISSON_FAMILY",
    "PoissonFamilyAdapter",
    "_ObservedPoissonAccumulator",
    "_binary_child_conditional_logit",
    "_largest_eigenvalue_psd",
    "_matlab_debias",
    "_normal_sample_statistics",
    "_soft_threshold",
    "_solve_gaussian_weighted_l1",
    "_solve_gaussian_weighted_l1_from_statistics",
    "_solve_bernoulli_tlp_batched",
    "_solve_bernoulli_tlp_rowwise",
    "_solve_poisson_tlp_rowwise",
    "_tlp_penalty",
    "_tlp_penalty_value_grad",
    "_update_deep_normal_observed",
    "register_parametric_family",
    "resolve_parametric_family",
]
