"""
Family-adapted penalized SAEM for D-layer parametric DDEs, plus the
Normal-working-response semiparametric outer iteration.
 
Three-mode marginal PMF computation for P(A_1):
  Mode A  (all-dense):   every K_d <= max_enum_bits  →  exact, full 2^{K_1} vector
  Mode B  (dense+sparse): only K_1 is large, K_2 enumerable  →  exact for query patterns
  Mode C  (dense+MC):     multiple large layers  →  MC forward sampling + analytic last step
 
Model (1-based layers):
  Layer D (top):   A_D ~ Bernoulli(prop)
  Layer d (1<d<D): A_d | A_{d+1} ~ Bernoulli(logistic(B_{d+1} @ [1, A_{d+1}]))
  Layer 0 (obs):   X | A_1 follows the selected parametric family adapter
 
Python list indexing (0-based):
  B_list[d] = B_{d+1}     d=0 => B_1 (J, K_1+1);  d>=1 => B_{d+1} (K_d, K_{d+1}+1)
  A_list[d] = A_{d+1}     (N, K_{d+1})
  pen_list[d]              penalty magnitude for B_list[d]
  K[d] = K_{d+1}           number of nodes in layer d+1
"""
 
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp as _logsumexp
from dde.utils import binary, logistic, patterns_to_int, thres, g_est_rversion
from .normal_init_semi import Normal_init_semi
from .parametric_families import (
    BERNOULLI_FAMILY,
    NORMAL_FAMILY,
    POISSON_FAMILY,
    _ObservedNormalAccumulator,
    _binary_child_conditional_logit,
    _matlab_debias,
    _normal_sample_statistics,
    _solve_bernoulli_tlp_batched,
    _tlp_penalty,
    _tlp_penalty_value_grad,
    _update_deep_normal_observed,
    resolve_parametric_family,
)


# MARK: Original SA objective accumulation built a nested closure chain, so each
# objective evaluation became more expensive as the SAEM inner iteration grew.
# def _sa_accumulate(f_old, f_new, t):
#     alpha = 1.0 / t
#     def g(x, _fo=f_old, _fn=f_new, _a=alpha):
#         return (1.0 - _a) * _fo(x) + _a * _fn(x)
#     return g


def _flatten_bernoulli_children(children, C):
    """Align fixed or Monte Carlo binary children with flattened parents."""
    children = np.asarray(children)
    if children.ndim == 2:
        if int(C) == 1:
            return children
        return np.repeat(children, int(C), axis=0)
    if children.ndim == 3:
        return np.moveaxis(children[:, 1:, :], 1, 2).reshape(
            -1, children.shape[1] - 1
        )
    raise ValueError("Bernoulli children must be a 2D matrix or 3D MC array.")


def _batched_grouped_bernoulli_value_and_grad(
    design,
    counts,
    sum_y,
    coefficients,
    *,
    row_chunk_size=2048,
):
    """Evaluate all independent Bernoulli responses with shared designs.

    The returned scalar is the sum of the existing per-response objectives,
    and the returned matrix stacks their analytic gradients.  Chunking caps
    the temporary ``n_patterns x n_responses`` logits used by the batched
    matrix multiplications.
    """
    design = np.asarray(design, dtype=float)
    counts = np.asarray(counts, dtype=float)
    sum_y = np.asarray(sum_y, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if design.ndim != 2 or coefficients.ndim != 2:
        raise ValueError("Bernoulli batch design and coefficients must be matrices.")
    if counts.shape != (design.shape[0],):
        raise ValueError("Bernoulli batch counts have an incompatible shape.")
    if sum_y.shape != (design.shape[0], coefficients.shape[0]):
        raise ValueError("Bernoulli batch response sums have an incompatible shape.")
    if coefficients.shape[1] != design.shape[1]:
        raise ValueError("Bernoulli batch coefficient width is incompatible.")
    if row_chunk_size is None:
        row_chunk_size = max(1, design.shape[0])
    row_chunk_size = max(1, int(row_chunk_size))

    value = 0.0
    gradient = np.zeros_like(coefficients)
    for start in range(0, design.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, design.shape[0])
        design_chunk = design[start:stop]
        counts_chunk = counts[start:stop, None]
        response_chunk = sum_y[start:stop]
        raw_eta = design_chunk @ coefficients.T
        eta = np.clip(raw_eta, -500.0, 500.0)
        value += float(
            np.sum(
                -response_chunk * eta
                + counts_chunk * np.logaddexp(0.0, eta)
            )
        )
        residual = counts_chunk * logistic(eta) - response_chunk
        residual *= (raw_eta > -500.0) & (raw_eta < 500.0)
        gradient += residual.T @ design_chunk
    return value, gradient


class _BernoulliLayerAccumulator:
    """Layer-shared Bernoulli statistics aggregated by parent pattern."""

    def __init__(self, K_upper, n_children):
        self.K_upper = int(K_upper)
        self.n_children = int(n_children)
        self.codes = np.empty(0, dtype=np.int64)
        self.counts = np.empty(0, dtype=float)
        self.sum_y = np.empty((0, self.n_children), dtype=float)
        self._parent_states = None
        self._design = None

    def update(self, A_lower_samples, A_upper_samples, t):
        alpha = 1.0 / t
        C = A_upper_samples.shape[2]
        parents = np.moveaxis(A_upper_samples[:, 1:, :], 1, 2).reshape(-1, self.K_upper)
        children = _flatten_bernoulli_children(A_lower_samples, C)
        weights = np.int64(1) << np.arange(self.K_upper - 1, -1, -1, dtype=np.int64)
        codes_flat = parents.astype(np.int64) @ weights

        new_codes, inverse = np.unique(codes_flat, return_inverse=True)
        new_counts = np.bincount(inverse).astype(float) / C
        new_sum_y = np.zeros((new_codes.size, self.n_children), dtype=float)
        np.add.at(new_sum_y, inverse, children)
        new_sum_y /= C

        if alpha >= 1.0:
            # Step size one (burn-in sweep): the objective is the current
            # draw alone, so the history is dropped instead of kept at
            # weight zero.
            self.codes = new_codes
            self.counts = new_counts
            self.sum_y = new_sum_y
            self._parent_states = None
            self._design = None
            return

        all_codes = np.union1d(self.codes, new_codes)
        merged_counts = np.zeros(all_codes.size, dtype=float)
        merged_sum_y = np.zeros(
            (all_codes.size, self.n_children), dtype=float
        )

        if self.codes.size:
            old_pos = np.searchsorted(all_codes, self.codes)
            merged_counts[old_pos] = (1.0 - alpha) * self.counts
            merged_sum_y[old_pos, :] = (1.0 - alpha) * self.sum_y
        new_pos = np.searchsorted(all_codes, new_codes)
        merged_counts[new_pos] += alpha * new_counts
        merged_sum_y[new_pos, :] += alpha * new_sum_y

        self.codes = all_codes
        self.counts = merged_counts
        self.sum_y = merged_sum_y
        self._parent_states = None
        self._design = None

    @property
    def parent_states(self):
        if self._parent_states is None:
            self._parent_states = binary(self.codes, self.K_upper)
        return self._parent_states

    def value(self, child_index, x):
        eta = self.design @ x
        eta = np.clip(eta, -500, 500)
        return float(
            -(self.sum_y[:, child_index] @ eta)
            + self.counts @ np.logaddexp(0.0, eta)
        )

    @property
    def design(self):
        if self._design is None:
            self._design = np.column_stack(
                [np.ones(self.codes.size, dtype=float), self.parent_states]
            )
        return self._design

    def value_and_grad(self, child_index, x):
        raw_eta = self.design @ x
        eta = np.clip(raw_eta, -500, 500)
        sum_y = self.sum_y[:, child_index]
        value = float(-(sum_y @ eta) + self.counts @ np.logaddexp(0.0, eta))
        residual = self.counts * logistic(eta) - sum_y
        clip_derivative = ((raw_eta > -500.0) & (raw_eta < 500.0)).astype(float)
        grad = self.design.T @ (residual * clip_derivative)
        return value, grad

    def batched_value_and_grad(self, coefficients, *, row_chunk_size=2048):
        return _batched_grouped_bernoulli_value_and_grad(
            self.design,
            self.counts,
            self.sum_y,
            coefficients,
            row_chunk_size=row_chunk_size,
        )

    def weighted_design_gram(self):
        """Return the SA-weighted design cross-product for this layer."""
        return self.design.T @ (self.counts[:, None] * self.design)


class _DirectAddressBernoulliLayerAccumulator:
    """Exact dense statistics for layers with few binary parents.

    For small ``K_upper``, the complete parent-pattern table is tiny. Direct
    integer addressing avoids sorting, unions, aligned temporary arrays, and
    repeated design reconstruction in the grouped backend.
    """

    def __init__(self, K_upper, n_children):
        self.K_upper = int(K_upper)
        self.n_children = int(n_children)
        self.n_patterns = 1 << self.K_upper
        self.counts = np.zeros(self.n_patterns, dtype=float)
        self.sum_y = np.zeros(
            (self.n_patterns, self.n_children), dtype=float
        )
        self.parent_states = binary(
            np.arange(self.n_patterns, dtype=np.int64), self.K_upper
        )
        self.design = np.column_stack(
            [np.ones(self.n_patterns, dtype=float), self.parent_states]
        )
        self._code_weights = np.int64(1) << np.arange(
            self.K_upper - 1, -1, -1, dtype=np.int64
        )

    def update(self, A_lower_samples, A_upper_samples, t):
        alpha = 1.0 / t
        C = A_upper_samples.shape[2]
        parents = np.moveaxis(
            A_upper_samples[:, 1:, :], 1, 2
        ).reshape(-1, self.K_upper)
        children = _flatten_bernoulli_children(A_lower_samples, C)
        codes = parents.astype(np.int64) @ self._code_weights

        current_counts = np.bincount(
            codes, minlength=self.n_patterns
        ).astype(float) / C
        current_sum_y = np.empty_like(self.sum_y)
        for child_index in range(self.n_children):
            current_sum_y[:, child_index] = np.bincount(
                codes,
                weights=children[:, child_index],
                minlength=self.n_patterns,
            ) / C

        self.counts *= 1.0 - alpha
        self.counts += alpha * current_counts
        self.sum_y *= 1.0 - alpha
        self.sum_y += alpha * current_sum_y

    @property
    def storage_nbytes(self):
        return int(
            self.counts.nbytes
            + self.sum_y.nbytes
            + self.parent_states.nbytes
            + self.design.nbytes
            + self._code_weights.nbytes
        )

    def value(self, child_index, x):
        eta = np.clip(self.design @ x, -500, 500)
        return float(
            -(self.sum_y[:, child_index] @ eta)
            + self.counts @ np.logaddexp(0.0, eta)
        )

    def value_and_grad(self, child_index, x):
        raw_eta = self.design @ x
        eta = np.clip(raw_eta, -500, 500)
        sum_y = self.sum_y[:, child_index]
        value = float(
            -(sum_y @ eta) + self.counts @ np.logaddexp(0.0, eta)
        )
        residual = self.counts * logistic(eta) - sum_y
        clip_derivative = (
            (raw_eta > -500.0) & (raw_eta < 500.0)
        ).astype(float)
        grad = self.design.T @ (residual * clip_derivative)
        return value, grad

    def batched_value_and_grad(self, coefficients, *, row_chunk_size=2048):
        return _batched_grouped_bernoulli_value_and_grad(
            self.design,
            self.counts,
            self.sum_y,
            coefficients,
            row_chunk_size=row_chunk_size,
        )

    def weighted_design_gram(self):
        """Return the SA-weighted design cross-product for this layer."""
        return self.design.T @ (self.counts[:, None] * self.design)


class _BinaryHistoryBernoulliLayerAccumulator:
    """Exact SA objective history for layers with wide binary parents.

    Grouping by a full parent pattern is effective when the parent layer is
    enumerable, but becomes wasteful when almost every high-dimensional
    pattern is unique.  This backend keeps append-only binary Monte Carlo
    rows instead. Child states are bit-packed and the intercept is handled
    separately, avoiding the duplicated float64 ``parent_states``/``design``
    caches required by the grouped backend.

    The stored scalar chunk weights reproduce the same recursive stochastic
    average used by ``_BernoulliLayerAccumulator`` for any sequence of updates.
    """

    def __init__(self, K_upper, n_children):
        self.K_upper = int(K_upper)
        self.n_children = int(n_children)
        self._packed_child_width = (self.n_children + 7) // 8
        self._size = 0
        self._parent_chunks = []
        self._packed_child_chunks = []
        self._row_weight = 0.0
        self._draws_per_update = None
        self._response_stats = np.zeros(
            (self.n_children, self.K_upper + 1), dtype=float
        )

    def _iter_float_chunks(self, row_chunk_size=None):
        """Yield bounded float parents with aligned packed child rows."""
        if row_chunk_size is not None:
            row_chunk_size = max(1, int(row_chunk_size))
        for parent_chunk, packed_child_chunk in zip(
            self._parent_chunks, self._packed_child_chunks
        ):
            chunk_size = (
                parent_chunk.shape[0]
                if row_chunk_size is None
                else row_chunk_size
            )
            for start in range(0, parent_chunk.shape[0], chunk_size):
                stop = min(start + chunk_size, parent_chunk.shape[0])
                yield (
                    parent_chunk[start:stop].astype(float),
                    packed_child_chunk[start:stop],
                )

    def update(self, A_lower_samples, A_upper_samples, t):
        alpha = 1.0 / t
        C = A_upper_samples.shape[2]
        new_row_weight = alpha / C
        if alpha >= 1.0 and self._size:
            # Step size one (burn-in sweep): the stored history would carry
            # weight zero, so it is discarded and the sequence restarts.
            self._size = 0
            self._parent_chunks = []
            self._packed_child_chunks = []
            self._row_weight = 0.0
            self._draws_per_update = None
            self._response_stats = np.zeros_like(self._response_stats)
        if self._size:
            old_row_weight = (1.0 - alpha) * self._row_weight
            if self._draws_per_update != C or not np.isclose(
                old_row_weight, new_row_weight, rtol=1e-14, atol=1e-15
            ):
                raise ValueError(
                    "Binary-history scalar weights require sequential theta_t=1/t "
                    "updates with a fixed number of Monte Carlo draws."
                )
        else:
            self._draws_per_update = C
        self._row_weight = new_row_weight

        parents = np.moveaxis(
            A_upper_samples[:, 1:, :], 1, 2
        ).reshape(-1, self.K_upper)
        children = _flatten_bernoulli_children(A_lower_samples, C)
        parents = np.ascontiguousarray(parents, dtype=np.uint8)
        packed_children = np.packbits(
            np.ascontiguousarray(children, dtype=np.uint8),
            axis=1,
            bitorder="little",
        )

        current_response_stats = np.empty_like(self._response_stats)
        current_response_stats[:, 0] = np.sum(
            children, axis=0, dtype=np.float64
        ) / C
        current_response_stats[:, 1:] = 0.0
        for start in range(0, parents.shape[0], 2048):
            stop = min(start + 2048, parents.shape[0])
            parent_chunk = parents[start:stop].astype(float)
            child_chunk = children[start:stop].astype(float)
            current_response_stats[:, 1:] += (
                child_chunk.T @ parent_chunk
            )
        current_response_stats[:, 1:] /= C
        self._response_stats = (
            (1.0 - alpha) * self._response_stats
            + alpha * current_response_stats
        )

        self._parent_chunks.append(parents)
        self._packed_child_chunks.append(packed_children)
        self._size += parents.shape[0]

    @staticmethod
    def _child_column(packed_children, child_index):
        byte_index, bit_index = divmod(int(child_index), 8)
        return (
            (packed_children[:, byte_index] >> np.uint8(bit_index))
            & np.uint8(1)
        ).astype(float)

    @property
    def storage_nbytes(self):
        return int(
            sum(chunk.nbytes for chunk in self._parent_chunks)
            + sum(chunk.nbytes for chunk in self._packed_child_chunks)
            + self._response_stats.nbytes
        )

    def value(self, child_index, x):
        x = np.asarray(x, dtype=float)
        value = -float(self._response_stats[child_index] @ x)
        for parents, packed_children in self._iter_float_chunks():
            raw_eta = x[0] + parents @ x[1:]
            eta = np.clip(raw_eta, -500.0, 500.0)
            value += self._row_weight * float(
                np.sum(np.logaddexp(0.0, eta))
            )
            clipped = (raw_eta <= -500.0) | (raw_eta >= 500.0)
            if np.any(clipped):
                children = self._child_column(
                    packed_children, child_index
                )
                value += self._row_weight * float(
                    children @ (raw_eta - eta)
                )
        return value

    def value_and_grad(self, child_index, x):
        x = np.asarray(x, dtype=float)
        value = -float(self._response_stats[child_index] @ x)
        grad = -self._response_stats[child_index].copy()
        for parents, packed_children in self._iter_float_chunks():
            raw_eta = x[0] + parents @ x[1:]
            eta = np.clip(raw_eta, -500.0, 500.0)
            probability = logistic(eta)
            positive_residual = self._row_weight * probability
            value += self._row_weight * float(
                np.sum(np.logaddexp(0.0, eta))
            )
            grad[0] += np.sum(positive_residual)
            grad[1:] += parents.T @ positive_residual

            clipped = (raw_eta <= -500.0) | (raw_eta >= 500.0)
            if np.any(clipped):
                children = self._child_column(
                    packed_children, child_index
                )
                value += self._row_weight * float(
                    children @ (raw_eta - eta)
                )
                correction = self._row_weight * (children - probability)
                correction *= clipped
                grad[0] += np.sum(correction)
                grad[1:] += parents.T @ correction
        return value, grad

    def batched_value_and_grad(self, coefficients, *, row_chunk_size=2048):
        """Return the summed objective and all response gradients.

        In the usual interior case the response-dependent linear term comes
        from the exact running sufficient statistic.  The packed child
        history is unpacked only for coordinates whose natural parameter is
        clipped, matching ``value_and_grad`` at the numerical guard rails.
        """
        coefficients = np.asarray(coefficients, dtype=float)
        expected_shape = (self.n_children, self.K_upper + 1)
        if coefficients.shape != expected_shape:
            raise ValueError(
                "Bernoulli batch coefficients must have shape "
                f"{expected_shape}; got {coefficients.shape}."
            )
        if row_chunk_size is None:
            row_chunk_size = max(1, self._size)
        row_chunk_size = max(1, int(row_chunk_size))

        value = 0.0
        gradient = np.zeros_like(coefficients)
        for parents, packed_children in self._iter_float_chunks(
            row_chunk_size
        ):
            raw_eta = (
                coefficients[:, 0][None, :]
                + parents @ coefficients[:, 1:].T
            )
            eta = np.clip(raw_eta, -500.0, 500.0)
            probability = logistic(eta)
            weighted_probability = self._row_weight * probability
            value += self._row_weight * float(
                np.sum(np.logaddexp(0.0, eta))
            )
            gradient[:, 0] += np.sum(weighted_probability, axis=0)
            gradient[:, 1:] += weighted_probability.T @ parents

            clipped = (raw_eta <= -500.0) | (raw_eta >= 500.0)
            if np.any(clipped):
                children = np.unpackbits(
                    packed_children,
                    axis=1,
                    count=self.n_children,
                    bitorder="little",
                ).astype(float, copy=False)
                value += self._row_weight * float(
                    np.sum(children * (raw_eta - eta))
                )
                correction = self._row_weight * (children - probability)
                correction *= clipped
                gradient[:, 0] += np.sum(correction, axis=0)
                gradient[:, 1:] += correction.T @ parents

        value -= float(np.sum(self._response_stats * coefficients))
        gradient -= self._response_stats
        return value, gradient

    def weighted_design_gram(self):
        """Return the SA-weighted cross-product without expanding a design."""
        gram = np.zeros((self.K_upper + 1, self.K_upper + 1), dtype=float)
        gram[0, 0] = self._size
        for parents, _ in self._iter_float_chunks():
            parent_sums = np.sum(parents, axis=0)
            gram[0, 1:] += parent_sums
            gram[1:, 0] += parent_sums
            gram[1:, 1:] += parents.T @ parents
        gram *= self._row_weight
        return gram


_DIRECT_ADDRESS_MAX_BITS = 10
_GROUPED_PATTERN_MAX_BITS = 20


def _make_bernoulli_layer_accumulator(K_upper, n_children):
    K_upper = int(K_upper)
    if K_upper <= _DIRECT_ADDRESS_MAX_BITS:
        return _DirectAddressBernoulliLayerAccumulator(K_upper, n_children)
    if K_upper <= _GROUPED_PATTERN_MAX_BITS:
        return _BernoulliLayerAccumulator(K_upper, n_children)
    return _BinaryHistoryBernoulliLayerAccumulator(K_upper, n_children)


def _compute_bernoulli_probabilities_parametric(
    X,
    prop,
    B_list,
    dispersion,
    A_list,
    *,
    family,
    row_chunk_size=2048,
    coordinate_chunk_size=1,
    eligible_widths=None,
    temperature=1.0,
):
    """Compute one synchronous latent-probability sweep for a parametric DDE.

    This is algebraically equivalent to the original coordinate loops: every
    layer is conditioned on the same incoming ``A_list`` rather than on values
    sampled earlier in the sweep.
    """
    family = resolve_parametric_family(family)
    X = np.asarray(X, dtype=float)
    prop = np.asarray(prop, dtype=float)
    N = X.shape[0]
    D = len(B_list)
    K = [B.shape[1] - 1 for B in B_list]
    ber = []
    if row_chunk_size is None:
        row_chunk_size = N
    row_chunk_size = max(1, int(row_chunk_size))
    coordinate_chunk_size = max(1, int(coordinate_chunk_size))
    if eligible_widths is None:
        eligible_widths = K
    eligible_widths = tuple(int(width) for width in eligible_widths)
    if len(eligible_widths) != D or any(
        width < 1 or width > maximum
        for width, maximum in zip(eligible_widths, K)
    ):
        raise ValueError(
            "eligible_widths must contain one value in [1, K_d] per layer."
        )

    for d_idx in range(D):
        width = eligible_widths[d_idx]
        if d_idx == D - 1:
            prop_clipped = np.clip(prop, 1e-15, 1.0 - 1e-15)
            prior_logit = np.log(
                prop_clipped[:width] / (1.0 - prop_clipped[:width])
            )
        else:
            B_above = B_list[d_idx + 1]
            parent_width = eligible_widths[d_idx + 1]
            prior_logit = (
                B_above[:width, 0][None, :]
                + A_list[d_idx + 1][:, :parent_width]
                @ B_above[:width, 1 : parent_width + 1].T
            )

        if d_idx == 0:
            like_logit = family.observed_conditional_logit(
                X,
                B_list[0],
                dispersion,
                A_list[0],
                width=width,
                row_chunk_size=row_chunk_size,
                coordinate_chunk_size=coordinate_chunk_size,
            )
        else:
            B_cur = B_list[d_idx]
            A_cur = A_list[d_idx]
            A_below = A_list[d_idx - 1]
            child_width = eligible_widths[d_idx - 1]
            like_logit = _binary_child_conditional_logit(
                A_below,
                B_cur,
                A_cur,
                parent_width=width,
                child_width=child_width,
                row_chunk_size=row_chunk_size,
                coordinate_chunk_size=coordinate_chunk_size,
                sparse="auto" if family.key == "bernoulli" else False,
            )

        like_logit += prior_logit
        like_logit *= float(temperature)
        expit(like_logit, out=like_logit)
        if width == K[d_idx]:
            ber.append(like_logit)
        else:
            layer_probability = np.zeros((N, K[d_idx]), dtype=float)
            layer_probability[:, :width] = like_logit
            ber.append(layer_probability)

    return ber


def _observed_coordinate_loglik_delta(
    X,
    eta_zero,
    slope,
    dispersion,
    *,
    family,
    row_chunk_size=2048,
):
    """Return the observed log-likelihood change when one parent is set to 1.

    ``eta_zero`` is the observed natural parameter with the selected latent
    coordinate set to zero.  The adapter supplies a family-specific algebraic
    form so this hot path avoids constructing two full log-probability arrays.
    """
    family = resolve_parametric_family(family)
    X = np.asarray(X, dtype=float)
    eta_zero = np.asarray(eta_zero, dtype=float)
    slope = np.asarray(slope, dtype=float)
    N = X.shape[0]
    if eta_zero.shape != X.shape or slope.shape != (X.shape[1],):
        raise ValueError(
            "Observed coordinate likelihood received incompatible shapes."
        )
    if row_chunk_size is None:
        row_chunk_size = N
    row_chunk_size = max(1, int(row_chunk_size))
    delta = np.empty(N, dtype=float)
    for start in range(0, N, row_chunk_size):
        stop = min(start + row_chunk_size, N)
        delta[start:stop] = family.observed_coordinate_loglik_delta(
            X[start:stop],
            eta_zero[start:stop],
            slope,
            dispersion,
        )
    return delta


def _binary_coordinate_loglik_delta(
    children,
    eta_zero,
    slope,
    *,
    child_linear=None,
    row_chunk_size=None,
):
    """Return Bernoulli-child log-likelihood odds for one parent coordinate."""
    children = np.asarray(children, dtype=float)
    eta_zero = np.asarray(eta_zero, dtype=float)
    slope = np.asarray(slope, dtype=float)
    if eta_zero.shape != children.shape or slope.shape != (children.shape[1],):
        raise ValueError(
            "Binary coordinate likelihood received incompatible shapes."
        )
    if child_linear is None:
        cached_linear = None
    else:
        cached_linear = np.asarray(child_linear, dtype=float)
        if cached_linear.shape != (children.shape[0],):
            raise ValueError(
                "Cached Bernoulli child-linear term has an incompatible shape."
            )
    if row_chunk_size is None:
        row_chunk_size = max(1, children.shape[0])
    row_chunk_size = max(1, int(row_chunk_size))
    delta = np.empty(children.shape[0], dtype=float)
    for start in range(0, children.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, children.shape[0])
        eta_zero_chunk = eta_zero[start:stop]
        linear_chunk = (
            children[start:stop] @ slope
            if cached_linear is None
            else cached_linear[start:stop]
        )
        delta[start:stop] = linear_chunk - np.sum(
            np.logaddexp(0.0, eta_zero_chunk + slope[None, :])
            - np.logaddexp(0.0, eta_zero_chunk),
            axis=1,
        )
    return delta


def _coordinate_prior_logit(
    prop,
    B_list,
    A_list,
    layer,
    coordinate,
    *,
    eligible_widths=None,
):
    """Return the prior log-odds shared by dense and sparse Gibbs paths."""
    if layer == len(A_list) - 1:
        probability = np.clip(prop[coordinate], 1e-15, 1.0 - 1e-15)
        return np.log(probability / (1.0 - probability))
    coefficient_above = np.asarray(B_list[layer + 1], dtype=float)
    parent_width = (
        coefficient_above.shape[1] - 1
        if eligible_widths is None
        else int(eligible_widths[layer + 1])
    )
    return (
        coefficient_above[coordinate, 0]
        + np.asarray(A_list[layer + 1], dtype=float)[:, :parent_width]
        @ coefficient_above[coordinate, 1 : parent_width + 1]
    )


def _sparse_bernoulli_coordinate_probability(
    X,
    prop,
    B_list,
    A_list,
    layer,
    coordinate,
    child_eta,
    active_children,
    *,
    child_linear=None,
    row_chunk_size=2048,
    eligible_widths=None,
    temperature=1.0,
):
    """Evaluate one Bernoulli full conditional on nonzero child slopes only."""
    state = np.asarray(A_list[layer], dtype=np.uint8)
    coefficient = np.asarray(B_list[layer], dtype=float)
    active_children = np.asarray(active_children, dtype=np.intp)
    if active_children.ndim != 1:
        raise ValueError("Active Bernoulli child indices must be one-dimensional.")

    if active_children.size:
        slope = coefficient[active_children, coordinate + 1]
        eta_zero_active = (
            np.asarray(child_eta[:, active_children], dtype=float)
            - state[:, coordinate, None] * slope[None, :]
        )
        children = (
            np.asarray(X)[:, active_children]
            if layer == 0
            else np.asarray(A_list[layer - 1])[:, active_children]
        )
        likelihood_logit = _binary_coordinate_loglik_delta(
            children,
            eta_zero_active,
            slope,
            child_linear=child_linear,
            row_chunk_size=row_chunk_size,
        )
    else:
        eta_zero_active = np.empty((state.shape[0], 0), dtype=float)
        likelihood_logit = np.zeros(state.shape[0], dtype=float)

    prior_logit = _coordinate_prior_logit(
        prop,
        B_list,
        A_list,
        layer,
        coordinate,
        eligible_widths=eligible_widths,
    )
    return (
        expit(float(temperature) * (prior_logit + likelihood_logit)),
        eta_zero_active,
    )


def _sparse_poisson_coordinate_probability(
    X,
    prop,
    B_list,
    A_list,
    coordinate,
    child_eta,
    active_children,
    *,
    row_chunk_size=2048,
    eligible_widths=None,
    temperature=1.0,
):
    """Evaluate a Poisson observed-coordinate update on nonzero slopes."""
    X = np.asarray(X, dtype=float)
    state = np.asarray(A_list[0], dtype=np.uint8)
    coefficient = np.asarray(B_list[0], dtype=float)
    active_children = np.asarray(active_children, dtype=np.intp)
    if active_children.ndim != 1:
        raise ValueError("Active Poisson child indices must be one-dimensional.")

    if active_children.size:
        slope = coefficient[active_children, coordinate + 1]
        eta_zero_active = (
            np.asarray(child_eta[:, active_children], dtype=float)
            - state[:, coordinate, None] * slope[None, :]
        )
        if row_chunk_size is None:
            row_chunk_size = X.shape[0]
        row_chunk_size = max(1, int(row_chunk_size))
        likelihood_logit = np.empty(X.shape[0], dtype=float)
        for start in range(0, X.shape[0], row_chunk_size):
            stop = min(start + row_chunk_size, X.shape[0])
            likelihood_logit[start:stop] = (
                POISSON_FAMILY.observed_coordinate_loglik_delta(
                    X[start:stop][:, active_children],
                    eta_zero_active[start:stop],
                    slope,
                    None,
                )
            )
    else:
        eta_zero_active = np.empty((state.shape[0], 0), dtype=float)
        likelihood_logit = np.zeros(state.shape[0], dtype=float)

    prior_logit = _coordinate_prior_logit(
        prop,
        B_list,
        A_list,
        0,
        coordinate,
        eligible_widths=eligible_widths,
    )
    return (
        expit(float(temperature) * (prior_logit + likelihood_logit)),
        eta_zero_active,
    )


def _sparse_observed_child_linear_cache(X, coefficient, active_children):
    """Cache ``X @ slope`` for the observed Bernoulli Gibbs layer."""
    X = np.asarray(X)
    coefficient = np.asarray(coefficient, dtype=float)
    cache = np.zeros((X.shape[0], len(active_children)), dtype=float)
    for coordinate, indices in enumerate(active_children):
        if indices.size:
            cache[:, coordinate] = (
                X[:, indices] @ coefficient[indices, coordinate + 1]
            )
    return cache


def _latent_coordinate_probability_parametric(
    X,
    prop,
    B_list,
    dispersion,
    A_list,
    layer,
    coordinate,
    *,
    family,
    child_eta=None,
    row_chunk_size=2048,
    sparse_bernoulli=True,
    eligible_widths=None,
    temperature=1.0,
):
    """Return one exact full conditional and its zero-coordinate child eta."""
    family = resolve_parametric_family(family)
    X = np.asarray(X, dtype=float)
    prop = np.asarray(prop, dtype=float)
    layer = int(layer)
    coordinate = int(coordinate)
    state = np.asarray(A_list[layer], dtype=np.uint8)
    coefficient = np.asarray(B_list[layer], dtype=float)
    parent_width = (
        state.shape[1]
        if eligible_widths is None
        else int(eligible_widths[layer])
    )
    child_width = (
        coefficient.shape[0]
        if layer == 0 or eligible_widths is None
        else int(eligible_widths[layer - 1])
    )
    coefficient_active = coefficient[
        :child_width, : parent_width + 1
    ]
    if child_eta is None:
        child_eta = (
            coefficient_active[:, 0][None, :]
            + state[:, :parent_width] @ coefficient_active[:, 1:].T
        )
    else:
        child_eta = np.asarray(child_eta, dtype=float)
        if child_eta.shape != (state.shape[0], child_width):
            raise ValueError(
                "Child natural parameters have an incompatible shape."
            )
    slope = coefficient_active[:, coordinate + 1]

    if family.key == "bernoulli" and sparse_bernoulli:
        active_children = np.flatnonzero(slope != 0.0)
        probability, eta_zero_active = (
            _sparse_bernoulli_coordinate_probability(
                X,
                prop,
                B_list,
                A_list,
                layer,
                coordinate,
                child_eta,
                active_children,
                row_chunk_size=row_chunk_size,
                eligible_widths=eligible_widths,
                temperature=temperature,
            )
        )
        eta_zero = child_eta.copy()
        eta_zero[:, active_children] = eta_zero_active
        return probability, eta_zero

    eta_zero = child_eta - state[:, coordinate, None] * slope[None, :]

    if layer == 0:
        likelihood_logit = _observed_coordinate_loglik_delta(
            X,
            eta_zero,
            slope,
            dispersion,
            family=family,
            row_chunk_size=row_chunk_size,
        )
    else:
        likelihood_logit = _binary_coordinate_loglik_delta(
            np.asarray(A_list[layer - 1])[:, :child_width],
            eta_zero,
            slope,
        )

    prior_logit = _coordinate_prior_logit(
        prop,
        B_list,
        A_list,
        layer,
        coordinate,
        eligible_widths=eligible_widths,
    )
    return (
        expit(float(temperature) * (prior_logit + likelihood_logit)),
        eta_zero,
    )


def _sample_latent_layers_parametric(
    X,
    prop,
    B_list,
    dispersion,
    A_list,
    C,
    *,
    family,
    row_chunk_size=2048,
    rng=None,
    sparse_bernoulli=True,
    eligible_widths=None,
    temperature=1.0,
):
    """Draw a persistent sequence of systematic coordinate-Gibbs sweeps.

    Observations are vectorised, while dependent latent coordinates are updated
    sequentially.  Each Monte Carlo draw continues from the preceding sweep,
    and the final draw becomes the next persistent SAEM state.
    """
    family = resolve_parametric_family(family)
    X = np.asarray(X, dtype=float)
    C = int(C)
    if C < 1:
        raise ValueError("C must be a positive integer.")
    initial_state = [np.asarray(A, dtype=np.uint8) for A in A_list]
    widths = tuple(state.shape[1] for state in initial_state)
    if eligible_widths is None:
        eligible_widths = widths
    eligible_widths = tuple(int(width) for width in eligible_widths)
    if len(eligible_widths) != len(widths) or any(
        width < 1 or width > maximum
        for width, maximum in zip(eligible_widths, widths)
    ):
        raise ValueError(
            "eligible_widths must contain one value in [1, K_d] per layer."
        )
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    N = X.shape[0]
    samples = [
        np.zeros((N, state.shape[1] + 1, C), dtype=np.uint8)
        for state in initial_state
    ]
    for sample in samples:
        sample[:, 0, :] = 1
    top_probability_sum = np.zeros_like(initial_state[-1], dtype=float)
    random = np.random.random if rng is None else rng.random
    state = [A.copy() for A in initial_state]
    for layer, width in enumerate(eligible_widths):
        state[layer][:, width:] = 0
    use_sparse_updates = (
        bool(sparse_bernoulli) and family.key in {"bernoulli", "poisson"}
    )
    active_children_by_layer = None
    observed_child_linear = None
    if use_sparse_updates:
        active_children_by_layer = [
            [
                np.flatnonzero(
                    np.asarray(coefficient)[
                        : (
                            np.asarray(coefficient).shape[0]
                            if layer == 0
                            else eligible_widths[layer - 1]
                        ),
                        coordinate + 1,
                    ]
                    != 0.0
                )
                for coordinate in range(eligible_widths[layer])
            ]
            for layer, coefficient in enumerate(B_list)
        ]
        if family.key == "bernoulli":
            observed_child_linear = _sparse_observed_child_linear_cache(
                X, B_list[0], active_children_by_layer[0]
            )

    for draw_index in range(C):
        for layer, coefficient in enumerate(B_list):
            coefficient = np.asarray(coefficient, dtype=float)
            parent_width = eligible_widths[layer]
            child_width = (
                coefficient.shape[0]
                if layer == 0
                else eligible_widths[layer - 1]
            )
            coefficient_active = coefficient[
                :child_width, : parent_width + 1
            ]
            child_eta = (
                coefficient_active[:, 0][None, :]
                + state[layer][:, :parent_width]
                @ coefficient_active[:, 1:].T
            )
            for coordinate in range(parent_width):
                if use_sparse_updates:
                    active_children = active_children_by_layer[layer][coordinate]
                    if layer == 0 and family.key == "poisson":
                        probability, eta_zero_active = (
                            _sparse_poisson_coordinate_probability(
                                X,
                                prop,
                                B_list,
                                state,
                                coordinate,
                                child_eta,
                                active_children,
                                row_chunk_size=row_chunk_size,
                                eligible_widths=eligible_widths,
                                temperature=temperature,
                            )
                        )
                    else:
                        probability, eta_zero_active = (
                            _sparse_bernoulli_coordinate_probability(
                                X,
                                prop,
                                B_list,
                                state,
                                layer,
                                coordinate,
                                child_eta,
                                active_children,
                                child_linear=(
                                    observed_child_linear[:, coordinate]
                                    if layer == 0
                                    and observed_child_linear is not None
                                    else None
                                ),
                                row_chunk_size=row_chunk_size,
                                eligible_widths=eligible_widths,
                                temperature=temperature,
                            )
                        )
                else:
                    probability, eta_zero = _latent_coordinate_probability_parametric(
                        X,
                        prop,
                        B_list,
                        dispersion,
                        state,
                        layer,
                        coordinate,
                        family=family,
                        child_eta=child_eta,
                        row_chunk_size=row_chunk_size,
                        sparse_bernoulli=False,
                        eligible_widths=eligible_widths,
                        temperature=temperature,
                    )
                updated = (random(N) < probability).astype(np.uint8)
                state[layer][:, coordinate] = updated
                if use_sparse_updates:
                    if active_children.size:
                        active_slope = coefficient_active[
                            active_children, coordinate + 1
                        ]
                        child_eta[:, active_children] = (
                            eta_zero_active
                            + updated[:, None] * active_slope[None, :]
                        )
                else:
                    slope = coefficient_active[:, coordinate + 1]
                    eta_zero += updated[:, None] * slope[None, :]
                    child_eta = eta_zero
                if layer == len(state) - 1:
                    top_probability_sum[:, coordinate] += probability
        for layer, layer_state in enumerate(state):
            samples[layer][:, 1:, draw_index] = layer_state

    return samples, top_probability_sum / C


def _postprocess_and_measure_fixed_update(
    prop,
    prop_update,
    B_list,
    B_updates,
    dispersion,
    dispersion_update,
    *,
    family,
    tau,
    depth,
):
    """Postprocess an M-step and return its parameter-normalised change."""
    family = resolve_parametric_family(family)
    B_processed = family.postprocess_fixed_coefficients(
        B_updates, tau=tau, depth=depth
    )
    squared_error = (
        np.sum((np.asarray(prop) - np.asarray(prop_update)) ** 2)
        + sum(
            np.sum((np.asarray(old) - np.asarray(new)) ** 2)
            for old, new in zip(B_list, B_processed)
        )
        + family.dispersion_distance_sq(dispersion, dispersion_update)
    )
    n_parameters = np.asarray(prop).size + sum(
        np.asarray(B).size for B in B_list
    )
    if dispersion is not None:
        n_parameters += np.asarray(dispersion).size
    return B_processed, float(np.sqrt(squared_error / n_parameters))


def _owned_array_nbytes(owner):
    """Return bytes owned by arrays already materialized on an accumulator."""
    seen = set()
    total = 0
    for value in vars(owner).values():
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            if isinstance(item, np.ndarray) and id(item) not in seen:
                seen.add(id(item))
                total += item.nbytes
    return int(total)


def _fixed_accumulator_trace_summary(accumulator):
    """Return scalar-only diagnostics without materializing lazy designs."""
    if hasattr(accumulator, "pattern_count"):
        pattern_count = int(accumulator.pattern_count)
    elif hasattr(accumulator, "patterns"):
        pattern_count = int(accumulator.patterns.shape[0])
    elif hasattr(accumulator, "codes"):
        pattern_count = int(accumulator.codes.size)
    elif hasattr(accumulator, "_size"):
        pattern_count = int(accumulator._size)
    elif hasattr(accumulator, "counts"):
        pattern_count = int(np.count_nonzero(accumulator.counts))
    else:
        pattern_count = 0
    return {
        "backend": type(accumulator).__name__,
        "pattern_count": pattern_count,
        "storage_bytes": _owned_array_nbytes(accumulator),
    }


def _compute_bernoulli_probabilities_normal(
    X,
    prop,
    B_list,
    gamma,
    A_list,
    *,
    row_chunk_size=2048,
    coordinate_chunk_size=1,
    eligible_widths=None,
    temperature=1.0,
):
    """Compatibility wrapper for the Normal observed-family probability sweep."""
    return _compute_bernoulli_probabilities_parametric(
        X,
        prop,
        B_list,
        gamma,
        A_list,
        family=NORMAL_FAMILY,
        row_chunk_size=row_chunk_size,
        coordinate_chunk_size=coordinate_chunk_size,
        eligible_widths=eligible_widths,
        temperature=temperature,
    )
 
 
# ============================================================
#  Inner SAEM (parametric observed layer, D latent layers)
# ============================================================
 
def get_SAEM_parametric(X, prop_in, B_list_in, dispersion_in,
                        pen_list, tau, A_list_in, C, tol,
                        max_iter=50, bound_val=10.0, verbose=True,
                        estep_row_chunk_size=2048, trace=None, *, family,
                        sa_burn_in=0):
    """Run fixed-width SAEM with observed behavior supplied by ``family``.

    ``tol`` is compared with the root-mean-square parameter change after the
    family coefficient postprocessor has been applied.

    ``sa_burn_in`` is the number of initial sweeps run with step size one
    (plain Monte Carlo EM: every accumulated objective is replaced by the
    current draw) before the Robbins-Monro averaging ``1/(t - sa_burn_in)``
    starts.  With a step size of ``1/t`` from the first sweep, the draws made
    from a poor initialization keep a weight of order ``1/t`` in every later
    objective; the burn-in lets the latent states move first.  The
    convergence test starts after the burn-in.
    """
    sa_burn_in = int(sa_burn_in)
    if sa_burn_in < 0:
        raise ValueError("sa_burn_in must be a nonnegative integer.")
    family = resolve_parametric_family(family)
    N, J = X.shape
    D = len(B_list_in)
    K = [B_list_in[d].shape[1] - 1 for d in range(D)]
 
    prop   = np.array(prop_in, dtype=float).copy()
    B_list = [np.array(B, dtype=float).copy() for B in B_list_in]
    dispersion = family.copy_dispersion(dispersion_in)
    # Latent states are binary. Keeping persistent states in uint8 reduces
    # their storage eightfold; floating-point design matrices are still formed
    # at the exact locations where the likelihood needs them.
    A_list = [np.array(A, dtype=np.uint8).copy() for A in A_list_in]
 
    # Only the deepest-layer running probabilities contribute to ``prop``.
    # Keeping the unused lower-layer arrays costs N * sum(K[:-1]) float64s.
    p_top = np.zeros((N, K[-1]))

    # Response-only quantities are constant across SAEM iterations.  The
    # adapter decides which quantities its observed M-step needs.
    response_cache = family.make_response_cache(X)
 
    # MARK: Original accumulators stored Python functions that were repeatedly
    # wrapped by _sa_accumulate, creating an increasingly deep objective chain.
    # _zero = lambda x: 0.0
    # f_acc_obs = [_zero] * J
    # f_acc_bern = []
    # for d in range(1, D):
    #     f_acc_bern.append([_zero] * B_list[d].shape[0])
    f_acc_obs = family.make_fixed_accumulator(J, K[0] + 1)
    f_acc_bern = [
        _make_bernoulli_layer_accumulator(K[d], B_list[d].shape[0])
        for d in range(1, D)
    ]
 
    t = 1
    while True:
        iteration_started = time.perf_counter()
        estep_started = time.perf_counter()
        A_samples, top_probability = _sample_latent_layers_parametric(
            X,
            prop,
            B_list,
            dispersion,
            A_list,
            C,
            family=family,
            row_chunk_size=estep_row_chunk_size,
        )
        estep_seconds = time.perf_counter() - estep_started

        t_sa = max(1, t - sa_burn_in)
        p_top = (
            (1.0 - 1.0 / t_sa) * p_top
            + (1.0 / t_sa) * top_probability
        )
        prop_update = np.mean(p_top, axis=0)

        observed_mstep_started = time.perf_counter()
        B1_update, dispersion_update = family.fixed_mstep(
            X,
            A_samples[0],
            B_list[0],
            dispersion,
            f_acc_obs,
            iteration=t_sa,
            depth=D,
            penalty=pen_list[0],
            tau=tau,
            bound_val=bound_val,
            response_cache=response_cache,
        )
        mstep_seconds = [time.perf_counter() - observed_mstep_started]
 
        B_updates = [B1_update]
        for d_idx in range(1, D):
            latent_mstep_started = time.perf_counter()
            n_rows = B_list[d_idx].shape[0]
            bidx = d_idx - 1
            f_acc_bern[bidx].update(
                A_samples[d_idx - 1], A_samples[d_idx], t_sa
            )
            bounds = family.fixed_latent_bounds(K[d_idx] + 1, bound_val)
            if family.key in {"bernoulli", "poisson"}:
                B_d_upd = _solve_bernoulli_tlp_batched(
                    f_acc_bern[bidx],
                    B_list[d_idx],
                    pen_list[d_idx],
                    tau,
                    bounds,
                    max_iter=10,
                )
            else:
                B_d_upd = np.zeros_like(B_list[d_idx])
                for k in range(n_rows):
                    _pen = pen_list[d_idx]

                    def obj(x, __k=k, __pen=_pen):
                        value, grad = f_acc_bern[bidx].value_and_grad(__k, x)
                        pen_value, pen_grad = _tlp_penalty_value_grad(
                            x[1:], __pen, tau
                        )
                        grad[1:] += pen_grad
                        return value + pen_value, grad

                    res = minimize(
                        obj,
                        B_list[d_idx][k, :],
                        method="L-BFGS-B",
                        jac=True,
                        bounds=bounds,
                        options={"maxiter": 10},
                    )
                    B_d_upd[k, :] = res.x
            B_updates.append(B_d_upd)
            mstep_seconds.append(time.perf_counter() - latent_mstep_started)
 
        B_updates, err = _postprocess_and_measure_fixed_update(
            prop,
            prop_update,
            B_list,
            B_updates,
            dispersion,
            dispersion_update,
            family=family,
            tau=tau,
            depth=D,
        )

        if trace is not None:
            trace.append(
                {
                    "iteration": int(t),
                    "error": float(err),
                    "prop_update_rmse": float(
                        np.sqrt(np.mean((prop_update - prop) ** 2))
                    ),
                    "layer_update_rmse": [
                        float(np.sqrt(np.mean((new - old) ** 2)))
                        for old, new in zip(B_list, B_updates)
                    ],
                    "e_step_seconds": float(estep_seconds),
                    "m_step_seconds": [float(value) for value in mstep_seconds],
                    "iteration_seconds": float(
                        time.perf_counter() - iteration_started
                    ),
                    "accumulators": {
                        "observed": _fixed_accumulator_trace_summary(f_acc_obs),
                        "latent": [
                            _fixed_accumulator_trace_summary(accumulator)
                            for accumulator in f_acc_bern
                        ],
                    },
                }
            )
 
        A_list_new = [
            A_samples[d][:, 1:, -1].copy() for d in range(D)
        ]
 
        prop = prop_update.copy()
        B_list = [B.copy() for B in B_updates]
        dispersion = family.copy_dispersion(dispersion_update)
        A_list = A_list_new
 
        t += 1
        if verbose:
            print(f'EM Iteration {t},\t Err {err:.4g}')
        if (t > sa_burn_in + 1 and abs(err) <= tol) or t >= max_iter:
            break
 
    return prop, B_list, dispersion, A_list, t


def get_SAEM_normal(X, prop_in, B_list_in, gamma_in,
                    pen_list, tau, A_list_in, C, tol,
                    max_iter=50, bound_val=10.0, verbose=True,
                    estep_row_chunk_size=2048):
    """Compatibility wrapper for the fixed-width Normal SAEM kernel."""
    return get_SAEM_parametric(
        X,
        prop_in,
        B_list_in,
        gamma_in,
        pen_list,
        tau,
        A_list_in,
        C,
        tol,
        max_iter=max_iter,
        bound_val=bound_val,
        verbose=verbose,
        estep_row_chunk_size=estep_row_chunk_size,
        family=NORMAL_FAMILY,
    )


def get_SAEM_bernoulli(X, prop_in, B_list_in,
                       pen_list, tau, A_list_in, C, tol,
                       max_iter=50, bound_val=10.0, verbose=True,
                       estep_row_chunk_size=2048):
    """Compatibility wrapper for the fixed-width Bernoulli SAEM kernel."""
    return get_SAEM_parametric(
        X,
        prop_in,
        B_list_in,
        None,
        pen_list,
        tau,
        A_list_in,
        C,
        tol,
        max_iter=max_iter,
        bound_val=bound_val,
        verbose=verbose,
        estep_row_chunk_size=estep_row_chunk_size,
        family=BERNOULLI_FAMILY,
    )


def get_SAEM_poisson(X, prop_in, B_list_in,
                     pen_list, tau, A_list_in, C, tol,
                     max_iter=50, bound_val=10.0, verbose=True,
                     estep_row_chunk_size=2048):
    """Compatibility wrapper for the arbitrary-depth fixed Poisson kernel."""
    return get_SAEM_parametric(
        X,
        prop_in,
        B_list_in,
        None,
        pen_list,
        tau,
        A_list_in,
        C,
        tol,
        max_iter=max_iter,
        bound_val=bound_val,
        verbose=verbose,
        estep_row_chunk_size=estep_row_chunk_size,
        family=POISSON_FAMILY,
    )
 
 
# ============================================================
#  Marginal PMF of A_1  —  three modes
# ============================================================
 
def _logits_to_log_sigmoid(logits):
    """log sigma(x) and log(1-sigma(x)) via softplus, no intermediate array."""
    sp = np.logaddexp(0, logits)
    log1mPi = -sp
    logPi = logits + log1mPi
    return logPi, log1mPi
 
 
def _marginalize_one_layer_dense(pmf_upper, B, K_lower, K_upper):
    """
    Dense marginalization over all 2^{K_lower} output states.
    Requires K_lower ≤ ~20.
    """
    n_upper = pmf_upper.shape[0]
    A_upper_states = binary(np.arange(n_upper, dtype=np.int64), K_upper)
 
    n_lower = 1 << K_lower
    A_lower_states = binary(np.arange(n_lower, dtype=np.int64), K_lower)
 
    logits = B[:, 0:1] + B[:, 1:] @ A_upper_states.T
    del A_upper_states
    logPi, log1mPi = _logits_to_log_sigmoid(logits)
    del logits
 
    diff = logPi - log1mPi
    base = log1mPi.sum(axis=0)
    del logPi, log1mPi
 
    log_cond = A_lower_states @ diff + base[None, :]
    del diff
    np.exp(log_cond, out=log_cond)
    return log_cond @ pmf_upper
 
 
def _marginalize_last_layer_sparse(pmf_upper, B, K_lower, K_upper,
                                   query_patterns, chunk_size=64):
    """
    Exact sparse marginalization for the final layer.
    Requires the *parent* layer to be fully enumerable (pmf_upper has 2^{K_upper} entries).
    Only evaluates P(A_lower = a) for requested query_patterns.
 
    Memory: ~(K_lower + chunk_size) * 2^{K_upper} * 8 bytes.
    """
    n_upper = pmf_upper.shape[0]
    A_upper_states = binary(np.arange(n_upper, dtype=np.int64), K_upper)
 
    logits = B[:, 0:1] + B[:, 1:] @ A_upper_states.T
    del A_upper_states
    logPi, log1mPi = _logits_to_log_sigmoid(logits)
    del logits
 
    diff = logPi - log1mPi
    base = log1mPi.sum(axis=0)
    del logPi, log1mPi
 
    M = query_patterns.shape[0]
    probs = np.empty(M)
    Q = query_patterns.astype(float)
 
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        log_cond = Q[start:end] @ diff + base[None, :]
        np.exp(log_cond, out=log_cond)
        probs[start:end] = log_cond @ pmf_upper
 
    return probs
 
 
def _marginalize_mc_forward(pmf_enumerable, K_enum, B_chain, K_chain,
                            B_last, query_patterns,
                            mc_samples=50000, query_chunk=64):
    """
    Monte-Carlo forward-sampling marginalization for layers that are
    too large to enumerate.
 
    Strategy:
      1. Sample parent-layer states from the categorical distribution
         pmf_enumerable  (exact, no approximation loss here).
      2. Forward-sample through B_chain (one or more Bernoulli layers)
         to get A_2 samples.
      3. Evaluate P(A_1 = a | A_2) analytically for each query pattern,
         in log-space with logsumexp for numerical stability.
 
    Parameters
    ----------
    pmf_enumerable : (2**K_enum,)
        Exact PMF of the last enumerable layer.
    K_enum : int
        Number of nodes in that layer.
    B_chain : list of ndarrays
        Sequence of B matrices for forward propagation.
        B_chain[0] maps from the enumerable layer to the next layer below, etc.
        Each B has shape (K_output, K_input + 1).
    K_chain : list of int
        K_chain[i] = number of output nodes for B_chain[i] = B_chain[i].shape[0].
    B_last : (K_1, K_2 + 1) ndarray
        Coefficient matrix connecting layer 2 → layer 1.
    query_patterns : (M, K_1) ndarray
        Binary A_1 patterns to evaluate.
    mc_samples : int
        Number of Monte-Carlo forward samples.
    query_chunk : int
        Chunk size over query patterns to limit memory.
 
    Returns
    -------
    probs : (M,)   estimated P(A_1 = a) for each query pattern
    """
    S = mc_samples
 
    # ---- 1. Sample from enumerable layer (exact categorical) ----
    state_indices = np.random.choice(pmf_enumerable.shape[0], size=S,
                                     p=pmf_enumerable)
    A_current = binary(state_indices, K_enum)                # (S, K_enum)
 
    # ---- 2. Forward-sample through the chain ----
    for B_fwd in B_chain:
        K_out = B_fwd.shape[0]
        A_aug = np.column_stack([np.ones(S), A_current])     # (S, K_in + 1)
        logits = A_aug @ B_fwd.T                             # (S, K_out)
        probs_fwd = logistic(logits)
        A_current = (np.random.rand(S, K_out) < probs_fwd).astype(float)
        del A_aug, logits, probs_fwd
 
    # A_current is now A_2 samples: (S, K_2)
    # ---- 3. Analytic evaluation of P(A_1 = a | A_2) in log-space ----
    K_1 = B_last.shape[0]
    K_2 = B_last.shape[1] - 1
    assert A_current.shape[1] == K_2
 
    A2_aug = np.column_stack([np.ones(S), A_current])        # (S, K_2+1)
    eta = A2_aug @ B_last.T                                  # (S, K_1)
    del A2_aug, A_current
 
    # softplus(eta) per sample, summed over K_1 dimension
    sp_sum = np.logaddexp(0, eta).sum(axis=1)                # (S,)
 
    # For query a:  log P(a | A_2^(s)) = a @ eta^(s) - sp_sum^(s)
    # P(A_1=a) ≈ (1/S) sum_s P(A_1=a | A_2^(s))
    #           = exp( logsumexp_s(log P(a | A_2^(s))) - log S )
 
    M = query_patterns.shape[0]
    queries = query_patterns.astype(float)                   # (M, K_1)
    log_pmf = np.empty(M)
 
    # chunk over queries to keep (chunk, S) matrices bounded
    for start in range(0, M, query_chunk):
        end = min(start + query_chunk, M)
        score = queries[start:end] @ eta.T                   # (cs, S)
        log_prob = score - sp_sum[None, :]                   # (cs, S)
        log_pmf[start:end] = _logsumexp(log_prob, axis=1) - np.log(S)
        del score, log_prob
 
    return np.exp(log_pmf)
 
 
def compute_marginal_pmf_A1(prop, B_list, query_patterns=None,
                            max_enum_bits=20, mc_samples=50000):
    """
    Compute marginal PMF of A_1.
 
    Automatically selects the best strategy:
 
      Mode A (all-dense):
          All K_d ≤ max_enum_bits.  Returns full (2^{K_1},) vector.
 
      Mode B (dense + exact sparse):
          Only K_1 > max_enum_bits, but K_2 is enumerable.
          Returns (M,) exact probabilities for query_patterns.
          Memory: ~ (K_1 + 64) * 2^{K_2} * 8 bytes.
 
      Mode C (dense + MC forward):
          Multiple large layers (K_d > max_enum_bits for d ≤ some d*).
          Dense propagation to the last enumerable layer, then MC forward
          sampling through the large layers + analytic last step.
          Returns (M,) estimated probabilities for query_patterns.
          Memory: ~ (S * K_1 + M_chunk * S) * 8 bytes.
 
    Parameters
    ----------
    prop            : (K_D,)
    B_list          : list of D arrays
    query_patterns  : (M, K_1) float/int, required when K_1 > max_enum_bits
    max_enum_bits   : int (default 20)
    mc_samples      : int, MC sample count for Mode C (default 50000)
 
    Returns
    -------
    If Mode A:  (2**K_1,) ndarray
    If Mode B/C: (M,) ndarray
    """
    prop = np.asarray(prop, dtype=float)
    D = len(B_list)
    K = [B_list[d].shape[1] - 1 for d in range(D)]  # K[d] = K_{d+1}
 
    # ---- Step 1: P(A_D) from product-Bernoulli ----
    K_D = K[D - 1]
    A_D_states = binary(np.arange(1 << K_D, dtype=np.int64), K_D)
    prop_c = np.clip(prop, 1e-15, 1.0 - 1e-15)
    log_pmf = (A_D_states * np.log(prop_c)
               + (1.0 - A_D_states) * np.log(1.0 - prop_c)).sum(axis=1)
    pmf = np.exp(log_pmf)
 
    if D == 1:
        K_1 = K[0]
        if K_1 <= max_enum_bits:
            return pmf
        if query_patterns is None:
            raise ValueError("query_patterns required when K_1 > max_enum_bits")
        A = query_patterns.astype(float)
        return np.exp((A * np.log(prop_c) + (1 - A) * np.log(1 - prop_c)).sum(axis=1))
 
    # ---- Step 2: Dense propagation downward ----
    # Process d_idx = D-1, D-2, ..., 2, 1
    # At d_idx: B_list[d_idx] connects layer (d_idx+1) → layer d_idx
    #           K_lower = B_list[d_idx].shape[0]  (nodes in layer d_idx)
    #           K_upper = K[d_idx]                 (nodes in layer d_idx+1)
    #           If K_lower ≤ max_enum_bits: enumerate → dense
    #           Else: switch to MC/sparse
    mc_start_d_idx = None
    for d_idx in range(D - 1, 0, -1):
        K_lower = B_list[d_idx].shape[0]
        K_upper = K[d_idx]
 
        if K_lower <= max_enum_bits:
            pmf = _marginalize_one_layer_dense(pmf, B_list[d_idx],
                                               K_lower, K_upper)
        else:
            mc_start_d_idx = d_idx
            break
 
    # ---- Step 3: Decide mode ----
    if mc_start_d_idx is None:
        # Mode A: all dense, pmf is P(A_1)
        return pmf
 
    # K_1 > max_enum_bits  →  need query_patterns
    if query_patterns is None:
        raise ValueError("query_patterns required when K_1 > max_enum_bits")
 
    if mc_start_d_idx == 1:
        # Mode B: only the last transition (A_2 → A_1) is large.
        # pmf is over layer 2, fully enumerable.
        # K_2 = K[1], which passed the dense check at d_idx=2  →  K_2 ≤ max_enum_bits.
        K_1 = K[0]
        K_2 = K[1]
        return _marginalize_last_layer_sparse(pmf, B_list[1], K_1, K_2,
                                              query_patterns)
 
    # Mode C: mc_start_d_idx ≥ 2.  Multiple large layers.
    # pmf is over layer (mc_start_d_idx + 1), which is enumerable.
    # We need to MC-forward-sample layers mc_start_d_idx, ..., 2,
    # then analytically evaluate P(A_1 | A_2).
    K_enum = K[mc_start_d_idx]  # nodes in layer (mc_start_d_idx + 1)
 
    # Build the chain of B matrices for forward propagation:
    # d_idx = mc_start_d_idx  → produces layer mc_start_d_idx (K = B_list[d_idx].shape[0])
    # d_idx = mc_start_d_idx-1 → produces layer mc_start_d_idx-1
    # ...
    # d_idx = 2 → produces layer 2
    B_chain = [B_list[d_idx] for d_idx in range(mc_start_d_idx, 1, -1)]
    K_chain = [B_list[d_idx].shape[0] for d_idx in range(mc_start_d_idx, 1, -1)]
 
    B_last = B_list[1]  # B_2: connects layer 2 → layer 1
 
    return _marginalize_mc_forward(pmf, K_enum, B_chain, K_chain,
                                   B_last, query_patterns,
                                   mc_samples=mc_samples)


def _forward_sample_A1(prop, B_list, n_samples):
    """Draw ``n_samples`` first-layer states by sampling the hierarchy top-down."""
    prop = np.clip(np.asarray(prop, dtype=float), 1e-12, 1.0 - 1e-12)
    S = int(n_samples)
    A = (np.random.rand(S, prop.shape[0]) < prop[None, :]).astype(float)
    for B in reversed(B_list[1:]):
        logits = B[:, 0][None, :] + A @ B[:, 1:].T
        A = (np.random.rand(*logits.shape) < logistic(logits)).astype(float)
    return A


def _parent_set_g_estimate(
    Y, prop, B_list, A1, *, delta, mc_samples=50000, max_parents=20
):
    """Transformation estimate conditioning each variable on its own parents.

    Under the DDE, ``Y_j`` depends on the first latent layer only through the
    coordinates with nonzero loadings in row ``j`` of ``B_1``, so
    ``F_{Y_j | A_1 = a}`` equals ``F_{Y_j | A_1[pa(j)] = a[pa(j)]}`` and the
    mixture ``g_j(y) = sum_a pi(a) Phi^{-1}(F_{Y_j|a}(y))`` collapses to a sum
    over the parent sub-patterns.  Each class then pools every observation that
    shares the parent sub-pattern, which keeps the conditional CDFs estimable
    when ``K_1`` is far larger than ``log2 N`` (D=5 has ``K_1 = 162``), and
    the class weights ``P(A_1[pa(j)] = s)`` are estimated from forward
    samples of the fitted hierarchy.  Memory is O(N J) and time is
    O(J N log N) per call; no latent pattern is enumerated.
    """
    Y = np.asarray(Y, dtype=float)
    A1 = np.asarray(A1).astype(np.uint8)
    N, J = Y.shape
    B1 = np.asarray(B_list[0], dtype=float)
    samples = _forward_sample_A1(prop, B_list, mc_samples).astype(np.uint8)
    g_est = np.empty((N, J), dtype=float)
    for j in range(J):
        parents = np.flatnonzero(B1[j, 1:] != 0.0)
        if parents.size > max_parents:
            parents = parents[np.argsort(-np.abs(B1[j, 1 + parents]))[:max_parents]]
        if parents.size == 0:
            labels = np.zeros(N, dtype=int)
            weights = np.ones(1)
        else:
            classes, labels = np.unique(A1[:, parents], axis=0, return_inverse=True)
            labels = np.asarray(labels).reshape(-1)
            bits = 1 << np.arange(parents.size - 1, -1, -1, dtype=np.int64)
            codes_obs = classes.astype(np.int64) @ bits
            codes_smp = samples[:, parents].astype(np.int64) @ bits
            present = np.isin(codes_smp, codes_obs)
            counts = np.bincount(
                np.searchsorted(codes_obs, codes_smp[present]), minlength=codes_obs.size
            )
            weights = counts.astype(float)
            if weights.sum() <= 0.0:
                weights = np.bincount(labels, minlength=classes.shape[0]).astype(float)
            weights = weights / weights.sum()
        g_est[:, j] = g_est_rversion(
            Y[:, j : j + 1], labels, N, 1, weights.shape[0], weights, delta
        )[:, 0]
    return g_est


def _semiparametric_g_step(
    Y,
    X,
    prop,
    B_list,
    A1,
    *,
    delta,
    max_enum_bits=20,
    mc_samples=50000,
    g_update_rho=1.0,
    g_mode="auto",
):
    """Update the SemiDDE pseudo-observations for one outer iteration.

    Keeping this step independent of the inner estimator lets the fixed-width
    SAEM and CSP-MCEM paths use exactly the same semiparametric observation
    update.  ``B_list`` and ``A1`` may already be compressed to the currently
    active CSP coordinates.

    ``g_mode`` selects the transformation estimator: ``"pattern"`` conditions
    on the full first-layer pattern (dense enumeration when
    ``K_1 <= max_enum_bits``, otherwise unique observed patterns with
    marginal probabilities from :func:`compute_marginal_pmf_A1`);
    ``"parents"`` conditions each variable on its own parent coordinates
    (:func:`_parent_set_g_estimate`); ``"auto"`` uses ``"pattern"`` while
    ``K_1 <= max_enum_bits`` and ``"parents"`` beyond that.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    A1 = np.asarray(A1)
    N, J = Y.shape
    K1 = int(A1.shape[1])
    sparse_mode = K1 > int(max_enum_bits)
    g_mode = str(g_mode).strip().lower()
    if g_mode not in {"auto", "pattern", "parents"}:
        raise ValueError("g_mode must be 'auto', 'pattern', or 'parents'.")
    if g_mode == "auto":
        g_mode = "parents" if sparse_mode else "pattern"

    if g_mode == "parents":
        g_new = _parent_set_g_estimate(
            Y, prop, B_list, A1, delta=delta, mc_samples=mc_samples
        )
        X_next = (1.0 - g_update_rho) * X + g_update_rho * g_new
        diff = np.mean(np.abs(X_next - X)) / (np.mean(np.abs(X)) + 1e-8)
        return X_next, g_new, float(diff)

    if sparse_mode:
        # Row-wise unique patterns: an integer encoding overflows int64 once
        # K_1 exceeds 62 (D=5 has K_1=162), so classes are formed directly
        # from the binary rows.
        unique_patterns, dec = np.unique(
            A1.astype(np.uint8), axis=0, return_inverse=True
        )
        dec = np.asarray(dec).reshape(-1)
        pi_alpha = compute_marginal_pmf_A1(
            prop,
            B_list,
            query_patterns=unique_patterns.astype(float),
            max_enum_bits=max_enum_bits,
            mc_samples=mc_samples,
        )
        # The marginal mass of the observed patterns is far below one when
        # K_1 is large (each pattern has probability about 2**-K_1), and the
        # transformation estimator standardizes each column with an absolute
        # floor on the spread.  Renormalizing over the observed classes keeps
        # the class weights on the O(1) scale of the dense mode, where the
        # full PMF sums to one.
        total_mass = float(np.sum(pi_alpha))
        if total_mass > 0.0:
            pi_alpha = pi_alpha / total_mass
        else:
            pi_alpha = np.full(pi_alpha.shape[0], 1.0 / pi_alpha.shape[0])
        n_classes = unique_patterns.shape[0]
    else:
        weights = np.int64(1) << np.arange(
            K1 - 1, -1, -1, dtype=np.int64
        )
        dec = (A1.astype(np.int64) @ weights).astype(int)
        n_classes = 1 << K1
        pi_alpha = compute_marginal_pmf_A1(
            prop,
            B_list,
            max_enum_bits=max_enum_bits,
            mc_samples=mc_samples,
        )

    g_new = g_est_rversion(
        Y, dec, N, J, n_classes, pi_alpha, delta
    )
    X_next = (1.0 - g_update_rho) * X + g_update_rho * g_new
    diff = np.mean(np.abs(X_next - X)) / (
        np.mean(np.abs(X)) + 1e-8
    )
    return X_next, g_new, float(diff)
 
 
# ============================================================
#  Semiparametric SAEM
# ============================================================
 
def get_SAEM_semiparametricv1(
    Y, X_in, prop_in, B_list_in, gamma_in,
    pen_list, tau,
    A_list_in,
    tol, C,
    X_true=None,
    delta=None,
    max_outer_iter=10,
    max_inner_iter=50,
    max_enum_bits=20,
    mc_samples=50000,
    g_update_rho=1.0,
    g_mode="auto",
    g_tol=1e-2,
    final_refit=False,
    final_refit_B_list_true=None,
    final_refit_G_list_true=None,
    final_refit_epsilon=1e-4,
    final_refit_max_iter=None,
    verbose=True,
):
    """
    Semi-parametric SAEM for D-layer model.
 
    Outer loop: update g-functions.
    Inner loop: Normal SAEM on X = g(Y).
 
    When K_1 > max_enum_bits (sparse/MC mode):
        dec       : (N,) int — remapped class labels in [0, n_classes)
        n_classes : int      — number of *unique* A_1 patterns in the data
        pi_alpha  : (n_classes,) — marginal probability for each class
 
    When K_1 ≤ max_enum_bits (dense mode, backward compatible):
        dec       : (N,) int — integer encoding of A_1 in [0, 2^K_1)
        n_classes : int = 2^K_1
        pi_alpha  : (2^K_1,) — full marginal PMF
 
    Returns
    -------
    X, prop, B_list, gamma, A_list, itera
    """
    Y = np.asarray(Y, dtype=float)
    N, J = Y.shape
    X = np.asarray(X_in, dtype=float).copy()
    if not (0.0 < g_update_rho <= 1.0):
        raise ValueError("g_update_rho must be in (0, 1].")
 
    if delta is None:
        delta = min(1.0 / (4.0 * (N ** 0.25) * np.sqrt(np.pi * np.log(N))),
                    1e-12)
 
    D = len(B_list_in)
    K = [B_list_in[d].shape[1] - 1 for d in range(D)]
    K1 = K[0]
 
    prop   = np.asarray(prop_in, dtype=float).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list_in]
    gamma  = np.asarray(gamma_in, dtype=float).copy()
    A_list = [np.asarray(A, dtype=float).copy() for A in A_list_in]
 
    itera = 0
 
    for outer_iter in range(max_outer_iter):
        if verbose:
            print(f"\n===== N {N} | Outer iteration {outer_iter + 1} =====")
 
        prop, B_list, gamma, A_list, t = get_SAEM_normal(
            X, prop, B_list, gamma,
            pen_list, tau, A_list, C, tol,
            max_iter=max_inner_iter, verbose=verbose)
        itera += t
 
        if verbose:
            print("Updating g ...")
 
        X_next, g_new, diff = _semiparametric_g_step(
            Y,
            X,
            prop,
            B_list,
            A_list[0],
            delta=delta,
            max_enum_bits=max_enum_bits,
            mc_samples=mc_samples,
            g_update_rho=g_update_rho,
            g_mode=g_mode,
        )
 
        if X_true is not None and verbose:
            rmse = np.sqrt(np.mean((X_true - g_new) ** 2))
            print(f"RMSE(X_true, g_new) = {rmse:.4f}")
 
        # MARK: Original semiparametric outer loop fully replaced X by g_new at
        # every iteration, which can amplify deep-layer latent drift.
        # diff = np.mean(np.abs(g_new - X)) / (np.mean(np.abs(X)) + 1e-8)
        # if verbose:
        #     print(f"diff g = {diff:.3e}")
        # if diff < 1e-2:
        #     if verbose:
        #         print("Converged on g.")
        #     X = g_new.copy()
        #     break
        # X = g_new.copy()
        if verbose:
            print(f"diff g = {diff:.3e}")
        if diff < g_tol:
            if verbose:
                print("Converged on g.")
            X = X_next.copy()
            break
        X = X_next.copy()
 
    # MARK: Original semiparametric estimator returned the warm-start parameters
    # from the last outer iteration directly.
    # return X, prop, B_list, gamma, A_list, itera
    if final_refit:
        if final_refit_B_list_true is None or final_refit_G_list_true is None:
            raise ValueError(
                "final_refit=True requires final_refit_B_list_true and "
                "final_refit_G_list_true for the current research initializer."
            )
        prop_refit, B_list_refit, gamma_refit, A_list_refit, _ = Normal_init_semi(
            X,
            K1,
            K,
            final_refit_B_list_true,
            final_refit_G_list_true,
            final_refit_epsilon,
        )
        prop, B_list, gamma, A_list, t_refit = get_SAEM_normal(
            X,
            prop_refit,
            B_list_refit,
            gamma_refit,
            pen_list,
            tau,
            A_list_refit,
            C,
            tol,
            max_iter=max_inner_iter if final_refit_max_iter is None else final_refit_max_iter,
            verbose=verbose,
        )
        itera += t_refit

    return X, prop, B_list, gamma, A_list, itera
