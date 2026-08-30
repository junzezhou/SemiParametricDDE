"""
Penalized SAEM for D-layer Normal-observed deep discrete latent model
+ Semiparametric outer iteration for g-function estimation.
 
Three-mode marginal PMF computation for P(A_1):
  Mode A  (all-dense):   every K_d <= max_enum_bits  →  exact, full 2^{K_1} vector
  Mode B  (dense+sparse): only K_1 is large, K_2 enumerable  →  exact for query patterns
  Mode C  (dense+MC):     multiple large layers  →  MC forward sampling + analytic last step
 
Model (1-based layers):
  Layer D (top):   A_D ~ Bernoulli(prop)
  Layer d (1<d<D): A_d | A_{d+1} ~ Bernoulli(logistic(B_{d+1} @ [1, A_{d+1}]))
  Layer 0 (obs):   X | A_1 ~ Normal(mean = B_1 @ [1, A_1], var = gamma)
 
Python list indexing (0-based):
  B_list[d] = B_{d+1}     d=0 => B_1 (J, K_1+1);  d>=1 => B_{d+1} (K_d, K_{d+1}+1)
  A_list[d] = A_{d+1}     (N, K_{d+1})
  pen_list[d]              penalty magnitude for B_list[d]
  K[d] = K_{d+1}           number of nodes in layer d+1
"""
 
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp as _logsumexp
from dde.utils import TLP, binary, logistic, patterns_to_int, thres, g_est_rversion
from .normal_init_semi import Normal_init_semi


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
  
  
# MARK: Original SA objective accumulation built a nested closure chain, so each
# objective evaluation became more expensive as the SAEM inner iteration grew.
# def _sa_accumulate(f_old, f_new, t):
#     alpha = 1.0 / t
#     def g(x, _fo=f_old, _fn=f_new, _a=alpha):
#         return (1.0 - _a) * _fo(x) + _a * _fn(x)
#     return g


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
        x_sq_sum = np.sum(X ** 2, axis=0)
    residual_sum_squares = x_sq_sum - 2.0 * cross + quadratic
    gamma_update = residual_sum_squares / X.shape[0]
    return B_update, gamma_update


def _matlab_debias(x, tau):
    """Match the deep-Normal MATLAB ``debias.m`` elementwise."""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    out[x > tau] = x[x > tau] + tau
    out[x < tau] = x[x < tau] - tau
    out[np.abs(x) < tau] = 0.0
    return out


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
        children = np.moveaxis(A_lower_samples[:, 1:, :], 1, 2).reshape(
            -1, self.n_children
        )
        weights = np.int64(1) << np.arange(self.K_upper - 1, -1, -1, dtype=np.int64)
        codes_flat = parents.astype(np.int64) @ weights

        new_codes, inverse = np.unique(codes_flat, return_inverse=True)
        new_counts = np.bincount(inverse).astype(float) / C
        new_sum_y = np.zeros((new_codes.size, self.n_children), dtype=float)
        np.add.at(new_sum_y, inverse, children)
        new_sum_y /= C

        all_codes = np.union1d(self.codes, new_codes)
        old_counts = np.zeros(all_codes.size, dtype=float)
        old_sum_y = np.zeros((all_codes.size, self.n_children), dtype=float)
        new_counts_aligned = np.zeros(all_codes.size, dtype=float)
        new_sum_y_aligned = np.zeros((all_codes.size, self.n_children), dtype=float)

        if self.codes.size:
            old_pos = np.searchsorted(all_codes, self.codes)
            old_counts[old_pos] = self.counts
            old_sum_y[old_pos, :] = self.sum_y
        new_pos = np.searchsorted(all_codes, new_codes)
        new_counts_aligned[new_pos] = new_counts
        new_sum_y_aligned[new_pos, :] = new_sum_y

        self.codes = all_codes
        self.counts = (1.0 - alpha) * old_counts + alpha * new_counts_aligned
        self.sum_y = (1.0 - alpha) * old_sum_y + alpha * new_sum_y_aligned
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
        children = np.moveaxis(
            A_lower_samples[:, 1:, :], 1, 2
        ).reshape(-1, self.n_children)
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
        self._capacity = 0
        self._size = 0
        self._parents = np.empty((0, self.K_upper), dtype=float)
        self._packed_children = np.empty(
            (0, self._packed_child_width), dtype=np.uint8
        )
        self._row_weight = 0.0
        self._draws_per_update = None
        self._response_stats = np.zeros(
            (self.n_children, self.K_upper + 1), dtype=float
        )

    def _ensure_capacity(self, required):
        if required <= self._capacity:
            return
        new_capacity = (
            required if self._capacity == 0 else max(required, 2 * self._capacity)
        )
        parents = np.empty((new_capacity, self.K_upper), dtype=float)
        packed_children = np.empty(
            (new_capacity, self._packed_child_width), dtype=np.uint8
        )
        if self._size:
            parents[:self._size] = self._parents[:self._size]
            packed_children[:self._size] = self._packed_children[:self._size]
        self._parents = parents
        self._packed_children = packed_children
        self._capacity = new_capacity

    def update(self, A_lower_samples, A_upper_samples, t):
        alpha = 1.0 / t
        C = A_upper_samples.shape[2]
        new_row_weight = alpha / C
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
        children = np.moveaxis(
            A_lower_samples[:, 1:, :], 1, 2
        ).reshape(-1, self.n_children)
        parents = np.ascontiguousarray(parents, dtype=float)
        children_float = np.ascontiguousarray(children, dtype=float)
        packed_children = np.packbits(
            np.ascontiguousarray(children, dtype=np.uint8),
            axis=1,
            bitorder="little",
        )

        current_response_stats = np.empty_like(self._response_stats)
        current_response_stats[:, 0] = np.sum(children_float, axis=0) / C
        current_response_stats[:, 1:] = children_float.T @ parents / C
        self._response_stats = (
            (1.0 - alpha) * self._response_stats
            + alpha * current_response_stats
        )

        end = self._size + parents.shape[0]
        self._ensure_capacity(end)
        self._parents[self._size:end] = parents
        self._packed_children[self._size:end] = packed_children
        self._size = end

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
            self._parents.nbytes
            + self._packed_children.nbytes
            + self._response_stats.nbytes
        )

    def value(self, child_index, x):
        x = np.asarray(x, dtype=float)
        parents = self._parents[:self._size]
        raw_eta = x[0] + parents @ x[1:]
        eta = np.clip(raw_eta, -500, 500)
        positive_value = self._row_weight * np.sum(np.logaddexp(0.0, eta))
        if np.all((raw_eta > -500.0) & (raw_eta < 500.0)):
            return float(
                positive_value - self._response_stats[child_index] @ x
            )

        # Clipping makes the response term locally nonlinear. Retain the
        # packed child history solely for this rare boundary fallback so the
        # result stays identical to the previous implementation.
        y = self._child_column(
            self._packed_children[:self._size], child_index
        )
        return float(
            self._row_weight
            * np.sum(-y * eta + np.logaddexp(0.0, eta))
        )

    def value_and_grad(self, child_index, x):
        x = np.asarray(x, dtype=float)
        parents = self._parents[:self._size]
        raw_eta = x[0] + parents @ x[1:]
        eta = np.clip(raw_eta, -500, 500)
        interior = (raw_eta > -500.0) & (raw_eta < 500.0)
        if np.all(interior):
            positive_residual = self._row_weight * logistic(eta)
            grad = np.empty_like(x)
            grad[0] = np.sum(positive_residual)
            grad[1:] = parents.T @ positive_residual
            grad -= self._response_stats[child_index]
            value = float(
                self._row_weight * np.sum(np.logaddexp(0.0, eta))
                - self._response_stats[child_index] @ x
            )
            return value, grad

        y = self._child_column(
            self._packed_children[:self._size], child_index
        )
        value = float(
            self._row_weight
            * np.sum(-y * eta + np.logaddexp(0.0, eta))
        )
        residual = self._row_weight * (logistic(eta) - y)
        residual *= interior
        grad = np.empty_like(x)
        grad[0] = np.sum(residual)
        grad[1:] = parents.T @ residual
        return value, grad


_DIRECT_ADDRESS_MAX_BITS = 10
_GROUPED_PATTERN_MAX_BITS = 20


def _make_bernoulli_layer_accumulator(K_upper, n_children):
    K_upper = int(K_upper)
    if K_upper <= _DIRECT_ADDRESS_MAX_BITS:
        return _DirectAddressBernoulliLayerAccumulator(K_upper, n_children)
    if K_upper <= _GROUPED_PATTERN_MAX_BITS:
        return _BernoulliLayerAccumulator(K_upper, n_children)
    return _BinaryHistoryBernoulliLayerAccumulator(K_upper, n_children)


def _compute_bernoulli_probabilities_normal(
    X,
    prop,
    B_list,
    gamma,
    A_list,
    *,
    row_chunk_size=2048,
    temperature=1.0,
):
    """Compute one synchronous latent-probability sweep for the Normal DDE.

    This is algebraically equivalent to the original coordinate loops: every
    layer is conditioned on the same incoming ``A_list`` rather than on values
    sampled earlier in the sweep.
    """
    X = np.asarray(X, dtype=float)
    prop = np.asarray(prop, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    N = X.shape[0]
    D = len(B_list)
    K = [B.shape[1] - 1 for B in B_list]
    ber = []
    if row_chunk_size is None:
        row_chunk_size = N
    row_chunk_size = max(1, int(row_chunk_size))

    for d_idx in range(D):
        if d_idx == D - 1:
            prop_clipped = np.clip(prop, 1e-15, 1.0 - 1e-15)
            prior_logit = np.broadcast_to(
                np.log(prop_clipped / (1.0 - prop_clipped)),
                (N, K[d_idx]),
            )
        else:
            B_above = B_list[d_idx + 1]
            prior_logit = (
                B_above[:, 0][None, :]
                + A_list[d_idx + 1] @ B_above[:, 1:].T
            )

        if d_idx == 0:
            B1 = B_list[0]
            slopes = B1[:, 1:]
            weighted_slopes = slopes / gamma[:, None]
            slope_gram = slopes.T @ weighted_slopes
            slope_norms = np.diag(slope_gram)
            like_logit = np.empty((N, K[d_idx]), dtype=float)
            for start in range(0, N, row_chunk_size):
                stop = min(start + row_chunk_size, N)
                A_chunk = A_list[0][start:stop]
                data_linear = (
                    (X[start:stop] - B1[:, 0][None, :])
                    / gamma[None, :]
                ) @ slopes
                like_logit[start:stop] = (
                    data_linear
                    - A_chunk @ slope_gram
                    + (A_chunk - 0.5) * slope_norms[None, :]
                )
        else:
            B_cur = B_list[d_idx]
            A_cur = A_list[d_idx]
            A_below = A_list[d_idx - 1]
            like_logit = np.empty((N, K[d_idx]), dtype=float)

            # Broadcasting all coordinates at once would require an
            # N x K_{d-1} x K_d array. Loop over K_d while reusing base_eta
            # and over observation blocks to keep peak memory independent of N.
            for start in range(0, N, row_chunk_size):
                stop = min(start + row_chunk_size, N)
                A_cur_chunk = A_cur[start:stop]
                base_eta = (
                    B_cur[:, 0][None, :]
                    + A_cur_chunk @ B_cur[:, 1:].T
                )
                base_softplus = np.logaddexp(
                    0.0, np.clip(base_eta, -500, 500)
                )
                child_linear = A_below[start:stop] @ B_cur[:, 1:]
                for l in range(K[d_idx]):
                    slope_l = B_cur[:, l + 1]
                    # For binary a_l, one shifted softplus plus the shared
                    # base value exactly represents softplus(eta_1) -
                    # softplus(eta_0).  This halves the expensive logaddexp
                    # work without constructing an N x K_{d-1} x K_d array.
                    direction = 1.0 - 2.0 * A_cur_chunk[:, l].astype(float)
                    shifted_eta = np.clip(
                        base_eta + direction[:, None] * slope_l[None, :],
                        -500,
                        500,
                    )
                    log_normalizer_delta = direction * np.sum(
                        np.logaddexp(0.0, shifted_eta) - base_softplus,
                        axis=1,
                    )
                    like_logit[start:stop, l] = (
                        child_linear[:, l] - log_normalizer_delta
                    )

        ber.append(logistic(float(temperature) * (prior_logit + like_logit)))

    return ber
 
 
# ============================================================
#  Inner SAEM (Normal observed layer, D latent layers)
# ============================================================
 
def get_SAEM_normal(X, prop_in, B_list_in, gamma_in,
                    pen_list, tau, A_list_in, C, tol,
                    max_iter=50, bound_val=10.0, verbose=True,
                    estep_row_chunk_size=2048):
    N, J = X.shape
    D = len(B_list_in)
    K = [B_list_in[d].shape[1] - 1 for d in range(D)]
 
    prop   = np.array(prop_in, dtype=float).copy()
    B_list = [np.array(B, dtype=float).copy() for B in B_list_in]
    gamma  = np.array(gamma_in, dtype=float).copy()
    # Latent states are binary. Keeping persistent states in uint8 reduces
    # their storage eightfold; floating-point design matrices are still formed
    # at the exact locations where the likelihood needs them.
    A_list = [np.array(A, dtype=np.uint8).copy() for A in A_list_in]
 
    # Only the deepest-layer running probabilities contribute to ``prop``.
    # Keeping the unused lower-layer arrays costs N * sum(K[:-1]) float64s.
    p_top = np.zeros((N, K[-1]))

    # This response-only statistic is constant across SAEM iterations.  Passing
    # it to the closed-form update avoids allocating X ** 2 on every iteration.
    x_sq_sum = np.sum(X ** 2, axis=0)
 
    # MARK: Original accumulators stored Python functions that were repeatedly
    # wrapped by _sa_accumulate, creating an increasingly deep objective chain.
    # _zero = lambda x: 0.0
    # f_acc_obs = [_zero] * J
    # f_acc_bern = []
    # for d in range(1, D):
    #     f_acc_bern.append([_zero] * B_list[d].shape[0])
    f_acc_obs = _ObservedNormalAccumulator(J, K[0] + 1)
    f_acc_bern = [
        _make_bernoulli_layer_accumulator(K[d], B_list[d].shape[0])
        for d in range(1, D)
    ]
 
    t = 1
    while True:
        ber = _compute_bernoulli_probabilities_normal(
            X,
            prop,
            B_list,
            gamma,
            A_list,
            row_chunk_size=estep_row_chunk_size,
        )
 
        A_samples = []
        for d in range(D):
            raw = np.random.rand(N, K[d], C) < ber[d][:, :, None]
            augmented = np.empty((N, K[d] + 1, C), dtype=np.uint8)
            augmented[:, 0, :] = 1
            augmented[:, 1:, :] = raw
            A_samples.append(augmented)
 
        p_top = (1.0 - 1.0 / t) * p_top + (1.0 / t) * ber[-1]
        prop_update = np.mean(p_top, axis=0)

        Q_current, q_current = f_acc_obs.update(X, A_samples[0], t)
        if D >= 3:
            B1_update, gamma_update = _update_deep_normal_observed(
                X, Q_current, q_current, tau, x_sq_sum=x_sq_sum
            )
        else:
            B1_update = np.zeros_like(B_list[0])
            gamma_update = np.zeros(J)
            for j in range(J):
                Xj = X[:, j]
                _pen = pen_list[0]
                def obj(x, __j=j, __pen=_pen):
                    value, grad = f_acc_obs.value_and_grad(__j, x)
                    pen_value, pen_grad = _tlp_penalty_value_grad(
                        x[1:], __pen, tau
                    )
                    grad[1:] += pen_grad
                    return value + pen_value, grad
                bounds = [(-bound_val, bound_val)] * (K[0] + 1)
                res = minimize(obj, B_list[0][j, :], method='L-BFGS-B',
                               jac=True, bounds=bounds, options={'maxiter': 10})
                B1_update[j, :] = res.x
                eta = np.tensordot(A_samples[0], res.x, axes=([1], [0]))
                gamma_update[j] = np.sum((Xj[:, None] - eta) ** 2) / (N * C)
 
        B_updates = [B1_update]
        for d_idx in range(1, D):
            n_rows = B_list[d_idx].shape[0]
            B_d_upd = np.zeros_like(B_list[d_idx])
            bidx = d_idx - 1
            f_acc_bern[bidx].update(
                A_samples[d_idx - 1], A_samples[d_idx], t
            )
            for k in range(n_rows):
                _pen = pen_list[d_idx]
                def obj(x, __k=k, __pen=_pen):
                    value, grad = f_acc_bern[bidx].value_and_grad(__k, x)
                    pen_value, pen_grad = _tlp_penalty_value_grad(
                        x[1:], __pen, tau
                    )
                    grad[1:] += pen_grad
                    return value + pen_value, grad
                bounds = [(-bound_val, bound_val)] * (K[d_idx] + 1)
                res = minimize(obj, B_list[d_idx][k, :], method='L-BFGS-B',
                               jac=True, bounds=bounds, options={'maxiter': 10})
                B_d_upd[k, :] = res.x
            B_updates.append(B_d_upd)
 
        err = np.sqrt(
            np.sum((prop - prop_update) ** 2)
            + sum(np.sum((B_list[d] - B_updates[d]) ** 2) for d in range(D))
            + np.sum((gamma - gamma_update) ** 2))
 
        A_list_new = []
        for d in range(D):
            A_list_new.append(
                (np.mean(A_samples[d][:, 1:, :], axis=2) > 0.5).astype(np.uint8))
 
        prop = prop_update.copy()
        for d in range(D):
            if D >= 3:
                B_updates[d][:, 1:] = _matlab_debias(
                    B_updates[d][:, 1:], tau
                )
            else:
                B_updates[d][:, 1:] = thres(B_updates[d][:, 1:], tau)
        B_list = [B.copy() for B in B_updates]
        gamma = gamma_update.copy()
        A_list = A_list_new
 
        t += 1
        if verbose:
            print(f'EM Iteration {t},\t Err {err:.1f}')
        if abs(err) <= tol or t >= max_iter:
            break
 
    return prop, B_list, gamma, A_list, t
 
 
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
    sparse_mode = (K1 > max_enum_bits)
 
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
 
        if sparse_mode:
            dec_full = patterns_to_int(A_list[0])
            unique_vals, inverse = np.unique(dec_full, return_inverse=True)
            n_classes = unique_vals.shape[0]
 
            unique_patterns = binary(unique_vals, K1)
            unique_probs = compute_marginal_pmf_A1(
                prop, B_list,
                query_patterns=unique_patterns,
                max_enum_bits=max_enum_bits,
                mc_samples=mc_samples)
 
            dec = inverse
            pi_alpha = unique_probs
        else:
            weights = np.int64(1) << np.arange(K1 - 1, -1, -1, dtype=np.int64)
            dec = (A_list[0].astype(np.int64) @ weights).astype(int)
            n_classes = 1 << K1
            pi_alpha = compute_marginal_pmf_A1(
                prop, B_list,
                max_enum_bits=max_enum_bits)
 
        g_new = g_est_rversion(Y, dec, N, J, n_classes, pi_alpha, delta)
 
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
        X_next = (1.0 - g_update_rho) * X + g_update_rho * g_new
        diff = np.mean(np.abs(X_next - X)) / (np.mean(np.abs(X)) + 1e-8)
        if verbose:
            print(f"diff g = {diff:.3e}")
        if diff < 1e-2:
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
