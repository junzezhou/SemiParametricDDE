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
from semidde.utils import binary, g_est_rversion
from .normal_init_semi import Normal_init_semi
 
 
# ============================================================
#  Utility functions
# ============================================================
 
def logistic(x):
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
 
 
def TLP(x, tau):
    return np.sum(np.minimum(np.abs(x) / tau, 1.0))
 
 
def thres(B, tau):
    return B * (np.abs(B) >= tau)
 
 
def binary(indices, n_bits):
    """Integer indices → (M, n_bits) binary matrix (MSB first)."""
    indices = np.asarray(indices, dtype=np.int64)
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.int64)
    return ((indices[:, None] >> shifts) & 1).astype(float)
 
 
def patterns_to_int(A):
    """(M, K) binary matrix → (M,) int64  (MSB-first, K ≤ 62)."""
    A = np.asarray(A, dtype=np.int64)
    K = A.shape[1]
    weights = np.int64(1) << np.arange(K - 1, -1, -1, dtype=np.int64)
    return A @ weights
 
 
# ============================================================
#  Negative log-likelihood factories
# ============================================================
 
def _make_obs_negloglik(Xj, A1_samples, N, K1, C):
    def f(x):
        eta = np.tensordot(A1_samples, x, axes=([1], [0]))
        return np.sum((Xj[:, None] - eta) ** 2) / C
    return f
 
 
def _make_bern_negloglik(Yk, A_upper_samples, N, K_upper, C):
    def f(x):
        eta = np.tensordot(A_upper_samples, x, axes=([1], [0]))
        eta = np.clip(eta, -500, 500)
        return -(np.sum(Yk * eta) - np.sum(np.log(1.0 + np.exp(eta)))) / C
    return f
 
 
# MARK: Original SA objective accumulation built a nested closure chain, so each
# objective evaluation became more expensive as the SAEM inner iteration grew.
# def _sa_accumulate(f_old, f_new, t):
#     alpha = 1.0 / t
#     def g(x, _fo=f_old, _fn=f_new, _a=alpha):
#         return (1.0 - _a) * _fo(x) + _a * _fn(x)
#     return g


class _ObsQuadraticAccumulator:
    """Running-average quadratic objective for the observed Normal layer."""

    def __init__(self, Xj, n_features):
        self.Xj = np.asarray(Xj, dtype=float)
        self.Q = np.zeros((n_features, n_features), dtype=float)
        self.q = np.zeros(n_features, dtype=float)

    def update(self, A_samples, t):
        alpha = 1.0 / t
        C = A_samples.shape[2]
        Q_new = np.einsum("nkc,nlc->kl", A_samples, A_samples, optimize=True) / C
        q_new = np.einsum("nkc,n->k", A_samples, self.Xj, optimize=True) / C
        self.Q = (1.0 - alpha) * self.Q + alpha * Q_new
        self.q = (1.0 - alpha) * self.q + alpha * q_new

    def __call__(self, x):
        return float(x @ self.Q @ x - 2.0 * self.q @ x)


class _BernoulliPatternAccumulator:
    """Running-average Bernoulli objective aggregated by parent latent pattern."""

    def __init__(self, K_upper):
        self.K_upper = int(K_upper)
        self.codes = np.empty(0, dtype=np.int64)
        self.counts = np.empty(0, dtype=float)
        self.sum_y = np.empty(0, dtype=float)

    def update(self, Yk, A_upper_samples, t):
        alpha = 1.0 / t
        C = A_upper_samples.shape[2]
        parents = np.moveaxis(A_upper_samples[:, 1:, :], 1, 2).reshape(-1, self.K_upper)
        weights = np.int64(1) << np.arange(self.K_upper - 1, -1, -1, dtype=np.int64)
        codes_flat = parents.astype(np.int64) @ weights
        y_flat = np.asarray(Yk, dtype=float).reshape(-1)

        new_codes, inverse = np.unique(codes_flat, return_inverse=True)
        new_counts = np.bincount(inverse).astype(float) / C
        new_sum_y = np.bincount(inverse, weights=y_flat).astype(float) / C

        all_codes = np.union1d(self.codes, new_codes)
        old_counts = np.zeros(all_codes.size, dtype=float)
        old_sum_y = np.zeros(all_codes.size, dtype=float)
        new_counts_aligned = np.zeros(all_codes.size, dtype=float)
        new_sum_y_aligned = np.zeros(all_codes.size, dtype=float)

        if self.codes.size:
            old_pos = np.searchsorted(all_codes, self.codes)
            old_counts[old_pos] = self.counts
            old_sum_y[old_pos] = self.sum_y
        new_pos = np.searchsorted(all_codes, new_codes)
        new_counts_aligned[new_pos] = new_counts
        new_sum_y_aligned[new_pos] = new_sum_y

        self.codes = all_codes
        self.counts = (1.0 - alpha) * old_counts + alpha * new_counts_aligned
        self.sum_y = (1.0 - alpha) * old_sum_y + alpha * new_sum_y_aligned

    def __call__(self, x):
        parent_states = binary(self.codes, self.K_upper)
        eta = x[0] + parent_states @ x[1:]
        eta = np.clip(eta, -500, 500)
        return float(-(self.sum_y @ eta) + self.counts @ np.logaddexp(0.0, eta))
 
 
# ============================================================
#  Inner SAEM (Normal observed layer, D latent layers)
# ============================================================
 
def get_SAEM_normal(X, prop_in, B_list_in, gamma_in,
                    pen_list, tau, A_list_in, C, tol,
                    max_iter=50, bound_val=10.0, verbose=True):
    N, J = X.shape
    D = len(B_list_in)
    K = [B_list_in[d].shape[1] - 1 for d in range(D)]
 
    prop   = np.array(prop_in, dtype=float).copy()
    B_list = [np.array(B, dtype=float).copy() for B in B_list_in]
    gamma  = np.array(gamma_in, dtype=float).copy()
    A_list = [np.array(A, dtype=float).copy() for A in A_list_in]
 
    p = [np.zeros((N, K[d])) for d in range(D)]
 
    # MARK: Original accumulators stored Python functions that were repeatedly
    # wrapped by _sa_accumulate, creating an increasingly deep objective chain.
    # _zero = lambda x: 0.0
    # f_acc_obs = [_zero] * J
    # f_acc_bern = []
    # for d in range(1, D):
    #     f_acc_bern.append([_zero] * B_list[d].shape[0])
    f_acc_obs = [_ObsQuadraticAccumulator(X[:, j], K[0] + 1) for j in range(J)]
    f_acc_bern = [
        [_BernoulliPatternAccumulator(K[d]) for _ in range(B_list[d].shape[0])]
        for d in range(1, D)
    ]
 
    t = 1
    while True:
        ber = [np.zeros((N, K[d])) for d in range(D)]
 
        for d_idx in range(D):
            for l in range(K[d_idx]):
                if d_idx == D - 1:
                    p_l = np.clip(prop[l], 1e-15, 1.0 - 1e-15)
                    prior_logit = np.full(N, np.log(p_l / (1.0 - p_l)))
                else:
                    B_above = B_list[d_idx + 1]
                    A_above = A_list[d_idx + 1]
                    A_above_aug = np.column_stack([np.ones(N), A_above])
                    prior_logit = A_above_aug @ B_above[l, :]
 
                if d_idx == 0:
                    B1 = B_list[0]; A1 = A_list[0]
                    cols_keep = [c for c in range(K[0] + 1) if c != l + 1]
                    A1_minus_l = np.delete(A1, l, axis=1)
                    A1_aug_minus_l = np.column_stack([np.ones(N), A1_minus_l])
                    residual = X - A1_aug_minus_l @ B1[:, cols_keep].T
                    like_logit = -np.sum(
                        (B1[:, l + 1] ** 2
                         - 2.0 * residual * B1[None, :, l + 1])
                        / gamma[None, :], axis=1) / 2.0
                else:
                    B_cur = B_list[d_idx]
                    A_below = A_list[d_idx - 1]
                    A_cur = A_list[d_idx]
                    A_cur_1 = A_cur.copy(); A_cur_1[:, l] = 1.0
                    A_cur_0 = A_cur.copy(); A_cur_0[:, l] = 0.0
                    A_aug_1 = np.column_stack([np.ones(N), A_cur_1])
                    A_aug_0 = np.column_stack([np.ones(N), A_cur_0])
                    eta_1 = np.clip(A_aug_1 @ B_cur.T, -500, 500)
                    eta_0 = np.clip(A_aug_0 @ B_cur.T, -500, 500)
                    like_logit = (
                        np.sum(A_below * B_cur[:, l + 1][None, :], axis=1)
                        - np.sum(np.log((1.0 + np.exp(eta_1))
                                        / (1.0 + np.exp(eta_0))), axis=1))
 
                ber[d_idx][:, l] = logistic(prior_logit + like_logit)
 
        A_samples = []
        for d in range(D):
            raw = (np.random.rand(N, K[d], C)
                   < ber[d][:, :, None]).astype(float)
            A_samples.append(
                np.concatenate([np.ones((N, 1, C)), raw], axis=1))
 
        for d in range(D):
            p[d] = (1.0 - 1.0 / t) * p[d] + (1.0 / t) * ber[d]
 
        prop_update = np.mean(p[D - 1], axis=0)
 
        B1_update = np.zeros_like(B_list[0])
        gamma_update = np.zeros(J)
        for j in range(J):
            Xj = X[:, j]
            # MARK: Original code accumulated closures, causing objective
            # evaluation time to grow with the number of inner SAEM iterations.
            # f_loglik = _make_obs_negloglik(Xj, A_samples[0], N, K[0], C)
            # f_acc_obs[j] = _sa_accumulate(f_acc_obs[j], f_loglik, t)
            # _fj, _pen = f_acc_obs[j], pen_list[0]
            f_acc_obs[j].update(A_samples[0], t)
            _fj, _pen = f_acc_obs[j], pen_list[0]
            def obj(x, __fj=_fj, __pen=_pen):
                return __fj(x) + __pen * TLP(x[1:], tau)
            bounds = [(-bound_val, bound_val)] * (K[0] + 1)
            res = minimize(obj, B_list[0][j, :], method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 10})
            B1_update[j, :] = res.x
            eta = np.tensordot(A_samples[0], res.x, axes=([1], [0]))
            gamma_update[j] = np.sum((Xj[:, None] - eta) ** 2) / (N * C)
 
        B_updates = [B1_update]
        for d_idx in range(1, D):
            n_rows = B_list[d_idx].shape[0]
            B_d_upd = np.zeros_like(B_list[d_idx])
            for k in range(n_rows):
                Yk = A_samples[d_idx - 1][:, k + 1, :]
                # MARK: Original code accumulated Bernoulli objectives as nested
                # closures. The replacement stores running averages of pattern
                # counts and successes, yielding the same logistic objective with
                # constant evaluation cost per optimizer call.
                # f_loglik = _make_bern_negloglik(
                #     Yk, A_samples[d_idx], N, K[d_idx], C)
                # bidx = d_idx - 1
                # f_acc_bern[bidx][k] = _sa_accumulate(
                #     f_acc_bern[bidx][k], f_loglik, t)
                # _fk, _pen = f_acc_bern[bidx][k], pen_list[d_idx]
                bidx = d_idx - 1
                f_acc_bern[bidx][k].update(Yk, A_samples[d_idx], t)
                _fk, _pen = f_acc_bern[bidx][k], pen_list[d_idx]
                def obj(x, __fk=_fk, __pen=_pen):
                    return __fk(x) + __pen * TLP(x[1:], tau)
                bounds = [(-bound_val, bound_val)] * (K[d_idx] + 1)
                res = minimize(obj, B_list[d_idx][k, :], method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 10})
                B_d_upd[k, :] = res.x
            B_updates.append(B_d_upd)
 
        err = np.sqrt(
            np.sum((prop - prop_update) ** 2)
            + sum(np.sum((B_list[d] - B_updates[d]) ** 2) for d in range(D))
            + np.sum((gamma - gamma_update) ** 2))
 
        A_list_new = []
        for d in range(D):
            A_list_new.append(
                (np.mean(A_samples[d][:, 1:, :], axis=2) > 0.5).astype(float))
 
        prop = prop_update.copy()
        for d in range(D):
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
