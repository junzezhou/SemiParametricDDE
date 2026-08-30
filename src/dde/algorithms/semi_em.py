import numpy as np
from itertools import product as iproduct
from scipy.special import expit, logit, logsumexp
from scipy.optimize import minimize
from scipy.stats import norm

from dde.utils import TLP, binary, compute_propA_U, g_est_rversion, thres
from dde.algorithms.f1_saem import F_1_SAEM
from dde.algorithms.f2_saem import F_2_SAEM


def _logistic(x):
    """Numerically stable logistic sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def get_EM_est(X, prop_in, B_list, gamma_in, pen_list, tau, tol,
               A_list_in=None, C=10, max_iter=50, use_exact=None):
    """
    Penalized EM for Normal-D-latent-layer DDEs.

    When the K values are small enough for exhaustive enumeration
    (all 2^K_d are manageable), the exact PEM is used.
    When any K_d is too large, a Stochastic Approximation EM (SAEM)
    with Gibbs sampling is used automatically.

    Parameters
    ----------
    X        : (N, J)  Obs matrix
    prop_in  : (K_D,)  Initial Bernoulli probabilities for D-th layer.
    B_list   : list of D arrays
                 B_list[0] : (J,  K_1+1)
                 B_list[d] : (K_d, K_{d+1}+1) for d=1,...,D-1
    gamma_in : (J,)    Initial conditional variance for Obs features.
    pen_list : list of D floats — TLP regularization weights.
    tau      : float   TLP truncation threshold.
    tol      : float   Convergence threshold.
    A_list_in: list of D arrays, optional
                 A_list_in[d] : (N, K_{d+1})  initial latent binary matrices.
                 Required for SAEM mode. If None and SAEM is needed, initialised randomly.
    C        : int     Number of MC samples per SAEM iteration (default 10).
    max_iter : int     Maximum iterations (default 50).
    use_exact: bool or None
                 If True, force exact EM. If False, force SAEM.
                 If None (default), auto-detect based on K sizes.

    Returns
    -------
    prop, B_list, gamma, loglik_or_err, A_list_or_phi, itera
        - In exact mode: returns (prop, B_list, gamma, loglik, phi, itera)
        - In SAEM mode:  returns (prop, B_list, gamma, err, A_list, itera)
    """
    X = np.asarray(X, dtype=float)
    N, J = X.shape

    prop  = np.asarray(prop_in, dtype=float).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list]
    gamma = np.asarray(gamma_in, dtype=float).copy()

    D = len(B_list)
    K = []
    K.append(B_list[0].shape[1] - 1)          # K_1
    for d in range(1, D):
        K.append(B_list[d].shape[1] - 1)      # K_{d+1}
    # K = [K_1, K_2, ..., K_D]

    K_D = K[-1]
    assert K_D == prop.shape[0]

    # Auto-detect whether exact enumeration is feasible
    EXACT_THRESHOLD = 20   # max K_d for which 2^K_d is tractable
    if use_exact is None:
        use_exact = all(k <= EXACT_THRESHOLD for k in K)

    if use_exact:
        print("Using exact EM (exhaustive enumeration).")
        return _get_EM_est_exact(X, prop, B_list, gamma, pen_list, tau, tol,
                                 D, K, N, J, max_iter)
    else:
        print("Using SAEM (Gibbs sampling).")
        return _get_SAEM_est(X, prop, B_list, gamma, pen_list, tau, tol,
                             D, K, N, J, A_list_in, C, max_iter)


# =============================================================================
#  SAEM version — Gibbs sampling for arbitrary K sizes
# =============================================================================

def _get_SAEM_est(X, prop, B_list, gamma, pen_list, tau, tol,
                  D, K, N, J, A_list_in, C, max_iter):
    """
    SAEM (Stochastic Approximation EM) for D-layer Normal DDE.

    Uses coordinate-wise Gibbs sampling to draw latent variables,
    then stochastic approximation + M-step to update parameters.
    """

    # --- Initialise latent variables A_list[d]: (N, K_{d+1}) ---
    # Convention: A_list[d] corresponds to layer d+1, shape (N, K[d])
    # K = [K_1, ..., K_D], so A_list[d] has shape (N, K[d])
    if A_list_in is not None:
        A_list = [np.asarray(A, dtype=float).copy() for A in A_list_in]
    else:
        A_list = [(np.random.rand(N, K[d]) > 0.5).astype(float) for d in range(D)]

    # --- optimisation bounds ---
    bounds_list = []
    for d in range(D):
        dim = K[d] + 1 if d == 0 else K[d] + 1
        lb = -10 * np.ones(dim)
        ub =  10 * np.ones(dim)
        bounds_list.append(list(zip(lb, ub)))

    # --- MC sample containers ---
    # A_sample_long[d]: (N, K[d]+1, C) — with leading intercept column
    A_sample_long = [np.zeros((N, K[d]+1, C)) for d in range(D)]

    # --- Bernoulli conditional probabilities ---
    ber = [np.full((N, K[d]), 0.5) for d in range(D)]

    # --- stochastic approximation accumulators ---
    p_sa = [np.full((N, K[d]), 0.5) for d in range(D)]

    # --- old objective closures for SA ---
    f_old_B0 = [lambda x: 0.0 for _ in range(J)]
    f_old_Bd = [[lambda x: 0.0 for _ in range(K[d-1])] for d in range(1, D)]
    # f_old_Bd[d-1][k] for layer d, unit k

    # penalty helpers
    def ftn_pen(d):
        def _pen(x):
            return pen_list[d] * TLP(x[1:], tau)
        return _pen

    t = 1
    iter_indicator = True
    err = 0.0

    B_list_update = [np.zeros_like(B) for B in B_list]
    gamma_update = gamma.copy()
    prop_update = prop.copy()

    while iter_indicator:
        # ========== Simulation step (Gibbs sampling) ==========
        A_old = [A.copy() for A in A_list]

        for i in range(N):
            # --- sample top layer (layer D) ---
            # A_list[D-1][i, :] are the K_D binary variables
            for l in range(K[D-1]):
                # P(A_D[i,l]=1 | A_D[i,-l], A_{D-1}[i,:], prop)

                # prior log-odds
                logit_prior = np.log(prop[l] / (1.0 - prop[l] + 1e-15) + 1e-15)

                # If D >= 2: contribution from layer D-1 (child)
                # A_{D-1}_k | A_D ~ Bernoulli(logistic(B_list[D-1][k,:] @ [1, A_D]))
                if D >= 2:
                    A_D_i = A_old[D-1][i, :].copy()
                    A_D_i[l] = 1.0
                    eta_1 = B_list[D-1] @ np.concatenate(([1.0], A_D_i))  # (K_{D-1},)
                    A_D_i[l] = 0.0
                    eta_0 = B_list[D-1] @ np.concatenate(([1.0], A_D_i))  # (K_{D-1},)

                    # sum_k [ A_{D-1}[i,k] * B_{D-1}[k, l+1] ]
                    term_child = np.sum(A_old[D-2][i, :] * B_list[D-1][:, l+1])
                    term_norm  = np.sum(np.log1p(np.exp(eta_1)) - np.log1p(np.exp(eta_0)))
                    logit_val = logit_prior + term_child - term_norm
                else:
                    # D==1: top layer is also the only layer, child is obs X
                    # contribution from obs layer:
                    # X | A_1 ~ N(B_list[0] @ [1, A_1], diag(gamma))
                    mask = np.ones(K[0], dtype=bool)
                    mask[l] = False
                    A_1_i = A_old[0][i, :].copy()
                    part1 = B_list[0][:, l+1] ** 2
                    part2 = 2.0 * (X[i, :] - B_list[0][:, np.concatenate(([True], mask))] @
                                   np.concatenate(([1.0], A_1_i[mask])))
                    term_b = np.sum((part1 - part2 * B_list[0][:, l+1]) / gamma)
                    logit_val = logit_prior - term_b / 2.0

                ber[D-1][i, l] = _logistic(logit_val)

            # --- sample intermediate layers d = D-2, D-3, ..., 0 (layers D-1, ..., 1) ---
            # For layer d (0-indexed), its parent is layer d+1 and child is layer d-1
            # (or obs X if d==0)
            for d_idx in range(D-2, -1, -1):
                # d_idx = index in A_list/K arrays
                # layer (d_idx+1) in the model
                # parent: A_list[d_idx+1], B_list[d_idx+1]
                # child: A_list[d_idx-1] (or X if d_idx==0), B_list[d_idx]
                for k in range(K[d_idx]):
                    if d_idx == 0:
                        # --- obs layer child ---
                        # X | A_1 ~ N(B_list[0] @ [1, A_1], diag(gamma))
                        # logit from parent layer:
                        logit_parent = np.dot(
                            np.concatenate(([1.0], A_old[d_idx+1][i, :])),
                            B_list[d_idx+1][k, :] if D > 1 else np.zeros(1)
                        ) if D > 1 else 0.0

                        # If D==1, parent contribution is just the prior (already handled above)
                        # For D>1, parent = layer 2
                        if D > 1:
                            logit_from_parent = np.dot(
                                np.concatenate(([1.0], A_old[1][i, :])),
                                B_list[1][k, :]
                            )
                        else:
                            logit_from_parent = np.log(prop[k] / (1.0 - prop[k] + 1e-15) + 1e-15)

                        # contribution from obs X
                        mask = np.ones(K[0], dtype=bool)
                        mask[k] = False
                        A_1_i = A_old[0][i, :].copy()
                        part1 = B_list[0][:, k+1] ** 2
                        part2 = 2.0 * (X[i, :] - B_list[0][:, np.concatenate(([True], mask))] @
                                       np.concatenate(([1.0], A_1_i[mask])))
                        term_obs = np.sum((part1 - part2 * B_list[0][:, k+1]) / gamma)

                        logit_val = logit_from_parent - term_obs / 2.0

                    else:
                        # --- intermediate layer ---
                        # parent: layer d_idx+1, B_list[d_idx+1]
                        # child:  layer d_idx-1, B_list[d_idx]

                        # logit from parent
                        logit_from_parent = np.dot(
                            np.concatenate(([1.0], A_old[d_idx+1][i, :])),
                            B_list[d_idx+1][k, :]
                        )

                        # contribution from child layer d_idx-1
                        # A_{d_idx-1}_m | A_{d_idx} ~ Bernoulli(logistic(B_list[d_idx][m,:] @ [1, A_{d_idx}]))
                        A_d_i = A_old[d_idx][i, :].copy()
                        A_d_i[k] = 1.0
                        eta_1 = B_list[d_idx] @ np.concatenate(([1.0], A_d_i))  # (K[d_idx-1],)
                        A_d_i[k] = 0.0
                        eta_0 = B_list[d_idx] @ np.concatenate(([1.0], A_d_i))  # (K[d_idx-1],)

                        term_child = np.sum(A_old[d_idx-1][i, :] * B_list[d_idx][:, k+1])
                        term_norm  = np.sum(np.log1p(np.exp(eta_1)) - np.log1p(np.exp(eta_0)))

                        logit_val = logit_from_parent + term_child - term_norm

                    ber[d_idx][i, k] = _logistic(logit_val)

        # ---- draw MC samples ----
        for c in range(C):
            for d in range(D):
                samples = (np.random.rand(N, K[d]) < ber[d]).astype(float)
                A_sample_long[d][:, :, c] = np.column_stack((np.ones(N), samples))

        # ---- stochastic approximation update ----
        step = 1.0 / t
        for d in range(D):
            p_sa[d] = (1.0 - step) * p_sa[d] + step * ber[d]

        # ========== M-step ==========

        # -- update prop (top layer) --
        prop_update = np.mean(p_sa[D-1], axis=0)  # (K_D,)

        # -- update B_list[0] and gamma (obs layer) --
        B_new_0 = np.zeros_like(B_list[0])
        for j in range(J):
            Xj = X[:, j:j+1]  # (N, 1)
            f_loglik = F_1_SAEM(Xj, A_sample_long[0], N, K[0], C)
            # stochastic approximation of objective
            _f_old = f_old_B0[j]
            f_old_B0[j] = lambda x, _fo=_f_old, _fl=f_loglik: \
                (1.0 - step) * _fo(x) + step * _fl(x)
            _pen = ftn_pen(0)
            f_j = lambda x, _fsa=f_old_B0[j]: _fsa(x) + _pen(x)

            res = minimize(f_j, B_list[0][j, :], method='L-BFGS-B',
                           bounds=bounds_list[0],
                           options=dict(maxiter=20, disp=False))
            opt = res.x
            B_new_0[j, :] = opt

            # update gamma_j
            eta = np.sum(A_sample_long[0] * opt.reshape(1, K[0]+1, 1), axis=1)  # (N, C)
            gamma_update[j] = float(np.sum((Xj - eta) ** 2) / (N * C))

        B_list_update[0] = B_new_0

        # -- update B_list[d] for d = 1, ..., D-1 (logistic layers) --
        for d in range(1, D):
            Kd = K[d-1]    # child layer size
            B_new_d = np.zeros_like(B_list[d])
            for k in range(Kd):
                # A_child_k samples: (N, C)
                A_child_k = A_sample_long[d-1][:, k+1, :].reshape(N, C)
                f_loglik = F_2_SAEM(A_child_k, A_sample_long[d], N, K[d], C)

                _f_old_dk = f_old_Bd[d-1][k]
                f_old_Bd[d-1][k] = lambda x, _fo=_f_old_dk, _fl=f_loglik: \
                    (1.0 - step) * _fo(x) + step * _fl(x)
                _pen = ftn_pen(d)
                f_k = lambda x, _fsa=f_old_Bd[d-1][k]: _fsa(x) + _pen(x)

                res2 = minimize(f_k, B_list[d][k, :], method='L-BFGS-B',
                                bounds=bounds_list[d],
                                options=dict(maxiter=20, disp=False))
                B_new_d[k, :] = res2.x

            B_list_update[d] = B_new_d

        # ========== Convergence check ==========
        err = np.sqrt(
            np.linalg.norm(prop - prop_update) ** 2 +
            sum(np.linalg.norm(B_list[d] - B_list_update[d]) ** 2 for d in range(D)) +
            np.linalg.norm(gamma - gamma_update) ** 2
        )

        # Update latent hard assignments from MC samples
        A_list = [
            (np.mean(A_sample_long[d][:, 1:, :], axis=2) > 0.5).astype(float)
            for d in range(D)
        ]
        A_old = A_list  # for next Gibbs sweep

        # Apply parameter updates + TLP thresholding
        prop = prop_update.copy()
        for d in range(D):
            Bu = B_list_update[d]
            B_list[d] = np.column_stack([Bu[:, :1], thres(Bu[:, 1:], tau)])
        gamma = gamma_update.copy()

        t += 1
        print(f"SAEM Iteration {t},\t Err {err:.4f}")
        iter_indicator = (abs(err) > tol) and (t < max_iter)

    return prop, B_list, gamma, float(err), A_list, t - 1


# =============================================================================
#  Exact EM version — only for small K (exhaustive enumeration)
# =============================================================================

def _get_EM_est_exact(X, prop, B_list, gamma, pen_list, tau, tol,
                      D, K, N, J, max_iter):
    """
    Original exact PEM for Normal-D-latent-layer DDEs.
    Only feasible when all K[d] are small (e.g. <= 20).
    """

    K_D = K[-1]

    # --- enumerate all states for each layer ---
    states      = [binary(np.arange(2**k), k) for k in K]
    states_long = [np.column_stack([np.ones((2**k, 1)), states[d]])
                   for d, k in enumerate(K)]
    sizes = [2**k for k in K]

    phi_shape = (N,) + tuple(sizes)

    # ---- box bounds for each layer ----
    bounds_list = []
    for d in range(D):
        dim = K[d] + 1
        lb = np.concatenate([-10*np.ones(1), -10*np.ones(dim-1)])
        ub = 10*np.ones(dim)
        bounds_list.append(tuple((l, u) for l, u in zip(lb, ub)))

    itera = 0
    loglik = 0.0
    iter_indicator = True

    while iter_indicator:
        old_loglik = loglik

        # ========== E-step ==========
        phi = np.zeros(phi_shape, dtype=float)

        log_gamma_sum = np.sum(np.log(gamma))
        inv_gamma     = 1.0 / gamma

        for idx in iproduct(*[range(s) for s in sizes]):
            a1 = idx[0]
            lam = states_long[0][a1] @ B_list[0].T
            diff = X - lam.reshape(1, J)
            log_obs = -0.5 * (
                np.sum((diff**2) * inv_gamma.reshape(1, J), axis=1)
                + log_gamma_sum
            )

            log_mid = 0.0
            for d in range(1, D):
                eta = states_long[d][idx[d]] @ B_list[d].T
                a_child = states[d-1][idx[d-1]]
                log_mid += float(
                    np.sum(a_child * eta) - np.sum(np.log1p(np.exp(eta)))
                )

            a_top = states[D-1][idx[D-1]]
            log_prior = float(
                np.sum(np.log(prop) * a_top + np.log(1 - prop) * (1 - a_top))
            )

            phi[(slice(None),) + idx] = np.exp(log_obs + log_mid + log_prior)

        sum_all = phi.reshape(N, -1).sum(axis=1)
        loglik  = float(np.sum(np.log(sum_all)))
        phi    /= sum_all.reshape((N,) + (1,)*D)

        # ========== M-step ==========
        axes_except_top = tuple(range(1, D))
        phi_top = phi.sum(axis=axes_except_top)
        xi_top  = phi_top.sum(axis=0)
        prop = (xi_top.reshape(1, -1) @ states[D-1]).reshape(K_D,) / float(N)

        axes_not_1 = tuple(range(2, D+1))
        pi_1 = phi.sum(axis=axes_not_1)

        B_list_update = [None] * D
        B_new_0 = np.zeros_like(B_list[0])
        for j in range(J):
            Xj = X[:, j:j+1]
            def obj_B0(x, j=j):
                x   = np.asarray(x, dtype=float)
                pen = pen_list[0] * TLP(x, tau)
                eta = x @ states_long[0].T
                resid = Xj - eta.reshape(1, -1)
                return float(pen + np.sum((resid**2) * pi_1))

            res = minimize(obj_B0, B_list[0][j, :], method="L-BFGS-B",
                           bounds=bounds_list[0], options=dict(maxiter=20, disp=False))
            opt = res.x
            B_new_0[j, :] = opt

            eta_opt = opt @ states_long[0].T
            resid   = Xj - eta_opt.reshape(1, -1)
            gamma[j] = float(np.sum((resid**2) * pi_1) / float(N))

        B_list_update[0] = B_new_0

        for d in range(1, D):
            Kd   = K[d-1]
            Kd1  = K[d]

            keep_axes = (0, d, d+1)
            sum_axes  = tuple(i for i in range(D+1) if i not in keep_axes)
            phi_dd1   = phi.sum(axis=sum_axes)
            phi_dd1_n = phi_dd1.sum(axis=0)

            psi_d = phi_dd1_n.T @ states[d-1]
            psi_2_d = phi_dd1_n.sum(axis=0)

            B_new_d = np.zeros_like(B_list[d])
            for k in range(Kd):
                psi_k = psi_d[:, k]
                def obj_Bd(x, psi_k=psi_k):
                    x   = np.asarray(x, dtype=float)
                    pen = pen_list[d] * TLP(x, tau)
                    eta = x @ states_long[d].T
                    obj = float(np.sum(eta * psi_k) - np.sum(np.log1p(np.exp(eta)) * psi_2_d))
                    return float(pen - obj)

                res2 = minimize(obj_Bd, B_list[d][k, :], method="L-BFGS-B",
                                bounds=bounds_list[d],
                                options=dict(maxiter=20, disp=False))
                B_new_d[k, :] = res2.x

            B_list_update[d] = B_new_d

        err = loglik - old_loglik
        for d in range(D):
            Bu = B_list_update[d]
            B_list[d] = np.column_stack([Bu[:, :1], thres(Bu[:, 1:], tau)])

        itera += 1
        iter_indicator = (abs(err) > tol) and (itera < max_iter)

    return prop, B_list, gamma, float(loglik), phi, itera
