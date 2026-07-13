"""D-layer rank-likelihood SAEM with CSP shrinkage for DDE copulas.

This module is the canonical package implementation of the MATLAB-architecture
D-layer CopDDE CSP kernel, using the ``K_list`` / ``B_list`` / ``A_list``
convention.  The package rule thresholds only the first two layers.

Layer convention
----------------
``B_list[0]`` maps ``A_list[0]`` to the latent Gaussian observation ``Z`` and
has shape ``(J, K_1 + 1)``.

For ``d >= 1``, ``B_list[d]`` maps parent layer ``A_list[d]`` to child layer
``A_list[d-1]`` and has shape ``(K_d, K_{d+1} + 1)``.

``prop`` is the independent Bernoulli probability vector for the deepest layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import norm

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None


_EPS_PROB = 1e-12
_FLOAT_TINY = np.finfo(float).tiny


@dataclass(frozen=True)
class _RankGroups:
    missing: np.ndarray
    groups: Tuple[np.ndarray, ...]


def _as_arrays(B_list, A_list):
    return [np.asarray(B, dtype=float).copy() for B in B_list], [
        np.asarray(A, dtype=float).copy() for A in A_list
    ]


def _validate_inputs(Z, prop, gamma, B_list, A_list):
    D = len(B_list)
    if len(A_list) != D:
        raise ValueError("B_list and A_list must have the same length.")
    if D < 1:
        raise ValueError("At least one latent layer is required.")

    N, J = Z.shape
    K = [A.shape[1] for A in A_list]
    if B_list[0].shape != (J, K[0] + 1):
        raise ValueError(
            f"B_list[0] shape {B_list[0].shape} must be {(J, K[0] + 1)}."
        )
    for d in range(1, D):
        expected = (K[d - 1], K[d] + 1)
        if B_list[d].shape != expected:
            raise ValueError(f"B_list[{d}] shape {B_list[d].shape} must be {expected}.")
    if prop.shape[0] != K[-1]:
        raise ValueError("prop length must match the deepest latent width.")
    if gamma.shape[0] != J:
        raise ValueError("gamma length must match the observed dimension J.")
    for d, A in enumerate(A_list):
        if A.shape != (N, K[d]):
            raise ValueError(f"A_list[{d}] shape {A.shape} must be {(N, K[d])}.")
    return D, N, J, K


def _prepare_rank_groups(R: np.ndarray, Rlevels: np.ndarray) -> Tuple[_RankGroups, ...]:
    R = np.asarray(R, dtype=float)
    levels = np.asarray(Rlevels).astype(int).reshape(-1)
    out = []
    for j in range(R.shape[1]):
        rj = R[:, j]
        missing = np.flatnonzero(np.isnan(rj))
        groups = tuple(np.flatnonzero(rj == r) for r in range(1, levels[j] + 1))
        out.append(_RankGroups(missing=missing, groups=groups))
    return tuple(out)


def _sample_rank_likelihood_Z(
    Z: np.ndarray,
    mu: np.ndarray,
    gamma: np.ndarray,
    temp: float,
    rank_groups: Sequence[_RankGroups],
) -> None:
    """In-place rank-likelihood Gaussian resampling."""
    for j in range(Z.shape[1]):
        mu_j = mu[:, j]
        sd_j = np.sqrt(gamma[j] / temp)
        Zj = Z[:, j].copy()
        groups = rank_groups[j].groups

        miss = rank_groups[j].missing
        if miss.size:
            Zj[miss] = mu_j[miss] + sd_j * np.random.randn(miss.size)

        n_levels = len(groups)
        if n_levels:
            group_min = np.empty(n_levels)
            group_max = np.empty(n_levels)
            for rr, idx in enumerate(groups):
                if idx.size:
                    vals = Zj[idx]
                    group_min[rr] = np.min(vals)
                    group_max[rr] = np.max(vals)
                else:
                    group_min[rr] = np.inf
                    group_max[rr] = -np.inf

            for rr, idx in enumerate(groups):
                if idx.size == 0:
                    continue
                lb = -np.inf if rr == 0 or group_max[rr - 1] == -np.inf else group_max[rr - 1]
                ub = np.inf if rr == n_levels - 1 or group_min[rr + 1] == np.inf else group_min[rr + 1]
                pL = norm.cdf(lb, loc=mu_j[idx], scale=sd_j)
                pU = norm.cdf(ub, loc=mu_j[idx], scale=sd_j)
                pL = np.clip(pL, _EPS_PROB, 1.0 - _EPS_PROB)
                pU = np.clip(pU, _EPS_PROB, 1.0 - _EPS_PROB)
                bad = pU <= pL
                if np.any(bad):
                    mid = 0.5 * (pL[bad] + pU[bad])
                    pL[bad] = mid
                    pU[bad] = mid
                u = pL + (pU - pL) * np.random.rand(idx.size)
                Zj[idx] = norm.ppf(u, loc=mu_j[idx], scale=sd_j)
                group_min[rr] = np.min(Zj[idx])
                group_max[rr] = np.max(Zj[idx])

        Z[:, j] = Zj


def _logistic_objective_value_grad(x, X, y, weights, temp, sample_count=1):
    eta = X @ x
    scale = temp / sample_count
    value = scale * np.sum(np.logaddexp(0.0, eta) - y * eta)
    grad = scale * (X.T @ (expit(eta) - y))
    if weights is not None:
        value += np.sum(weights[1:] * np.abs(x[1:]))
        grad[1:] += weights[1:] * np.sign(x[1:])
    return float(value), grad


def _gaussian_objective_value_grad_scaled(x, X, y, weights, scale):
    resid = X @ x - y
    value = 0.5 * scale * np.dot(resid, resid)
    grad = scale * (X.T @ resid)
    if weights is not None:
        value += np.sum(weights[1:] * np.abs(x[1:]))
        grad[1:] += weights[1:] * np.sign(x[1:])
    return float(value), grad


def _tlp(x, tau):
    return float(np.sum(np.minimum(np.abs(x), tau)))


def _bounded_minimize_value_grad(value_grad, x0, lb, ub, maxiter=50):
    res = minimize(
        value_grad,
        np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lb, ub)),
        options={"maxiter": int(maxiter), "disp": False},
    )
    return np.asarray(res.x, dtype=float)


def _bounded_minimize(fun, x0, lb, ub, maxiter=50):
    res = minimize(
        fun,
        np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        bounds=list(zip(lb, ub)),
        options={"maxiter": int(maxiter), "disp": False, "maxls": 10},
    )
    return np.asarray(res.x, dtype=float)


def _maybe_parallel_map(func: Callable, items: Iterable, n_jobs: int):
    items = list(items)
    if n_jobs is None or n_jobs == 1 or len(items) <= 1 or Parallel is None:
        return [func(item) for item in items]
    return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(func)(item) for item in items)


def _threshold(x, tau):
    out = np.asarray(x, dtype=float).copy()
    out[np.abs(out) < tau] = 0.0
    return out


def _update_bernoulli_probabilities_D(Z, A_old, B_list, gamma, prop, temp):
    """Mean-field coordinate conditional probabilities for all D latent layers."""
    D = len(A_old)
    N = Z.shape[0]
    K = [A.shape[1] for A in A_old]
    ber = [np.zeros((N, Kd), dtype=float) for Kd in K]

    prop_clip = np.clip(prop, _EPS_PROB, 1.0 - _EPS_PROB)

    for d in range(D):
        for k in range(K[d]):
            if d == D - 1:
                logit_val = np.full(N, np.log(prop_clip[k] / (1.0 - prop_clip[k])))
            else:
                parent_design = np.column_stack((np.ones(N), A_old[d + 1]))
                logit_val = parent_design @ B_list[d + 1][k, :].T

            if d == 0:
                B1 = B_list[0]
                A1 = A_old[0]
                cols_keep = [c for c in range(K[0] + 1) if c != k + 1]
                A_minus = np.delete(A1, k, axis=1)
                design_minus = np.column_stack((np.ones(N), A_minus))
                residual = Z - design_minus @ B1[:, cols_keep].T
                b_col = B1[:, k + 1]
                child = -0.5 * np.sum(
                    (b_col[None, :] ** 2 - 2.0 * residual * b_col[None, :])
                    / gamma[None, :],
                    axis=1,
                )
            else:
                B_child = B_list[d]
                A_child = A_old[d - 1]
                A_curr = A_old[d]
                A1 = A_curr.copy()
                A0 = A_curr.copy()
                A1[:, k] = 1.0
                A0[:, k] = 0.0
                eta1 = np.column_stack((np.ones(N), A1)) @ B_child.T
                eta0 = np.column_stack((np.ones(N), A0)) @ B_child.T
                child = (
                    A_child @ B_child[:, k + 1]
                    - np.sum(np.logaddexp(0.0, eta1) - np.logaddexp(0.0, eta0), axis=1)
                )

            ber[d][:, k] = expit(temp * (logit_val + child))

    return ber


def _sample_latent_layers(ber, C):
    N = ber[0].shape[0]
    A_sample_long = []
    for probs in ber:
        Kd = probs.shape[1]
        samples = np.zeros((N, Kd + 1, C), dtype=float)
        samples[:, 0, :] = 1.0
        for c in range(C):
            samples[:, 1:, c] = (np.random.rand(N, Kd) < probs).astype(float)
        A_sample_long.append(samples)
    return A_sample_long


def _make_bounds(K, D):
    lb = [np.concatenate(([-10.0], -10.0 * np.ones(K[0])))]
    ub = [10.0 * np.ones(K[0] + 1)]
    for d in range(1, D):
        lb.append(np.concatenate(([-5.0], -10.0 * np.ones(K[d]))))
        ub.append(10.0 * np.ones(K[d] + 1))
    return lb, ub


def _optimize_gaussian_layer(
    Z,
    A_sample,
    B1,
    gamma,
    theta_or_penalty,
    temp,
    lb,
    ub,
    maxiter,
    n_jobs,
    penalty_kind,
    tau,
):
    N, K1_plus_1, C = A_sample.shape
    X_design = np.transpose(A_sample, (0, 2, 1)).reshape(N * C, K1_plus_1)
    J = Z.shape[1]

    def solve_row(j):
        y = np.repeat(Z[:, j], C)
        scale = temp / (gamma[j] * C)
        if penalty_kind == "weighted_l1":
            weights = np.concatenate(([0.0], 1.0 / np.asarray(theta_or_penalty[j], dtype=float)))
            return _bounded_minimize_value_grad(
                lambda x: _gaussian_objective_value_grad_scaled(x, X_design, y, weights, scale),
                B1[j, :],
                lb,
                ub,
                maxiter=maxiter,
            )

        pen = float(theta_or_penalty)

        def obj(x):
            value, _ = _gaussian_objective_value_grad_scaled(x, X_design, y, None, scale)
            return value + pen * _tlp(x[1:], tau)

        return _bounded_minimize(obj, B1[j, :], lb, ub, maxiter=maxiter)

    B_update = np.vstack(_maybe_parallel_map(solve_row, range(J), n_jobs))
    eta = np.tensordot(A_sample, B_update, axes=([1], [1]))
    sse = np.sum((Z[:, None, :] - eta) ** 2, axis=(0, 1))
    gamma_update = (1.0 + 0.5 * sse) / (1.0 + 0.5 * N * C + 1.0)
    return B_update, gamma_update


def _optimize_logistic_layer(
    A_child_sample,
    A_parent_sample,
    B_d,
    theta_or_penalty,
    temp,
    lb,
    ub,
    maxiter,
    n_jobs,
    penalty_kind,
    tau,
):
    N, K_parent_plus_1, C = A_parent_sample.shape
    X_design = np.transpose(A_parent_sample, (0, 2, 1)).reshape(N * C, K_parent_plus_1)
    K_child = A_child_sample.shape[1] - 1

    def solve_row(k):
        y = A_child_sample[:, k + 1, :].reshape(N * C)
        if penalty_kind == "weighted_l1":
            weights = np.concatenate(([0.0], 1.0 / np.asarray(theta_or_penalty[k], dtype=float)))
            return _bounded_minimize_value_grad(
                lambda x: _logistic_objective_value_grad(
                    x, X_design, y, weights, temp, sample_count=C
                ),
                B_d[k, :],
                lb,
                ub,
                maxiter=maxiter,
            )

        pen = float(theta_or_penalty)

        def obj(x):
            value, _ = _logistic_objective_value_grad(
                x, X_design, y, None, temp, sample_count=C
            )
            return value + pen * _tlp(x[1:], tau)

        return _bounded_minimize(obj, B_d[k, :], lb, ub, maxiter=maxiter)

    return np.vstack(_maybe_parallel_map(solve_row, range(K_child), n_jobs))


def _update_csp_layer(B_d, omega, theta_inf, theta_slab, num_rows):
    Kd = B_d.shape[1] - 1
    L = Kd + 1
    abs_sum = np.sum(np.abs(B_d[:, 1:]), axis=0)
    log_dens_spike = -num_rows * np.log(2.0 * theta_inf) - abs_sum / theta_inf
    log_dens_slab = (
        np.log(5.0)
        - num_rows * np.log(2.0)
        + gammaln(1.0 + num_rows)
        - (1.0 + num_rows) * np.log(5.0 + abs_sum)
    )
    log_omega = np.log(np.maximum(omega, _FLOAT_TINY))

    prob_mat = np.empty((Kd, L), dtype=float)
    for h in range(Kd):
        log_probs = log_omega.copy()
        log_probs[: h + 1] += log_dens_spike[h]
        log_probs[h + 1 :] += log_dens_slab[h]
        probs = np.exp(log_probs - np.max(log_probs))
        prob_mat[h, :] = probs / np.sum(probs)

    z = np.argmax(prob_mat, axis=1) + 1
    alpha = Kd
    v = np.zeros(L, dtype=float)
    for ell in range(1, Kd + 1):
        n_l = np.sum(z == ell)
        n_gtl = np.sum(z > ell)
        v[ell - 1] = np.random.beta(1.0 + n_l, alpha + n_gtl)
    v[L - 1] = 1.0
    omega = v * np.concatenate(([1.0], np.cumprod(1.0 - v[:-1])))

    theta_slab = theta_slab.copy()
    for h in range(Kd):
        if z[h] > (h + 1):
            shape_post = 1.0 + num_rows
            rate_post = 5.0 + np.sum(np.abs(B_d[:, h + 1]))
            theta_slab[h] = 1.0 / np.random.gamma(shape_post, 1.0 / rate_post)

    theta_col = np.where(z <= np.arange(1, Kd + 1), theta_inf, theta_slab)
    theta = np.broadcast_to(theta_col[None, :], (B_d.shape[0], Kd)).copy()
    return z, theta, theta_slab, omega


def get_SAEM_RL_CSP_D(
    X,
    Z,
    R,
    Rlevels,
    prop_in,
    gamma_in,
    B_list_in,
    A_list_in,
    C,
    it,
    t_spike,
    temp,
    tau,
    *,
    optimizer_maxiter: int = 15,
    n_jobs: int = 1,
    verbose: bool = True,
    convergence_window: int = 7,
    convergence_std: float = 0.005,
    burn: int = 5,
    trace_callback=None,
):
    """MATLAB-architecture D-layer rank-likelihood SAEM with CSP shrinkage.

    Parameters use the package list convention.  The only deliberate departure
    from the MATLAB D-layer routine is thresholding: only the first two layers
    are thresholded, regardless of total depth.
    """
    X = np.asarray(X)
    Z = np.asarray(Z, dtype=float).copy()
    R = np.asarray(R, dtype=float)
    Rlevels = np.asarray(Rlevels).astype(int).reshape(-1)
    prop = np.asarray(prop_in, dtype=float).reshape(-1).copy()
    gamma = np.asarray(gamma_in, dtype=float).reshape(-1).copy()
    B_list, A_list = _as_arrays(B_list_in, A_list_in)
    D, N, J, K = _validate_inputs(Z, prop, gamma, B_list, A_list)

    t_spike = np.asarray(t_spike, dtype=float).reshape(-1)
    if t_spike.size == 1:
        t_spike = np.repeat(t_spike.item(), D)
    if t_spike.size != D:
        raise ValueError("t_spike must be a scalar or a length-D sequence.")

    rank_groups = _prepare_rank_groups(R, Rlevels)
    lb, ub = _make_bounds(K, D)

    theta_slab = [np.ones(Kd, dtype=float) for Kd in K]
    omega = [np.ones(Kd + 1, dtype=float) / (Kd + 1) for Kd in K]
    z_cusp = [np.floor(Kd / 2) * np.ones(Kd, dtype=int) for Kd in K]
    theta = [
        np.full((J if d == 0 else K[d - 1], K[d]), t_spike[d], dtype=float)
        for d in range(D)
    ]
    num_act = np.array(K, dtype=int)
    rel_hist = [[] for _ in range(D)]

    A_sample_long = None
    ber = None
    it_num = 0

    while it_num < it:
        ber = _update_bernoulli_probabilities_D(Z, A_list, B_list, gamma, prop, temp)
        A_sample_long = _sample_latent_layers(ber, C)

        mu = np.mean(A_sample_long[0], axis=2) @ B_list[0].T
        _sample_rank_likelihood_Z(Z, mu, gamma, temp, rank_groups)

        for d in range(D - 1, -1, -1):
            num_rows = J if d == 0 else int(num_act[d - 1])
            z_cusp[d], theta[d], theta_slab[d], omega[d] = _update_csp_layer(
                B_list[d], omega[d], t_spike[d], theta_slab[d], num_rows
            )

        prop_update = np.mean(ber[-1], axis=0)
        B_update = [None] * D
        B_update[0], gamma_update = _optimize_gaussian_layer(
            Z,
            A_sample_long[0],
            B_list[0],
            gamma,
            theta[0],
            temp,
            lb[0],
            ub[0],
            optimizer_maxiter,
            n_jobs,
            "weighted_l1",
            tau,
        )
        for d in range(1, D):
            B_update[d] = _optimize_logistic_layer(
                A_sample_long[d - 1],
                A_sample_long[d],
                B_list[d],
                theta[d],
                temp,
                lb[d],
                ub[d],
                optimizer_maxiter,
                n_jobs,
                "weighted_l1",
                tau,
            )

        it_num += 1
        ind_inact = []
        for d in range(D):
            inactive = np.where(z_cusp[d] <= np.arange(1, K[d] + 1))[0]
            ind_inact.append(inactive)
            num_act[d] = K[d] - inactive.size

        for d in range(min(2, D)):
            B_update[d][:, 1:] = _threshold(B_update[d][:, 1:], tau)

        for d in range(D - 1):
            if ind_inact[d].size:
                B_update[d + 1][ind_inact[d], :] = 0.0

        if it_num > burn:
            temp = min(1.0, temp + 0.01 * ((it_num - burn) - 1))

        stop_now = True
        for d in range(D):
            rel = np.linalg.norm(B_update[d][:, 1:] - B_list[d][:, 1:], ord="fro") / (
                np.linalg.norm(B_list[d][:, 1:], ord="fro") + 1e-6
            )
            rel_hist[d].append(rel)
            if (
                len(rel_hist[d]) < convergence_window
                or np.std(rel_hist[d][-convergence_window:]) >= convergence_std
            ):
                stop_now = False

        if trace_callback is not None:
            trace_callback(
                {
                    "iteration": it_num,
                    "B_list": [B.copy() for B in B_update],
                    "gamma": gamma_update.copy(),
                    "prop": prop_update.copy(),
                    "Z": Z.copy(),
                    "num_active": num_act.copy(),
                    "z_cusp": [z.copy() for z in z_cusp],
                    "rel": [float(hist[-1]) for hist in rel_hist],
                    "temp": float(temp),
                    "will_stop": bool(stop_now),
                }
            )

        A_list = [
            (np.mean(A_sample_long[d][:, 1:, :], axis=2) > 0.5).astype(float)
            for d in range(D)
        ]
        B_list = [B.copy() for B in B_update]
        gamma = gamma_update.copy()
        prop = prop_update.copy()

        if stop_now:
            for d in range(D):
                inactive = ind_inact[d]
                if inactive.size:
                    B_list[d][:, inactive + 1] = 0.0
            if verbose:
                print("ending algorithm")
            break

    if it_num >= it:
        for d in range(D):
            inactive = np.where(z_cusp[d] <= np.arange(1, K[d] + 1))[0]
            if inactive.size:
                B_list[d][:, inactive + 1] = 0.0
        if verbose:
            print("Ending algorithm for iteration budget")

    return {
        "A_list": A_list,
        "B_list": B_list,
        "Z": Z,
        "gamma": gamma,
        "prop": prop,
        "theta": theta,
        "theta_slab": theta_slab,
        "z_cusp": z_cusp,
        "omega": omega,
        "num_act": num_act,
        "it_num": it_num,
        "A_sample_long": A_sample_long,
        "p": ber,
    }


__all__ = ["get_SAEM_RL_CSP_D"]
