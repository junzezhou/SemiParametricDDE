"""D-layer rescaling helpers for DDE copulas."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from dde.utils import validate_dde_lists


def _binary_configurations(k: int) -> np.ndarray:
    if k < 0:
        raise ValueError("k must be non-negative.")
    if k == 0:
        return np.zeros((1, 0), dtype=float)
    values = np.arange(2 ** k, dtype=np.uint64)[:, None]
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint64)[None, :]
    return ((values >> shifts) & 1).astype(float)


def iter_latent_configurations_D(prop, B_list):
    """Yield ``(A_list, probability)`` over all D-layer latent configurations."""
    prop = np.asarray(prop, dtype=float).reshape(-1)
    B_list = [np.asarray(B, dtype=float) for B in B_list]
    J = B_list[0].shape[0]
    validate_dde_lists(np.zeros((1, J)), B_list, prop=prop)
    D = len(B_list)
    K_list = [B_list[0].shape[1] - 1] + [B_list[d].shape[1] - 1 for d in range(1, D)]

    def descend(d, A_parent_list, prob_so_far):
        if d < 0:
            yield list(A_parent_list), float(prob_so_far)
            return

        if d == D - 1:
            probs = prop
        else:
            parent = A_parent_list[d + 1]
            probs = expit(B_list[d + 1] @ np.concatenate(([1.0], parent)))

        for A_d in _binary_configurations(K_list[d]):
            p_d = np.prod(probs ** A_d * (1.0 - probs) ** (1.0 - A_d))
            A_next = list(A_parent_list)
            A_next[d] = A_d
            yield from descend(d - 1, A_next, prob_so_far * p_d)

    yield from descend(D - 1, [None] * D, 1.0)


def implied_Z_moments_D(prop, B_list, gamma):
    """Compute model-implied marginal mean and sd of the latent Gaussian layer."""
    prop = np.asarray(prop, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    B_list = [np.asarray(B, dtype=float) for B in B_list]
    J = B_list[0].shape[0]
    validate_dde_lists(np.zeros((1, J)), B_list, prop=prop, gamma=gamma)

    mu = np.zeros(J, dtype=float)
    second = np.zeros(J, dtype=float)
    for A_list, p_joint in iter_latent_configurations_D(prop, B_list):
        A1 = A_list[0]
        mean_A = B_list[0] @ np.concatenate(([1.0], A1))
        mu += p_joint * mean_A
        second += p_joint * (mean_A ** 2 + gamma)

    var = np.maximum(second - mu ** 2, 0.0)
    sd = np.sqrt(var)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def rescale_B1_D(prop, B_list, gamma, emp=False, Z=None):
    """Rescale ``B_list[0]`` so each marginal latent Gaussian has mean 0 and sd 1.

    Deeper logistic layers are copied unchanged.  When ``emp=True``, empirical
    moments are taken from ``Z``; otherwise moments are computed by exact latent
    configuration enumeration.
    """
    prop = np.asarray(prop, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    B_scaled = [np.asarray(B, dtype=float).copy() for B in B_list]
    J = B_scaled[0].shape[0]
    validate_dde_lists(np.zeros((1, J)), B_scaled, prop=prop, gamma=gamma)

    if emp:
        if Z is None:
            raise ValueError("Z is required when emp=True.")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != J:
            raise ValueError(f"Z must have shape (N, {J}).")
        mu_used = np.nanmean(Z, axis=0)
        sd_used = np.nanstd(Z, axis=0, ddof=1)
        sd_used[sd_used < 1e-12] = 1.0
    else:
        mu_used, sd_used = implied_Z_moments_D(prop, B_scaled, gamma)

    for j in range(J):
        B_scaled[0][j, 0] = (B_scaled[0][j, 0] - mu_used[j]) / sd_used[j]
        B_scaled[0][j, 1:] = B_scaled[0][j, 1:] / sd_used[j]
    gamma_scaled = gamma / (sd_used ** 2)

    return B_scaled, gamma_scaled, mu_used, sd_used


__all__ = ["implied_Z_moments_D", "iter_latent_configurations_D", "rescale_B1_D"]
