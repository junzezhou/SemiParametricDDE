"""Python translation of rescale_B1.m.

This function rescales rows of B1 so that the implied marginal latent
Gaussian variable Z_j has mean 0 and variance 1.
"""

from __future__ import annotations

import numpy as np


def _binary_configurations(k: int) -> np.ndarray:
    """Match MATLAB dec2bin(0:(2^K-1)) - '0' with MSB-first columns."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return np.zeros((1, 0), dtype=int)
    values = np.arange(2 ** k, dtype=np.uint64)[:, None]
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint64)[None, :]
    return ((values >> shifts) & 1).astype(int)


def rescale_B1(prop_true, B2, B1, gamma_vec, emp, Z=None):
    """Rescale rows of B1 so the implied marginal Z_j has mean 0 and variance 1.

    Parameters
    ----------
    prop_true : array_like, shape (K2,)
        Top-layer Bernoulli probabilities.
    B2 : array_like, shape (K1, K2 + 1)
        Coefficient matrix for A1 | A2 logistic model.
    B1 : array_like, shape (p, K1 + 1)
        Coefficient matrix for Z | A1 Gaussian model.
    gamma_vec : array_like, shape (p,)
        Conditional variances for each Z_j.
    emp : bool
        If True, use empirical means and standard deviations from Z.
        If False, use model-implied moments by enumerating all A2/A1 configurations.
    Z : array_like, shape (N, p), optional
        Latent Gaussian draws, required when emp=True.

    Returns
    -------
    B1_scaled : ndarray, shape (p, K1 + 1)
    gamma_scaled : ndarray, shape (p,)
    mu_used : ndarray, shape (p,)
    sd_used : ndarray, shape (p,)
    """
    prop_true = np.asarray(prop_true, dtype=float).reshape(-1)
    B2 = np.asarray(B2, dtype=float)
    B1 = np.asarray(B1, dtype=float)
    gamma_vec = np.asarray(gamma_vec, dtype=float).reshape(-1)

    K2 = prop_true.size
    K1 = B2.shape[0]
    p = B1.shape[0]

    if B2.shape[1] != K2 + 1:
        raise ValueError("B2 must have shape (K1, K2 + 1).")
    if B1.shape[1] != K1 + 1:
        raise ValueError("B1 must have shape (p, K1 + 1).")
    if gamma_vec.size != p:
        raise ValueError("gamma_vec must have length p, where p = B1.shape[0].")

    if emp:
        if Z is None:
            raise ValueError("When emp=True, you must supply Z.")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != p:
            raise ValueError("Z must have p columns, where p = B1.shape[0].")

        # MATLAB: mean(Z, 1, 'omitnan')' and std(Z, 0, 1, 'omitnan')'
        # std(..., 0, ...) uses sample standard deviation, i.e., ddof=1.
        mu_used = np.nanmean(Z, axis=0)
        sd_used = np.nanstd(Z, axis=0, ddof=1)

    else:
        A2_configs = _binary_configurations(K2)
        A1_configs = _binary_configurations(K1)

        mu_used = np.zeros(p, dtype=float)
        second_moment = np.zeros(p, dtype=float)

        for A2 in A2_configs:
            A2 = A2.astype(float)
            pA2 = np.prod(prop_true ** A2 * (1.0 - prop_true) ** (1.0 - A2))

            logits = B2 @ np.concatenate(([1.0], A2))
            pA1_given_A2 = 1.0 / (1.0 + np.exp(-logits))

            for A1 in A1_configs:
                A1 = A1.astype(float)
                pA1 = np.prod(pA1_given_A2 ** A1 * (1.0 - pA1_given_A2) ** (1.0 - A1))
                pJoint = pA2 * pA1

                A1_with_intercept = np.concatenate(([1.0], A1))
                mu_A1 = B1 @ A1_with_intercept
                second_moment += pJoint * (mu_A1 ** 2 + gamma_vec)
                mu_used += pJoint * mu_A1

        varZ = second_moment - mu_used ** 2
        sd_used = np.sqrt(varZ)

    # Guard against zero or tiny standard deviations.
    sd_used = np.asarray(sd_used, dtype=float)
    sd_used[sd_used < 1e-12] = 1.0

    # Rescale each row of B1.
    intercept_shift = np.zeros(K1 + 1, dtype=float)
    B1_scaled = np.zeros_like(B1, dtype=float)
    for j in range(p):
        intercept_shift[0] = mu_used[j]
        B1_scaled[j, :] = (B1[j, :] - intercept_shift) / sd_used[j]

    # Variance rescales by sd^2.
    gamma_scaled = gamma_vec / (sd_used ** 2)

    return B1_scaled, gamma_scaled, mu_used, sd_used
