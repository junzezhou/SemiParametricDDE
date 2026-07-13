"""Python translation of generate_X_Cop_marg_emp.m.

This function generates data from the DDE-copula simulation model used in the
Matlab simulation script.  The translation keeps the original input/output order:

    Z, X, A1, A2, p_true, Q = generate_X_Cop_marg_emp(...)

Notes on Matlab -> Python differences:
- Matlab arrays are 1-based; NumPy arrays are 0-based.  The marginal cycling
  ``mod(j-1, 3)+1`` is reproduced by ``j % 3`` for Python column index ``j``.
- Matlab ``binornd(1, p, ...)`` is reproduced by ``rng.binomial(1, p, ...)``.
- Matlab ``tiedrank`` is reproduced by ``scipy.stats.rankdata(..., method='average')``.
- Matlab ``poissinv`` and ``gaminv`` are reproduced by SciPy inverse CDF/PPF.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.special import expit
from scipy.stats import gamma as gamma_dist
from scipy.stats import poisson, rankdata


def generate_X_Cop_marg_emp(
    N: int,
    prop_true,
    lambdas,
    gamma,
    B1,
    B2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate DDE-copula simulation data with empirical marginal transform.

    Parameters
    ----------
    N : int
        Number of observations.
    prop_true : array-like, shape (K2,)
        Bernoulli probabilities for the top latent layer A2.
    lambdas : array-like, shape (p,)
        Marginal parameters used for zero-inflated Poisson, Gamma scale, and
        standard Poisson variables.  This matches the Matlab code where the
        same vector is cycled through the three marginal types.
    gamma : array-like, shape (p,)
        Feature-specific Gaussian variances for the latent Gaussian layer Z.
    B1 : array-like, shape (p, K1 + 1)
        First-layer loading matrix, including intercept in the first column.
    B2 : array-like, shape (K1, K2 + 1)
        Second-layer loading matrix, including intercept in the first column.
    rng : numpy.random.Generator, optional
        Random number generator.  If omitted, ``np.random.default_rng()`` is used.

    Returns
    -------
    Z : ndarray, shape (N, p)
        Latent Gaussian data.
    X : ndarray, shape (N, p)
        Observed data after marginal inverse-CDF transforms.
    A1 : ndarray, shape (N, K1)
        First latent binary layer.
    A2 : ndarray, shape (N, K2)
        Second latent binary layer.
    p_true : ndarray, shape (N, K1)
        Conditional Bernoulli probabilities for A1 given A2.
    Q : ndarray, shape (N, p)
        Empirical CDF values for Z, computed columnwise by tied ranks.
    """

    if rng is None:
        rng = np.random.default_rng()

    prop_true = np.asarray(prop_true, dtype=float).reshape(-1)
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    B1 = np.asarray(B1, dtype=float)
    B2 = np.asarray(B2, dtype=float)

    K2 = prop_true.shape[0]

    # Generate A2: independent Bernoulli latent binary variables.
    A2 = np.zeros((N, K2), dtype=float)
    for k in range(K2):
        A2[:, k] = rng.binomial(1, prop_true[k], size=N)

    # Generate A1: conditionally independent Bernoulli variables given A2.
    A2_design = np.column_stack((np.ones(N), A2))
    p_true = expit(A2_design @ B2.T)
    A1 = rng.binomial(1, p_true).astype(float)

    # Latent Gaussian layer: Z = mu + sqrt(gamma) * epsilon.
    A1_design = np.column_stack((np.ones(N), A1))
    mu = A1_design @ B1.T
    Z = np.sqrt(gamma.reshape(1, -1)) * rng.standard_normal(size=mu.shape) + mu
    p = Z.shape[1]

    # Empirical estimate of Q via columnwise rank transform:
    # Q_ij ~= F_j(Z_ij), using tied ranks and scaling by N + 1 to keep Q in (0, 1).
    Q = np.zeros((N, p), dtype=float)
    for j in range(p):
        r = rankdata(Z[:, j], method="average")
        Q[:, j] = r / (N + 1.0)

    # Allocate observed data matrix X.
    X = np.zeros((N, p), dtype=float)

    # Cycle through marginal types:
    #   Python j % 3 == 0: Matlab modtype = 1, zero-inflated Poisson
    #   Python j % 3 == 1: Matlab modtype = 2, right-skewed Gamma
    #   Python j % 3 == 2: Matlab modtype = 3, standard Poisson
    for j in range(p):
        modtype = j % 3

        if modtype == 0:
            # Zero-inflated marginal with extra zero probability pi0.
            pi0 = 0.3
            qj = Q[:, j]
            idx_nonzero = qj >= pi0
            if np.any(idx_nonzero):
                u_adj = (qj[idx_nonzero] - pi0) / (1.0 - pi0)
                X[idx_nonzero, j] = poisson.ppf(u_adj, lambdas[j])

        elif modtype == 1:
            # Right-skewed marginal: rounded Gamma inverse CDF.
            shape = 2.0
            scale = lambdas[j]
            X[:, j] = np.round(gamma_dist.ppf(Q[:, j], a=shape, scale=scale))

        else:
            # Standard Poisson marginal.
            X[:, j] = poisson.ppf(Q[:, j], lambdas[j])

    return Z, X, A1, A2, p_true, Q
