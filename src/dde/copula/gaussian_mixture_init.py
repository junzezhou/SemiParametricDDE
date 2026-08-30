"""Python translation of simulate_gaussian_mixture.m.

Simulate Gaussian-mixture-based columns using truncated SVD, k-means
clustering with elbow detection, and rank/quantile matching.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


RandomStateLike = Optional[Union[int, np.random.Generator]]


def _zscore_columns(X: np.ndarray) -> np.ndarray:
    """Mimic MATLAB normalize(X) default behavior for numeric matrix columns."""
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0, ddof=1)
    sigma = np.where((sigma == 0) | ~np.isfinite(sigma), 1.0, sigma)
    return (X - mu) / sigma


def _as_rng(random_state: RandomStateLike = None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def simulate_gaussian_mixture(X, J, random_state: RandomStateLike = None):
    """Simulate Gaussian-mixture-based columns with quantile matching.

    Parameters
    ----------
    X : array_like, shape (N, P)
        Input data matrix.
    J : int
        Number of columns to simulate, corresponding to MATLAB columns
        ``1:J``. In Python this means columns ``0:J``.
    random_state : int or numpy.random.Generator, optional
        Random seed or generator for reproducibility. This is an optional
        Python-only convenience; omitting it preserves stochastic behavior.

    Returns
    -------
    Sim_all : ndarray, shape (N, J)
        Simulated matrix whose columns are rank/quantile matched to the
        corresponding columns of ``X``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    N, P = X.shape
    J = int(J)
    if J < 0 or J > P:
        raise ValueError(f"J must be between 0 and the number of columns in X ({P}).")

    rng = _as_rng(random_state)

    # --- Step 1: Normalize and compute economy SVD ---
    X_norm = _zscore_columns(X)
    U, sing_vals, Vt = np.linalg.svd(X_norm, full_matrices=False)

    # --- Step 2: Select number of components covering 80% variance ---
    total_var = np.sum(sing_vals ** 2)
    if total_var <= 0 or not np.isfinite(total_var):
        K1 = 1
    else:
        cum_var = np.cumsum(sing_vals ** 2) / total_var
        # MATLAB find(cum_var >= 0.8, 1), converted to 0-based index + 1.
        K1 = int(np.argmax(cum_var >= 0.8) + 1)

    # --- Step 3: Truncated rank-K1 reconstruction ---
    U_k = U[:, :K1]
    S_k = np.diag(sing_vals[:K1])
    V_k = Vt.T[:, :K1]
    A_k = U_k @ S_k @ V_k.T

    # --- Step 4: Distance matrix among right singular vectors ---
    D = cdist(V_k, V_k)
    np.fill_diagonal(D, np.inf)

    maxK = 10
    maxK_eff = min(maxK, N)

    # --- Step 5: Initialize output matrix ---
    Sim_all = np.zeros((N, J), dtype=float)

    # --- Step 6: Loop over columns ---
    for j in range(J):
        # Find index of closest column to j.
        closest_col = int(np.argmin(D[j, :]))

        # Subset rank-K1 approximation.
        noisy_X = A_k[:, [j, closest_col]]

        # --- Step 7: Find optimal k via elbow method ---
        wcss = np.zeros(maxK_eff, dtype=float)
        for k in range(1, maxK_eff + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans_tmp = KMeans(n_clusters=k, n_init=1, random_state=None)
                kmeans_tmp.fit(noisy_X)
            # sklearn inertia_ is the within-cluster sum of squared distances,
            # corresponding to MATLAB sum(sumd).
            wcss[k - 1] = float(kmeans_tmp.inertia_)

        x = np.arange(1, maxK_eff + 1, dtype=float)
        y = wcss
        x1, y1 = x[0], y[0]
        x2, y2 = x[-1], y[-1]
        denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if denom == 0:
            elbow_k = 1
        else:
            dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
            elbow_k = int(np.argmax(dist) + 1)

        # --- Step 8: Fit k-means with optimal k ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=elbow_k, n_init=1, random_state=None)
            idx = kmeans.fit_predict(noisy_X)
        C = kmeans.cluster_centers_

        # --- Step 9: Cluster proportions ---
        counts = np.bincount(idx, minlength=elbow_k).astype(float)
        proportions = counts / counts.sum()

        # --- Step 10: Covariance per cluster ---
        covariances = []
        for k in range(elbow_k):
            cluster_points = noisy_X[idx == k, :]
            if cluster_points.shape[0] > 1:
                covariances.append(np.cov(cluster_points, rowvar=False, ddof=1))
            else:
                covariances.append(np.eye(noisy_X.shape[1]))

        # --- Step 11: Simulate Gaussian mixture ---
        sim = np.zeros((N, 2), dtype=float)
        for i in range(N):
            cluster = int(rng.choice(elbow_k, p=proportions))
            cov = np.asarray(covariances[cluster], dtype=float)
            # Numerical safeguard: mvnrnd in MATLAB can tolerate semi-definite
            # covariance in many cases; add tiny jitter if NumPy complains.
            try:
                sim[i, :] = rng.multivariate_normal(C[cluster, :], cov)
            except np.linalg.LinAlgError:
                sim[i, :] = rng.multivariate_normal(
                    C[cluster, :], cov + 1e-8 * np.eye(cov.shape[0])
                )

        # --- Step 12: Quantile matching using the first simulated column ---
        orderX = np.argsort(X[:, j], kind="mergesort")
        sorted_sim = np.sort(sim[:, 0], kind="mergesort")

        matched_sim1 = np.zeros(N, dtype=float)
        matched_sim1[orderX] = sorted_sim

        # --- Step 13: Store result ---
        Sim_all[:, j] = matched_sim1

    return Sim_all
