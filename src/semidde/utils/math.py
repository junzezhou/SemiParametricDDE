"""Numerical helpers shared across DDE estimators."""

from __future__ import annotations

import numpy as np


def logistic(x):
    """Numerically stable logistic sigmoid."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def logit(x):
    """Inverse logistic transform with boundary clipping."""
    x = np.asarray(x, dtype=float)
    eps = np.finfo(float).eps
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def binary(x, k=None):
    """Return a matrix of binary representations, most significant bit first."""
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    if k is None:
        k = int(np.max(np.floor(np.log2(np.maximum(x, 1)))) + 1)
    shifts = np.arange(k - 1, -1, -1, dtype=np.int64)
    return ((x[:, None] >> shifts) & 1).astype(float)


def patterns_to_int(A):
    """Convert a binary pattern matrix to integer codes, MSB first."""
    A = np.asarray(A, dtype=np.int64)
    if A.ndim != 2:
        raise ValueError("A must be a 2D binary matrix.")
    k = A.shape[1]
    weights = np.int64(1) << np.arange(k - 1, -1, -1, dtype=np.int64)
    return A @ weights


def compute_propA_U(prop_U, A_U):
    """Joint PMF over binary configurations for independent Bernoulli entries."""
    prop_U = np.asarray(prop_U, dtype=float)
    A_U = np.asarray(A_U, dtype=float)
    prop_U = np.clip(prop_U, 1e-12, 1.0 - 1e-12)
    log_p = A_U * np.log(prop_U) + (1.0 - A_U) * np.log(1.0 - prop_U)
    return np.exp(log_p.sum(axis=1))
