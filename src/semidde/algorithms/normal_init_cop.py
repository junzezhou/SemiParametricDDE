"""CopDDE Normal initialization matching MATLAB ``Normal_init_D``.

This is an additive reproduction-oriented initializer for
``DDE-Copula-m/Utilities/Normal_init_D.m`` and ``binary_init.m``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import svd
from scipy.special import logit

from semidde.utils import thres, validate_dde_lists


def _right_inverse_xtx(X):
    return np.linalg.pinv(X.T @ X)


def _varimax(Phi, gamma=1.0, q=1000, tol=1e-6):
    Phi = np.asarray(Phi, dtype=float)
    if Phi.ndim == 1:
        Phi = Phi.reshape(-1, 1)

    p, k = Phi.shape
    R = np.eye(k)
    d_old = 0.0
    for _ in range(q):
        Lambda = Phi @ R
        tmp = Phi.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)))
        U, s, Vt = np.linalg.svd(tmp, full_matrices=False)
        R = U @ Vt
        d_new = np.sum(s)
        if d_old != 0.0 and d_new / d_old < 1.0 + tol:
            break
        d_old = d_new
    return Phi @ R


def _matlab_sign_flip(B):
    signs = 2.0 * (np.mean(B, axis=0) > 0).astype(float) - 1.0
    return B * signs.reshape(1, -1)


def _binary_init_matlab(N, K1, K2, A1_est, factor=2.0, epsilon=1e-4):
    """Python version of ``DDE-Copula-m/Utilities/binary_init.m``."""
    N = int(N)
    K1 = int(K1)
    K2 = int(K2)
    A1_est = np.asarray(A1_est, dtype=float)
    if A1_est.shape != (N, K1):
        raise ValueError(f"A1_est shape {A1_est.shape} must be {(N, K1)}.")

    U, S, Vt = svd(A1_est, full_matrices=False)
    m = max(K2 * 2, int(np.sum(S > 1.01 * np.sqrt(N))))
    m = min(m, len(S))
    A1_top_m = U[:, :m] @ np.diag(S[:m]) @ Vt[:m, :]
    A1_trunc = np.clip(A1_top_m, epsilon, 1.0 - epsilon)

    A_inv = logit(A1_trunc)
    A_inv_adj = A_inv - np.mean(A_inv, axis=0, keepdims=True)
    _, _, Vt_adj = svd(A_inv_adj, full_matrices=False)
    V_adj = Vt_adj.T

    if K2 == 1:
        B2_est = V_adj[:, :1]
        B2_est = _matlab_sign_flip(B2_est)
        A2_est = A_inv_adj @ B2_est @ _right_inverse_xtx(B2_est)
        A2_est = (A2_est > 0).astype(float)
        A2_centered = A2_est - np.mean(A2_est, axis=0, keepdims=True)
        B2_re_est = (_right_inverse_xtx(A2_centered) @ A2_centered.T @ A_inv_adj).T / factor
    else:
        R_V = _varimax(V_adj[:, :K2], q=1000)
        B2_est = thres(R_V, max(0.05, 1.0 / 2.5 / np.sqrt(K1)))
        B2_est = _matlab_sign_flip(B2_est)

        A2_est = A_inv_adj @ B2_est @ _right_inverse_xtx(B2_est)
        A2_est = (A2_est > 0).astype(float)
        A2_centered = A2_est - np.mean(A2_est, axis=0, keepdims=True)
        B2_re_est = (_right_inverse_xtx(A2_centered) @ A2_centered.T @ A_inv_adj).T / factor

    b2 = np.mean(A_inv_adj, axis=0) - B2_re_est @ np.mean(A2_est, axis=0)
    B2_ini = np.column_stack((b2, B2_re_est))
    return A2_est, B2_ini


def Normal_init_cop(X, K_list: Sequence[int], epsilon=1e-4):
    """Initialize CopDDE D-layer parameters using MATLAB ``Normal_init_D`` rules."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")

    N, J = X.shape
    K_tuple = tuple(int(k) for k in K_list)
    if len(K_tuple) < 1 or any(k < 1 for k in K_tuple):
        raise ValueError("K_list must contain at least one positive layer width.")
    D = len(K_tuple)

    A_list = [None] * D
    B_list = [None] * D

    K1 = K_tuple[0]
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    _, _, Vt = svd(X_centered, full_matrices=False)
    V_top_m = Vt.T[:, :K1]
    R_V = _varimax(V_top_m, q=1000)
    B1_est = thres(R_V, 1.0 / 2.5 / np.sqrt(J))
    B1_est = _matlab_sign_flip(B1_est)

    A1_est = X_centered @ B1_est @ _right_inverse_xtx(B1_est)
    A1_est = (A1_est > 0).astype(float)
    G1_est = B1_est != 0

    A1_centered = A1_est - np.mean(A1_est, axis=0, keepdims=True)
    B1_re_est = (_right_inverse_xtx(A1_centered) @ A1_centered.T @ X_centered).T
    B1_re_est = thres(B1_re_est * G1_est, 0)
    b1 = np.mean(X, axis=0) - B1_re_est @ np.mean(A1_est, axis=0)

    B_list[0] = np.column_stack((b1, B1_re_est))
    gamma = np.mean((X - np.column_stack((np.ones(N), A1_est)) @ B_list[0].T) ** 2, axis=0)
    gamma = np.maximum(gamma, 1e-8)
    A_list[0] = A1_est

    for d in range(1, D):
        A_list[d], B_list[d] = _binary_init_matlab(
            N,
            K_tuple[d - 1],
            K_tuple[d],
            A_list[d - 1],
            factor=2.0,
            epsilon=epsilon,
        )

    prop = np.clip(np.mean(A_list[-1] > 0, axis=0), 1e-8, 1.0 - 1e-8)
    validate_dde_lists(X, B_list, A_list=A_list, prop=prop, gamma=gamma, K_list=K_tuple)
    return prop, B_list, gamma, A_list


normal_init_cop = Normal_init_cop


__all__ = [
    "Normal_init_cop",
    "normal_init_cop",
]
