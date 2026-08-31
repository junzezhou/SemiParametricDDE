import numpy as np
from scipy.sparse.linalg import svds as truncated_svds
from scipy.linalg import svd
from dde.utils import logit, thres
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, norm
from sklearn.cluster import KMeans

try:
    from factor_analyzer.rotator import Rotator
except Exception:  # pragma: no cover - optional dependency fallback
    Rotator = None


def _varimax(Phi, gamma=1.0, q=1000, tol=1e-6):
    """Small varimax fallback used when factor-analyzer is unavailable."""
    Phi = np.asarray(Phi, dtype=float)
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


def _rotate_varimax(loadings):
    if Rotator is not None:
        return Rotator(method="varimax", max_iter=1000).fit_transform(loadings)
    return _varimax(loadings, q=1000)


# ──────────────────────────────────────────────────────────────
# Helper: safe pseudo-inverse–based least-squares  (A @ x ≈ b)
# ──────────────────────────────────────────────────────────────
def _safe_lstsq(A, B, rcond=1e-10):
    """Solve A @ X = B in least-squares sense, robust to rank deficiency."""
    return np.linalg.lstsq(A, B, rcond=rcond)[0]


def _safe_proj(A, B):
    """Compute (A^T A)^{-1} A^T B robustly via lstsq."""
    return _safe_lstsq(A, B)


def _permute_columns_hungarian(B_est, B_ref_slice):
    """
    Find the best column permutation of B_est to match B_ref_slice
    using scipy's C-level Hungarian algorithm (much faster than pure-Python munkres).

    B_est       : (rows, K)  estimated matrix whose columns to permute
    B_ref_slice : (rows, K)  reference columns (the non-intercept part of B_list[d])

    Returns permuted B_est.
    """
    K = B_est.shape[1]
    K_ref = B_ref_slice.shape[1]
    K_use = min(K, K_ref)

    costmat = np.zeros((K_use, K_use))
    for k in range(K_use):
        pos_idx = np.where(B_ref_slice[:, k] > 0)[0]
        if len(pos_idx) > 0:
            costmat[k, :] = -np.sum(B_est[pos_idx, :K_use], axis=0)

    row_ind, col_ind = linear_sum_assignment(costmat)
    perm = col_ind  # col_ind[k] = which column of B_est goes to position k
    return B_est[:, perm], perm


def _truncated_svd(M, k):
    """
    Compute the top-k SVD of M robustly.
    Falls back to full SVD when k >= min(M.shape) - 1 (svds limitation).
    """
    m, n = M.shape
    k = min(k, m, n)
    if k <= 0:
        return (np.zeros((m, 0)), np.array([]), np.zeros((0, n)))
    # scipy.sparse.linalg.svds requires k < min(m, n)
    if k < min(m, n) - 1:
        try:
            U, S, Vt = truncated_svds(np.asarray(M, dtype=np.float64), k=k)
            # svds returns in ascending order; flip to descending
            idx = np.argsort(-S)
            return U[:, idx], S[idx], Vt[idx, :]
        except Exception:
            pass
    # fallback: full SVD, then truncate
    U, S, Vt = svd(M, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


def _normal_residual_variance(X, A, B, row_chunk_size=2048):
    """Columnwise residual variance without a full N-by-J residual copy."""
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if row_chunk_size is None or N <= int(row_chunk_size):
        design = np.column_stack([np.ones(N), A])
        return np.mean((X - design @ B.T) ** 2, axis=0)

    row_chunk_size = max(1, int(row_chunk_size))
    residual_sum_squares = np.zeros(X.shape[1], dtype=float)
    for start in range(0, N, row_chunk_size):
        stop = min(start + row_chunk_size, N)
        design = np.column_stack([np.ones(stop - start), A[start:stop]])
        residual = X[start:stop] - design @ B.T
        residual_sum_squares += np.sum(residual ** 2, axis=0)
    return residual_sum_squares / N


def initialize_binary_parent(A_prev, K_parent, epsilon=1e-4):
    """Spectrally initialize one binary parent from active child states."""
    A_prev = np.asarray(A_prev, dtype=np.uint8)
    if A_prev.ndim != 2 or A_prev.shape[0] < 1 or A_prev.shape[1] < 1:
        raise ValueError("A_prev must be a nonempty binary matrix.")
    K_parent = int(K_parent)
    if K_parent < 1 or K_parent > A_prev.shape[1]:
        raise ValueError("K_parent must lie between 1 and A_prev.shape[1].")

    N, K_child = A_prev.shape
    n_sv = min(K_parent + 2, min(N, K_child))
    U_sv, S_sv, Vt_sv = _truncated_svd(A_prev, n_sv)
    m = max(K_parent + 1, int(np.sum(S_sv > 1.01 * np.sqrt(N))))
    m = min(m, len(S_sv))
    A_top_m = U_sv[:, :m] @ np.diag(S_sv[:m]) @ Vt_sv[:m, :]
    np.clip(A_top_m, epsilon, 1.0 - epsilon, out=A_top_m)
    A_inverse = logit(A_top_m)
    A_inverse -= np.mean(A_inverse, axis=0, keepdims=True)

    n_sv2 = min(K_parent + 1, min(N, K_child))
    _, _, Vt_adj = _truncated_svd(A_inverse, n_sv2)
    if K_parent == 1:
        B_est = Vt_adj.T[:, :1]
    else:
        B_est = thres(
            _rotate_varimax(Vt_adj.T[:, :K_parent]),
            1.0 / (3.5 * np.sqrt(K_child)),
        )
    sign_flip = np.sign(np.mean(B_est, axis=0))
    sign_flip[sign_flip == 0] = 1
    B_est *= sign_flip

    parent_nonbinary = A_inverse @ _safe_lstsq(
        B_est.T @ B_est, B_est.T
    ).T
    A_parent = (parent_nonbinary > 0).astype(np.uint8)
    parent_centered = A_parent - np.mean(A_parent, axis=0, keepdims=True)
    B_reestimated = _safe_proj(parent_centered, A_inverse).T / 2.0
    B_reestimated = thres(B_reestimated * (B_est != 0), 0)
    intercept = (
        np.mean(A_inverse, axis=0).reshape(-1, 1)
        - B_reestimated
        @ np.mean(A_parent, axis=0).reshape(-1, 1)
    )
    return np.hstack([intercept, B_reestimated]), A_parent


def Normal_init_semi(
    X,
    K1_max,
    K_list,
    B_list=None,
    G_list=None,
    epsilon=1e-4,
    *,
    mode=None,
):
    """
    Algorithm 1 (extended): Spectral initialization for Normal-D-latent-layer DDEs

    Parameters
    ----------
    X       : (N, J)
    K1_max  : int          upper bound for K1
    K_list  : (D,) int     [K_1, K_2, ..., K_D]
    B_list  : (D,) arrays, optional
                B_list[0] : (J,  K_1+1)
                B_list[d] : (K_d, K_{d+1}+1)  for d = 1,...,D-1
              Required only when mode="oracle".
    G_list  : (D,) arrays, optional
              Same shapes as B_list but binary (structure). Required only
              when mode="oracle".
    epsilon : float        truncation threshold for logit
    mode    : {"data", "oracle"}, optional
              "data" uses only X and K_list. "oracle" uses B_list/G_list
              for simulation-only column alignment and initialization
              structure diagnostics. If omitted, mode is inferred from
              whether B_list and G_list are supplied.

    Returns
    -------
    prop_ini     : (K_D,)
    B_list_ini   : list of D arrays (same shapes as B_list)
    gamma_ini    : (J,)
    A_list_ini   : list of D arrays, each (N, K_d)
    err_G_list   : list of D floats in oracle mode; list of None in data mode
    """
    N, J = X.shape
    D = len(K_list)
    if mode is None:
        mode = "oracle" if B_list is not None or G_list is not None else "data"
    mode = str(mode).lower()
    if mode not in {"data", "oracle"}:
        raise ValueError("mode must be 'data' or 'oracle'.")
    use_oracle = mode == "oracle"
    if use_oracle:
        if B_list is None or G_list is None:
            raise ValueError("B_list and G_list are required when mode='oracle'.")
        assert len(B_list) == D
        assert len(G_list) == D

    B_list_ini = [None] * D
    A_list_ini = [None] * D
    err_G_list = [None] * D

    # ============================================================
    # Layer 1: X -> A_1  (using K1_max)
    # ============================================================
    K1 = K_list[0]

    X_centered = X - np.mean(X, axis=0, keepdims=True)

    # --- Truncated SVD: only need top K1_max right singular vectors ---
    _, _, Vt_top = _truncated_svd(X_centered, K1_max)
    V_top_m = Vt_top.T  # (J, K1_max)

    R_V = _rotate_varimax(V_top_m)
    B1_est = thres(R_V, 1 / (2.5 * np.sqrt(J)))

    sign_flip = np.sign(np.mean(B1_est, axis=0))
    sign_flip[sign_flip == 0] = 1
    B1_est = B1_est * sign_flip

    if use_oracle:
        # Simulation-only alignment to true columns for MSE/structure summaries.
        B1_est, _ = _permute_columns_hungarian(B1_est, B_list[0][:, 1:])

        err_G_list[0] = np.mean(
            (B1_est[:, :min(K1, K1_max)] != 0) == G_list[0][:, :min(K1, K1_max)]
        )

    # estimate A1 via least-squares projection (robust to rank deficiency)
    BtB = B1_est.T @ B1_est
    A1_est_nonbin = X_centered @ _safe_lstsq(BtB, B1_est.T).T
    A1_est = (A1_est_nonbin > 0).astype(np.uint8)
    G1_est = (B1_est != 0)

    # re-estimate B1 via least-squares
    A1_centered = A1_est - np.mean(A1_est, axis=0, keepdims=True)
    B1_re_est = _safe_proj(A1_centered, X_centered).T
    B1_re_est = thres(B1_re_est * G1_est, 0)
    b1 = (np.mean(X, axis=0).reshape(-1, 1)
          - B1_re_est @ np.mean(A1_est, axis=0).reshape(-1, 1))
    B_list_ini[0] = np.hstack([b1, B1_re_est])
    gamma_ini = _normal_residual_variance(X, A1_est, B_list_ini[0])
    A_list_ini[0] = A1_est   # (N, K1)

    # These arrays are specific to the first-layer spectral fit and otherwise
    # remain live for the duration of the recursive initialization.
    del X_centered, A1_est_nonbin, A1_centered

    # ============================================================
    # Layers d = 2, ..., D:  logit(A_{d-1}) -> A_d
    # ============================================================
    for d in range(1, D):
        Kd   = K_list[d - 1]   # child layer dim  (= rows of B_list[d])
        Kd1  = K_list[d]       # parent layer dim
        A_prev = A_list_ini[d - 1]   # (N, Kd),  values in {0,1}

        # --- Truncated SVD of A_prev, then logit ---
        n_sv = min(Kd1 + 2, min(N, Kd))
        U_sv, S_sv, Vt_sv = _truncated_svd(A_prev, n_sv)
        m = max(Kd1 + 1, int(np.sum(S_sv > 1.01 * np.sqrt(N))))
        m = min(m, len(S_sv))
        A_top_m = U_sv[:, :m] @ np.diag(S_sv[:m]) @ Vt_sv[:m, :]
        np.clip(A_top_m, epsilon, 1 - epsilon, out=A_top_m)
        A_inv_adj = logit(A_top_m)
        A_inv_adj -= np.mean(A_inv_adj, axis=0, keepdims=True)

        # --- Truncated SVD of logit-transformed layer ---
        n_sv2 = min(Kd1 + 1, min(N, Kd))
        _, _, Vt_adj = _truncated_svd(A_inv_adj, n_sv2)
        V_adj = Vt_adj.T   # (Kd, n_sv2)

        if Kd1 == 1:
            # trivial single-column case
            Bd_est = V_adj[:, 0:1]
            sign_flip = np.sign(np.mean(Bd_est, axis=0))
            sign_flip[sign_flip == 0] = 1
            Bd_est = Bd_est * sign_flip

            BdtBd = Bd_est.T @ Bd_est
            Ad_est = A_inv_adj @ _safe_lstsq(BdtBd, Bd_est.T).T
            Ad_est = (Ad_est > 0).astype(np.uint8)
            Ad_centered = Ad_est - np.mean(Ad_est, axis=0, keepdims=True)
            Bd_re_est = _safe_proj(Ad_centered, A_inv_adj).T / 2
            if not use_oracle:
                Gd_est = (Bd_est != 0)
                Bd_re_est = thres(Bd_re_est * Gd_est, 0)

        else:
            # varimax rotation
            R_V = _rotate_varimax(V_adj[:, :Kd1])
            Bd_est = thres(R_V, 1 / (3.5 * np.sqrt(Kd)))

            sign_flip = np.sign(np.mean(Bd_est, axis=0))
            sign_flip[sign_flip == 0] = 1
            Bd_est = Bd_est * sign_flip

            if use_oracle:
                # Simulation-only alignment to true columns.
                Bd_est, _ = _permute_columns_hungarian(Bd_est, B_list[d][:, 1:])

            # estimate Ad via robust projection
            BdtBd = Bd_est.T @ Bd_est
            Ad_est = A_inv_adj @ _safe_lstsq(BdtBd, Bd_est.T).T
            Ad_est = (Ad_est > 0).astype(np.uint8)

            # re-estimate Bd via least-squares
            Ad_centered = Ad_est - np.mean(Ad_est, axis=0, keepdims=True)
            Bd_re_est = _safe_proj(Ad_centered, A_inv_adj).T / 2

            Gd_est = (Bd_est != 0)
            if use_oracle:
                # Re-align only in oracle/simulation mode.
                Bd_re_est, _ = _permute_columns_hungarian(Bd_re_est, B_list[d][:, 1:])

                # Match the original simulation initializer's stabilizing convention.
                diag_size = min(Kd1, Bd_re_est.shape[0], Bd_re_est.shape[1])
                for k in range(diag_size):
                    Bd_re_est[k, k] = max(Bd_re_est[k, k], 2.5)
            else:
                Bd_re_est = thres(Bd_re_est * Gd_est, 0)

        # final assembly for layer d
        bd = (np.mean(A_inv_adj, axis=0).reshape(-1, 1)
              - Bd_re_est @ np.mean(Ad_est, axis=0).reshape(-1, 1))
        B_list_ini[d] = np.hstack([bd, Bd_re_est])

        if use_oracle:
            err_G_list[d] = np.mean(
                (B_list_ini[d][:, 1:Kd1 + 1] != 0) == G_list[d][:, :Kd1]
            )

        A_list_ini[d] = Ad_est   # (N, Kd1)

    # ============================================================
    # prop_ini: top-layer marginal
    # ============================================================
    prop_ini = np.mean(A_list_ini[-1] > 0, axis=0)   # (K_D,)

    return prop_ini, B_list_ini, gamma_ini, A_list_ini, err_G_list


normal_init_semi = Normal_init_semi


def rank_gaussianize_columns(Y, clip_eps=1e-6, standardize=True):
    """
    Input:
        Y: (N, J) float
    Output:
        X0: (N, J) float
    """
    N, J = Y.shape
    X0 = np.zeros_like(Y, dtype=float)

    for j in range(J):
        y = Y[:, j]
        mask = ~np.isnan(y)
        yv = y[mask]
        if len(yv) == 0:
            continue

        # 计算秩并归一化到 (0,1)
        ranks = np.argsort(np.argsort(yv)) + 1
        u = (ranks - 0.5) / len(yv)
        u = np.clip(u, clip_eps, 1 - clip_eps)

        # 用 scipy.stats.norm.ppf 替代 erfinv
        x = norm.ppf(u)

        if standardize:
            x = (x - x.mean()) / x.std()

        X0[mask, j] = x

    return X0

__all__ = [
    "Normal_init_semi",
    "initialize_binary_parent",
    "normal_init_semi",
    "rank_gaussianize_columns",
]
