"""Hyperparameter-selection helper functions for SemiDDE and CopDDE."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from dde.algorithms.copula import prepare_rank_likelihood_inputs
from dde.algorithms.normal_init_semi import _rotate_varimax, rank_gaussianize_columns
from dde.algorithms.semi_saem import binary, compute_marginal_pmf_A1
from dde.results import DDEResult
from dde.simulation import simulate_latent_layers_D
from dde.utils import logit, thres, validate_K_list_identifiability


def _singular_values(M: np.ndarray) -> np.ndarray:
    return np.linalg.svd(np.asarray(M, dtype=float), compute_uv=False)


def _full_svd(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(np.asarray(M, dtype=float), full_matrices=False)


def _select_by_spectral_ratio(
    singular_values: np.ndarray,
    grid: Sequence[int],
    *,
    tolerance: float = 0.0,
) -> tuple[int, Dict[str, float]]:
    s = np.asarray(singular_values, dtype=float).reshape(-1)
    ratios: Dict[str, float] = {}
    best_k = None
    best_ratio = -np.inf
    for k_raw in grid:
        k = int(k_raw)
        if k < 1 or k >= s.size:
            continue
        denom = max(float(s[k]), 1e-12)
        ratio = float(s[k - 1] / denom)
        ratios[str(k)] = ratio
        if ratio > best_ratio:
            best_ratio = ratio
            best_k = k
    if best_k is None:
        raise ValueError("No valid spectral-ratio candidate was available for K selection.")
    if tolerance > 0.0:
        threshold = best_ratio - float(tolerance)
        best_k = min(int(k) for k, ratio in ratios.items() if ratio >= threshold)
    return best_k, ratios


def _default_k1_grid(N: int, J: int) -> List[int]:
    upper = max(1, min(J // 3, N - 2, J - 1))
    lower = 1 if upper < 4 else max(1, J // 4 + 1)
    if lower > upper:
        lower = 1
    return list(range(lower, upper + 1))


def _estimate_A_from_linearized(Z0: np.ndarray, K: int, *, threshold: float) -> np.ndarray:
    _, _, Vt = _full_svd(Z0)
    loadings = _rotate_varimax(Vt[:K, :].T)
    B_est = thres(loadings, threshold)
    sign_flip = np.sign(np.mean(B_est, axis=0))
    sign_flip[sign_flip == 0] = 1
    B_est = B_est * sign_flip
    BtB = B_est.T @ B_est
    A_nonbin = Z0 @ np.linalg.lstsq(BtB, B_est.T, rcond=1e-10)[0].T
    return (A_nonbin > 0).astype(float)


def _estimate_next_A_matlab_binary_init(A_est: np.ndarray, K_next: int, *, epsilon: float) -> np.ndarray:
    """Replicate MATLAB binary_init_1.m for layerwise K selection."""
    A_est = np.asarray(A_est, dtype=float)
    N, K_prev = A_est.shape
    U, S, Vt = _full_svd(A_est)
    m = max(min(int(K_next) * 2, K_prev - 2), int(np.sum(S > 1.01 * np.sqrt(N))))
    m = min(max(m, 1), len(S))
    A_top_m = U[:, :m] @ np.diag(S[:m]) @ Vt[:m, :]
    A_trunc = np.clip(A_top_m, epsilon, 1.0 - epsilon)
    A_inv = logit(A_trunc)
    A_inv_adj = A_inv - np.mean(A_inv, axis=0, keepdims=True)
    _, _, Vt_adj = _full_svd(A_inv_adj)

    if int(K_next) == 1:
        B_est = Vt_adj[:1, :].T
    else:
        loadings = _rotate_varimax(Vt_adj[: int(K_next), :].T)
        B_est = thres(loadings, max(0.05, 1.0 / (2.5 * np.sqrt(K_prev))))

    sign_flip = np.sign(np.mean(B_est, axis=0))
    sign_flip[sign_flip == 0] = 1
    B_est = B_est * sign_flip
    BtB = B_est.T @ B_est
    A_next = A_inv_adj @ np.linalg.lstsq(BtB, B_est.T, rcond=1e-10)[0].T
    return (A_next > 0).astype(float)


def estimate_K_list_semidde(
    Y,
    D: int,
    *,
    k1_grid: Optional[Sequence[int]] = None,
    epsilon: float = 1e-4,
    spectral_ratio_tolerance: float = 0.02,
    return_details: bool = False,
):
    """Estimate SemiDDE layer widths with the DDE spectral-ratio heuristic."""
    if D is None:
        raise ValueError("D is required to estimate K_list.")
    D = int(D)
    if D < 1:
        raise ValueError("D must be a positive integer.")

    X = rank_gaussianize_columns(np.asarray(Y, dtype=float))
    N, J = X.shape
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    min_k1 = 3 ** (D - 1)
    if k1_grid is None:
        default_grid_1 = _default_k1_grid(N, J)
        lower_1 = min(default_grid_1[0], min_k1)
        upper_1 = default_grid_1[-1]
        grid_1 = list(range(lower_1, upper_1 + 1))
    else:
        grid_1 = list(k1_grid)
    grid_1 = [int(k) for k in grid_1 if int(k) >= min_k1]
    if not grid_1:
        raise ValueError(f"Data support D at most 1 under the K_list identifiability rule.")
    K1, ratios_1 = _select_by_spectral_ratio(
        _singular_values(X_centered),
        grid_1,
        tolerance=spectral_ratio_tolerance,
    )
    K_list = [K1]
    details = [
        {
            "layer": 1,
            "grid": grid_1,
            "ratios": ratios_1,
            "selected": K1,
        }
    ]

    A_est = _estimate_A_from_linearized(
        X_centered,
        K1,
        threshold=1.0 / (2.5 * np.sqrt(J)),
    )
    if K1 < 3 and D > 1:
        raise ValueError(f"Data support D at most 1 under the K_list identifiability rule.")
    K_prev = K1
    for layer in range(2, D + 1):
        lower = 1 if layer == D else max(int(np.ceil(K_prev / 4.0)), 1)
        if layer < D:
            lower = max(lower, 3 ** (D - layer))
        upper = min(int(np.floor(K_prev / 3)), min(A_est.shape) - 1)
        if upper < 1:
            raise ValueError(f"Data support D at most {layer - 1} under the K_list identifiability rule.")
        if upper < lower:
            raise ValueError(f"Data support D at most {layer} under the K_list identifiability rule.")
        grid = list(range(lower, upper + 1))
        U, S, Vt = _full_svd(A_est)
        m = max(upper + 1, int(np.sum(S > 1.01 * np.sqrt(N))))
        m = min(m, len(S))
        A_top_m = U[:, :m] @ np.diag(S[:m]) @ Vt[:m, :]
        A_trunc = np.clip(A_top_m, epsilon, 1.0 - epsilon)
        A_inv = logit(A_trunc)
        A_inv_adj = A_inv - np.mean(A_inv, axis=0, keepdims=True)
        K_d, ratios_d = _select_by_spectral_ratio(
            _singular_values(A_inv_adj),
            grid,
            tolerance=spectral_ratio_tolerance,
        )
        K_list.append(K_d)
        details.append(
            {
                "layer": layer,
                "grid": grid,
                "ratios": ratios_d,
                "selected": K_d,
            }
        )
        if K_d < 3 and layer < D:
            raise ValueError(f"Data support D at most {layer} under the K_list identifiability rule.")
        if layer < D:
            A_est = _estimate_next_A_matlab_binary_init(A_est, K_d, epsilon=epsilon)
            K_prev = K_d

    validate_K_list_identifiability(K_list, D=D)
    if return_details:
        return K_list, details
    return K_list


def estimate_K_list_copula(Y, D: int) -> List[int]:
    """Initialize CopDDE maximal widths as ``floor(J / 3**d)``."""
    if D is None:
        raise ValueError("D is required to estimate K_list.")
    D = int(D)
    if D < 1:
        raise ValueError("D must be a positive integer.")
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be a 2D array; got shape {Y.shape}.")
    J = int(Y.shape[1])
    K_list = []
    prev_width = J
    for layer in range(1, D + 1):
        K_d = int(prev_width // 3)
        if K_d < 1:
            raise ValueError(f"Data support D at most {layer - 1} under the K_list identifiability rule.")
        K_list.append(K_d)
        if K_d < 3 and layer < D:
            raise ValueError(f"Data support D at most {layer} under the K_list identifiability rule.")
        prev_width = K_d
    return list(validate_K_list_identifiability(K_list, D=D, J=J))


def copula_tau_default(N: int) -> float:
    """Return the CopDDE paper's threshold default ``max(0.3, 3*N**(-0.3))``."""
    return float(max(0.3, 3.0 * (int(N) ** -0.3)))


def auto_grid_copula_fixed(N: int, D: int) -> List[Dict[str, Any]]:
    """Return the compact CopDDE fixed-penalty grid."""
    D = int(D)
    tau = copula_tau_default(N)
    candidates = []
    for temp in (0.7, 0.9):
        for lambda_exp in (1 / 8, 2 / 8, 3 / 8):
            pen = float(int(N) ** float(lambda_exp))
            candidates.append(
                {
                    "temp": float(temp),
                    "pen_list": [pen] * D,
                    "tau": tau,
                    "lambda_exp": float(lambda_exp),
                }
            )
    return candidates


def _default_csp_t_spike_values(D: int) -> List[List[float]]:
    if D <= 2:
        base = [[0.01, 0.03, 0.05], [0.03, 0.05, 0.10]]
        return [base[d] for d in range(D)]
    values = [[0.01, 0.03, 0.05], [0.01, 0.03, 0.05], [0.003, 0.005, 0.01]]
    while len(values) < D:
        values.append([0.003, 0.005, 0.01])
    return values[:D]


def _default_csp_t_spike_base(D: int, values: Sequence[Sequence[float]]) -> List[float]:
    if D == 3:
        return [0.01, 0.03, 0.005]
    return [float(layer_values[len(layer_values) // 2]) for layer_values in values]


def auto_grid_copula_csp(
    N: int,
    D: int,
    *,
    strategy: str = "coordinate",
    t_spike_values: Optional[Sequence[Sequence[float]]] = None,
    base_t_spike: Optional[Sequence[float]] = None,
) -> Any:
    """Return a CopDDE CSP full grid or sequential coordinate-search plan."""
    D = int(D)
    values = [list(map(float, vals)) for vals in (t_spike_values or _default_csp_t_spike_values(D))]
    if len(values) != D:
        raise ValueError("t_spike_values must contain one sequence per layer.")
    tau = copula_tau_default(N)
    temp = 0.7 if D <= 2 else 0.9
    strategy = strategy.lower()
    if strategy == "coordinate" and D >= 3:
        base = list(map(float, base_t_spike or _default_csp_t_spike_base(D, values)))
        if len(base) != D:
            raise ValueError("base_t_spike must have length D.")
        return {
            "strategy": "coordinate",
            "temp": temp,
            "tau": tau,
            "base_t_spike": base,
            "t_spike_values": values,
        }
    if strategy not in {"coordinate", "full"}:
        raise ValueError("strategy must be 'coordinate' or 'full'.")
    return [
        {
            "temp": temp,
            "tau": tau,
            "t_spike": list(vals),
        }
        for vals in product(*values)
    ]


def auto_grid(
    N: int,
    D: int,
    *,
    layer_scales: Optional[Sequence[float]] = None,
    lambda_exponents: Sequence[float] = (2 / 8, 3 / 8),
    tau_exponents: Sequence[float] = (-2 / 8, -3 / 8),
    tau_values: Sequence[float] = (0.3,),
) -> List[Dict[str, Any]]:
    """Return the DDE-style grid over ``pen_list`` and ``tau``."""
    D = int(D)
    if D < 1:
        raise ValueError("D must be a positive integer.")
    if layer_scales is None:
        layer_scales_arr = np.ones(D, dtype=float)
    else:
        layer_scales_arr = np.asarray(layer_scales, dtype=float).reshape(-1)
        if layer_scales_arr.shape != (D,):
            raise ValueError("layer_scales must have length D.")
    candidates = []
    tau_grid = [(float(2.0 * (N ** float(tau_exp))), float(tau_exp)) for tau_exp in tau_exponents]
    tau_grid.extend((float(tau), None) for tau in tau_values)
    for lambda_exp in lambda_exponents:
        lam = float(N ** float(lambda_exp))
        pen_list = (lam * layer_scales_arr).tolist()
        for tau, tau_exp in tau_grid:
            candidates.append(
                {
                    "pen_list": pen_list,
                    "tau": tau,
                    "lambda_exp": float(lambda_exp),
                    "tau_exp": tau_exp,
                }
            )
    return candidates


def _normal_score_matrix(Y: np.ndarray, model: str) -> np.ndarray:
    if model == "copula":
        Z, _, _ = prepare_rank_likelihood_inputs(Y)
        return Z
    return rank_gaussianize_columns(np.asarray(Y, dtype=float))


def normal_mixture_loglik_score(
    Y: np.ndarray,
    result: DDEResult,
    *,
    model: Optional[str] = None,
    max_enum_bits: int = 20,
    mc_samples: int = 50_000,
    pattern_chunk_size: int = 4096,
) -> float:
    """Score observations by the fitted latent-normal mixture pseudo log-likelihood.

    The score is an average holdout log-likelihood after deterministic normal-score
    transformation of the observed columns.  It uses no simulation truth.
    """
    if result.prop is None or result.B_list is None or result.gamma is None:
        raise ValueError("result must contain prop, B_list, and gamma to be scored.")

    chosen = (model or result.model).lower()
    X = _normal_score_matrix(np.asarray(Y), "copula" if chosen in {"copula", "copdde"} else "semi")
    B_list = [np.asarray(B, dtype=float) for B in result.B_list]
    gamma = np.maximum(np.asarray(result.gamma, dtype=float).reshape(-1), 1e-8)
    K1 = B_list[0].shape[1] - 1
    if K1 > max_enum_bits:
        raise ValueError(
            "normal_mixture_loglik_score currently requires K_1 <= max_enum_bits "
            "for exact holdout scoring."
        )

    patterns = binary(np.arange(1 << K1, dtype=np.int64), K1)
    pmf = np.asarray(
        compute_marginal_pmf_A1(
            result.prop,
            B_list,
            max_enum_bits=max_enum_bits,
            mc_samples=mc_samples,
        ),
        dtype=float,
    )
    pmf = np.clip(pmf, 1e-300, None)
    pmf = pmf / pmf.sum()
    chunk_size = max(1, int(pattern_chunk_size))
    log_pmf = np.log(pmf)
    logdet = float(np.sum(np.log(2.0 * np.pi * gamma)))
    x_quad = np.sum((X * X) / gamma[None, :], axis=1)
    x_scaled = X / gamma[None, :]
    obs_loglik = np.full(X.shape[0], -np.inf, dtype=float)

    for start in range(0, patterns.shape[0], chunk_size):
        end = min(start + chunk_size, patterns.shape[0])
        means = np.column_stack([np.ones(end - start), patterns[start:end]]) @ B_list[0].T
        means_quad = np.sum((means * means) / gamma[None, :], axis=1)
        cross = x_scaled @ means.T
        log_norm = -0.5 * (x_quad[:, None] + means_quad[None, :] - 2.0 * cross + logdet)
        chunk_loglik = logsumexp(log_pmf[start:end][None, :] + log_norm, axis=1)
        obs_loglik = np.logaddexp(obs_loglik, chunk_loglik)

    return float(np.mean(obs_loglik))


def _as_rng(random_state=None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _empirical_inverse_cdf_columns(reference: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float)
    out = np.zeros_like(uniforms, dtype=float)
    for j in range(reference.shape[1]):
        obs = reference[:, j]
        obs = obs[np.isfinite(obs)]
        if obs.size == 0:
            raise ValueError(f"Y column {j} contains no finite values for empirical marginal mapping.")
        obs = np.sort(obs)
        idx = np.clip(np.ceil(uniforms[:, j] * obs.size).astype(int) - 1, 0, obs.size - 1)
        out[:, j] = obs[idx]
    return out


def simulate_copula_predictive(
    Y_reference: np.ndarray,
    result: DDEResult,
    *,
    n_samples: Optional[int] = None,
    random_state: Optional[object] = None,
) -> np.ndarray:
    """Generate synthetic observed data from a fitted CopDDE model."""
    if result.prop is None or result.B_list is None or result.gamma is None:
        raise ValueError("result must contain prop, B_list, and gamma to simulate predictive data.")
    rng = _as_rng(random_state)
    Y_reference = np.asarray(Y_reference, dtype=float)
    N = int(n_samples or Y_reference.shape[0])
    B_list = [np.asarray(B, dtype=float) for B in result.B_list]
    gamma = np.maximum(np.asarray(result.gamma, dtype=float).reshape(-1), 1e-8)
    A_list, _ = simulate_latent_layers_D(N, result.prop, B_list, rng)
    A1_design = np.column_stack((np.ones(N), A_list[0]))
    mu = A1_design @ B_list[0].T
    Z = mu + np.sqrt(gamma.reshape(1, -1)) * rng.standard_normal(size=mu.shape)
    uniforms = np.zeros_like(Z, dtype=float)
    for j in range(Z.shape[1]):
        uniforms[:, j] = rankdata(Z[:, j], method="average") / (N + 1.0)
    return _empirical_inverse_cdf_columns(Y_reference, uniforms)


def _rf_ready_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        fill = float(np.median(col[finite])) if np.any(finite) else 0.0
        col[~finite] = fill
        X[:, j] = col
    return X


def copula_predictive_pmse_score(
    Y: np.ndarray,
    result: DDEResult,
    *,
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    n_estimators: int = 100,
    n_splits: int = 5,
) -> tuple[float, List[float]]:
    """Compute the CopDDE predictive-evaluation pMSE score."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    Y = np.asarray(Y, dtype=float)
    rng = _as_rng(random_state)
    pmse_values: List[float] = []
    for _ in range(int(n_repeats)):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        synth = simulate_copula_predictive(Y, result, n_samples=Y.shape[0], random_state=seed)
        X_rf = _rf_ready_matrix(np.vstack([Y, synth]))
        label = np.concatenate([np.ones(Y.shape[0], dtype=int), np.zeros(synth.shape[0], dtype=int)])
        splits = min(int(n_splits), int(np.sum(label == 0)), int(np.sum(label == 1)))
        if splits < 2:
            raise ValueError("pMSE evaluation needs at least two observations in each class.")
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        clf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=seed,
            n_jobs=1,
            class_weight=None,
        )
        p_hat = cross_val_predict(clf, X_rf, label, cv=cv, method="predict_proba")[:, 1]
        pmse_values.append(float(np.mean((p_hat - 0.5) ** 2)))
    return float(np.mean(pmse_values)), pmse_values


__all__ = [
    "auto_grid",
    "auto_grid_copula_csp",
    "auto_grid_copula_fixed",
    "copula_predictive_pmse_score",
    "copula_tau_default",
    "estimate_K_list_copula",
    "estimate_K_list_semidde",
    "normal_mixture_loglik_score",
    "simulate_copula_predictive",
]
