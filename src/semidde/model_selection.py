"""Data-driven hyperparameter selection for SemiDDE and CopDDE."""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from .api import fit
from .algorithms.copula_D import prepare_rank_likelihood_inputs_D
from .algorithms.normal_init_semi import Normal_init_semi, _rotate_varimax, rank_gaussianize_columns
from .algorithms.semi_saem import binary, compute_marginal_pmf_A1
from .results import DDEResult
from .simulation import simulate_latent_layers_D
from .utils import logit, thres, validate_K_list_identifiability


InitFactory = Callable[[np.ndarray, Sequence[int], Mapping[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class CVSelectionResult:
    """Summary of a cross-validation hyperparameter search."""

    best_index: int
    best_candidate: Dict[str, Any]
    cv_table: List[Dict[str, Any]]
    fold_scores: List[List[float]]


def _copy_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    copied = dict(candidate)
    if "K_list" in copied and copied["K_list"] is not None:
        copied["K_list"] = [int(k) for k in copied["K_list"]]
    if "pen_list" in copied and copied["pen_list"] is not None:
        copied["pen_list"] = list(copied["pen_list"])
    if "t_spike" in copied and copied["t_spike"] is not None:
        copied["t_spike"] = list(copied["t_spike"])
    return copied


def _split_fit_kwargs(candidate: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    fit_kwargs = _copy_candidate(candidate)
    meta = {}
    for key in ("name", "id", "label", "lambda_exp", "tau_exp", "coordinate_layer", "coordinate_value"):
        if key in fit_kwargs:
            meta[key] = fit_kwargs.pop(key)
    return fit_kwargs, meta


def _as_candidate_list(candidates: Optional[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    if candidates is None:
        return [{}]
    candidate_list = [_copy_candidate(c) for c in candidates]
    if not candidate_list:
        raise ValueError("candidates must contain at least one candidate.")
    return candidate_list


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


def _semi_data_init(
    Y: np.ndarray,
    K_list: Sequence[int],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    X = rank_gaussianize_columns(np.asarray(Y, dtype=float))
    K_tuple = tuple(int(k) for k in K_list)
    validate_K_list_identifiability(K_tuple, J=Y.shape[1])
    prop, B_list, gamma, A_list, _ = Normal_init_semi(
        X,
        K_tuple[0],
        K_tuple,
        epsilon=context.get("epsilon", 1e-4),
        mode="data",
    )
    return {
        "X_in": X,
        "prop_in": prop,
        "B_list_in": B_list,
        "gamma_in": gamma,
        "A_list_in": A_list,
    }


def _normal_score_matrix(Y: np.ndarray, model: str) -> np.ndarray:
    if model == "copula":
        Z, _, _ = prepare_rank_likelihood_inputs_D(Y)
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


def _model_df_and_capacity(result: DDEResult, tau: float = 0.0) -> tuple[int, int]:
    if result.prop is None or result.B_list is None or result.gamma is None:
        raise ValueError("result must contain prop, B_list, and gamma.")
    df = int(np.asarray(result.prop).size + np.asarray(result.gamma).size)
    capacity = df
    threshold = float(tau)
    for B in result.B_list:
        B_arr = np.asarray(B, dtype=float)
        df += B_arr.shape[0]
        capacity += B_arr.size
        slopes = B_arr[:, 1:]
        df += int(np.sum(np.abs(slopes) > threshold))
    return int(df), int(capacity)


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def _score_from_loglik(
    avg_loglik: float,
    *,
    criterion: str,
    df: int,
    capacity: int,
    n_model: int,
    n_score: int,
    ebic_gamma: float,
) -> float:
    criterion = criterion.lower()
    if criterion == "loglik":
        return float(avg_loglik)
    if criterion == "bic":
        return float(avg_loglik - 0.5 * df * np.log(n_model) / max(n_score, 1))
    if criterion == "ebic":
        penalty = df * np.log(n_model) + 2.0 * ebic_gamma * _log_comb(capacity, min(df, capacity))
        return float(avg_loglik - 0.5 * penalty / max(n_score, 1))
    raise ValueError("criterion must be 'bic', 'ebic', or 'loglik'.")


def _resolve_semidde_defaults(
    Y: np.ndarray,
    D: int,
    fixed_hp: Mapping[str, Any],
    *,
    k_estimator_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    N = Y.shape[0]
    defaults: Dict[str, Any] = {
        "C": 1,
        "max_outer_iter": 10,
        "max_inner_iter": 50,
        "max_enum_bits": 12,
        "mc_samples": 50_000,
        "g_update_rho": 1.0,
        "initialize": "data",
        "verbose": False,
    }
    diagnostics: Dict[str, Any] = {}
    if "K_list" in fixed_hp:
        defaults["K_list"] = list(validate_K_list_identifiability(fixed_hp["K_list"], D=D, J=Y.shape[1]))
        diagnostics["K_list_source"] = "fixed_hp"
    else:
        K_list, K_details = estimate_K_list_semidde(
            Y,
            D,
            return_details=True,
            **dict(k_estimator_kwargs or {}),
        )
        defaults["K_list"] = K_list
        diagnostics["K_list_source"] = "estimated"
        diagnostics["K_list_estimation"] = K_details
    defaults["tol"] = float(defaults["K_list"][-1]) / 2.0
    defaults["pen_list"] = [float(N ** (2 / 8))] * int(D)
    defaults["tau"] = 0.3
    return defaults, diagnostics


def _resolve_copula_defaults(
    Y: np.ndarray,
    D: int,
    fixed_hp: Mapping[str, Any],
    *,
    estimate_K: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    N = Y.shape[0]
    defaults: Dict[str, Any] = {
        "C": 1,
        "it": 50,
        "tol": 1e-3,
        "temp": 0.7 if int(D) <= 2 else 0.9,
        "tau": copula_tau_default(N),
        "pen_list": [float(N ** (2 / 8))] * int(D),
        "initialize": "data",
        "verbose": False,
    }
    diagnostics: Dict[str, Any] = {}
    if "K_list" in fixed_hp:
        defaults["K_list"] = list(validate_K_list_identifiability(fixed_hp["K_list"], D=D, J=Y.shape[1]))
        diagnostics["K_list_source"] = "fixed_hp"
    elif estimate_K:
        K_list = estimate_K_list_copula(Y, D)
        defaults["K_list"] = K_list
        diagnostics["K_list_source"] = "copula_maximal_width"
    return defaults, diagnostics


def _with_copula_method_defaults(hp: Mapping[str, Any], D: int) -> Dict[str, Any]:
    hp_out = dict(hp)
    method = str(hp_out.get("dimension_method", "csp")).lower()
    if method == "csp":
        hp_out.pop("pen_list", None)
        hp_out.pop("tol", None)
        if "t_spike" not in hp_out:
            hp_out["t_spike"] = _default_csp_t_spike_base(D, _default_csp_t_spike_values(D))
    elif method == "fixed":
        hp_out.pop("t_spike", None)
    return hp_out


def _resolve_dimension_method(model: str, dimension_method: Optional[str]) -> str:
    if model == "semi":
        method = "fixed" if dimension_method is None else str(dimension_method).lower()
        if method != "fixed":
            raise ValueError("SemiDDE supports only dimension_method='fixed'.")
        return method
    method = "csp" if dimension_method is None else str(dimension_method).lower()
    if method not in {"fixed", "csp"}:
        raise ValueError("CopDDE dimension_method must be 'fixed' or 'csp'.")
    return method


def _reject_embedded_dimension_method(
    *,
    candidates: Any,
    fixed_hp: Mapping[str, Any],
    base_fit_kwargs: Mapping[str, Any],
) -> None:
    locations = []
    if "dimension_method" in fixed_hp:
        locations.append("fixed_hp")
    if "dimension_method" in base_fit_kwargs:
        locations.append("base_fit_kwargs")
    if _is_coordinate_plan(candidates):
        if "dimension_method" in candidates:
            locations.append("candidates")
    else:
        for candidate in _as_candidate_list(candidates):
            if "dimension_method" in candidate:
                locations.append("candidates")
                break
    if locations:
        where = ", ".join(locations)
        raise ValueError(
            "dimension_method is an algorithm-path option, not a hyperparameter; "
            f"pass it directly to select_hyperparameters(...). Found it in: {where}."
        )


def _validate_copula_path_hyperparameters(
    *,
    dimension_method: str,
    candidates: Any,
    fixed_hp: Mapping[str, Any],
    base_fit_kwargs: Mapping[str, Any],
) -> None:
    containers: List[Mapping[str, Any]] = [fixed_hp, base_fit_kwargs]
    if _is_coordinate_plan(candidates):
        containers.append(candidates)
    else:
        containers.extend(_as_candidate_list(candidates))
    if dimension_method == "csp" and any("pen_list" in container for container in containers):
        raise ValueError(
            "pen_list is only valid for CopDDE dimension_method='fixed'; "
            "pass dimension_method='fixed' to select_hyperparameters(...) for the fixed path."
        )
    if dimension_method == "fixed" and any("t_spike" in container or "t_spike_values" in container for container in containers):
        raise ValueError(
            "t_spike is only valid for CopDDE dimension_method='csp'; "
            "pass dimension_method='csp' to select_hyperparameters(...) for the CSP path."
        )


def _is_coordinate_plan(candidates: Any) -> bool:
    return isinstance(candidates, Mapping) and candidates.get("strategy") == "coordinate"


def _fit_copula_pmse_candidate(
    Y: np.ndarray,
    hp: Mapping[str, Any],
    *,
    n_repeats: int,
    random_state: Optional[int],
    pmse_n_estimators: int,
    scorer: Optional[Callable[[np.ndarray, DDEResult], float]] = None,
) -> tuple[DDEResult, float, List[float], Dict[str, float]]:
    fit_started = time.perf_counter()
    fitted = fit(Y, model="copula", **dict(hp))
    fit_seconds = time.perf_counter() - fit_started
    score_started = time.perf_counter()
    if scorer is not None:
        score = float(scorer(Y, fitted))
        score_seconds = time.perf_counter() - score_started
        return fitted, score, [score], {
            "fit_seconds": float(fit_seconds),
            "score_seconds": float(score_seconds),
            "total_seconds": float(fit_seconds + score_seconds),
        }
    pmse, pmse_values = copula_predictive_pmse_score(
        Y,
        fitted,
        n_repeats=n_repeats,
        random_state=random_state,
        n_estimators=pmse_n_estimators,
    )
    score_seconds = time.perf_counter() - score_started
    return fitted, -float(pmse), [-float(value) for value in pmse_values], {
        "fit_seconds": float(fit_seconds),
        "score_seconds": float(score_seconds),
        "total_seconds": float(fit_seconds + score_seconds),
    }


def _row_from_hp(
    *,
    candidate_index: int,
    meta: Mapping[str, Any],
    K_list: Sequence[int],
    criterion: str,
    scores: Sequence[float],
    hp: Mapping[str, Any],
) -> Dict[str, Any]:
    row = {
        "candidate_index": candidate_index,
        **dict(meta),
        "K_list": list(K_list),
        "criterion": criterion,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    }
    if criterion == "pmse":
        row["mean_pmse"] = float(-row["mean_score"])
        row["std_pmse"] = row["std_score"]
    for key, value in hp.items():
        if key not in row and np.isscalar(value):
            row[key] = value
        elif key in {"pen_list", "t_spike"}:
            row[key] = np.asarray(value, dtype=float).reshape(-1).tolist()
    return row


def _select_copula_pmse(
    Y: np.ndarray,
    *,
    D: int,
    candidates: Any,
    fixed: Mapping[str, Any],
    base: Mapping[str, Any],
    defaults: Mapping[str, Any],
    setup_diagnostics: Mapping[str, Any],
    dimension_method: str,
    refit: bool,
    scorer: Optional[Callable[[np.ndarray, DDEResult], float]],
    n_repeats: int,
    random_state: Optional[int],
    pmse_n_estimators: int,
) -> DDEResult:
    cv_table: List[Dict[str, Any]] = []
    fold_scores: List[List[float]] = []
    fitted_results: List[Optional[DDEResult]] = []
    timing_rows: List[Dict[str, Any]] = []
    evaluation_cache: Dict[tuple, tuple[DDEResult, float, List[float], Dict[str, float]]] = {}

    def complete_hp(candidate: Mapping[str, Any]) -> Dict[str, Any]:
        fit_candidate, _ = _split_fit_kwargs(candidate)
        hp = dict(defaults)
        hp.update(base)
        hp.update(fixed)
        hp.update(fit_candidate)
        hp["dimension_method"] = dimension_method
        hp = _with_copula_method_defaults(hp, D)
        K_list = hp.get("K_list")
        if K_list is None:
            raise ValueError("Each candidate, fixed_hp, base_fit_kwargs, or defaults must provide K_list.")
        hp["K_list"] = list(validate_K_list_identifiability(K_list, D=D, J=Y.shape[1]))
        return hp

    def cache_key(hp: Mapping[str, Any]) -> tuple:
        key_parts = []
        for key, value in sorted(hp.items()):
            if key in {"Z", "R", "Rlevels", "prop_in", "gamma_in", "B_list_in", "A_list_in"}:
                key_parts.append((key, id(value)))
            elif key in {"K_list", "pen_list", "t_spike"}:
                key_parts.append((key, tuple(np.asarray(value).reshape(-1).tolist())))
            elif np.isscalar(value) or value is None:
                key_parts.append((key, value))
            else:
                key_parts.append((key, id(value)))
        return tuple(key_parts)

    def evaluate_pmse_candidate(idx: int, hp: Mapping[str, Any]):
        key = cache_key(hp)
        if key in evaluation_cache:
            result, mean_score, scores, original_timing = evaluation_cache[key]
            timing = {
                "fit_seconds": 0.0,
                "score_seconds": 0.0,
                "total_seconds": 0.0,
                "cached_fit_seconds": float(original_timing["fit_seconds"]),
                "cached_score_seconds": float(original_timing["score_seconds"]),
                "cached_total_seconds": float(original_timing["total_seconds"]),
            }
            return result, mean_score, list(scores), timing, True
        result, mean_score, scores, timing = _fit_copula_pmse_candidate(
            Y,
            hp,
            n_repeats=n_repeats,
            random_state=None if random_state is None else random_state + idx,
            pmse_n_estimators=pmse_n_estimators,
            scorer=scorer,
        )
        evaluation_cache[key] = (result, mean_score, list(scores), dict(timing))
        return result, mean_score, scores, timing, False

    if _is_coordinate_plan(candidates):
        plan = dict(candidates)
        current = list(map(float, plan["base_t_spike"]))
        values = [list(map(float, vals)) for vals in plan["t_spike_values"]]
        best_result: Optional[DDEResult] = None
        best_row_index = -1
        for layer in range(D):
            layer_best_score = -np.inf
            layer_best_value = current[layer]
            for value in values[layer]:
                trial = list(current)
                trial[layer] = float(value)
                candidate = {
                    "temp": float(plan["temp"]),
                    "tau": float(plan["tau"]),
                    "t_spike": trial,
                    "coordinate_layer": layer,
                    "coordinate_value": float(value),
                }
                hp = complete_hp(candidate)
                fit_candidate, meta = _split_fit_kwargs(candidate)
                idx = len(cv_table)
                result, mean_score, scores, timing, from_cache = evaluate_pmse_candidate(idx, hp)
                row = _row_from_hp(
                    candidate_index=idx,
                    meta=meta,
                    K_list=hp["K_list"],
                    criterion="pmse",
                    scores=scores,
                    hp={**fit_candidate, "coordinate_layer": layer, "coordinate_value": float(value)},
                )
                row["fit_seconds"] = float(timing["fit_seconds"])
                row["score_seconds"] = float(timing["score_seconds"])
                row["elapsed_seconds"] = float(timing["total_seconds"])
                row["from_cache"] = bool(from_cache)
                if from_cache:
                    row["cached_fit_seconds"] = float(timing["cached_fit_seconds"])
                    row["cached_score_seconds"] = float(timing["cached_score_seconds"])
                    row["cached_elapsed_seconds"] = float(timing["cached_total_seconds"])
                cv_table.append(row)
                fold_scores.append(scores)
                fitted_results.append(result)
                timing_rows.append(
                    {
                        "candidate_index": int(idx),
                        "fold_index": None,
                        "n_train": int(Y.shape[0]),
                        "n_holdout": int(Y.shape[0]),
                        "fit_seconds": float(timing["fit_seconds"]),
                        "score_seconds": float(timing["score_seconds"]),
                        "total_seconds": float(timing["total_seconds"]),
                        "from_cache": bool(from_cache),
                    }
                )
                if mean_score > layer_best_score:
                    layer_best_score = mean_score
                    layer_best_value = float(value)
                    best_result = result
                    best_row_index = idx
            current[layer] = layer_best_value
        best_index = best_row_index
        best_candidate = dict(cv_table[best_index])
    else:
        candidate_list = _as_candidate_list(candidates)
        for idx, candidate in enumerate(candidate_list):
            hp = complete_hp(candidate)
            fit_candidate, meta = _split_fit_kwargs(candidate)
            result, _, scores, timing, from_cache = evaluate_pmse_candidate(idx, hp)
            row = _row_from_hp(
                candidate_index=idx,
                meta=meta,
                K_list=hp["K_list"],
                criterion="pmse",
                scores=scores,
                hp=hp,
            )
            row["fit_seconds"] = float(timing["fit_seconds"])
            row["score_seconds"] = float(timing["score_seconds"])
            row["elapsed_seconds"] = float(timing["total_seconds"])
            row["from_cache"] = bool(from_cache)
            if from_cache:
                row["cached_fit_seconds"] = float(timing["cached_fit_seconds"])
                row["cached_score_seconds"] = float(timing["cached_score_seconds"])
                row["cached_elapsed_seconds"] = float(timing["cached_total_seconds"])
            cv_table.append(row)
            fold_scores.append(scores)
            fitted_results.append(result)
            timing_rows.append(
                {
                    "candidate_index": int(idx),
                    "fold_index": None,
                    "n_train": int(Y.shape[0]),
                    "n_holdout": int(Y.shape[0]),
                    "fit_seconds": float(timing["fit_seconds"]),
                    "score_seconds": float(timing["score_seconds"]),
                    "total_seconds": float(timing["total_seconds"]),
                    "from_cache": bool(from_cache),
                }
            )
        best_index = int(np.argmax([row["mean_score"] for row in cv_table]))
        best_candidate = _copy_candidate(candidate_list[best_index])
        best_result = fitted_results[best_index]

    selection = CVSelectionResult(
        best_index=best_index,
        best_candidate=best_candidate,
        cv_table=cv_table,
        fold_scores=fold_scores,
    )
    diagnostics = {
        "hyperparameter_selection": selection,
        "cv_table": cv_table,
        "fold_scores": fold_scores,
        "timing_table": timing_rows,
        **dict(setup_diagnostics),
    }
    if _is_coordinate_plan(candidates):
        diagnostics["copula_grid_strategy"] = "coordinate"
        diagnostics["selected_t_spike"] = cv_table[best_index].get("t_spike", None)
    if not refit:
        return DDEResult(model="copula", diagnostics=diagnostics)
    final = best_result
    if final is None:
        raise RuntimeError("No CopDDE pMSE candidate was fitted.")
    final.diagnostics = dict(final.diagnostics)
    final.diagnostics.update(diagnostics)
    return final


def select_hyperparameters(
    Y,
    *,
    D: int,
    model: str,
    dimension_method: Optional[str] = None,
    candidates: Optional[Iterable[Mapping[str, Any]]] = None,
    fixed_hp: Optional[Mapping[str, Any]] = None,
    criterion: str = "bic",
    n_splits: int = 5,
    random_state: Optional[int] = None,
    base_fit_kwargs: Optional[Mapping[str, Any]] = None,
    init_factory: Optional[InitFactory] = None,
    refit: bool = True,
    scorer: Optional[Callable[[np.ndarray, DDEResult], float]] = None,
    ebic_gamma: float = 1.0,
    k_estimator_kwargs: Optional[Mapping[str, Any]] = None,
    n_repeats: int = 10,
    pmse_n_estimators: int = 100,
) -> DDEResult:
    """Select hyperparameters using K-fold cross-validation on observed data.

    For SemiDDE, ``D`` is required and ``dimension_method`` must be ``"fixed"``.
    For CopDDE, ``dimension_method`` is ``"csp"`` by default and may be set to
    ``"fixed"``. ``candidate > fixed_hp > package default`` determines the
    hyperparameters used for each fit.
    """
    Y = np.asarray(Y)
    if D is None:
        raise ValueError("D is required.")
    D = int(D)
    chosen = model.lower()
    if chosen in {"semidde"}:
        chosen = "semi"
    if chosen in {"copdde"}:
        chosen = "copula"
    if chosen not in {"semi", "copula"}:
        raise ValueError("model must be 'semi' or 'copula'.")
    criterion = criterion.lower()
    if criterion not in {"bic", "ebic", "loglik", "pmse"}:
        raise ValueError("criterion must be 'bic', 'ebic', 'loglik', or 'pmse'.")
    if criterion == "pmse" and chosen != "copula":
        raise ValueError("criterion='pmse' is only available for model='copula'.")
    method = _resolve_dimension_method(chosen, dimension_method)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    fixed = _copy_candidate(fixed_hp or {})
    base = dict(base_fit_kwargs or {})
    _reject_embedded_dimension_method(candidates=candidates, fixed_hp=fixed, base_fit_kwargs=base)
    if candidates is None:
        if chosen == "semi":
            candidates = auto_grid(Y.shape[0], D)
        elif method == "fixed":
            candidates = auto_grid_copula_fixed(Y.shape[0], D)
        else:
            candidates = auto_grid_copula_csp(Y.shape[0], D, strategy="coordinate")
    if chosen == "copula":
        _validate_copula_path_hyperparameters(
            dimension_method=method,
            candidates=candidates,
            fixed_hp=fixed,
            base_fit_kwargs=base,
        )

    candidate_list = [] if _is_coordinate_plan(candidates) else _as_candidate_list(candidates)
    setup_diagnostics: Dict[str, Any] = {}
    if chosen == "semi":
        defaults, setup_diagnostics = _resolve_semidde_defaults(
            Y,
            D,
            fixed,
            k_estimator_kwargs=k_estimator_kwargs,
        )
    else:
        candidate_has_all_K = (not _is_coordinate_plan(candidates)) and all("K_list" in candidate for candidate in candidate_list)
        defaults, setup_diagnostics = _resolve_copula_defaults(
            Y,
            D,
            fixed,
            estimate_K=not candidate_has_all_K,
        )
        if criterion == "pmse":
            return _select_copula_pmse(
                Y,
                D=D,
                candidates=candidates,
                fixed=fixed,
                base=base,
                defaults=defaults,
                setup_diagnostics=setup_diagnostics,
                dimension_method=method,
                refit=refit,
                scorer=scorer,
                n_repeats=n_repeats,
                random_state=random_state,
                pmse_n_estimators=pmse_n_estimators,
            )
    score_fn = scorer or (lambda holdout, fitted: normal_mixture_loglik_score(holdout, fitted, model=chosen))
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(Y))
    cv_table: List[Dict[str, Any]] = []
    fold_scores: List[List[float]] = []
    full_fit_results: List[Optional[DDEResult]] = []
    evaluated_candidates: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []

    def evaluate_candidate(c_idx: int, candidate: Mapping[str, Any]) -> tuple[Dict[str, Any], List[float], Optional[DDEResult], Dict[str, Any]]:
        candidate_started = time.perf_counter()
        fit_candidate, meta = _split_fit_kwargs(candidate)
        hp = dict(defaults)
        hp.update(base)
        hp.update(fixed)
        hp.update(fit_candidate)
        if chosen == "copula":
            hp["dimension_method"] = method
            hp = _with_copula_method_defaults(hp, D)
        K_list = hp.get("K_list")
        if K_list is None:
            raise ValueError("Each candidate, fixed_hp, or base_fit_kwargs must provide K_list.")
        K_list = list(validate_K_list_identifiability(K_list, D=D, J=Y.shape[1]))
        hp["K_list"] = K_list

        full_result = None
        if criterion == "ebic":
            full_kwargs = dict(hp)
            init_seconds = 0.0
            if chosen == "semi":
                factory = init_factory or _semi_data_init
                init_started = time.perf_counter()
                full_kwargs.update(factory(Y, full_kwargs["K_list"], {"epsilon": full_kwargs.get("epsilon", 1e-4)}))
                init_seconds = time.perf_counter() - init_started
            fit_started = time.perf_counter()
            full_result = fit(Y, model=chosen, **full_kwargs)
            fit_seconds = time.perf_counter() - fit_started
            score_started = time.perf_counter()
            avg_ll = float(score_fn(Y, full_result))
            score_seconds = time.perf_counter() - score_started
            df, capacity = _model_df_and_capacity(full_result, tau=float(full_kwargs.get("tau", 0.0)))
            scores = [
                _score_from_loglik(
                    avg_ll,
                    criterion="ebic",
                    df=df,
                    capacity=capacity,
                    n_model=Y.shape[0],
                    n_score=Y.shape[0],
                    ebic_gamma=ebic_gamma,
                )
            ]
            timing_rows.append(
                {
                    "candidate_index": c_idx,
                    "fold_index": None,
                    "n_train": int(Y.shape[0]),
                    "n_holdout": int(Y.shape[0]),
                    "init_seconds": float(init_seconds),
                    "fit_seconds": float(fit_seconds),
                    "score_seconds": float(score_seconds),
                    "total_seconds": float(init_seconds + fit_seconds + score_seconds),
                }
            )
        else:
            scores = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                train_Y = Y[train_idx]
                holdout_Y = Y[test_idx]
                kwargs = dict(hp)
                kwargs.setdefault("K_list", K_list)
                init_seconds = 0.0
                if chosen == "semi":
                    factory = init_factory or _semi_data_init
                    init_started = time.perf_counter()
                    kwargs.update(factory(train_Y, kwargs["K_list"], {"epsilon": kwargs.get("epsilon", 1e-4)}))
                    init_seconds = time.perf_counter() - init_started
                fit_started = time.perf_counter()
                fitted = fit(train_Y, model=chosen, **kwargs)
                fit_seconds = time.perf_counter() - fit_started
                score_started = time.perf_counter()
                avg_ll = float(score_fn(holdout_Y, fitted))
                score_seconds = time.perf_counter() - score_started
                df, capacity = _model_df_and_capacity(fitted, tau=float(kwargs.get("tau", 0.0)))
                scores.append(
                    _score_from_loglik(
                        avg_ll,
                        criterion=criterion,
                        df=df,
                        capacity=capacity,
                        n_model=len(train_idx),
                        n_score=len(test_idx),
                        ebic_gamma=ebic_gamma,
                    )
                )
                timing_rows.append(
                    {
                        "candidate_index": c_idx,
                        "fold_index": int(fold_idx),
                        "n_train": int(len(train_idx)),
                        "n_holdout": int(len(test_idx)),
                        "init_seconds": float(init_seconds),
                        "fit_seconds": float(fit_seconds),
                        "score_seconds": float(score_seconds),
                        "total_seconds": float(init_seconds + fit_seconds + score_seconds),
                    }
                )

        row = {
            "candidate_index": c_idx,
            **meta,
            "K_list": list(K_list),
            "criterion": criterion,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        }
        candidate_timings = [r for r in timing_rows if r["candidate_index"] == c_idx]
        row["elapsed_seconds"] = float(time.perf_counter() - candidate_started)
        row["init_seconds"] = float(sum(r["init_seconds"] for r in candidate_timings))
        row["fit_seconds"] = float(sum(r["fit_seconds"] for r in candidate_timings))
        row["score_seconds"] = float(sum(r["score_seconds"] for r in candidate_timings))
        for key, value in hp.items():
            if key not in row and np.isscalar(value):
                row[key] = value
            elif key in {"pen_list", "t_spike"}:
                row[key] = np.asarray(value, dtype=float).reshape(-1).tolist()
        return row, scores, full_result, fit_candidate

    if _is_coordinate_plan(candidates):
        plan = dict(candidates)
        current = list(map(float, plan["base_t_spike"]))
        values = [list(map(float, vals)) for vals in plan["t_spike_values"]]
        for layer in range(D):
            layer_best_score = -np.inf
            layer_best_value = current[layer]
            for value in values[layer]:
                trial = list(current)
                trial[layer] = float(value)
                candidate = {
                    "temp": float(plan["temp"]),
                    "tau": float(plan["tau"]),
                    "t_spike": trial,
                    "coordinate_layer": layer,
                    "coordinate_value": float(value),
                }
                idx = len(cv_table)
                row, scores, full_result, fit_candidate = evaluate_candidate(idx, candidate)
                cv_table.append(row)
                fold_scores.append(scores)
                full_fit_results.append(full_result)
                evaluated_candidates.append(fit_candidate)
                mean_score = float(row["mean_score"])
                if mean_score > layer_best_score:
                    layer_best_score = mean_score
                    layer_best_value = float(value)
            current[layer] = layer_best_value
    else:
        for c_idx, candidate in enumerate(candidate_list):
            row, scores, full_result, fit_candidate = evaluate_candidate(c_idx, candidate)
            cv_table.append(row)
            fold_scores.append(scores)
            full_fit_results.append(full_result)
            evaluated_candidates.append(fit_candidate)

    best_index = int(np.argmax([row["mean_score"] for row in cv_table]))
    best_candidate = evaluated_candidates[best_index]
    selection = CVSelectionResult(
        best_index=best_index,
        best_candidate=best_candidate,
        cv_table=cv_table,
        fold_scores=fold_scores,
    )

    if not refit:
        return DDEResult(
            model=chosen,
            diagnostics={
                "hyperparameter_selection": selection,
                "cv_table": cv_table,
                "fold_scores": fold_scores,
                "timing_table": timing_rows,
                **setup_diagnostics,
            },
        )

    if criterion == "ebic" and full_fit_results[best_index] is not None:
        final = full_fit_results[best_index]
        final.diagnostics = dict(final.diagnostics)
        final.diagnostics.update(
            {
                "hyperparameter_selection": selection,
                "cv_table": cv_table,
                "fold_scores": fold_scores,
                **setup_diagnostics,
            }
        )
        return final

    final_kwargs = dict(defaults)
    final_kwargs.update(base)
    final_kwargs.update(fixed)
    final_fit_candidate, _ = _split_fit_kwargs(best_candidate)
    final_kwargs.update(final_fit_candidate)
    if chosen == "copula":
        final_kwargs["dimension_method"] = method
        final_kwargs = _with_copula_method_defaults(final_kwargs, D)
    if chosen == "semi":
        factory = init_factory or _semi_data_init
        final_kwargs.update(factory(Y, final_kwargs["K_list"], {"epsilon": final_kwargs.get("epsilon", 1e-4)}))
    final = fit(Y, model=chosen, **final_kwargs)
    final.diagnostics = dict(final.diagnostics)
    final.diagnostics.update(
        {
            "hyperparameter_selection": selection,
            "cv_table": cv_table,
            "fold_scores": fold_scores,
            "timing_table": timing_rows,
            **setup_diagnostics,
        }
    )
    return final


__all__ = [
    "CVSelectionResult",
    "auto_grid",
    "auto_grid_copula_csp",
    "auto_grid_copula_fixed",
    "copula_predictive_pmse_score",
    "copula_tau_default",
    "estimate_K_list_copula",
    "estimate_K_list_semidde",
    "normal_mixture_loglik_score",
    "select_hyperparameters",
    "simulate_copula_predictive",
]
