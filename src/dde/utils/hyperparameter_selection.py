"""Hyperparameter-selection helper functions for SemiDDE and CopDDE."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.special import gammaln, logsumexp
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


def _layer_choices(values: Sequence[Any], D: int, name: str) -> List[List[float]]:
    """Accept common choices or one independent candidate sequence per layer."""
    raw = list(values)
    if not raw:
        raise ValueError(f"{name} must not be empty.")
    if isinstance(raw[0], (list, tuple, np.ndarray)):
        if len(raw) != D:
            raise ValueError(f"{name} must contain one choice sequence per layer.")
        out = [list(map(float, layer)) for layer in raw]
    else:
        out = [list(map(float, raw)) for _ in range(D)]
    if any(not layer or not np.all(np.isfinite(layer)) for layer in out):
        raise ValueError(f"{name} needs finite choices in every layer.")
    return out


def auto_grid_copula_fixed(
    N: int,
    D: int,
    *,
    lambda_exponents: Sequence[Any] = (1 / 8, 2 / 8, 3 / 8),
    temp_values: Sequence[float] = (0.7, 0.9),
    tau_values: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Cross the fixed-path penalty choices independently across layers."""
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")
    exponent_choices = _layer_choices(lambda_exponents, D, "lambda_exponents")
    tau_choices = [copula_tau_default(N)] if tau_values is None else list(map(float, tau_values))
    temp_choices = list(map(float, temp_values))
    if not tau_choices or not temp_choices or any(x <= 0 or not np.isfinite(x) for x in [*tau_choices, *temp_choices]):
        raise ValueError("tau_values and temp_values need positive finite choices.")
    candidates = []
    for temp in temp_choices:
        for tau in tau_choices:
            for lambda_exp in product(*exponent_choices):
                candidates.append(
                    {
                        "temp": float(temp),
                        "pen_list": [float(int(N) ** exponent) for exponent in lambda_exp],
                        "tau": tau,
                        "lambda_exp": list(lambda_exp),
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


# Fixed-width CopDDE temperature when ``temp`` is not given (see
# default_fit_hyperparameters).
COPDDE_FIXED_TEMP = 1.0


def default_fit_hyperparameters(model: str, dimension_method: str, N: int, D: int) -> Dict[str, Any]:
    """Hyperparameters that ``fit`` uses when they are not given.

    ``select_hyperparameters`` uses the same values for the hyperparameters
    it does not tune.  ``model`` is ``"dde"``, ``"semi"`` or ``"copula"``.

    * fixed widths: ``pen_list = N**(1/4)`` in every layer, the penalty of
      the simulation studies; CopDDE also ``tol = 0.005`` (the other
      fixed-width paths already default to it) and ``temp``
      (``COPDDE_FIXED_TEMP``);
    * CSP: ``t_spike`` = (0.03, 0.05) for ``D = 2``, (0.01, 0.03, 0.005)
      for ``D = 3`` (the CopDDE paper's D = 3 setting) and the middle of
      the default grid otherwise; ``temp`` = 0.7 for ``D <= 2`` and 0.9
      otherwise (the CopDDE paper's starting temperatures);
    * ``tau`` = 0.3, and for CopDDE the CopDDE paper's
      ``max(0.3, 3 N**-0.3)``.

    The fixed-path values were compared with EBIC/pMSE-selected ones in
    ``Results/hp_selection_validation_v2/REPORT.md`` (v3 section).
    """
    key = str(model).strip().lower()
    method = str(dimension_method).strip().lower()
    D = int(D)
    N = int(N)
    if key not in {"dde", "semi", "copula"} or method not in {"fixed", "csp"} or D < 1:
        raise ValueError("model must be 'dde', 'semi' or 'copula', dimension_method 'fixed' or 'csp'.")
    out: Dict[str, Any] = {"tau": copula_tau_default(N) if key == "copula" else 0.3}
    if method == "fixed":
        out["pen_list"] = [float(N ** (2 / 8))] * D
        if key == "copula":
            out["tol"] = 0.005
            out["temp"] = COPDDE_FIXED_TEMP
    else:
        out["t_spike"] = _default_csp_t_spike_base(D, _default_csp_t_spike_values(D))
        out["temp"] = 0.7 if D <= 2 else 0.9
    return out


def auto_grid_copula_csp(
    N: int,
    D: int,
    *,
    strategy: str = "full",
    t_spike_values: Optional[Sequence[Sequence[float]]] = None,
    base_t_spike: Optional[Sequence[float]] = None,
    temp_values: Optional[Sequence[float]] = None,
    tau_values: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Return the complete Cartesian grid of layer-specific CSP rates."""
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")
    values = [list(map(float, vals)) for vals in (t_spike_values if t_spike_values is not None else _default_csp_t_spike_values(D))]
    if len(values) != D:
        raise ValueError("t_spike_values must contain one sequence per layer.")
    if any(not layer or not np.all(np.isfinite(layer)) or any(x <= 0 for x in layer) for layer in values):
        raise ValueError("Each t_spike layer needs positive finite candidates.")
    tau_choices = [copula_tau_default(N)] if tau_values is None else list(map(float, tau_values))
    temp_choices = [0.7 if D <= 2 else 0.9] if temp_values is None else list(map(float, temp_values))
    if not tau_choices or not temp_choices or any(x <= 0 or not np.isfinite(x) for x in [*tau_choices, *temp_choices]):
        raise ValueError("tau_values and temp_values need positive finite choices.")
    if strategy.lower() != "full":
        raise ValueError("Only strategy='full' is supported; all combinations are evaluated.")
    if base_t_spike is not None:
        raise ValueError("base_t_spike is incompatible with a full Cartesian grid.")
    return [
        {"temp": temp, "tau": tau, "t_spike": list(vals)}
        for temp, tau in product(temp_choices, tau_choices)
        for vals in product(*values)
    ]


def auto_grid(
    N: int,
    D: int,
    *,
    layer_scales: Optional[Sequence[float]] = None,
    lambda_exponents: Sequence[Any] = (2 / 8, 3 / 8),
    tau_exponents: Sequence[float] = (-2 / 8, -3 / 8),
    tau_values: Sequence[float] = (0.3,),
) -> List[Dict[str, Any]]:
    """Cross each layer's penalty choices with every threshold choice."""
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
    exponent_choices = _layer_choices(lambda_exponents, D, "lambda_exponents")
    for lambda_exp in product(*exponent_choices):
        pen_list = [float(N ** float(exponent) * layer_scales_arr[d]) for d, exponent in enumerate(lambda_exp)]
        for tau, tau_exp in tau_grid:
            candidates.append(
                {
                    "pen_list": pen_list,
                    "tau": tau,
                    "lambda_exp": list(map(float, lambda_exp)),
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
    max_enum_bits: int = 12,
    mc_samples: int = 4096,
    pattern_chunk_size: int = 1024,
) -> float:
    """Average latent-normal mixture log-likelihood on normal-score observations."""
    chosen = (model or result.model).lower()
    observed_mask = np.isfinite(np.asarray(Y, dtype=float))
    X = _normal_score_matrix(np.asarray(Y), "copula" if chosen in {"copula", "copdde"} else "semi")
    return parametric_mixture_loglik_score(
        X, result, family="normal", max_enum_bits=max_enum_bits,
        mc_samples=mc_samples, pattern_chunk_size=pattern_chunk_size,
        observed_mask=observed_mask,
    )


def parametric_mixture_loglik_score(
    Y: np.ndarray,
    result: DDEResult,
    *,
    family: str,
    max_enum_bits: int = 12,
    mc_samples: int = 4096,
    pattern_chunk_size: int = 1024,
    random_state: Optional[int] = 0,
    observed_mask: Optional[np.ndarray] = None,
) -> float:
    """Observed-data mixture score for Normal, Bernoulli, or Poisson DDE.

    Enumerate shallow binary states when feasible. Above ``max_enum_bits``,
    sample latent states from the fitted hierarchy so scoring stays bounded.
    """
    if result.prop is None or result.B_list is None:
        raise ValueError("result must contain prop and B_list to be scored.")
    X = np.asarray(Y, dtype=float)
    if X.ndim != 2:
        raise ValueError("Y must be a two-dimensional matrix.")
    mask = np.isfinite(X) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
    if mask.shape != X.shape:
        raise ValueError("observed_mask must have the same shape as Y.")
    X = np.where(mask, X, 0.0)
    fully_observed = bool(np.all(mask))
    B_list = [np.asarray(B, dtype=float) for B in result.B_list]
    K1 = B_list[0].shape[1] - 1
    if K1 <= max_enum_bits:
        patterns = binary(np.arange(1 << K1, dtype=np.int64), K1)
        pmf = np.asarray(compute_marginal_pmf_A1(
            result.prop, B_list, max_enum_bits=max_enum_bits,
            mc_samples=mc_samples,
        ), dtype=float)
        pmf = np.clip(pmf, 1e-300, None)
        pmf /= pmf.sum()
    else:
        if mc_samples < 1:
            raise ValueError("mc_samples must be positive for sampled scoring.")
        patterns = np.asarray(simulate_latent_layers_D(
            int(mc_samples), result.prop, B_list, _as_rng(random_state)
        )[0][0], dtype=float)
        pmf = np.full(patterns.shape[0], 1.0 / patterns.shape[0])
    chunk_size = max(1, int(pattern_chunk_size))
    log_pmf = np.log(pmf)
    family = str(family).lower()
    if family not in {"normal", "bernoulli", "poisson"}:
        raise ValueError("family must be 'normal', 'bernoulli', or 'poisson'.")
    if X.shape[1] != B_list[0].shape[0]:
        raise ValueError("Y and fitted observed-layer widths differ.")
    if family == "normal":
        if result.gamma is None:
            raise ValueError("A Normal fit needs gamma for likelihood scoring.")
        gamma = np.maximum(np.asarray(result.gamma, dtype=float).reshape(-1), 1e-8)
        weights = mask / gamma[None, :]
        logdet = mask @ np.log(2.0 * np.pi * gamma)
        x_quad = np.sum((X * X) * weights, axis=1)
        x_scaled = X * weights
    elif family == "poisson":
        log_factorial = np.sum(gammaln(X + 1.0) * mask, axis=1)
    obs_loglik = np.full(X.shape[0], -np.inf, dtype=float)
    for start in range(0, patterns.shape[0], chunk_size):
        end = min(start + chunk_size, patterns.shape[0])
        eta = np.column_stack([np.ones(end - start), patterns[start:end]]) @ B_list[0].T
        if family == "normal":
            means_quad = (
                np.sum((eta * eta) / gamma[None, :], axis=1)[None, :]
                if fully_observed else weights @ (eta * eta).T
            )
            component_ll = -0.5 * (
                x_quad[:, None] + means_quad - 2.0 * (x_scaled @ eta.T) + logdet[:, None]
            )
        elif family == "bernoulli":
            softplus = np.logaddexp(0.0, eta)
            penalty = np.sum(softplus, axis=1)[None, :] if fully_observed else mask @ softplus.T
            component_ll = X @ eta.T - penalty
        else:
            rate = np.exp(np.clip(eta, -700, 700))
            penalty = np.sum(rate, axis=1)[None, :] if fully_observed else mask @ rate.T
            component_ll = X @ eta.T - penalty - log_factorial[:, None]
        chunk_loglik = logsumexp(log_pmf[start:end][None, :] + component_ll, axis=1)
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


def _copula_pmse_repeat(Y: np.ndarray, synth: np.ndarray, seed: int, n_estimators: int, n_splits: int) -> float:
    X_rf = _rf_ready_matrix(np.vstack([Y, synth]))
    label = np.concatenate([np.ones(Y.shape[0], dtype=int), np.zeros(synth.shape[0], dtype=int)])
    splits = min(int(n_splits), int(np.sum(label == 0)), int(np.sum(label == 1)))
    if splits < 2:
        raise ValueError("pMSE evaluation needs at least two observations in each class.")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators), random_state=seed, n_jobs=1,
        class_weight=None,
    )
    p_hat = cross_val_predict(clf, X_rf, label, cv=cv, method="predict_proba")[:, 1]
    return float(np.mean((p_hat - 0.5) ** 2))


def copula_predictive_pmse_score(
    Y: np.ndarray,
    result: DDEResult,
    *,
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    n_estimators: int = 100,
    n_splits: int = 5,
    n_jobs: int = 1,
) -> tuple[float, List[float]]:
    """Compute pMSE, optionally distributing independent repeats across jobs."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be at least 1.")
    Y = np.asarray(Y, dtype=float)
    rng = _as_rng(random_state)
    seeds = [int(rng.integers(0, np.iinfo(np.int32).max)) for _ in range(int(n_repeats))]
    synths = [simulate_copula_predictive(Y, result, n_samples=Y.shape[0], random_state=seed) for seed in seeds]
    if int(n_jobs) == 1:
        pmse_values = [
            _copula_pmse_repeat(Y, synth, seed, n_estimators, n_splits)
            for synth, seed in zip(synths, seeds)
        ]
    else:
        pmse_values = Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(_copula_pmse_repeat)(Y, synth, seed, n_estimators, n_splits)
            for synth, seed in zip(synths, seeds)
        )
    return float(np.mean(pmse_values)), pmse_values


__all__ = [
    "auto_grid",
    "auto_grid_copula_csp",
    "auto_grid_copula_fixed",
    "copula_predictive_pmse_score",
    "copula_tau_default",
    "default_fit_hyperparameters",
    "estimate_K_list_copula",
    "estimate_K_list_semidde",
    "normal_mixture_loglik_score",
    "parametric_mixture_loglik_score",
    "simulate_copula_predictive",
]
