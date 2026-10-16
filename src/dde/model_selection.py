"""Full-grid hyperparameter selection for DDE, SemiDDE, and CopDDE."""

from __future__ import annotations

import time
from itertools import product
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy.special import gammaln
from sklearn.model_selection import KFold

from .api import fit
from .algorithms.normal_init_semi import Normal_init_semi, rank_gaussianize_columns
from .results import DDEResult
from .utils import validate_K_list_identifiability
from .utils.hyperparameter_selection import (
    _default_csp_t_spike_base,
    _default_csp_t_spike_values,
    auto_grid,
    auto_grid_copula_csp,
    auto_grid_copula_fixed,
    copula_predictive_pmse_score,
    copula_tau_default,
    estimate_K_list_copula,
    estimate_K_list_semidde,
    normal_mixture_loglik_score,
    parametric_mixture_loglik_score,
    simulate_copula_predictive,
)


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


def _with_iteration_alias(hp: Dict[str, Any], chosen: str, *explicit: Mapping[str, Any]) -> Dict[str, Any]:
    if "max_iter" in hp:
        legacy = {"dde": "max_it", "semi": "max_inner_iter", "copula": "it"}[chosen]
        if not any(legacy in source for source in explicit):
            hp.pop(legacy, None)
    return hp


def _as_candidate_list(candidates: Optional[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    if candidates is None:
        return [{}]
    if isinstance(candidates, Mapping):
        if candidates.get("strategy") == "coordinate":
            raise ValueError("Coordinate search is unsupported; provide a full Cartesian grid.")
        axes = {}
        for key, value in candidates.items():
            if isinstance(value, Mapping) and set(value) == {"per_layer"}:
                axes[key] = [list(v) for v in product(*value["per_layer"])]
            elif isinstance(value, Mapping):
                raise ValueError(f"Candidate axis {key!r} must use a 'per_layer' mapping.")
            elif key in {"K_list", "pen_list", "t_spike"}:
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) and isinstance(value[0], (list, tuple, np.ndarray)):
                    axes[key] = [list(v) for v in value]
                else:
                    axes[key] = [value]
            elif isinstance(value, (list, tuple, np.ndarray)):
                axes[key] = list(value)
            else:
                axes[key] = [value]
            if not axes[key]:
                raise ValueError(f"Candidate axis {key!r} is empty.")
        return [_copy_candidate(dict(zip(axes, values))) for values in product(*axes.values())]
    candidate_list = [_copy_candidate(c) for c in candidates]
    if not candidate_list:
        raise ValueError("candidates must contain at least one candidate.")
    return candidate_list



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



def _model_df_and_capacity(result: DDEResult, tau: float = 0.0) -> tuple[int, int, int]:
    if result.prop is None or result.B_list is None:
        raise ValueError("result must contain prop and B_list.")
    df = int(np.asarray(result.prop).size + (0 if result.gamma is None else np.asarray(result.gamma).size))
    capacity = 0
    edges = 0
    threshold = float(tau)
    for B in result.B_list:
        B_arr = np.asarray(B, dtype=float)
        df += B_arr.shape[0]
        capacity += B_arr[:, 1:].size
        slopes = B_arr[:, 1:]
        active = int(np.sum(np.abs(slopes) > threshold))
        df += active
        edges += active
    return int(df), int(capacity), int(edges)


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
    edges: int,
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
        penalty = df * np.log(n_model) + 2.0 * ebic_gamma * _log_comb(capacity, edges)
        return float(avg_loglik - 0.5 * penalty / max(n_score, 1))
    raise ValueError("criterion must be 'bic', 'ebic', or 'loglik'.")


def _resolve_semidde_defaults(
    Y: np.ndarray,
    D: int,
    fixed_hp: Mapping[str, Any],
    *,
    k_estimator_kwargs: Optional[Mapping[str, Any]] = None,
    estimate_K: bool = True,
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
    elif estimate_K:
        K_list, K_details = estimate_K_list_semidde(
            Y,
            D,
            return_details=True,
            **dict(k_estimator_kwargs or {}),
        )
        defaults["K_list"] = K_list
        diagnostics["K_list_source"] = "estimated"
        diagnostics["K_list_estimation"] = K_details
    if "K_list" in defaults:
        defaults["tol"] = float(defaults["K_list"][-1]) / 2.0
    defaults["pen_list"] = [float(N ** (2 / 8))] * int(D)
    defaults["tau"] = 0.3
    return defaults, diagnostics


def _resolve_dde_defaults(
    Y: np.ndarray,
    D: int,
    fixed_hp: Mapping[str, Any],
    *,
    dimension_method: str,
    k_estimator_kwargs: Optional[Mapping[str, Any]] = None,
    estimate_K: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    defaults: Dict[str, Any] = {
        "C": 1, "max_it": 50, "tol": 0.005, "initialize": "data", "verbose": False,
        "tau": 0.3,
    }
    diagnostics: Dict[str, Any] = {}
    if dimension_method == "fixed":
        if "K_list" in fixed_hp:
            K_list = list(validate_K_list_identifiability(fixed_hp["K_list"], D=D, J=Y.shape[1]))
            diagnostics["K_list_source"] = "fixed_hp"
        elif estimate_K:
            K_list, details = estimate_K_list_semidde(
                Y, D, return_details=True, **dict(k_estimator_kwargs or {})
            )
            diagnostics.update(K_list_source="estimated", K_list_estimation=details)
        else:
            K_list = None
        if K_list is not None:
            defaults["K_list"] = K_list
        defaults["pen_list"] = [float(Y.shape[0] ** (2 / 8))] * D
    return defaults, diagnostics


def _resolve_semidde_csp_defaults(Y: np.ndarray, D: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    return {
        "C": 1, "max_outer_iter": 10, "max_inner_iter": 50,
        "max_enum_bits": 12, "mc_samples": 50_000, "g_update_rho": 1.0,
        "initialize": "data", "verbose": False, "tol": 0.005,
        "tau": 0.3, "temp": 0.7 if D <= 2 else 0.9,
    }, {"K_list_source": "maximal_identifiable_widths"}


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
    default = "csp" if model == "copula" else "fixed"
    method = default if dimension_method is None else str(dimension_method).lower()
    if method not in {"fixed", "csp"}:
        raise ValueError("dimension_method must be 'fixed' or 'csp'.")
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
    n_splits: int,
    n_jobs: int,
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
        n_splits=n_splits,
        n_jobs=n_jobs,
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
    n_splits: int,
    n_jobs: int,
) -> DDEResult:
    cv_table: List[Dict[str, Any]] = []
    fold_scores: List[List[float]] = []
    timing_rows: List[Dict[str, Any]] = []
    evaluation_cache: Dict[tuple, tuple[float, List[float], Dict[str, float]]] = {}
    best_result: Optional[DDEResult] = None
    best_score = -np.inf

    def complete_hp(candidate: Mapping[str, Any]) -> Dict[str, Any]:
        fit_candidate, _ = _split_fit_kwargs(candidate)
        hp = dict(defaults)
        hp.update(base)
        hp.update(fixed)
        hp.update(fit_candidate)
        hp = _with_iteration_alias(hp, "copula", base, fixed, fit_candidate)
        if random_state is not None:
            hp.setdefault("random_state", random_state)
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
            mean_score, scores, original_timing = evaluation_cache[key]
            timing = {
                "fit_seconds": 0.0,
                "score_seconds": 0.0,
                "total_seconds": 0.0,
                "cached_fit_seconds": float(original_timing["fit_seconds"]),
                "cached_score_seconds": float(original_timing["score_seconds"]),
                "cached_total_seconds": float(original_timing["total_seconds"]),
            }
            return None, mean_score, list(scores), timing, True
        result, mean_score, scores, timing = _fit_copula_pmse_candidate(
            Y,
            hp,
            n_repeats=n_repeats,
            random_state=random_state,
            pmse_n_estimators=pmse_n_estimators,
            n_splits=n_splits,
            n_jobs=n_jobs,
            scorer=scorer,
        )
        evaluation_cache[key] = (mean_score, list(scores), dict(timing))
        return result, mean_score, scores, timing, False

    candidate_list = _as_candidate_list(candidates)
    for idx, candidate in enumerate(candidate_list):
        hp = complete_hp(candidate)
        _, meta = _split_fit_kwargs(candidate)
        result, mean_score, scores, timing, from_cache = evaluate_pmse_candidate(idx, hp)
        if not np.isfinite(mean_score):
            raise ValueError(f"Candidate {idx} produced a non-finite pMSE score.")
        if mean_score > best_score:
            best_score = mean_score
            best_result = result
        row = _row_from_hp(
            candidate_index=idx, meta=meta, K_list=hp["K_list"],
            criterion="pmse", scores=scores, hp=hp,
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
        timing_rows.append({
            "candidate_index": int(idx), "fold_index": None,
            "n_train": int(Y.shape[0]), "n_holdout": int(Y.shape[0]),
            "fit_seconds": float(timing["fit_seconds"]),
            "score_seconds": float(timing["score_seconds"]),
            "total_seconds": float(timing["total_seconds"]),
            "from_cache": bool(from_cache),
        })
    best_index = int(np.argmax([row["mean_score"] for row in cv_table]))
    best_candidate = _copy_candidate(candidate_list[best_index])

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
    diagnostics["copula_grid_strategy"] = "full"
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
    method: Optional[str] = None,
    model: Optional[str] = None,
    family: Optional[str] = None,
    variable_types: Optional[Sequence[str]] = None,
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
    n_jobs: int = 1,
) -> DDEResult:
    """Evaluate every candidate and select the best DDE-family configuration.

    ``candidates`` may be explicit rows or a mapping of candidate axes. For
    layer-specific vectors, use ``{"pen_list": {"per_layer": [[...], ...]}}``
    or the same form with ``t_spike``. Every Cartesian combination is fitted.
    Candidate values override ``fixed_hp``, then ``base_fit_kwargs`` and defaults.
    """
    Y = np.asarray(Y)
    if Y.ndim != 2 or min(Y.shape) < 1:
        raise ValueError("Y must be a non-empty two-dimensional matrix.")
    if D is None:
        raise ValueError("D is required.")
    D = int(D)
    if D < 1:
        raise ValueError("D must be positive.")
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive.")
    from .api import _canonical_method, _infer_model
    if method is None and model is None:
        raise ValueError("method or model is required.")
    chosen = _canonical_method(method if method is not None else model)
    if model is not None and method is not None and chosen != _canonical_method(model):
        raise ValueError("method and model specify different DDE formulations.")
    if chosen == "auto":
        chosen = _infer_model(Y, variable_types)
    if chosen == "dde":
        if family is None:
            raise ValueError("family is required when method='DDE'.")
        from .algorithms.parametric_families import resolve_parametric_family
        family = resolve_parametric_family(family).name
    elif family is not None:
        raise ValueError("family is only used with method='DDE'.")
    criterion = criterion.lower()
    if criterion not in {"bic", "ebic", "loglik", "pmse"}:
        raise ValueError("criterion must be 'bic', 'ebic', 'loglik', or 'pmse'.")
    if criterion == "pmse" and chosen != "copula":
        raise ValueError("criterion='pmse' is only available for method='CopDDE'.")
    method = _resolve_dimension_method(chosen, dimension_method)

    if criterion != "ebic" and n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    fixed = _copy_candidate(fixed_hp or {})
    base = dict(base_fit_kwargs or {})
    candidate_list = None if candidates is None else _as_candidate_list(candidates)
    _reject_embedded_dimension_method(candidates=candidate_list, fixed_hp=fixed, base_fit_kwargs=base)
    if method == "csp" and ("K_list" in fixed or "K_list" in base):
        if chosen != "copula":
            raise ValueError("K_list must be omitted for DDE and SemiDDE CSP fits.")
    if candidates is None:
        if method == "fixed" and chosen in {"semi", "dde"}:
            candidates = auto_grid(Y.shape[0], D)
        elif method == "fixed":
            candidates = auto_grid_copula_fixed(Y.shape[0], D)
        else:
            # DDE and SemiDDE CSP: the family designs use temperatures between
            # 0.7 and 0.9, so the annealing temperature is a grid axis rather
            # than the CopDDE constant (0.7 for D<=2, 0.9 for D>=3).
            candidates = auto_grid_copula_csp(
                Y.shape[0], D,
                tau_values=None if chosen == "copula" else [0.3],
                temp_values=None if chosen == "copula" else [0.7, 0.9],
            )
        candidate_list = _as_candidate_list(candidates)
    if chosen == "copula":
        _validate_copula_path_hyperparameters(
            dimension_method=method,
            candidates=candidate_list,
            fixed_hp=fixed,
            base_fit_kwargs=base,
        )
    elif method == "csp":
        for candidate in candidate_list:
            if "pen_list" in candidate or "K_list" in candidate:
                raise ValueError("pen_list and K_list are only valid for fixed-width DDE and SemiDDE.")
        if "pen_list" in fixed or "pen_list" in base:
            raise ValueError("pen_list is only valid for fixed-width DDE and SemiDDE.")
    elif any("t_spike" in c or "temp" in c for c in [fixed, base, *candidate_list]):
        raise ValueError("t_spike and temp are only valid for CSP DDE and SemiDDE.")
    setup_diagnostics: Dict[str, Any] = {}
    known_hp = {**base, **fixed}
    candidate_has_all_K = all("K_list" in candidate for candidate in candidate_list)
    if chosen == "semi":
        if method == "fixed":
            defaults, setup_diagnostics = _resolve_semidde_defaults(
                Y, D, known_hp, k_estimator_kwargs=k_estimator_kwargs,
                estimate_K=not candidate_has_all_K,
            )
        else:
            defaults, setup_diagnostics = _resolve_semidde_csp_defaults(Y, D)
    elif chosen == "dde":
        defaults, setup_diagnostics = _resolve_dde_defaults(
            Y, D, known_hp, dimension_method=method,
            k_estimator_kwargs=k_estimator_kwargs,
            estimate_K=not candidate_has_all_K,
        )
    else:
        defaults, setup_diagnostics = _resolve_copula_defaults(
            Y,
            D,
            known_hp,
            estimate_K=not candidate_has_all_K,
        )
        if criterion == "pmse":
            return _select_copula_pmse(
                Y,
                D=D,
                candidates=candidate_list,
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
                n_splits=n_splits,
                n_jobs=n_jobs,
            )
    if scorer is not None:
        score_fn = scorer
    elif chosen == "dde":
        score_fn = lambda holdout, fitted: parametric_mixture_loglik_score(
            holdout, fitted, family=family,
        )
    else:
        score_fn = lambda holdout, fitted: normal_mixture_loglik_score(
            holdout, fitted, model=chosen,
        )
    folds = (
        [] if criterion == "ebic" else
        list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(Y))
    )
    cv_table: List[Dict[str, Any]] = []
    fold_scores: List[List[float]] = []
    best_full_result: Optional[DDEResult] = None
    best_full_score = -np.inf
    evaluated_candidates: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []

    def evaluate_candidate(c_idx: int, candidate: Mapping[str, Any]) -> tuple[Dict[str, Any], List[float], Optional[DDEResult], Dict[str, Any]]:
        candidate_started = time.perf_counter()
        fit_candidate, meta = _split_fit_kwargs(candidate)
        hp = dict(defaults)
        hp.update(base)
        hp.update(fixed)
        hp.update(fit_candidate)
        hp = _with_iteration_alias(hp, chosen, base, fixed, fit_candidate)
        if random_state is not None and (chosen == "copula" or method == "csp"):
            hp.setdefault("random_state", random_state)
        hp["dimension_method"] = method
        if chosen == "copula":
            hp = _with_copula_method_defaults(hp, D)
        K_list = hp.get("K_list")
        if chosen != "copula" and method == "csp":
            if K_list is not None:
                raise ValueError("K_list must be omitted for DDE and SemiDDE CSP fits.")
            K_list = estimate_K_list_copula(Y, D)
        else:
            if K_list is None:
                raise ValueError("Each candidate, fixed_hp, or base_fit_kwargs must provide K_list.")
            K_list = list(validate_K_list_identifiability(K_list, D=D, J=Y.shape[1]))
            hp["K_list"] = K_list
            if chosen == "semi" and method == "fixed" and "tol" not in fit_candidate and "tol" not in fixed and "tol" not in base:
                hp["tol"] = float(K_list[-1]) / 2.0

        full_result = None
        if criterion == "ebic":
            full_kwargs = dict(hp)
            init_seconds = 0.0
            if chosen == "semi" and method == "fixed":
                factory = init_factory or _semi_data_init
                init_started = time.perf_counter()
                full_kwargs.update(factory(Y, full_kwargs["K_list"], {"epsilon": full_kwargs.get("epsilon", 1e-4)}))
                init_seconds = time.perf_counter() - init_started
            fit_started = time.perf_counter()
            full_result = fit(Y, method=chosen, D=D, family=family, **full_kwargs)
            fit_seconds = time.perf_counter() - fit_started
            score_started = time.perf_counter()
            avg_ll = float(score_fn(Y, full_result))
            score_seconds = time.perf_counter() - score_started
            df, capacity, edges = _model_df_and_capacity(full_result, tau=float(full_kwargs.get("tau", 0.0)))
            scores = [
                _score_from_loglik(
                    avg_ll,
                    criterion="ebic",
                    df=df,
                    capacity=capacity,
                    edges=edges,
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
                init_seconds = 0.0
                if chosen == "semi" and method == "fixed":
                    factory = init_factory or _semi_data_init
                    init_started = time.perf_counter()
                    kwargs.update(factory(train_Y, kwargs["K_list"], {"epsilon": kwargs.get("epsilon", 1e-4)}))
                    init_seconds = time.perf_counter() - init_started
                fit_started = time.perf_counter()
                fitted = fit(train_Y, method=chosen, D=D, family=family, **kwargs)
                fit_seconds = time.perf_counter() - fit_started
                score_started = time.perf_counter()
                avg_ll = float(score_fn(holdout_Y, fitted))
                score_seconds = time.perf_counter() - score_started
                df, capacity, edges = _model_df_and_capacity(fitted, tau=float(kwargs.get("tau", 0.0)))
                scores.append(
                    _score_from_loglik(
                        avg_ll,
                        criterion=criterion,
                        df=df,
                        capacity=capacity,
                        edges=edges,
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
        if full_result is not None and full_result.K_list is not None:
            # Fitted widths of the full-data candidate fit (CSP diagnostics).
            row["fitted_K"] = [int(value) for value in full_result.K_list]
        if not np.isfinite(row["mean_score"]):
            raise ValueError(f"Candidate {c_idx} produced a non-finite {criterion} score.")
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

    for c_idx, candidate in enumerate(candidate_list):
        row, scores, full_result, fit_candidate = evaluate_candidate(c_idx, candidate)
        cv_table.append(row)
        fold_scores.append(scores)
        if full_result is not None and row["mean_score"] > best_full_score:
            best_full_result = full_result
            best_full_score = row["mean_score"]
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

    if criterion == "ebic" and best_full_result is not None:
        final = best_full_result
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

    final_kwargs = dict(defaults)
    final_kwargs.update(base)
    final_kwargs.update(fixed)
    final_fit_candidate, _ = _split_fit_kwargs(best_candidate)
    final_kwargs.update(final_fit_candidate)
    final_kwargs = _with_iteration_alias(final_kwargs, chosen, base, fixed, final_fit_candidate)
    if random_state is not None and (chosen == "copula" or method == "csp"):
        final_kwargs.setdefault("random_state", random_state)
    if chosen == "semi" and method == "fixed" and "tol" not in final_fit_candidate and "tol" not in fixed and "tol" not in base:
        final_kwargs["tol"] = float(final_kwargs["K_list"][-1]) / 2.0
    if chosen == "copula":
        final_kwargs["dimension_method"] = method
        final_kwargs = _with_copula_method_defaults(final_kwargs, D)
    else:
        final_kwargs["dimension_method"] = method
    if chosen == "semi" and method == "fixed":
        factory = init_factory or _semi_data_init
        final_kwargs.update(factory(Y, final_kwargs["K_list"], {"epsilon": final_kwargs.get("epsilon", 1e-4)}))
    refit_started = time.perf_counter()
    final = fit(Y, method=chosen, D=D, family=family, **final_kwargs)
    refit_seconds = time.perf_counter() - refit_started
    final.diagnostics = dict(final.diagnostics)
    final.diagnostics.update(
        {
            "hyperparameter_selection": selection,
            "cv_table": cv_table,
            "fold_scores": fold_scores,
            "timing_table": timing_rows,
            "refit_seconds": float(refit_seconds),
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
