"""Full-grid hyperparameter selection for DDE, SemiDDE, and CopDDE."""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy.special import gammaln
from sklearn.model_selection import KFold

from .api import ITERATION_LIMIT_NAMES, MODEL_NAMES, fit
from .results import DDEResult, data_column_names, record_column_names
from .utils import validate_K_list_identifiability
from .utils.hyperparameter_selection import (
    _default_csp_t_spike_base,
    _default_csp_t_spike_values,
    auto_grid,
    auto_grid_copula_csp,
    auto_grid_copula_fixed,
    copula_predictive_pmse_score,
    copula_tau_default,
    default_fit_hyperparameters,
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


def _normalize_iteration_limit(params: Dict[str, Any]) -> Dict[str, Any]:
    """Store any iteration-limit spelling under the manuscript's ``max_it``."""
    present = [(name, params.pop(name)) for name in ITERATION_LIMIT_NAMES if name in params]
    values = {int(value) for _, value in present if value is not None}
    if len(values) > 1:
        names = ", ".join(name for name, _ in present)
        raise ValueError(f"Conflicting iteration limits: {names}.")
    if values:
        params["max_it"] = values.pop()
    return params


def _copy_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    copied = dict(candidate)
    if "K_list" in copied and copied["K_list"] is not None:
        copied["K_list"] = [int(k) for k in copied["K_list"]]
    for key in ("pen_list", "t_spike"):
        # A scalar stands for the same value in every layer (broadcast once D
        # is known, see _broadcast_layer_values).
        if copied.get(key) is not None and not np.isscalar(copied[key]):
            copied[key] = list(copied[key])
    return _normalize_iteration_limit(copied)


def _broadcast_layer_values(params: Dict[str, Any], D: int) -> Dict[str, Any]:
    """Expand scalar ``pen_list`` and ``t_spike`` values to length-``D`` lists."""
    for key in ("pen_list", "t_spike"):
        if params.get(key) is not None and np.isscalar(params[key]):
            params[key] = [float(params[key])] * int(D)
    return params


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
    if isinstance(candidates, Mapping):
        if candidates.get("strategy") == "coordinate":
            raise ValueError("Coordinate search is unsupported; provide a full Cartesian grid.")
        axes = {}
        for key, value in candidates.items():
            if isinstance(value, Mapping) and set(value) == {"per_layer"}:
                axes[key] = [list(v) for v in product(*value["per_layer"])]
            elif isinstance(value, Mapping):
                raise ValueError(f"Candidate axis {key!r} must use a 'per_layer' mapping.")
            elif key in {"pen_list", "t_spike"}:
                # Manuscript Section 5.2: a flat list holds one candidate per
                # value, each used in every layer; nested lists are explicit
                # layer vectors.
                if isinstance(value, (list, tuple, np.ndarray)):
                    values = list(value)
                    if values and isinstance(values[0], (list, tuple, np.ndarray)):
                        axes[key] = [list(v) for v in values]
                    else:
                        axes[key] = values
                else:
                    axes[key] = [value]
            elif key == "K_list":
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
    """Data start of fixed-width SemiDDE candidates: the one ``fit`` uses.

    MARK: this called ``Normal_init_semi`` with its default projection
    binarization, while ``fit`` starts fixed-width SemiDDE from the
    anchor-mixture binarization (``Results/semidde_fixed_recon``), so a
    selected candidate and the same hyperparameters passed to ``fit``
    started from different states.
    """
    from .algorithms.semi import _initialize_semidde

    K_tuple = tuple(int(k) for k in K_list)
    validate_K_list_identifiability(K_tuple, J=Y.shape[1])
    state: Dict[str, Any] = {"verbose": False, "epsilon": context.get("epsilon", 1e-4)}
    _initialize_semidde(np.asarray(Y, dtype=float), K_tuple, state, dimension_method="fixed")
    return {key: state[key] for key in ("X_in", "prop_in", "B_list_in", "gamma_in", "A_list_in")}



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
        "max_it": 50,
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
    # The relative-change tolerance of the other fixed-width fits.  (The
    # MATLAB notebooks' ``tol = K_D / 2`` stopped every inner SAEM right
    # after its burn-in.)
    defaults["tol"] = 0.005
    shared = default_fit_hyperparameters("semi", "fixed", N, int(D))
    defaults["pen_list"] = shared["pen_list"]
    defaults["tau"] = shared["tau"]
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
    # CSP fits keep each kernel's own iteration budget and stopping rule (the
    # copula-backend Bernoulli/Poisson CSP runs 100 iterations without early
    # stop); fixed-width fits use 50 iterations and tol 0.005.
    defaults: Dict[str, Any] = {
        "C": 1, "initialize": "data", "verbose": False, "tau": 0.3,
    }
    diagnostics: Dict[str, Any] = {}
    if dimension_method == "fixed":
        defaults.update(max_it=50, tol=0.005)
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
        defaults["pen_list"] = default_fit_hyperparameters("dde", "fixed", Y.shape[0], D)["pen_list"]
    return defaults, diagnostics


def _resolve_semidde_csp_defaults(Y: np.ndarray, D: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    shared = default_fit_hyperparameters("semi", "csp", Y.shape[0], D)
    return {
        # The copula CSP path updates g after every iteration (no outer loop).
        "C": 1, "max_it": 50,
        "max_enum_bits": 12, "mc_samples": 50_000,
        "initialize": "data", "verbose": False, "tol": 0.005,
        "tau": shared["tau"], "temp": shared["temp"],
    }, {"K_list_source": "maximal_identifiable_widths"}


def _resolve_copula_defaults(
    Y: np.ndarray,
    D: int,
    fixed_hp: Mapping[str, Any],
    *,
    dimension_method: str,
    estimate_K: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    N = Y.shape[0]
    shared = default_fit_hyperparameters("copula", dimension_method, N, int(D))
    defaults: Dict[str, Any] = {
        "C": 1,
        "max_it": 50,
        # MARK: fixed-width candidates ran at temp 0.7 (D <= 2) / 0.9 and
        # tol 1e-3 while fit() runs them at its own defaults, so a selected
        # candidate refitted by fit() was a different fit.  Both now take
        # default_fit_hyperparameters (CSP: the CopDDE starting temperature).
        "temp": shared["temp"],
        "tau": shared["tau"],
        "initialize": "data",
        "verbose": False,
    }
    if dimension_method == "fixed":
        defaults["tol"] = shared["tol"]
        defaults["pen_list"] = shared["pen_list"]
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
        if "t_spike" not in hp_out:
            hp_out["t_spike"] = _default_csp_t_spike_base(D, _default_csp_t_spike_values(D))
    elif method == "fixed":
        hp_out.pop("t_spike", None)
    return hp_out


def _resolve_dimension_method(model: str, dimension_method: Optional[str], *, has_graph: bool = False) -> str:
    if dimension_method is None:
        from .api import _default_dimension_method

        return _default_dimension_method(model, has_graph, stacklevel=4)
    method = str(dimension_method).lower()
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
        return _record_selection(
            DDEResult(model=MODEL_NAMES["copula"], diagnostics=diagnostics),
            dimension_method=dimension_method, criterion="pmse", selection=selection,
        )
    final = best_result
    if final is None:
        raise RuntimeError("No CopDDE pMSE candidate was fitted.")
    final.diagnostics = dict(final.diagnostics)
    final.diagnostics.update(diagnostics)
    return _record_selection(
        final, dimension_method=dimension_method, criterion="pmse", selection=selection,
    )


def _record_selection(
    result: DDEResult,
    *,
    dimension_method: str,
    criterion: str,
    selection: CVSelectionResult,
) -> DDEResult:
    """Record the selected configuration in ``result.raw`` (manuscript Section 5.3)."""
    if isinstance(result.raw, Mapping):
        raw = dict(result.raw)
    else:
        raw = {} if result.raw is None else {"kernel_output": result.raw}
    raw.update(
        selection_method=dimension_method,
        selection_criterion=criterion,
        selected_hyperparameters=dict(selection.best_candidate),
        selected_index=int(selection.best_index),
    )
    result.raw = raw
    return result


def _semi_initial_state(
    Y: np.ndarray,
    kwargs: Mapping[str, Any],
    init_factory: Optional[InitFactory],
) -> Dict[str, Any]:
    """Initial state that fixed-width SemiDDE candidate fits share.

    With a ``G_list`` the fit's own graph-anchored initialization is used
    unless a custom ``init_factory`` is given.
    """
    if init_factory is None and kwargs.get("G_list") is not None:
        return {}
    factory = init_factory or _semi_data_init
    return factory(Y, kwargs["K_list"], {"epsilon": kwargs.get("epsilon", 1e-4)})


def _candidate_global_seed(random_state: int, c_idx: int) -> int:
    """Seed of the global NumPy generator for one candidate's fits."""
    sequence = np.random.SeedSequence([int(random_state) & 0xFFFFFFFFFFFFFFFF, int(c_idx)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _default_score_fn(chosen: str, family: Optional[str]):
    if chosen == "dde":
        return partial(_parametric_score, family=family)
    return partial(_normal_score, model=chosen)


def _parametric_score(holdout, fitted, *, family):
    return parametric_mixture_loglik_score(holdout, fitted, family=family)


def _normal_score(holdout, fitted, *, model):
    return normal_mixture_loglik_score(holdout, fitted, model=model)


def _evaluate_candidate_task(task):
    return _evaluate_candidate(*task)


def _evaluate_candidate(context: Mapping[str, Any], c_idx: int, candidate: Mapping[str, Any]):
    """Fit and score one candidate (all folds, or the full data for EBIC).

    A module-level function of explicit arguments so that a separate process
    can run it.  With a ``random_state`` the legacy global NumPy generator is
    seeded per candidate first: several kernels draw from it, and a seed that
    depends only on (random_state, candidate) makes each candidate's fit
    independent of the candidates evaluated before it, and so identical in
    serial and parallel runs.  Returns the CV-table row, the scores, the
    full-data fit (EBIC only), the fitted candidate values and its timing rows.
    """
    Y = context["Y"]; D = context["D"]; chosen = context["chosen"]; method = context["method"]
    family = context["family"]; criterion = context["criterion"]; defaults = context["defaults"]
    base = context["base"]; fixed = context["fixed"]; random_state = context["random_state"]
    init_factory = context["init_factory"]; ebic_gamma = context["ebic_gamma"]; folds = context["folds"]
    score_fn = context["scorer"] or _default_score_fn(chosen, family)
    timing_rows: List[Dict[str, Any]] = []
    if random_state is not None:
        np.random.seed(_candidate_global_seed(random_state, c_idx))
    candidate_started = time.perf_counter()
    fit_candidate, meta = _split_fit_kwargs(candidate)
    hp = dict(defaults)
    hp.update(base)
    hp.update(fixed)
    hp.update(fit_candidate)
    if random_state is not None and (chosen == "copula" or method == "csp"):
        hp.setdefault("random_state", random_state)
    hp["dimension_method"] = method
    if chosen == "copula":
        hp = _with_copula_method_defaults(hp, D)
    K_list = hp.get("K_list")
    if chosen != "copula" and method == "csp":
        if K_list is None:
            # the maximal identifiable widths the CSP fit starts from
            K_list = estimate_K_list_copula(Y, D)
        else:
            K_list = list(validate_K_list_identifiability(K_list, D=D, J=Y.shape[1]))
            hp["K_list"] = K_list
    else:
        if K_list is None:
            raise ValueError("Each candidate, fixed_hp, or base_fit_kwargs must provide K_list.")
        K_list = list(validate_K_list_identifiability(K_list, D=D, J=Y.shape[1]))
        hp["K_list"] = K_list

    full_result = None
    if criterion == "ebic":
        full_kwargs = dict(hp)
        init_seconds = 0.0
        if chosen == "semi" and method == "fixed":
            init_started = time.perf_counter()
            full_kwargs.update(_semi_initial_state(Y, full_kwargs, init_factory))
            init_seconds = time.perf_counter() - init_started
        fit_started = time.perf_counter()
        full_result = fit(Y, model=chosen, D=D, family=family, **full_kwargs)
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
                init_started = time.perf_counter()
                kwargs.update(_semi_initial_state(train_Y, kwargs, init_factory))
                init_seconds = time.perf_counter() - init_started
            fit_started = time.perf_counter()
            fitted = fit(train_Y, model=chosen, D=D, family=family, **kwargs)
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
    if criterion == "ebic":
        # Standard EBIC (smaller is better): -2 loglik + df log N + 2 gamma log C.
        row["ebic"] = float(-2.0 * Y.shape[0] * row["mean_score"])
    if full_result is not None and full_result.K_list is not None:
        # Fitted widths of the full-data candidate fit (CSP diagnostics).
        row["fitted_K"] = [int(value) for value in full_result.K_list]
    if not np.isfinite(row["mean_score"]):
        raise ValueError(f"Candidate {c_idx} produced a non-finite {criterion} score.")
    candidate_timings = timing_rows
    row["elapsed_seconds"] = float(time.perf_counter() - candidate_started)
    row["init_seconds"] = float(sum(r["init_seconds"] for r in candidate_timings))
    row["fit_seconds"] = float(sum(r["fit_seconds"] for r in candidate_timings))
    row["score_seconds"] = float(sum(r["score_seconds"] for r in candidate_timings))
    for key, value in hp.items():
        if key not in row and np.isscalar(value):
            row[key] = value
        elif key in {"pen_list", "t_spike"}:
            row[key] = np.asarray(value, dtype=float).reshape(-1).tolist()
    return row, scores, full_result, fit_candidate, timing_rows


def select_hyperparameters(
    Y,
    *,
    D: int,
    model: Optional[str] = None,
    family: Optional[str] = None,
    dimension_method: Optional[str] = None,
    candidates: Optional[Iterable[Mapping[str, Any]]] = None,
    criterion: Optional[str] = None,
    method: Optional[str] = None,
    variable_types: Optional[Sequence[str]] = None,
    fixed_hp: Optional[Mapping[str, Any]] = None,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    base_fit_kwargs: Optional[Mapping[str, Any]] = None,
    init_factory: Optional[InitFactory] = None,
    refit: bool = True,
    scorer: Optional[Callable[[np.ndarray, DDEResult], float]] = None,
    ebic_gamma: float = 1.0,
    k_estimator_kwargs: Optional[Mapping[str, Any]] = None,
    n_repeats: int = 25,
    pmse_n_estimators: int = 100,
    n_jobs: int = 1,
    **fit_kwargs: Any,
) -> DDEResult:
    """Evaluate every candidate and select the best DDE-family configuration.

    The interface follows Section 5.2 of the package manuscript: ``model``,
    ``family``, ``D`` and ``dimension_method`` choose the fit as in
    :func:`dde.fit`; ``candidates`` is the tuning grid; every other keyword
    (``K_list``, ``G_list``, ``C``, ``tol``, ``max_it``, ...) is passed to
    :func:`dde.fit` unchanged for every candidate.

    ``candidates`` maps each tuned argument to its candidate values, e.g.
    ``{"pen_list": [.1, .05], "tau": [.1, .3]}`` for fixed widths or
    ``{"t_spike": [.1, .05], "tau": [.2], "temp": [.7, .9]}`` for CSP; every
    Cartesian combination is fitted.  A flat list of ``pen_list`` or
    ``t_spike`` values holds one candidate per value, used in every layer;
    nested lists (``[[.02, .06], [.04, .08]]``) or
    ``{"per_layer": [[...], ...]}`` give layer-specific vectors.  A list of
    explicit candidate rows is also accepted.

    ``criterion`` defaults to ``"ebic"`` for the DDE and SemiDDE (the
    candidate with the smallest EBIC, reported in the ``"ebic"`` column of
    ``diagnostics["cv_table"]``, is returned) and to ``"pmse"`` for the
    CopDDE (the smallest mean propensity-score MSE over ``n_repeats``
    synthetic data sets, each scored by ``n_splits``-fold cross-validated
    random forests).  ``"bic"`` and ``"loglik"`` select by ``n_splits``-fold
    cross-validated likelihood instead.

    The selected fit is returned (refitted to the full data when the
    criterion is cross-validated).  ``raw["selection_method"]`` holds the
    ``dimension_method``, ``raw["selected_hyperparameters"]`` the selected
    candidate, and ``diagnostics["hyperparameter_selection"]`` and
    ``diagnostics["cv_table"]`` the full search.

    Advanced arguments: ``fixed_hp`` and ``base_fit_kwargs`` are older ways
    to pass fixed ``fit`` arguments (a direct keyword may not repeat a
    ``fixed_hp`` entry); candidate values override them.  ``n_jobs > 1``
    evaluates the candidates in that many worker processes (spawned); a
    custom ``scorer`` or ``init_factory`` must then be picklable.  With a
    ``random_state``, the global NumPy generator is seeded per candidate
    from (``random_state``, candidate index) before its fits, so the results
    do not depend on the evaluation order and are identical for every
    ``n_jobs``.  ``method`` is the deprecated spelling of ``model``.
    """
    column_names = data_column_names(Y)
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
    from .api import _infer_model, _resolve_model
    if method is None and model is None:
        raise ValueError("model is required.")
    chosen = _resolve_model(model, method)
    if chosen == "auto":
        chosen = _infer_model(Y, variable_types)
    if chosen == "dde":
        if family is None:
            raise ValueError("family is required when model='DDE'.")
        from .algorithms.parametric_families import resolve_parametric_family
        family = resolve_parametric_family(family).name
    elif family is not None:
        raise ValueError("family is only used with model='DDE'.")
    if criterion is None:
        criterion = "pmse" if chosen == "copula" else "ebic"
    criterion = criterion.lower()
    if criterion not in {"bic", "ebic", "loglik", "pmse"}:
        raise ValueError("criterion must be 'bic', 'ebic', 'loglik', or 'pmse'.")
    if criterion == "pmse" and chosen != "copula":
        raise ValueError("criterion='pmse' is only available for model='CopDDE'.")

    if criterion != "ebic" and n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    fixed = _copy_candidate(fixed_hp or {})
    for key, value in _copy_candidate(fit_kwargs).items():
        if key in fixed:
            raise ValueError(f"{key!r} is given both as an argument and in fixed_hp.")
        fixed[key] = value
    fixed = _broadcast_layer_values(fixed, D)
    base = _broadcast_layer_values(_normalize_iteration_limit(dict(base_fit_kwargs or {})), D)
    method = _resolve_dimension_method(
        chosen, dimension_method,
        has_graph=fixed.get("G_list") is not None or base.get("G_list") is not None,
    )
    candidate_list = None if candidates is None else _as_candidate_list(candidates)
    _reject_embedded_dimension_method(candidates=candidate_list, fixed_hp=fixed, base_fit_kwargs=base)
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
    candidate_list = [_broadcast_layer_values(candidate, D) for candidate in candidate_list]
    if chosen == "copula":
        _validate_copula_path_hyperparameters(
            dimension_method=method,
            candidates=candidate_list,
            fixed_hp=fixed,
            base_fit_kwargs=base,
        )
    elif method == "csp":
        if any("pen_list" in c for c in [fixed, base, *candidate_list]):
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
            dimension_method=method,
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

    context = {
        "Y": Y, "D": D, "chosen": chosen, "method": method, "family": family,
        "criterion": criterion, "defaults": defaults, "base": base, "fixed": fixed,
        "random_state": random_state, "init_factory": init_factory, "scorer": scorer,
        "ebic_gamma": ebic_gamma, "folds": folds,
    }
    tasks = [(context, c_idx, candidate) for c_idx, candidate in enumerate(candidate_list)]
    workers = min(int(n_jobs), len(tasks))
    if workers > 1:
        # Candidates are independent fits; the results are collected in
        # candidate order, so the selection is the serial one.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            outcomes = list(executor.map(_evaluate_candidate_task, tasks))
    else:
        outcomes = map(_evaluate_candidate_task, tasks)

    for row, scores, full_result, fit_candidate, candidate_timing in outcomes:
        cv_table.append(row)
        fold_scores.append(scores)
        timing_rows.extend(candidate_timing)
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
    _record = partial(_record_selection, dimension_method=method, criterion=criterion, selection=selection)

    def record(result):
        return record_column_names(_record(result), column_names)

    if not refit:
        return record(DDEResult(
            model=MODEL_NAMES[chosen],
            diagnostics={
                "hyperparameter_selection": selection,
                "cv_table": cv_table,
                "fold_scores": fold_scores,
                "timing_table": timing_rows,
                **setup_diagnostics,
            },
        ))

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
        return record(final)

    final_kwargs = dict(defaults)
    final_kwargs.update(base)
    final_kwargs.update(fixed)
    final_fit_candidate, _ = _split_fit_kwargs(best_candidate)
    final_kwargs.update(final_fit_candidate)
    if random_state is not None and (chosen == "copula" or method == "csp"):
        final_kwargs.setdefault("random_state", random_state)
    final_kwargs["dimension_method"] = method
    if chosen == "copula":
        final_kwargs = _with_copula_method_defaults(final_kwargs, D)
    if chosen == "semi" and method == "fixed":
        final_kwargs.update(_semi_initial_state(Y, final_kwargs, init_factory))
    if random_state is not None:
        # the refit draws from the global generator like the candidate fits;
        # it uses the selected candidate's seed, whatever ran before it
        np.random.seed(_candidate_global_seed(random_state, best_index))
    refit_started = time.perf_counter()
    final = fit(Y, model=chosen, D=D, family=family, **final_kwargs)
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
    return record(final)


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
