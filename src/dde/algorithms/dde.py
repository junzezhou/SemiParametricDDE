"""Parametric Deep Discrete Encoder fitting wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dde.results import DDEResult
from dde.utils import validate_K_list_identifiability, validate_dde_lists
from dde.utils.result_compression import compress_csp_result_components

from .csp import maximal_identifiable_widths
from .normal_csp import fit_parametric_csp_mcem
from .parametric_families import (
    BERNOULLI_FAMILY,
    NORMAL_FAMILY,
    POISSON_FAMILY,
    resolve_parametric_family,
)
from .semi_saem import get_SAEM_parametric


DEFAULT_PARAMETRIC_CONVERGENCE_TOL = 0.005
_BERNOULLI_CSP_PENALTY_REFERENCE_N = 1000.0
_BERNOULLI_CSP_PENALTY_POWER = 2.0 / 3.0
_BERNOULLI_CSP_SHRINKAGE_BURN = 10


def _resolve_mcem_shrinkage_defaults(
    *,
    family_key: str,
    D: int,
    N: int,
    backend: str,
    max_iter: int,
    penalty_strength,
    shrinkage_burn,
):
    """Resolve validated family-specific MCEM shrinkage defaults."""
    adaptive_bernoulli = (
        family_key == "bernoulli" and D >= 3 and backend == "mcem"
    )
    penalty_was_automatic = penalty_strength is None
    burn_was_automatic = shrinkage_burn is None

    if penalty_was_automatic:
        penalty_strength = (
            max(
                1.0,
                (N / _BERNOULLI_CSP_PENALTY_REFERENCE_N)
                ** _BERNOULLI_CSP_PENALTY_POWER,
            )
            if adaptive_bernoulli
            else 1.0
        )
    if burn_was_automatic:
        shrinkage_burn = (
            _BERNOULLI_CSP_SHRINKAGE_BURN
            if adaptive_bernoulli
            and max_iter >= 2 * _BERNOULLI_CSP_SHRINKAGE_BURN
            else 0
        )

    return (
        penalty_strength,
        shrinkage_burn,
        {
            "name": (
                "bernoulli_depth_adaptive"
                if adaptive_bernoulli
                and (penalty_was_automatic or burn_was_automatic)
                else "legacy_or_explicit"
            ),
            "penalty_strength_was_automatic": penalty_was_automatic,
            "shrinkage_burn_was_automatic": burn_was_automatic,
        },
    )


def _validate_pen_list(pen_list: Sequence[float], D: int) -> list[float]:
    if isinstance(pen_list, (str, bytes)):
        raise ValueError("pen_list must be a length-D sequence.")
    try:
        values = np.asarray(pen_list, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("pen_list must be a length-D numeric sequence.") from exc
    if values.ndim != 1:
        raise ValueError("pen_list must be a one-dimensional length-D sequence.")
    penalties = values.tolist()
    if len(penalties) != D:
        raise ValueError(f"pen_list must have length D={D}; got {len(penalties)}.")
    if not np.all(np.isfinite(penalties)) or any(value < 0 for value in penalties):
        raise ValueError("pen_list entries must be finite and nonnegative.")
    return penalties


def fit_parametric_dde(
    Y,
    *,
    family,
    D: int,
    K_list=None,
    dimension_method: str = "fixed",
    pen_list=None,
    t_spike=None,
    temp=None,
    tau: float,
    C: int,
    tol: float = DEFAULT_PARAMETRIC_CONVERGENCE_TOL,
    max_it: int = 50,
    initialize: str = "data",
    epsilon: float = 1e-4,
    prop_in=None,
    B_list_in=None,
    gamma_in=None,
    A_list_in=None,
    B_list_true=None,
    G_list_true=None,
    bound_val: float = 10.0,
    random_state=None,
    optimizer_maxiter: int = 100,
    optimizer_tol: float = 1e-6,
    convergence_window: int = 7,
    convergence_mean_tol: float = 0.01,
    mcem_stage_min_iter: int = 20,
    mcem_stage_width_window: int = 10,
    mcem_stage_freeze_deeper: bool = True,
    mcem_penalty_strength=None,
    mcem_shrinkage_burn=None,
    mcem_reorder_after_shrinkage_burn: bool = True,
    mcem_stage_spectral_gap_threshold=None,
    csp_backend: str = "mcem",
    burn: int = 5,
    estep_row_chunk_size: int = 2048,
    estep_coordinate_chunk_size: int = 1,
    keep_trace: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> DDEResult:
    """Fit a fixed-width or CSP-shrunk parametric DDE.

    The family adapter validates the observed matrix, prepares initialization,
    and supplies observed-layer operations to the fixed-width SAEM kernel or
    one of the CSP MCEM backends.
    In contrast, SemiDDE and CopDDE construct latent-Gaussian
    pseudo-observations before using their respective kernels.

    ``initialize="oracle"`` is intended only for simulation studies that need
    estimates aligned with known generating parameters.  Ordinary data fits
    should use the default ``initialize="data"``.

    For fixed-width fits, ``tol`` is the postprocessed parameter-update RMSE.
    Its default is ``0.005`` for a dimension-independent convergence scale.

    Parametric CSP defaults to the simultaneous ``csp_backend="mcem"`` path.
    The explicit ``"mcem_reinitialize"`` backend performs staged parent
    rebuilding, while ``"mcem_cap_only"`` applies staged child-implied caps
    and preserves eligible parent state.
    """
    family_adapter = resolve_parametric_family(family)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(
            f"Unexpected {family_adapter.name} DDE argument(s): {unknown}."
        )

    X = family_adapter.validate_y(Y)

    D = int(D)
    if D < 1:
        raise ValueError("D must be a positive integer.")
    method = str(dimension_method).strip().lower()
    if method not in {"fixed", "csp"}:
        raise ValueError(
            f"{family_adapter.name} DDE dimension_method must be 'fixed' or 'csp'."
        )
    if method not in family_adapter.supported_dimension_methods:
        supported = " or ".join(
            repr(value) for value in family_adapter.supported_dimension_methods
        )
        raise NotImplementedError(
            f"{family_adapter.name} DDE currently supports "
            f"dimension_method={supported} only."
        )
    if method == "fixed":
        if K_list is None:
            raise ValueError("K_list is required when dimension_method='fixed'.")
        if t_spike is not None or temp is not None:
            raise ValueError(
                "t_spike and temp are only used by dimension_method='csp'."
            )
        K_tuple = validate_K_list_identifiability(K_list, D=D, J=X.shape[1])
        penalties = _validate_pen_list(pen_list, D)
    elif method == "csp":
        if K_list is not None:
            raise ValueError(
                "K_list must be omitted when dimension_method='csp'; maximal "
                "identifiable widths are derived recursively from J."
            )
        if pen_list is not None:
            raise ValueError("pen_list is only used by dimension_method='fixed'.")
        if t_spike is None or temp is None:
            raise ValueError(
                f"{family_adapter.name} DDE with dimension_method='csp' "
                "requires t_spike and temp."
            )
        K_tuple = maximal_identifiable_widths(X.shape[1], D)
        penalties = None

    tau = float(tau)
    tol = float(tol)
    epsilon = float(epsilon)
    bound_val = float(bound_val)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    if not np.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly between 0 and 0.5.")
    if not np.isfinite(bound_val) or bound_val <= 0:
        raise ValueError("bound_val must be finite and positive.")
    C = int(C)
    max_it = int(max_it)
    if C < 1:
        raise ValueError("C must be a positive integer.")
    if max_it < 1:
        raise ValueError("max_it must be a positive integer.")

    supplied_names = ["prop_in", "B_list_in"]
    supplied = [prop_in, B_list_in]
    if family_adapter.has_dispersion:
        supplied_names.append("gamma_in")
        supplied.append(gamma_in)
    elif gamma_in is not None:
        raise ValueError(
            f"{family_adapter.name} has no dispersion parameter; gamma_in must be omitted."
        )
    supplied_names.append("A_list_in")
    supplied.append(A_list_in)
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        joined_names = ", ".join(supplied_names[:-1])
        raise ValueError(
            f"{joined_names}, and {supplied_names[-1]} must be supplied together."
        )

    if all(value is not None for value in supplied):
        initialization = "provided"
        validate_dde_lists(
            X,
            B_list_in,
            A_list=A_list_in,
            prop=prop_in,
            gamma=None,
            K_list=K_tuple,
            observed_name="Y",
            require_binary_A=True,
        )
        prop = np.asarray(prop_in, dtype=float)
        B_list = [np.asarray(B, dtype=float) for B in B_list_in]
        gamma = (
            family_adapter.validate_dispersion(gamma_in, X.shape[1])
            if family_adapter.has_dispersion
            else None
        )
        A_list = [np.asarray(A, dtype=float) for A in A_list_in]
    else:
        initialization = str(initialize).lower()
        if initialization == "data":
            if B_list_true is not None or G_list_true is not None:
                raise ValueError(
                    "B_list_true and G_list_true are only used with initialize='oracle'."
                )
            prop, B_list, gamma, A_list, _ = family_adapter.initialize(
                X,
                K_tuple,
                epsilon=epsilon,
                mode="data",
                random_state=random_state if method == "csp" else None,
            )
        elif initialization == "oracle":
            if B_list_true is None or G_list_true is None:
                raise ValueError(
                    "initialize='oracle' requires B_list_true and G_list_true."
                )
            prop, B_list, gamma, A_list, _ = family_adapter.initialize(
                X,
                K_tuple,
                B_list_true=B_list_true,
                G_list_true=G_list_true,
                epsilon=epsilon,
                mode="oracle",
                random_state=random_state if method == "csp" else None,
            )
        else:
            raise ValueError("initialize must be 'data' or 'oracle'.")

    if method == "fixed":
        if random_state is not None:
            raise ValueError(
                "random_state is currently supported by the CSP path only."
            )
        fixed_trace = [] if keep_trace else None
        prop, B_list, gamma, A_list, iterations = get_SAEM_parametric(
            X,
            prop,
            B_list,
            gamma,
            penalties,
            tau,
            A_list,
            C,
            tol,
            max_iter=max_it,
            bound_val=bound_val,
            verbose=bool(verbose),
            estep_row_chunk_size=estep_row_chunk_size,
            trace=fixed_trace,
            family=family_adapter,
        )

        X_result = X.copy()
        diagnostics = {
            "iterations": iterations,
            "D": D,
            "K_list": list(K_tuple),
            "family": family_adapter.name,
            "dimension_method": "fixed",
            "initialization": initialization,
            "latent_transition": "systematic_coordinate_gibbs",
            "mc_state_update": "last_draw",
            "convergence_metric": "postprocessed_parameter_rmse",
            "convergence_tol": tol,
        }
        if fixed_trace is not None:
            diagnostics["trace"] = fixed_trace
        return DDEResult(
            model="DDE",
            K_list=list(K_tuple),
            prop=prop,
            B_list=B_list,
            gamma=gamma,
            A_list=A_list,
            transformed=X_result,
            diagnostics=diagnostics,
            raw=(X_result, prop, B_list, gamma, A_list, iterations),
        )

    backend = str(csp_backend).strip().lower()
    backend_configuration = {
        "mcem": (False, "none"),
        "mcem_reinitialize": (True, "reinitialize"),
        "mcem_cap_only": (True, "cap_only"),
    }
    if backend not in backend_configuration:
        raise ValueError(
            "csp_backend must be 'mcem', 'mcem_reinitialize', or " "'mcem_cap_only'."
        )
    stagewise, stage_transition_mode = backend_configuration[backend]
    if backend != "mcem_reinitialize" and mcem_stage_spectral_gap_threshold is not None:
        raise ValueError(
            "mcem_stage_spectral_gap_threshold is only valid when "
            "csp_backend='mcem_reinitialize'."
        )
    (
        resolved_penalty_strength,
        resolved_shrinkage_burn,
        shrinkage_default_policy,
    ) = _resolve_mcem_shrinkage_defaults(
        family_key=family_adapter.key,
        D=D,
        N=X.shape[0],
        backend=backend,
        max_iter=max_it,
        penalty_strength=mcem_penalty_strength,
        shrinkage_burn=mcem_shrinkage_burn,
    )
    raw = fit_parametric_csp_mcem(
        X,
        prop,
        B_list,
        gamma,
        A_list,
        C=C,
        max_iter=max_it,
        t_spike=t_spike,
        temperature=temp,
        threshold=tau,
        convergence_tol=tol,
        convergence_window=convergence_window,
        burn=burn,
        optimizer_maxiter=optimizer_maxiter,
        optimizer_tol=optimizer_tol,
        bound_val=bound_val,
        estep_row_chunk_size=estep_row_chunk_size,
        estep_coordinate_chunk_size=estep_coordinate_chunk_size,
        random_state=random_state,
        keep_trace=keep_trace,
        verbose=bool(verbose),
        convergence_mean_tol=convergence_mean_tol,
        stagewise=stagewise,
        stage_min_iter=mcem_stage_min_iter,
        stage_width_window=mcem_stage_width_window,
        stage_freeze_deeper=bool(mcem_stage_freeze_deeper),
        penalty_strength=resolved_penalty_strength,
        shrinkage_burn=resolved_shrinkage_burn,
        reorder_after_shrinkage_burn=bool(
            mcem_reorder_after_shrinkage_burn
        ),
        stage_transition_mode=stage_transition_mode,
        stage_spectral_gap_threshold=mcem_stage_spectral_gap_threshold,
        family=family_adapter,
    )

    X_result = X.copy()
    latent_transition = raw["latent_transition"]
    diagnostics = {
        "iterations": raw["iterations"],
        "D": D,
        "K_list": tuple(raw["num_active"].tolist()),
        "K_max_list": K_tuple,
        "original_K_list": K_tuple,
        "family": family_adapter.name,
        "dimension_method": "csp",
        "initialization": initialization,
        "backend": backend,
        "kernel": (
            "systematic_gibbs_mcem_csp"
            if latent_transition == "systematic_coordinate_gibbs"
            else "approximate_mcem_csp"
        ),
        "latent_transition": latent_transition,
        "mc_state_update": raw["mc_state_update"],
        "transition_mode": stage_transition_mode,
        "threshold_rule": "first_two_layers",
        "canonicalized_initial_order": raw[
            "canonicalized_initial_order"
        ],
        "initial_permutations": raw["initial_permutations"],
        "num_active": raw["num_active"].copy(),
        "allocation": [value.copy() for value in raw["allocation"]],
        "omega": [value.copy() for value in raw["omega"]],
        "slab_scale": [value.copy() for value in raw["slab_scale"]],
        "relative_history": raw["relative_history"],
        "temperature": raw["temperature"],
    }
    diagnostics.update(
        {
            "stagewise": raw["stagewise"],
            "stage_transitions": raw["stage_transitions"],
            "final_stage": raw["final_stage"],
            "stage_freeze_deeper": raw["stage_freeze_deeper"],
            "penalty_strength": raw["penalty_strength"],
            "shrinkage_burn": raw["shrinkage_burn"],
            "shrinkage_default_policy": shrinkage_default_policy,
            "reorder_after_shrinkage_burn": raw[
                "reorder_after_shrinkage_burn"
            ],
            "warmup_reordered": raw["warmup_reordered"],
            "stage_spectral_gap_threshold": raw["stage_spectral_gap_threshold"],
            "stage_transition_events": raw["stage_transition_events"],
            "stage_caps": raw["stage_caps"],
            "stopped_by_convergence": raw["stopped_by_convergence"],
        }
    )
    if keep_trace:
        diagnostics["trace"] = raw["trace"]

    compressed = compress_csp_result_components(
        prop=raw["prop"],
        B_list=raw["B_list"],
        A_list=raw["A_list"],
        diagnostics=diagnostics,
        raw=raw,
        active_masks=raw["active_masks"],
        min_active=False,
    )
    if raw["canonicalized_initial_order"]:
        active_precanonical = {}
        active_canonical = compressed["diagnostics"][
            "active_columns_original_index"
        ]
        for layer, permutation in enumerate(raw["initial_permutations"]):
            canonical_indices = np.asarray(
                active_canonical[f"layer_{layer + 1}"], dtype=int
            )
            active_precanonical[f"layer_{layer + 1}"] = (
                np.asarray(permutation, dtype=int)[canonical_indices - 1] + 1
            ).tolist()
        compressed["diagnostics"][
            "active_columns_precanonical_index"
        ] = active_precanonical
    return DDEResult(
        model="DDE",
        K_list=list(compressed["diagnostics"]["K_list"]),
        prop=compressed["prop"],
        B_list=compressed["B_list"],
        gamma=raw["gamma"],
        A_list=compressed["A_list"],
        transformed=X_result,
        diagnostics=compressed["diagnostics"],
        raw=compressed["raw"],
    )


def fit_dde_normal(Y, **kwargs) -> DDEResult:
    """Compatibility wrapper for the public Normal DDE fitting function."""
    if "family" in kwargs:
        raise TypeError("Unexpected Normal DDE argument(s): family.")
    return fit_parametric_dde(Y, family=NORMAL_FAMILY, **kwargs)


def fit_dde_bernoulli(Y, **kwargs) -> DDEResult:
    """Compatibility wrapper for the public Bernoulli DDE fitting function."""
    if "family" in kwargs:
        raise TypeError("Unexpected Bernoulli DDE argument(s): family.")
    return fit_parametric_dde(Y, family=BERNOULLI_FAMILY, **kwargs)


def fit_dde_poisson(Y, **kwargs) -> DDEResult:
    """Compatibility wrapper for the public Poisson DDE fitting function."""
    if "family" in kwargs:
        raise TypeError("Unexpected Poisson DDE argument(s): family.")
    return fit_parametric_dde(Y, family=POISSON_FAMILY, **kwargs)


__all__ = [
    "fit_dde_bernoulli",
    "fit_dde_normal",
    "fit_dde_poisson",
    "fit_parametric_dde",
]
