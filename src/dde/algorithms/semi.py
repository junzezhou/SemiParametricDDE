"""Package wrappers for fixed-width and CSP SemiDDE estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from dde.results import DDEResult
from dde.utils import validate_K_list_identifiability, validate_dde_lists
from dde.utils.result_compression import compress_csp_result_components

from .csp import maximal_identifiable_widths
from .fixed_graph import graph_anchored_initialize, resolve_graph_list
from .normal_init_semi import Normal_init_semi, rank_gaussianize_columns
from .parametric_families import NORMAL_FAMILY
from .semi_csp import get_SAEM_semiparametric_csp_copula
from .semi_em import get_EM_est
from .semi_saem import get_SAEM_normal, get_SAEM_semiparametricv1


_INIT_KEYS = ("X_in", "prop_in", "B_list_in", "gamma_in", "A_list_in")


def _initialize_semidde(Y, K_list, kwargs, *, dimension_method, support_list=None):
    """Resolve supplied, data, or oracle initialization in one place."""
    graph_fixed = support_list is not None and any(
        support is not None for support in support_list
    )
    supplied = [kwargs.get(name) for name in _INIT_KEYS]
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError(
            "X_in, prop_in, B_list_in, gamma_in, and A_list_in must be "
            "supplied together."
        )

    verbose = bool(kwargs.get("verbose", True))
    initialize = kwargs.pop("initialize", "data")
    legacy_mode = kwargs.pop(
        "initialization_mode", kwargs.pop("init_mode", None)
    )
    if legacy_mode is not None:
        initialize = legacy_mode

    if all(value is not None for value in supplied):
        validate_dde_lists(
            kwargs["X_in"],
            kwargs["B_list_in"],
            A_list=kwargs["A_list_in"],
            prop=kwargs["prop_in"],
            gamma=kwargs["gamma_in"],
            K_list=K_list,
            observed_name="X_in",
            require_binary_A=True,
        )
        kwargs.pop("init_binarize", None)
        if verbose:
            print(
                "SemiDDE initialization: not triggered; using supplied "
                "initialized arguments."
            )
        return "provided"

    if initialize is True:
        mode = (
            "oracle"
            if "B_list_true" in kwargs or "G_list_true" in kwargs
            else "data"
        )
    elif initialize in {False, None}:
        mode = "none"
    else:
        mode = str(initialize).lower()
    if mode == "none":
        raise ValueError(
            "initialize=False requires complete X_in, prop_in, B_list_in, "
            "gamma_in, and A_list_in values."
        )
    if mode not in {"data", "oracle"}:
        raise ValueError("initialize must be 'data' or 'oracle'.")
    if graph_fixed and mode != "data":
        raise ValueError(
            "G_list labels the latent columns itself; use initialize='data' "
            "(the graph-anchored initialization) or supply a full state."
        )

    epsilon = float(kwargs.pop("epsilon", 1e-4))
    if dimension_method == "fixed":
        # The fixed kernel's deep-layer warm start also needs epsilon.
        kwargs["epsilon"] = epsilon
    # MARK: Original initializer thresholded the projected scores at zero for
    # both dimension methods; the fixed-width default is now the Normal fixed
    # kernel's anchor-mixture binarization (rare columns otherwise start near
    # 40 % prevalence and some collapse; D=3 design, true prevalence 0.14).
    binarize = kwargs.pop(
        "init_binarize",
        "anchor_mixture" if dimension_method == "fixed" else "projection",
    )
    X_init = rank_gaussianize_columns(Y)
    if verbose:
        print(f"SemiDDE initialization: triggered using initialize='{mode}'.")

    if mode == "oracle":
        try:
            B_list_true = kwargs.pop("B_list_true")
            G_list_true = kwargs.pop("G_list_true")
        except KeyError as exc:
            raise ValueError(
                "initialize='oracle' requires B_list_true and G_list_true."
            ) from exc
        prop, B_list, gamma, A_list, _ = Normal_init_semi(
            X_init,
            K_list[0],
            K_list,
            B_list_true,
            G_list_true,
            epsilon,
            mode="oracle",
            binarize=binarize,
        )
        if dimension_method == "fixed":
            # Simulation-only labels for the deep-layer initializer of the
            # staged schedule, as in the Normal fixed kernel's oracle mode.
            kwargs.setdefault("B_list_true", B_list_true)
            kwargs.setdefault("G_list_true", G_list_true)
        if dimension_method == "fixed" and kwargs.get("final_refit", False):
            kwargs.setdefault("final_refit_B_list_true", B_list_true)
            kwargs.setdefault("final_refit_G_list_true", G_list_true)
    elif graph_fixed:
        kwargs.pop("init_binarize", None)
        prop, B_list, gamma, A_list, _ = graph_anchored_initialize(
            X_init, support_list, K_list, family=NORMAL_FAMILY, epsilon=epsilon
        )
        mode = "data_graph"
    else:
        if kwargs.get("final_refit", False):
            raise ValueError("final_refit=True requires initialize='oracle'.")
        prop, B_list, gamma, A_list, _ = Normal_init_semi(
            X_init,
            K_list[0],
            K_list,
            epsilon=epsilon,
            mode="data",
            binarize=binarize,
        )

    kwargs.update(
        X_in=X_init,
        prop_in=prop,
        B_list_in=B_list,
        gamma_in=gamma,
        A_list_in=A_list,
    )
    return mode


def _fit_semidde_fixed(Y, K_list, kwargs, *, initialization, support_list=None):
    if support_list is not None and any(support is not None for support in support_list):
        # Fixed layers are neither penalized nor thresholded; with every
        # layer fixed, pen_list and tau are not used.
        if all(support is not None for support in support_list):
            kwargs.setdefault("pen_list", [0.0] * len(K_list))
            if kwargs.get("tau") is None:
                kwargs["tau"] = 1.0
        # The graph-anchored start pins the layer-1 states (see the Normal
        # DDE default in fit_parametric_dde).
        kwargs.setdefault("sa_burn_in", 0)
        kwargs["support_list"] = support_list
    if kwargs.get("C") is None:
        kwargs["C"] = 1
    if kwargs.get("tol") is None:
        # The relative-change tolerance of the fixed-width DDE kernels.
        kwargs["tol"] = 0.005
    required = [
        "X_in",
        "prop_in",
        "B_list_in",
        "gamma_in",
        "pen_list",
        "tau",
        "A_list_in",
        "C",
    ]
    missing = [name for name in required if kwargs.get(name) is None]
    if missing:
        raise ValueError(
            "fit_semidde fixed mode needs initialized algorithm arguments: "
            + ", ".join(missing)
        )

    X, prop, B_list, gamma, A_list, iterations, kernel_diagnostics = (
        get_SAEM_semiparametricv1(Y=Y, return_diagnostics=True, **kwargs)
    )
    fixed_layers = [
        support is not None for support in (support_list or [None] * len(K_list))
    ]
    return DDEResult(
        model="SemiDDE",
        K_list=[int(B.shape[1] - 1) for B in B_list],
        prop=prop,
        B_list=B_list,
        gamma=gamma,
        A_list=A_list,
        transformed=X,
        diagnostics={
            "iterations": iterations,
            "D": len(K_list),
            "K_list": tuple(K_list),
            "dimension_method": "fixed",
            "initialization": initialization,
            "pseudo_observation_update": "semiparametric_g",
            "fixed_graph_layers": fixed_layers,
            **kernel_diagnostics,
        },
        raw={
            "X": X, "prop": prop, "B_list": B_list, "gamma": gamma,
            "A_list": A_list, "iterations": iterations,
        },
    )


def _pop_inner_iterations(kwargs):
    aliases = [
        (name, kwargs.pop(name))
        for name in ("max_inner_iter", "max_it", "it")
        if name in kwargs
    ]
    if not aliases:
        return 50
    values = {int(value) for _, value in aliases}
    if len(values) != 1:
        names = ", ".join(name for name, _ in aliases)
        raise ValueError(f"Conflicting SemiDDE CSP iteration controls: {names}.")
    return values.pop()


def _fit_semidde_csp(Y, K_max, kwargs, *, initialization):
    if "pen_list" in kwargs:
        raise ValueError(
            "pen_list is only used by SemiDDE dimension_method='fixed'."
        )
    if kwargs.pop("final_refit", False):
        raise ValueError("final_refit is only available for fixed-width SemiDDE.")
    if kwargs.get("C") is None:
        kwargs["C"] = 1

    required = [
        "X_in",
        "prop_in",
        "B_list_in",
        "gamma_in",
        "A_list_in",
        "t_spike",
        "temp",
        "tau",
        "C",
    ]
    missing = [name for name in required if kwargs.get(name) is None]
    if missing:
        raise ValueError(
            "fit_semidde csp mode needs algorithm arguments: "
            + ", ".join(missing)
        )

    max_inner_iter = _pop_inner_iterations(kwargs)
    backend = str(kwargs.pop("csp_backend", "copula")).strip().lower()
    if backend != "copula":
        raise ValueError(
            "SemiDDE CSP runs on the copula kernel (csp_backend='copula', the "
            "default); the MCEM path was removed (git tag csp-mcem-last)."
        )
    return _fit_semidde_csp_copula(Y, K_max, kwargs, max_inner_iter, initialization=initialization)


def _fit_semidde_csp_copula(Y, K_max, kwargs, max_inner_iter, *, initialization):
    accepted = {
        "X_in", "prop_in", "B_list_in", "gamma_in", "A_list_in", "C", "t_spike", "temp", "tau", "tol",
        "g_update_every", "g_estimator", "g_mode", "max_enum_bits", "mc_samples", "delta", "random_state",
        "optimizer_maxiter", "keep_trace", "verbose",
    }
    engine = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith("csp_")}
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        mcem_only = sorted(set(unknown) & {
            "max_outer_iter", "g_update_rho", "optimizer_tol", "convergence_window",
            "convergence_mean_tol", "burn", "bound_val", "estep_row_chunk_size",
            "estep_coordinate_chunk_size",
        })
        hint = (
            "; " + ", ".join(mcem_only) + " belonged to the removed MCEM path"
            if mcem_only else ""
        )
        raise TypeError("Unexpected SemiDDE CSP argument(s): " + ", ".join(unknown) + hint + ".")
    engine = {key[len("csp_"):]: value for key, value in engine.items()}
    raw = get_SAEM_semiparametric_csp_copula(Y=Y, max_iter=max_inner_iter, **kwargs, **engine)
    diagnostics = {
        "iterations": raw["iterations"],
        "D": len(K_max),
        "K_list": tuple(raw["num_active"].tolist()),
        "K_max_list": tuple(K_max),
        "original_K_list": tuple(K_max),
        "dimension_method": "csp",
        "initialization": initialization,
        "backend": "copula",
        "kernel": "semiparametric_pseudo_observation_copula_csp",
        "pseudo_observation_update": "semiparametric_g_every_iteration",
        "g_estimator": raw["g_estimator"],
        "threshold_rule": raw["threshold_rule"],
        "num_active": raw["num_active"].copy(),
        "allocation": [value.copy() for value in raw["allocation"]],
        "omega": [value.copy() for value in raw["omega"]],
        "slab_scale": [value.copy() for value in raw["slab_scale"]],
        "relative_history": raw["relative_history"],
        "g_relative_history": raw["g_relative_history"],
        "temperature": raw["temperature"],
        "latent_transition": raw["latent_transition"],
        "mc_state_update": raw["mc_state_update"],
    }
    compressed = compress_csp_result_components(
        prop=raw["prop"], B_list=raw["B_list"], A_list=raw["A_list"], diagnostics=diagnostics, raw=raw,
        min_active=True,
    )
    return DDEResult(
        model="SemiDDE",
        K_list=list(compressed["diagnostics"]["K_list"]),
        prop=compressed["prop"],
        B_list=compressed["B_list"],
        gamma=raw["gamma"],
        A_list=compressed["A_list"],
        transformed=raw["X"],
        diagnostics=compressed["diagnostics"],
        raw=compressed["raw"],
    )


def fit_semidde(
    Y,
    *,
    D=None,
    K_list=None,
    dimension_method="fixed",
    G_list=None,
    **kwargs: Any,
) -> DDEResult:
    """Fit fixed-width or CSP-selected SemiDDE.

    ``G_list`` (fixed mode) fixes a user-specified graph of the shallowest
    layers as in :func:`dde.algorithms.dde.fit_parametric_dde`: the fixed
    layers are fitted on the graph without penalty or threshold in every
    inner SAEM fit of the pseudo-observations, start from the
    graph-anchored initialization, and run without the unit-step burn-in
    (``sa_burn_in=0``) unless it is given.  ``pen_list`` and ``tau`` may be
    omitted when every layer is fixed.

    Fixed mode uses the supplied ``K_list``.  CSP mode starts from the
    maximal widths ``K_list`` when given and otherwise from the recursive
    maximum identifiable widths ``floor(J/3), floor(floor(J/3)/3), ...``
    (``D`` is then required) before shrinking them.  ``C`` defaults to 1.
    CSP mode
    runs on the copula kernel by default (``csp_backend="copula"``: the
    Normal-family engine with a ``g`` step after every committed
    iteration); the former MCEM path was removed.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[0] < 1 or Y.shape[1] < 1:
        raise ValueError("Y must be a non-empty two-dimensional matrix.")
    method = str(dimension_method).strip().lower()
    if method not in {"fixed", "csp"}:
        raise ValueError("SemiDDE dimension_method must be 'fixed' or 'csp'.")

    support_list = None
    if G_list is not None and method != "fixed":
        raise ValueError(
            "G_list fixes the graph and the widths; use dimension_method='fixed'."
        )
    if method == "fixed":
        if G_list is not None:
            if D is None:
                D = len(K_list) if K_list is not None else len(G_list)
            K_tuple, support_list = resolve_graph_list(
                G_list, D=int(D), J=Y.shape[1], K_list=K_list
            )
        else:
            if K_list is None:
                raise ValueError("K_list is required when dimension_method='fixed'.")
            K_tuple = validate_K_list_identifiability(
                K_list, D=D, J=Y.shape[1]
            )
        if "t_spike" in kwargs or "temp" in kwargs:
            raise ValueError(
                "t_spike and temp are only used by dimension_method='csp'."
            )
    else:
        if D is None:
            if K_list is None:
                raise ValueError("D is required when SemiDDE dimension_method='csp'.")
            D = len(K_list)
        # Optional maximal widths; by default the recursive maximal
        # identifiable widths floor(J/3), floor(K_1/3), ...
        K_tuple = (
            maximal_identifiable_widths(Y.shape[1], int(D))
            if K_list is None
            else validate_K_list_identifiability(K_list, D=int(D), J=Y.shape[1])
        )

    fit_kwargs = dict(kwargs)
    initialization = _initialize_semidde(
        Y, K_tuple, fit_kwargs, dimension_method=method, support_list=support_list
    )
    if method == "fixed":
        return _fit_semidde_fixed(
            Y, K_tuple, fit_kwargs, initialization=initialization,
            support_list=support_list,
        )
    return _fit_semidde_csp(
        Y, K_tuple, fit_kwargs, initialization=initialization
    )


__all__ = [
    "Normal_init_semi",
    "fit_semidde",
    "get_EM_est",
    "get_SAEM_normal",
    "get_SAEM_semiparametricv1",
    "rank_gaussianize_columns",
]
