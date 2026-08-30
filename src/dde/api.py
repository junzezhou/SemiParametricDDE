"""User-facing fitting entry points."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .results import DDEResult


def _infer_model(Y: np.ndarray, variable_types: Optional[Sequence[str]]) -> str:
    """Choose a conservative default model from observed variable types."""
    if variable_types is None:
        return "semi"
    normalized = {str(v).lower() for v in variable_types}
    if normalized <= {"continuous", "numeric", "real"}:
        return "semi"
    return "copula"


def _canonical_method(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "auto": "auto",
        "dde": "dde",
        "semi": "semi",
        "semidde": "semi",
        "copula": "copula",
        "copdde": "copula",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "method must be 'DDE', 'SemiDDE', 'CopDDE', or 'auto'."
        ) from exc


def fit(
    Y,
    *,
    method: Optional[str] = None,
    model: Optional[str] = None,
    D: Optional[int] = None,
    family: Optional[str] = None,
    K_list=None,
    variable_types: Optional[Sequence[str]] = None,
    dimension_method: Optional[str] = None,
    **kwargs: Any,
) -> DDEResult:
    """Fit a DDE-family model through a unified package interface.

    Parameters
    ----------
    Y:
        Observed data matrix, with observations in rows and variables in columns.
    method:
        ``"DDE"``, ``"SemiDDE"``, ``"CopDDE"``, or ``"auto"``. ``"auto"``
        uses SemiDDE for all-continuous data and CopDDE otherwise.
    model:
        Alias for ``method`` retained for the manuscript's current examples.
        Supplying conflicting values raises an error.
    D:
        Number of stochastic binary latent layers. Required for DDE fits.
    family:
        Parametric observed-data family. Required when ``method="DDE"``.
    K_list:
        Layer widths ordered from the shallowest latent layer to the deepest.
        Required for fixed-width fits and omitted for Normal-DDE CSP fits,
        whose maximal widths are derived recursively from the observed width.
    variable_types:
        Optional variable type labels used when ``model="auto"``.
    dimension_method:
        ``"fixed"`` for supplied widths, or ``"csp"`` for CSP-based width
        selection in supported Normal DDE and copula estimators. If omitted, SemiDDE uses
        ``"fixed"`` and CopDDE uses ``"csp"``.
    **kwargs:
        Algorithm-specific keyword arguments.
    """
    Y = np.asarray(Y)
    method_value = "auto" if method is None and model is None else (
        method if method is not None else model
    )
    chosen = _canonical_method(method_value)
    if method is not None and model is not None:
        method_chosen = _canonical_method(method)
        model_chosen = _canonical_method(model)
        if method_chosen != model_chosen:
            raise ValueError("method and model specify different DDE formulations.")
    if chosen == "auto":
        chosen = _infer_model(Y, variable_types)

    if chosen == "dde":
        from .algorithms.dde import fit_dde_normal

        if D is None:
            raise ValueError("D is required when method='DDE'.")
        if family is None:
            raise ValueError("family is required when method='DDE'.")
        family_name = str(family).strip().lower()
        if family_name not in {"normal", "gaussian"}:
            raise NotImplementedError(
                "The current DDE interface implements family='Normal' only."
            )
        dimension = "fixed" if dimension_method is None else str(dimension_method).lower()
        return fit_dde_normal(
            Y,
            D=D,
            K_list=K_list,
            dimension_method=dimension,
            **kwargs,
        )

    if chosen in {"semi", "semidde"}:
        from .algorithms.semi import fit_semidde

        method = "fixed" if dimension_method is None else str(dimension_method).lower()
        if method != "fixed":
            raise ValueError("SemiDDE supports only dimension_method='fixed'.")
        return fit_semidde(Y, K_list=K_list, dimension_method=method, **kwargs)

    if chosen in {"copula", "copdde"}:
        from .algorithms.copula import fit_copula_dde

        method = "csp" if dimension_method is None else str(dimension_method).lower()
        if method not in {"fixed", "csp"}:
            raise ValueError("CopDDE dimension_method must be 'fixed' or 'csp'.")
        return fit_copula_dde(Y, K_list=K_list, dimension_method=method, **kwargs)

    raise RuntimeError(f"Unhandled DDE method: {chosen}.")
