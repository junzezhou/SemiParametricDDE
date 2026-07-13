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


def fit(
    Y,
    *,
    model: str = "auto",
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
    model:
        ``"semi"``, ``"copula"``, or ``"auto"``. ``"auto"`` uses SemiDDE for
        all-continuous data and CopDDE otherwise.
    K_list:
        Layer widths ordered from the shallowest latent layer to the deepest.
        Future CSP support will allow ``None`` entries.
    variable_types:
        Optional variable type labels used when ``model="auto"``.
    dimension_method:
        ``"fixed"`` for supplied widths, or ``"csp"`` for CSP-based width
        selection in supported copula estimators. If omitted, SemiDDE uses
        ``"fixed"`` and CopDDE uses ``"csp"``.
    **kwargs:
        Algorithm-specific keyword arguments.
    """
    Y = np.asarray(Y)
    chosen = _infer_model(Y, variable_types) if model == "auto" else model.lower()

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

    raise ValueError("model must be 'auto', 'semi', or 'copula'.")
