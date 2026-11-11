"""User-facing fitting entry points."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np

from .results import DDEResult, data_column_names, record_column_names


# Manuscript names of the three formulations, keyed by the internal path key.
MODEL_NAMES = {"dde": "DDE", "semi": "SemiDDE", "copula": "CopDDE"}

# ``max_it`` is the manuscript's iteration limit for every model; the other
# names are the kernels' own spellings, accepted as aliases.
ITERATION_LIMIT_NAMES = ("max_it", "max_iter", "it", "max_inner_iter")
_KERNEL_ITERATION_NAME = {"dde": "max_it", "semi": "max_inner_iter", "copula": "it"}


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
            "model must be 'DDE', 'SemiDDE', 'CopDDE', or 'auto'."
        ) from exc


def _resolve_model(model: Optional[str], method: Optional[str], *, stacklevel: int = 3) -> str:
    """Internal path key of ``model``; ``method`` is its deprecated spelling."""
    if method is not None:
        warnings.warn(
            "the 'method' argument is deprecated; use 'model' "
            "(model='DDE', 'SemiDDE' or 'CopDDE').",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        if model is not None and _canonical_method(method) != _canonical_method(model):
            raise ValueError("method and model specify different DDE formulations.")
        if model is None:
            model = method
    return "auto" if model is None else _canonical_method(model)


def _default_dimension_method(chosen: str, has_graph: bool, *, stacklevel: int = 3) -> str:
    """Path taken when ``dimension_method`` is omitted (the manuscript requires it)."""
    default = "csp" if chosen == "copula" and not has_graph else "fixed"
    warnings.warn(
        "dimension_method was not given; using "
        f"dimension_method={default!r} for model={MODEL_NAMES[chosen]!r}. "
        "Pass dimension_method='fixed' or 'csp' explicitly; omitting it "
        "will become an error.",
        FutureWarning,
        stacklevel=stacklevel,
    )
    return default


def _pop_iteration_limit(kwargs: dict, max_it: Optional[int]) -> Optional[int]:
    """Remove every iteration-limit spelling from ``kwargs`` and reconcile them."""
    supplied = [] if max_it is None else [("max_it", max_it)]
    for name in ITERATION_LIMIT_NAMES[1:]:
        if name in kwargs:
            value = kwargs.pop(name)
            if value is not None:
                supplied.append((name, value))
    values = {int(value) for _, value in supplied}
    if len(values) > 1:
        names = ", ".join(name for name, _ in supplied)
        raise ValueError(f"Conflicting iteration limits: {names}.")
    if not values:
        return None
    limit = values.pop()
    if limit < 1:
        raise ValueError("max_it must be a positive integer.")
    return limit


_RECORDED_ARGUMENTS = ("pen_list", "tau", "t_spike", "temp", "C", "tol")


def _fill_default_hyperparameters(kwargs, *, chosen, dimension, N, D, K_list, has_graph):
    """Fill the hyperparameters that were not given; return their names.

    The values are :func:`dde.utils.hyperparameter_selection.default_fit_hyperparameters`.
    A CopDDE fit on a restricting ``G_list`` keeps its own ``tol=0`` and
    ``temp=1`` defaults (the MATLAB ``get_SAEM_RL_G.m`` setting).
    """
    from .utils.hyperparameter_selection import default_fit_hyperparameters

    depth = D if D is not None else (len(K_list) if K_list is not None else None)
    if depth is None:
        return []  # the model reports the missing depth itself
    defaults = default_fit_hyperparameters(chosen, dimension, N, int(depth))
    if chosen == "copula" and has_graph:
        defaults.pop("tol", None)
        defaults.pop("temp", None)
    filled = []
    for name, value in defaults.items():
        if kwargs.get(name) is None:
            kwargs[name] = value
            filled.append(name)
    return sorted(filled)


def _record_active_structure(result: DDEResult) -> DDEResult:
    """Add the active latent columns and edges of every layer to the diagnostics."""
    if result.B_list:
        slopes = [np.asarray(B, dtype=float)[:, 1:] != 0 for B in result.B_list]
        result.diagnostics["active_by_layer"] = [int(s.any(axis=0).sum()) for s in slopes]
        result.diagnostics["n_active_edges"] = [int(s.sum()) for s in slopes]
    return result


def fit(
    Y,
    *,
    model: Optional[str] = None,
    D: Optional[int] = None,
    family: Optional[str] = None,
    dimension_method: Optional[str] = None,
    K_list=None,
    max_it: Optional[int] = None,
    variable_types: Optional[Sequence[str]] = None,
    method: Optional[str] = None,
    **kwargs: Any,
) -> DDEResult:
    """Fit a DDE-family model through a unified package interface.

    The arguments follow Table 1 of the package manuscript.

    Parameters
    ----------
    Y:
        ``N x J`` observed data matrix, observations in rows.
    model:
        ``"DDE"`` (parametric), ``"SemiDDE"`` or ``"CopDDE"``.  ``"auto"``
        uses SemiDDE for all-continuous data and CopDDE otherwise (see
        ``variable_types``).  The lower-case spellings ``"dde"``,
        ``"semi"`` and ``"copula"`` are accepted.
    D:
        Number of binary latent layers.  Fixed-width SemiDDE and CopDDE fits
        can infer it from ``K_list``, and fits on a fixed graph from
        ``K_list`` or ``G_list``.
    family:
        Observed-data family of the parametric DDE: ``"Normal"``,
        ``"Bernoulli"`` or ``"Poisson"``.  Required when ``model="DDE"``.
    dimension_method:
        ``"fixed"`` (widths given by ``K_list``, penalized SAEM) or
        ``"csp"`` (widths inferred by cumulative shrinkage, Bayesian SAEM).
        Omitting it is deprecated; the fit then uses ``"fixed"`` (CopDDE:
        ``"csp"``, or ``"fixed"`` with ``G_list``) and warns.
    K_list:
        Layer widths ``[K_1, ..., K_D]`` from the shallowest layer to the
        deepest.  Required for ``"fixed"``.  For ``"csp"`` it is optional and
        gives the maximal widths at which the fit starts; by default these
        are the maximal identifiable widths ``floor(J/3), floor(K_1/3), ...``.
    max_it:
        Maximum number of iterations of the fitting algorithm.  For SemiDDE
        fixed it bounds each inner SAEM fit; there
        ``diagnostics["iterations"]`` counts the sweeps of every stage and
        ``diagnostics["final_iterations"]`` those of the last fit.  The
        kernel spellings ``max_iter``, ``it`` and ``max_inner_iter`` are
        accepted as aliases.
    variable_types:
        Optional variable type labels used when ``model="auto"``.
    method:
        Deprecated spelling of ``model``.
    **kwargs:
        The remaining arguments of Table 1 and algorithm-specific options:

        * ``C`` -- latent draws per stochastic E-step (default 1);
        * ``tol`` -- convergence tolerance (see each kernel for its
          criterion);
        * ``tau`` -- truncation threshold of the coefficients;
        * ``pen_list`` -- length-``D`` truncated-lasso penalties
          (``"fixed"``);
        * ``t_spike``, ``temp`` -- length-``D`` CSP spike parameters and the
          initial annealing temperature (``"csp"``);
        * ``G_list`` (``"fixed"``) -- a user-specified graph: binary
          matrices ``G_list[0]`` of shape ``J x K_1`` and ``G_list[d]`` of
          shape ``K_d x K_{d+1}``; a shorter list fixes the shallowest
          layers only.  See :func:`dde.algorithms.dde.fit_parametric_dde` and
          :func:`dde.algorithms.copula.fit_copula_dde`.

        ``pen_list``, ``tau``, ``t_spike``, ``temp`` and the fixed-width
        CopDDE ``tol`` default to
        :func:`dde.utils.hyperparameter_selection.default_fit_hyperparameters`
        (``pen_list = N**(1/4)``, ``tau = 0.3`` or the CopDDE
        ``max(0.3, 3 N**-0.3)``, the default CSP ``t_spike`` and
        ``temp = 0.7`` for ``D <= 2``, else 0.9).

        Every fit draws from NumPy's global generator: ``np.random.seed``
        before the call reproduces it.  CSP and CopDDE fits also accept
        ``random_state``, which seeds their start only.

    Returns
    -------
    DDEResult
        ``model`` is ``"DDE"``, ``"SemiDDE"`` or ``"CopDDE"``;
        ``diagnostics["active_by_layer"]`` and
        ``diagnostics["n_active_edges"]`` count the nonzero slope columns and
        entries of every coefficient matrix.  When ``Y`` is a data frame,
        ``diagnostics["variable_names"]`` keeps its column names (the default
        labels of the observed variables in :mod:`dde.visualization`), and
        ``diagnostics["fit_arguments"]`` the hyperparameters given to the call
        (``pen_list``, ``tau``, ``t_spike``, ``temp``, ``C``, ``tol``).
    """
    column_names = data_column_names(Y)
    Y = np.asarray(Y)
    chosen = _resolve_model(model, method)
    if chosen == "auto":
        chosen = _infer_model(Y, variable_types)

    limit = _pop_iteration_limit(kwargs, max_it)
    if limit is not None:
        kwargs[_KERNEL_ITERATION_NAME[chosen]] = limit

    has_graph = kwargs.get("G_list") is not None
    if dimension_method is None:
        dimension = _default_dimension_method(chosen, has_graph)
    else:
        dimension = str(dimension_method).strip().lower()
    if dimension not in {"fixed", "csp"}:
        raise ValueError(
            f"{MODEL_NAMES[chosen]} dimension_method must be 'fixed' or 'csp'."
        )

    if D is None and has_graph:
        D = len(K_list) if K_list is not None else len(kwargs["G_list"])

    defaulted = _fill_default_hyperparameters(
        kwargs, chosen=chosen, dimension=dimension, N=Y.shape[0], D=D, K_list=K_list,
        has_graph=has_graph,
    )
    fit_arguments = {
        name: kwargs[name] for name in _RECORDED_ARGUMENTS if kwargs.get(name) is not None
    }

    if chosen == "dde":
        from .algorithms.dde import fit_parametric_dde

        if D is None:
            raise ValueError("D is required when model='DDE'.")
        if family is None:
            raise ValueError("family is required when model='DDE'.")
        family_name = str(family).strip().lower()
        result = fit_parametric_dde(
            Y,
            family=family_name,
            D=D,
            K_list=K_list,
            dimension_method=dimension,
            **kwargs,
        )
    elif chosen == "semi":
        from .algorithms.semi import fit_semidde

        result = fit_semidde(
            Y,
            D=D,
            K_list=K_list,
            dimension_method=dimension,
            **kwargs,
        )
    elif chosen == "copula":
        from .algorithms.copula import fit_copula_dde
        from .algorithms.csp import maximal_identifiable_widths

        if K_list is None and dimension == "csp":
            if D is None:
                raise ValueError("D is required for CopDDE CSP when K_list is omitted.")
            K_list = maximal_identifiable_widths(Y.shape[1], int(D))
        if D is not None and K_list is not None and len(K_list) != int(D):
            raise ValueError("K_list length must equal D.")
        result = fit_copula_dde(Y, K_list=K_list, dimension_method=dimension, **kwargs)
    else:
        raise RuntimeError(f"Unhandled DDE model: {chosen}.")
    result = record_column_names(_record_active_structure(result), column_names)
    result.diagnostics.setdefault("fit_arguments", fit_arguments)
    result.diagnostics.setdefault("default_arguments", defaulted)
    return result
