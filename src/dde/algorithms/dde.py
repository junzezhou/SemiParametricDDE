"""Parametric Deep Discrete Encoder fitting wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dde.results import DDEResult
from dde.utils import validate_K_list_identifiability, validate_dde_lists
from dde.utils.result_compression import compress_csp_result_components

from .csp import maximal_identifiable_widths
from .normal_init_semi import Normal_init_semi
from .normal_csp import fit_normal_csp_mcem, fit_normal_csp_saem
from .semi_saem import get_SAEM_normal


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


def fit_dde_normal(
    Y,
    *,
    D: int,
    K_list=None,
    dimension_method: str = "fixed",
    pen_list=None,
    t_spike=None,
    temp=None,
    tau: float,
    C: int,
    tol: float,
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
    width_stability_window: int = 10,
    csp_min_iter: int = 30,
    csp_backend: str = "mcem",
    burn: int = 5,
    estep_row_chunk_size: int = 2048,
    keep_trace: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> DDEResult:
    """Fit a fixed-width or CSP-shrunk parametric Normal DDE.

    The observed matrix is passed directly to the Normal initialization and
    either the fixed-width SAEM kernel or a CSP MCEM/SAEM kernel.  In
    contrast, SemiDDE and CopDDE construct latent-Gaussian pseudo-observations
    before using their respective kernels.

    ``initialize="oracle"`` is intended only for simulation studies that need
    estimates aligned with known generating parameters.  Ordinary data fits
    should use the default ``initialize="data"``.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected Normal DDE argument(s): {unknown}.")

    X = np.asarray(Y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Y must be a two-dimensional matrix; got shape {X.shape}.")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("Y must contain at least one observation and one variable.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Normal DDE requires finite values in Y.")

    D = int(D)
    if D < 1:
        raise ValueError("D must be a positive integer.")
    method = str(dimension_method).strip().lower()
    if method == "fixed":
        if K_list is None:
            raise ValueError("K_list is required when dimension_method='fixed'.")
        if t_spike is not None or temp is not None:
            raise ValueError("t_spike and temp are only used by dimension_method='csp'.")
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
                "Normal DDE with dimension_method='csp' requires t_spike and temp."
            )
        K_tuple = maximal_identifiable_widths(X.shape[1], D)
        penalties = None
    else:
        raise ValueError("Normal DDE dimension_method must be 'fixed' or 'csp'.")

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

    supplied = [prop_in, B_list_in, gamma_in, A_list_in]
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError(
            "prop_in, B_list_in, gamma_in, and A_list_in must be supplied together."
        )

    if all(value is not None for value in supplied):
        initialization = "provided"
        validate_dde_lists(
            X,
            B_list_in,
            A_list=A_list_in,
            prop=prop_in,
            gamma=gamma_in,
            K_list=K_tuple,
            observed_name="Y",
            require_binary_A=True,
        )
        prop = np.asarray(prop_in, dtype=float)
        B_list = [np.asarray(B, dtype=float) for B in B_list_in]
        gamma = np.asarray(gamma_in, dtype=float)
        A_list = [np.asarray(A, dtype=float) for A in A_list_in]
    else:
        initialization = str(initialize).lower()
        if initialization == "data":
            if B_list_true is not None or G_list_true is not None:
                raise ValueError(
                    "B_list_true and G_list_true are only used with initialize='oracle'."
                )
            prop, B_list, gamma, A_list, _ = Normal_init_semi(
                X,
                K_tuple[0],
                K_tuple,
                epsilon=epsilon,
                mode="data",
            )
        elif initialization == "oracle":
            if B_list_true is None or G_list_true is None:
                raise ValueError(
                    "initialize='oracle' requires B_list_true and G_list_true."
                )
            prop, B_list, gamma, A_list, _ = Normal_init_semi(
                X,
                K_tuple[0],
                K_tuple,
                B_list_true,
                G_list_true,
                epsilon,
                mode="oracle",
            )
        else:
            raise ValueError("initialize must be 'data' or 'oracle'.")

    if method == "fixed":
        if random_state is not None:
            raise ValueError("random_state is currently supported by the CSP path only.")
        prop, B_list, gamma, A_list, iterations = get_SAEM_normal(
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
        )

        X_result = X.copy()
        return DDEResult(
            model="DDE",
            K_list=list(K_tuple),
            prop=prop,
            B_list=B_list,
            gamma=gamma,
            A_list=A_list,
            transformed=X_result,
            diagnostics={
                "iterations": iterations,
                "D": D,
                "K_list": list(K_tuple),
                "family": "Normal",
                "dimension_method": "fixed",
                "initialization": initialization,
            },
            raw=(X_result, prop, B_list, gamma, A_list, iterations),
        )

    backend = str(csp_backend).strip().lower()
    if backend not in {"saem", "mcem"}:
        raise ValueError("csp_backend must be 'saem' or 'mcem'.")
    csp_kernel = (
        fit_normal_csp_saem if backend == "saem" else fit_normal_csp_mcem
    )
    csp_options = {}
    if backend == "saem":
        csp_options.update(
            convergence_mean_tol=convergence_mean_tol,
            width_stability_window=width_stability_window,
            min_iter=csp_min_iter,
        )
    raw = csp_kernel(
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
        random_state=random_state,
        keep_trace=keep_trace,
        verbose=bool(verbose),
        **csp_options,
    )

    X_result = X.copy()
    diagnostics = {
        "iterations": raw["iterations"],
        "D": D,
        "K_list": tuple(raw["num_active"].tolist()),
        "K_max_list": K_tuple,
        "original_K_list": K_tuple,
        "family": "Normal",
        "dimension_method": "csp",
        "initialization": initialization,
        "backend": "saem_csp" if backend == "saem" else "approximate_mcem_csp",
        "threshold_rule": "first_two_layers",
        "num_active": raw["num_active"].copy(),
        "allocation": [value.copy() for value in raw["allocation"]],
        "omega": [value.copy() for value in raw["omega"]],
        "slab_scale": [value.copy() for value in raw["slab_scale"]],
        "relative_history": raw["relative_history"],
        "temperature": raw["temperature"],
    }
    if backend == "saem":
        diagnostics.update(
            {
                "sa_iterations": raw["sa_iterations"],
                "activation_permutations": raw[
                    "activation_permutations"
                ],
                "csp_start_iteration": raw["csp_start_iteration"],
                "stopped_by_convergence": raw["stopped_by_convergence"],
                "sa_relative_history": raw["sa_relative_history"],
                "width_history": raw["width_history"],
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


__all__ = ["fit_dde_normal"]
