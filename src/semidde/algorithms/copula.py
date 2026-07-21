"""CopDDE workflow interface."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import norm, rankdata

from semidde.copula import simulate_gaussian_mixture
from semidde.results import DDEResult
from semidde.utils import validate_dde_lists
from semidde.utils.result_compression import compress_csp_result_components

from .get_SAEM_RL_CSP_D import get_SAEM_RL_CSP_D
from .get_SAEM_RL_D import get_SAEM_RL_D
from .normal_init_cop import Normal_init_cop


def prepare_rank_likelihood_inputs(X):
    """Create ``Z``, ``R``, and ``Rlevels`` for D-layer CopDDE fitting."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {X.shape}.")
    N, J = X.shape
    R = np.full((N, J), np.nan, dtype=float)
    Z = np.zeros((N, J), dtype=float)
    Rlevels = np.zeros(J, dtype=int)

    for j in range(J):
        xj = X[:, j]
        obs = ~np.isnan(xj)
        if not np.any(obs):
            Rlevels[j] = 0
            Z[:, j] = 0.0
            continue

        values, inverse = np.unique(xj[obs], return_inverse=True)
        Rlevels[j] = values.size
        R[obs, j] = inverse + 1

        ranks = rankdata(xj[obs], method="average")
        u = np.clip((ranks - 0.5) / np.sum(obs), 1e-6, 1.0 - 1e-6)
        Z[obs, j] = norm.ppf(u)
        if np.any(~obs):
            Z[~obs, j] = 0.0

    return Z, R, Rlevels


def normal_initialize_copula_dde(
    X,
    K_list: Sequence[int],
    *,
    Z=None,
    K1_max: Optional[int] = None,
    B_list_oracle=None,
    G_list_oracle=None,
    random_state: Optional[object] = None,
    epsilon: float = 1e-4,
):
    """Initialize D-layer CopDDE with mixture ``Z`` and MATLAB CopDDE rules."""
    if K1_max is not None and int(K1_max) != int(K_list[0]):
        raise ValueError("CopDDE Normal_init_D rules require K1_max == K_list[0].")
    if B_list_oracle is not None or G_list_oracle is not None:
        raise ValueError("CopDDE normal initialization does not use oracle B/G lists.")
    if Z is None:
        Z = simulate_gaussian_mixture(X, np.asarray(X).shape[1], random_state=random_state)

    prop, B_list, gamma, A_list = Normal_init_cop(Z, K_list, epsilon=epsilon)
    return {
        "prop": prop,
        "B_list": B_list,
        "gamma": gamma,
        "A_list": A_list,
        "used_oracle": False,
    }


def fit_copula_dde(
    X,
    *,
    K_list: Optional[Sequence[int]] = None,
    dimension_method: str = "csp",
    Z=None,
    R=None,
    Rlevels=None,
    prop_in=None,
    gamma_in=None,
    B_list_in=None,
    A_list_in=None,
    initialize: str = "data",
    random_state: Optional[object] = None,
    K1_max: Optional[int] = None,
    B_list_oracle=None,
    G_list_oracle=None,
    epsilon: float = 1e-4,
    **kwargs: Any,
) -> DDEResult:
    """Fit CopDDE through the D-layer rank-likelihood kernel."""
    X = np.asarray(X)
    verbose = bool(kwargs.get("verbose", True))
    if Z is None or R is None or Rlevels is None:
        _, R0, Rlevels0 = prepare_rank_likelihood_inputs(X)
        if Z is None:
            Z = simulate_gaussian_mixture(X, X.shape[1], random_state=random_state)
        if R is None:
            R = R0
        if Rlevels is None:
            Rlevels = Rlevels0

    if B_list_in is None or A_list_in is None or prop_in is None or gamma_in is None:
        if initialize is True:
            init_method = "data"
        elif initialize in {False, None}:
            init_method = "none"
        else:
            init_method = str(initialize).lower()
        if init_method == "normal":
            init_method = "data"
        if init_method != "data":
            raise ValueError(
                "initialize must be 'data'. Pass prop_in, gamma_in, B_list_in, "
                "and A_list_in to bypass initialization."
            )
        if K_list is None:
            raise ValueError("K_list is required when initialized arguments are incomplete.")
        if verbose:
            print("CopDDE initialization: triggered using initialize='data'.")
        init = normal_initialize_copula_dde(
            X,
            K_list,
            Z=Z,
            K1_max=K1_max,
            B_list_oracle=B_list_oracle,
            G_list_oracle=G_list_oracle,
            random_state=random_state,
            epsilon=epsilon,
        )
        prop_in = init["prop"] if prop_in is None else prop_in
        gamma_in = init["gamma"] if gamma_in is None else gamma_in
        B_list_in = init["B_list"] if B_list_in is None else B_list_in
        A_list_in = init["A_list"] if A_list_in is None else A_list_in
    else:
        if verbose:
            print("CopDDE initialization: not triggered; using supplied initialized arguments.")

    shape = validate_dde_lists(
        Z,
        B_list_in,
        A_list=A_list_in,
        prop=prop_in,
        gamma=gamma_in,
        K_list=K_list,
    )

    method = str(dimension_method).lower()
    common = dict(
        X=X,
        Z=Z,
        R=R,
        Rlevels=Rlevels,
        prop_in=prop_in,
        gamma_in=gamma_in,
        B_list_in=B_list_in,
        A_list_in=A_list_in,
    )

    if method == "fixed":
        required = ["C", "tol", "it", "temp", "pen_list", "tau"]
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise ValueError(
                "fit_copula_dde fixed mode needs initialized arguments and algorithm controls: "
                + ", ".join(missing)
            )
        raw = get_SAEM_RL_D(**common, **kwargs)
        diagnostics = {
            "iterations": raw["it_num"],
            "err": raw["err"],
            "p": raw["p"],
            "A_sample_long": raw["A_sample_long"],
            "K_list": shape.K_list,
        }
    elif method == "csp":
        required = ["C", "it", "t_spike", "temp", "tau"]
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise ValueError(
                "fit_copula_dde csp mode needs initialized arguments and algorithm controls: "
                + ", ".join(missing)
            )
        raw = get_SAEM_RL_CSP_D(**common, **kwargs)
        diagnostics = {
            "iterations": raw["it_num"],
            "num_active": raw["num_act"],
            "theta": raw["theta"],
            "theta_slab": raw["theta_slab"],
            "z_cusp": raw["z_cusp"],
            "omega": raw["omega"],
            "p": raw["p"],
            "A_sample_long": raw["A_sample_long"],
            "K_list": shape.K_list,
            "original_K_list": shape.K_list,
            "backend": "csp",
            "threshold_rule": "first_two_layers",
        }
        compressed = compress_csp_result_components(
            prop=raw["prop"],
            B_list=raw["B_list"],
            A_list=raw["A_list"],
            diagnostics=diagnostics,
            raw=raw,
        )
        raw = compressed["raw"]
        diagnostics = compressed["diagnostics"]
    else:
        raise ValueError("dimension_method must be 'fixed' or 'csp'.")

    return DDEResult(
        model="copula",
        prop=raw["prop"],
        B_list=raw["B_list"],
        gamma=raw["gamma"],
        A_list=raw["A_list"],
        transformed=raw["Z"],
        diagnostics=diagnostics,
        raw=raw,
    )


__all__ = [
    "Normal_init_cop",
    "fit_copula_dde",
    "normal_initialize_copula_dde",
    "prepare_rank_likelihood_inputs",
]
