"""Package wrappers for SemiDDE and Normal-DDE estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from semidde.results import DDEResult

from .normal_init_semi import Normal_init_semi, rank_gaussianize_columns
from .semi_em import get_EM_est
from .semi_saem import get_SAEM_normal, get_SAEM_semiparametricv1


def fit_semidde(Y, *, K_list=None, dimension_method="fixed", **kwargs: Any) -> DDEResult:
    """Fit the SemiDDE estimator.

    If initialized parameters are incomplete, ``initialize`` selects the
    initialization path: ``"data"`` for observed-data spectral initialization
    or ``"oracle"`` for simulation-only true-structure alignment.
    """
    Y = np.asarray(Y, dtype=float)
    if dimension_method != "fixed":
        raise NotImplementedError("SemiDDE currently supports fixed K_list only.")

    init_keys = ["X_in", "prop_in", "B_list_in", "gamma_in", "A_list_in"]
    needs_initialization = any(name not in kwargs for name in init_keys)
    initialize = kwargs.pop("initialize", "data")
    legacy_mode = kwargs.pop("initialization_mode", kwargs.pop("init_mode", None))
    if legacy_mode is not None:
        initialize = legacy_mode

    if needs_initialization:
        if K_list is None:
            raise ValueError("K_list is required when initialized arguments are incomplete.")
        if initialize is True:
            initialization_mode = (
                "oracle"
                if "B_list_true" in kwargs or "G_list_true" in kwargs
                else "data"
            )
        elif initialize in {False, None}:
            initialization_mode = "none"
        else:
            initialization_mode = str(initialize).lower()
        epsilon = kwargs.pop("epsilon", 1e-4)
        X_init = rank_gaussianize_columns(Y)
        if initialization_mode == "oracle":
            print("SemiDDE initialization: triggered using initialize='oracle'.")
            B_list_true = kwargs.pop("B_list_true")
            G_list_true = kwargs.pop("G_list_true")
            prop, B_list, gamma, A_list, _ = Normal_init_semi(
                X_init,
                K_list[0],
                K_list,
                B_list_true,
                G_list_true,
                epsilon,
                mode="oracle",
            )
            if kwargs.get("final_refit", False):
                kwargs.setdefault("final_refit_B_list_true", B_list_true)
                kwargs.setdefault("final_refit_G_list_true", G_list_true)
        elif initialization_mode == "data":
            print("SemiDDE initialization: triggered using initialize='data'.")
            if kwargs.get("final_refit", False):
                raise ValueError("final_refit=True requires initialize='oracle'.")
            prop, B_list, gamma, A_list, _ = Normal_init_semi(
                X_init,
                K_list[0],
                K_list,
                epsilon=epsilon,
                mode="data",
            )
        elif initialization_mode == "none":
            print("SemiDDE initialization: not triggered; initialized arguments are incomplete.")
            pass
        else:
            raise ValueError("initialize must be 'data' or 'oracle'.")
        if initialization_mode in {"data", "oracle"}:
            kwargs.setdefault("X_in", X_init)
            kwargs.setdefault("prop_in", prop)
            kwargs.setdefault("B_list_in", B_list)
            kwargs.setdefault("gamma_in", gamma)
            kwargs.setdefault("A_list_in", A_list)
    else:
        print("SemiDDE initialization: not triggered; using supplied initialized arguments.")

    required = ["X_in", "prop_in", "B_list_in", "gamma_in", "pen_list", "tau", "A_list_in", "tol", "C"]
    missing = [name for name in required if name not in kwargs]
    if missing:
        raise ValueError(
            "fit_semidde needs initialized algorithm arguments for now: "
            + ", ".join(missing)
        )

    X, prop, B_list, gamma, A_list, itera = get_SAEM_semiparametricv1(Y=Y, **kwargs)
    return DDEResult(
        model="semi",
        prop=prop,
        B_list=B_list,
        gamma=gamma,
        A_list=A_list,
        transformed=X,
        diagnostics={"iterations": itera},
        raw=(X, prop, B_list, gamma, A_list, itera),
    )


__all__ = [
    "Normal_init_semi",
    "fit_semidde",
    "get_EM_est",
    "get_SAEM_normal",
    "get_SAEM_semiparametricv1",
    "rank_gaussianize_columns",
]
