"""Package wrappers for DDE-copula estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from semidde.results import DDEResult

from .copula_D import fit_copula_dde_D
from .normal_init_cop import Normal_init_cop


def fit_copula_dde(Y, *, K_list=None, dimension_method="csp", **kwargs: Any) -> DDEResult:
    """Fit the CopDDE rank-likelihood estimator through the D-layer kernel."""
    Y = np.asarray(Y)

    legacy_keys = {"B1_in", "B2_in", "A1_in", "A2_in"}
    if any(name in kwargs for name in legacy_keys):
        raise ValueError(
            "The legacy two-layer B1/B2 CopDDE interface has been moved out of "
            "the semidde package. Use B_list_in/A_list_in with the D-layer kernel."
        )

    return fit_copula_dde_D(Y, K_list=K_list, dimension_method=dimension_method, **kwargs)


__all__ = [
    "Normal_init_cop",
    "fit_copula_dde",
]
