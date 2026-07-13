"""D-layer simulation helpers for DDE copulas."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import expit
from scipy.stats import gamma as gamma_dist
from scipy.stats import poisson, rankdata

from semidde.utils import validate_dde_lists


def _as_rng(random_state=None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_generator_inputs(N, prop_true, gamma, B_list):
    B_list = [np.asarray(B, dtype=float) for B in B_list]
    prop_true = np.asarray(prop_true, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    J = B_list[0].shape[0]
    validate_dde_lists(
        np.zeros((int(N), J)),
        B_list,
        prop=prop_true,
        gamma=gamma,
    )
    return int(N), prop_true, gamma, B_list


def simulate_latent_layers_D(N, prop_true, B_list, random_state=None):
    """Generate D binary latent layers from deepest to shallowest.

    ``B_list`` follows the package convention: ``B_list[0]`` maps ``A_1`` to
    the latent Gaussian observation layer, while ``B_list[d]`` for ``d >= 1``
    maps ``A_{d+1}`` to ``A_d``.
    """
    rng = _as_rng(random_state)
    B_list = [np.asarray(B, dtype=float) for B in B_list]
    prop_true = np.asarray(prop_true, dtype=float).reshape(-1)
    D = len(B_list)
    K_list = [B_list[0].shape[1] - 1] + [B_list[d].shape[1] - 1 for d in range(1, D)]
    if prop_true.shape != (K_list[-1],):
        raise ValueError(f"prop_true shape {prop_true.shape} must be ({K_list[-1]},).")

    A_list = [None] * D
    prob_list = [None] * D

    prob_list[-1] = np.broadcast_to(prop_true.reshape(1, -1), (N, K_list[-1])).copy()
    A_list[-1] = rng.binomial(1, prob_list[-1]).astype(float)

    for d in range(D - 2, -1, -1):
        parent_design = np.column_stack((np.ones(N), A_list[d + 1]))
        prob_list[d] = expit(parent_design @ B_list[d + 1].T)
        A_list[d] = rng.binomial(1, prob_list[d]).astype(float)

    return A_list, prob_list


def generate_X_Cop_marg_emp_D(
    N: int,
    prop_true,
    lambdas,
    gamma,
    B_list,
    random_state: Optional[object] = None,
):
    """Generate D-layer DDE-copula data with the empirical marginal transform.

    Returns a dictionary containing ``Z``, ``X``, ``A_list``, ``prob_list``,
    ``Q``, ``prop_true``, ``gamma``, and ``B_list``.
    """
    rng = _as_rng(random_state)
    N, prop_true, gamma, B_list = _validate_generator_inputs(N, prop_true, gamma, B_list)
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
    J = B_list[0].shape[0]
    if lambdas.shape != (J,):
        raise ValueError(f"lambdas shape {lambdas.shape} must be ({J},).")

    A_list, prob_list = simulate_latent_layers_D(N, prop_true, B_list, rng)
    A1_design = np.column_stack((np.ones(N), A_list[0]))
    mu = A1_design @ B_list[0].T
    Z = np.sqrt(gamma.reshape(1, -1)) * rng.standard_normal(size=mu.shape) + mu

    Q = np.zeros((N, J), dtype=float)
    for j in range(J):
        ranks = rankdata(Z[:, j], method="average")
        Q[:, j] = ranks / (N + 1.0)

    X = np.zeros((N, J), dtype=float)
    for j in range(J):
        modtype = j % 3
        if modtype == 0:
            pi0 = 0.3
            qj = Q[:, j]
            idx_nonzero = qj >= pi0
            if np.any(idx_nonzero):
                u_adj = (qj[idx_nonzero] - pi0) / (1.0 - pi0)
                X[idx_nonzero, j] = poisson.ppf(u_adj, lambdas[j])
        elif modtype == 1:
            X[:, j] = np.round(gamma_dist.ppf(Q[:, j], a=2.0, scale=lambdas[j]))
        else:
            X[:, j] = poisson.ppf(Q[:, j], lambdas[j])

    return {
        "Z": Z,
        "X": X,
        "A_list": A_list,
        "prob_list": prob_list,
        "Q": Q,
        "prop_true": prop_true,
        "gamma": gamma,
        "B_list": [B.copy() for B in B_list],
    }


__all__ = ["generate_X_Cop_marg_emp_D", "simulate_latent_layers_D"]

