"""D-layer rank-likelihood SAEM with fixed TLP penalties for DDE copulas."""

from __future__ import annotations

import numpy as np

try:
    from .get_SAEM_RL_CSP_D import (
        _make_bounds,
        _optimize_gaussian_layer,
        _optimize_logistic_layer,
        _prepare_rank_groups,
        _sample_latent_layers,
        _sample_rank_likelihood_Z,
        _threshold,
        _update_bernoulli_probabilities_D,
        _as_arrays,
        _validate_inputs,
    )
except Exception:  # pragma: no cover - direct script/notebook import
    from get_SAEM_RL_CSP_D import (
        _make_bounds,
        _optimize_gaussian_layer,
        _optimize_logistic_layer,
        _prepare_rank_groups,
        _sample_latent_layers,
        _sample_rank_likelihood_Z,
        _threshold,
        _update_bernoulli_probabilities_D,
        _as_arrays,
        _validate_inputs,
    )


def get_SAEM_RL_D(
    X,
    Z,
    R,
    Rlevels,
    prop_in,
    gamma_in,
    B_list_in,
    A_list_in,
    C,
    tol,
    it,
    temp,
    pen_list,
    tau,
    *,
    optimizer_maxiter: int = 50,
    n_jobs: int = 1,
    verbose: bool = True,
):
    """D-layer rank-likelihood SAEM using fixed TLP penalties.

    The interface follows the SemiDDE style: ``B_list_in`` and ``A_list_in`` are
    lists ordered from the shallowest latent layer to the deepest.
    """
    X = np.asarray(X)
    Z = np.asarray(Z, dtype=float).copy()
    R = np.asarray(R, dtype=float)
    Rlevels = np.asarray(Rlevels).astype(int).reshape(-1)
    prop = np.asarray(prop_in, dtype=float).reshape(-1).copy()
    gamma = np.asarray(gamma_in, dtype=float).reshape(-1).copy()
    B_list, A_list = _as_arrays(B_list_in, A_list_in)
    D, N, J, K = _validate_inputs(Z, prop, gamma, B_list, A_list)

    pen_list = np.asarray(pen_list, dtype=float).reshape(-1)
    if pen_list.size == 1:
        pen_list = np.repeat(pen_list.item(), D)
    if pen_list.size != D:
        raise ValueError("pen_list must be a scalar or a length-D sequence.")

    rank_groups = _prepare_rank_groups(R, Rlevels)
    lb, ub = _make_bounds(K, D)
    A_sample_long = None
    ber = None
    it_num = 0
    err = np.inf

    while it_num < it and abs(err) > tol:
        ber = _update_bernoulli_probabilities_D(Z, A_list, B_list, gamma, prop, temp)
        A_sample_long = _sample_latent_layers(ber, C)

        mu = np.mean(A_sample_long[0], axis=2) @ B_list[0].T
        _sample_rank_likelihood_Z(Z, mu, gamma, temp, rank_groups)

        prop_update = np.mean(ber[-1], axis=0)
        B_update = [None] * D
        B_update[0], gamma_update = _optimize_gaussian_layer(
            Z,
            A_sample_long[0],
            B_list[0],
            gamma,
            pen_list[0],
            temp,
            lb[0],
            ub[0],
            optimizer_maxiter,
            n_jobs,
            "tlp",
            tau,
        )
        for d in range(1, D):
            B_update[d] = _optimize_logistic_layer(
                A_sample_long[d - 1],
                A_sample_long[d],
                B_list[d],
                pen_list[d],
                temp,
                lb[d],
                ub[d],
                optimizer_maxiter,
                n_jobs,
                "tlp",
                tau,
            )

        for d in range(D):
            B_update[d][:, 1:] = _threshold(B_update[d][:, 1:], tau)

        err = np.sqrt(
            np.linalg.norm(prop - prop_update) ** 2
            + np.linalg.norm(gamma - gamma_update) ** 2
            + sum(np.linalg.norm(B_list[d] - B_update[d]) ** 2 for d in range(D))
        )

        A_list = [
            (np.mean(A_sample_long[d][:, 1:, :], axis=2) > 0.5).astype(float)
            for d in range(D)
        ]
        B_list = [B.copy() for B in B_update]
        gamma = gamma_update.copy()
        prop = prop_update.copy()
        it_num += 1

        if verbose:
            print(f"SAEM-RL-D Iteration {it_num}, Err {err:.4f}")

    return {
        "A_list": A_list,
        "B_list": B_list,
        "Z": Z,
        "gamma": gamma,
        "prop": prop,
        "err": float(err),
        "it_num": it_num,
        "A_sample_long": A_sample_long,
        "p": ber,
    }


__all__ = ["get_SAEM_RL_D"]

