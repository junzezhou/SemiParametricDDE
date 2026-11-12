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
        _StochasticApproximation,
        _optimize_gaussian_layer_sa,
        _optimize_logistic_layer_sa,
        _threshold,
        _update_bernoulli_probabilities_D,
        _as_arrays,
        _validate_inputs,
    )
    from .semi_saem import _sample_latent_layers_parametric
except Exception:  # pragma: no cover - direct script/notebook import
    from get_SAEM_RL_CSP_D import (
        _make_bounds,
        _optimize_gaussian_layer,
        _optimize_logistic_layer,
        _prepare_rank_groups,
        _sample_latent_layers,
        _sample_rank_likelihood_Z,
        _StochasticApproximation,
        _optimize_gaussian_layer_sa,
        _optimize_logistic_layer_sa,
        _threshold,
        _update_bernoulli_probabilities_D,
        _as_arrays,
        _validate_inputs,
    )
    from semi_saem import _sample_latent_layers_parametric


LATENT_TRANSITIONS = ("synchronous_mean_field", "systematic_coordinate_gibbs")


def _resolve_support_list(support_list, B_list):
    """Per-layer boolean slope masks; ``None`` marks a layer whose graph is estimated."""
    D = len(B_list)
    if support_list is None:
        return [None] * D
    entries = list(support_list)
    if len(entries) > D:
        raise ValueError(f"support_list has {len(entries)} entries for D={D} layers.")
    entries += [None] * (D - len(entries))
    resolved = []
    for d, support in enumerate(entries):
        if support is None:
            resolved.append(None)
            continue
        support = np.asarray(support).astype(bool)
        expected = (B_list[d].shape[0], B_list[d].shape[1] - 1)
        if support.shape != expected:
            raise ValueError(f"support_list[{d}] shape {support.shape} must be {expected}.")
        resolved.append(support)
    return resolved


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
    support_list=None,
    latent_transition: str = "synchronous_mean_field",
    sa_burn_in=None,
):
    """D-layer rank-likelihood SAEM using fixed TLP penalties.

    The interface follows the SemiDDE style: ``B_list_in`` and ``A_list_in`` are
    lists ordered from the shallowest latent layer to the deepest.

    The iterations stop when the relative parameter change
    ``||theta_new - theta_old|| / ||theta_old||`` (prevalences, variances
    and coefficients) is at most ``tol``, or after ``it`` iterations.

    ``support_list`` gives the allowable connections of some layers
    (manuscript Section 5.1): ``support_list[d]`` is a boolean
    ``(rows of B_list[d]) x K_d`` mask of allowable slopes, or ``None`` for a
    layer without a graph.  The slopes outside the mask are held at zero by
    the M-step bounds; the allowable slopes get the TLP penalty and the
    ``tau`` threshold like those of a layer without a graph.  (The MATLAB
    ``get_SAEM_RL_G.m`` instead refits an unpenalized model on a given
    graph.)

    ``latent_transition`` selects the latent draws: ``"synchronous_mean_field"``
    (every coordinate drawn from its conditional at the previous state, the
    CopDDE kernels' rule) or ``"systematic_coordinate_gibbs"`` (one sequential
    sweep of the exact full conditionals per draw, deepest layer first as in
    ``get_SAEM_RL_G.m``).

    ``sa_burn_in=None`` fits every M-step to the current draw alone (the
    CopDDE code, whose averaging line is commented out).  An integer turns
    on the stochastic-approximation averaging of the manuscript (eq. 18,
    21): ``sa_burn_in`` unit steps, then the M-step objective and the
    top-layer probabilities are averaged with step size
    ``1 / (t - sa_burn_in)``; the stopping test starts after the burn-in.
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
    latent_transition = str(latent_transition).strip().lower()
    if latent_transition not in LATENT_TRANSITIONS:
        raise ValueError("latent_transition must be one of " + ", ".join(LATENT_TRANSITIONS) + ".")
    gibbs = latent_transition == "systematic_coordinate_gibbs"

    support_list = _resolve_support_list(support_list, B_list)
    for d, support in enumerate(support_list):
        if support is not None:
            B_list[d][:, 1:] *= support

    rank_groups = _prepare_rank_groups(R, Rlevels)
    lb, ub = _make_bounds(K, D)
    sa = None if sa_burn_in is None else _StochasticApproximation(J, K, sa_burn_in)
    A_sample_long = None
    ber = None
    ber_top = None
    it_num = 0
    err = np.inf

    while it_num < it and (abs(err) > tol or (sa is not None and it_num <= sa.burn_in)):
        if gibbs:
            samples, ber_top = _sample_latent_layers_parametric(
                Z, prop, B_list, gamma, [A.astype(np.uint8) for A in A_list], C,
                family="normal", temperature=temp,
            )
            A_sample_long = [sample.astype(float) for sample in samples]
            ber = None
        else:
            ber = _update_bernoulli_probabilities_D(Z, A_list, B_list, gamma, prop, temp)
            A_sample_long = _sample_latent_layers(ber, C)
            ber_top = ber[-1]

        mu = np.mean(A_sample_long[0], axis=2) @ B_list[0].T
        _sample_rank_likelihood_Z(Z, mu, gamma, temp, rank_groups)

        B_update = [None] * D
        if sa is not None:
            prop_update = sa.update(it_num + 1, Z, A_sample_long, ber_top)
            B_update[0], gamma_update = _optimize_gaussian_layer_sa(
                sa.gaussian, Z.shape[0], A_sample_long[0].shape[2], B_list[0], gamma,
                pen_list[0], temp, lb[0], ub[0], optimizer_maxiter, n_jobs, "tlp", tau,
                support=support_list[0],
            )
            for d in range(1, D):
                B_update[d] = _optimize_logistic_layer_sa(
                    sa.logistic[d - 1], B_list[d], pen_list[d], temp, lb[d], ub[d],
                    optimizer_maxiter, n_jobs, "tlp", tau, support=support_list[d],
                )
        else:
            prop_update = np.mean(ber_top, axis=0)
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
                support=support_list[0],
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
                    support=support_list[d],
                )

        for d in range(D):
            B_update[d][:, 1:] = _threshold(B_update[d][:, 1:], tau)

        # Relative parameter change ||theta_new - theta_old|| / ||theta_old||.
        change = np.sqrt(
            np.linalg.norm(prop - prop_update) ** 2
            + np.linalg.norm(gamma - gamma_update) ** 2
            + sum(np.linalg.norm(B_list[d] - B_update[d]) ** 2 for d in range(D))
        )
        size = np.sqrt(
            np.linalg.norm(prop) ** 2
            + np.linalg.norm(gamma) ** 2
            + sum(np.linalg.norm(B_list[d]) ** 2 for d in range(D))
        )
        err = change / max(size, 1e-12)

        if gibbs:
            # the chain continues from its last sweep
            A_list = [sample[:, 1:, -1].copy() for sample in A_sample_long]
        else:
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
        "top_probabilities": ber_top,
        "support_list": support_list,
        "latent_transition": latent_transition,
    }


__all__ = ["LATENT_TRANSITIONS", "get_SAEM_RL_D"]
