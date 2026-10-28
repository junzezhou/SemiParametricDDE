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
        _bounded_minimize_value_grad,
        _maybe_parallel_map,
    )
    from .parametric_families import _solve_normal_on_support
    from .semi_saem import _sample_latent_layers_parametric
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
        _bounded_minimize_value_grad,
        _maybe_parallel_map,
    )
    from parametric_families import _solve_normal_on_support
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


def _gaussian_layer_on_support(Z, A_sample, B1, support, lb, ub, maxiter):
    """Unpenalized observed-layer M-step restricted to the graph.

    Each row is the least-squares fit on its intercept and graph parents (the
    objective's ``temp / gamma_j`` factor does not move the minimizer); rows
    whose solution leaves the box bounds are refitted by bounded L-BFGS-B.
    The variance update is the one of ``_optimize_gaussian_layer``.
    """
    N, K1_plus_1, C = A_sample.shape
    X_design = np.transpose(A_sample, (0, 2, 1)).reshape(N * C, K1_plus_1)
    Y = np.repeat(Z, C, axis=0)
    Q = X_design.T @ X_design
    q_all = (X_design.T @ Y).T
    B_update = _solve_normal_on_support(Q, q_all, support)
    outside = np.any((B_update < lb[None, :]) | (B_update > ub[None, :]), axis=1)
    for j in np.flatnonzero(outside):
        columns = np.concatenate(([0], 1 + np.flatnonzero(support[j])))
        Q_sub = Q[np.ix_(columns, columns)]
        q_sub = q_all[j, columns]

        def value_grad(x):
            Qx = Q_sub @ x
            return float(0.5 * x @ Qx - q_sub @ x), Qx - q_sub

        x0 = np.clip(B1[j, columns], lb[columns], ub[columns])
        B_update[j] = 0.0
        B_update[j, columns] = _bounded_minimize_value_grad(
            value_grad, x0, lb[columns], ub[columns], maxiter=maxiter
        )
    eta = np.tensordot(A_sample, B_update, axes=([1], [1]))
    sse = np.sum((Z[:, None, :] - eta) ** 2, axis=(0, 1))
    gamma_update = (1.0 + 0.5 * sse) / (1.0 + 0.5 * N * C + 1.0)
    return B_update, gamma_update


def _logistic_layer_on_support(A_child_sample, A_parent_sample, B_d, support, temp, lb, ub, maxiter, n_jobs):
    """Unpenalized latent-layer M-step restricted to the graph.

    The ``N * C`` design rows take at most ``2**K_parent`` distinct values,
    so each row's logistic likelihood is evaluated on the aggregated counts.
    """
    N, K_parent_plus_1, C = A_parent_sample.shape
    X_design = np.transpose(A_parent_sample, (0, 2, 1)).reshape(N * C, K_parent_plus_1)
    patterns, inverse, counts = np.unique(X_design, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    counts = counts.astype(float)
    K_child = A_child_sample.shape[1] - 1
    scale = temp / C

    def solve_row(k):
        columns = np.concatenate(([0], 1 + np.flatnonzero(support[k])))
        design = patterns[:, columns]
        ones = np.bincount(inverse, weights=A_child_sample[:, k + 1, :].reshape(N * C), minlength=patterns.shape[0])

        def value_grad(x):
            eta = design @ x
            value = scale * np.sum(counts * np.logaddexp(0.0, eta) - ones * eta)
            grad = scale * (design.T @ (counts / (1.0 + np.exp(-eta)) - ones))
            return float(value), grad

        row = np.zeros(K_parent_plus_1)
        row[columns] = _bounded_minimize_value_grad(
            value_grad, np.clip(B_d[k, columns], lb[columns], ub[columns]), lb[columns], ub[columns], maxiter=maxiter
        )
        return row

    return np.vstack(_maybe_parallel_map(solve_row, range(K_child), n_jobs))


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
):
    """D-layer rank-likelihood SAEM using fixed TLP penalties.

    The interface follows the SemiDDE style: ``B_list_in`` and ``A_list_in`` are
    lists ordered from the shallowest latent layer to the deepest.

    ``support_list`` fixes the graph of some layers: ``support_list[d]`` is a
    boolean ``(rows of B_list[d]) x K_d`` mask of allowed slopes, or ``None``
    for a layer whose graph is estimated.  A fixed layer is fitted on its
    graph without the TLP penalty or the ``tau`` threshold (its slopes off the
    graph stay zero), as in the MATLAB ``get_SAEM_RL_G.m``, which refits an
    unpenalized model on a given graph.

    ``latent_transition`` selects the latent draws: ``"synchronous_mean_field"``
    (every coordinate drawn from its conditional at the previous state, the
    CopDDE kernels' rule) or ``"systematic_coordinate_gibbs"`` (one sequential
    sweep of the exact full conditionals per draw, deepest layer first as in
    ``get_SAEM_RL_G.m``).
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
    A_sample_long = None
    ber = None
    ber_top = None
    it_num = 0
    err = np.inf

    while it_num < it and abs(err) > tol:
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

        prop_update = np.mean(ber_top, axis=0)
        B_update = [None] * D
        if support_list[0] is not None:
            B_update[0], gamma_update = _gaussian_layer_on_support(
                Z, A_sample_long[0], B_list[0], support_list[0], lb[0], ub[0], optimizer_maxiter
            )
        else:
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
            if support_list[d] is not None:
                B_update[d] = _logistic_layer_on_support(
                    A_sample_long[d - 1], A_sample_long[d], B_list[d], support_list[d],
                    temp, lb[d], ub[d], optimizer_maxiter, n_jobs,
                )
                continue
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
            if support_list[d] is None:
                B_update[d][:, 1:] = _threshold(B_update[d][:, 1:], tau)

        err = np.sqrt(
            np.linalg.norm(prop - prop_update) ** 2
            + np.linalg.norm(gamma - gamma_update) ** 2
            + sum(np.linalg.norm(B_list[d] - B_update[d]) ** 2 for d in range(D))
        )

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
