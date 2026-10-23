"""CSP dimension selection for semiparametric DDEs.

The SemiDDE observation layer is a latent Gaussian pseudo-observation
``X = g(Y)``.  Conditional on ``X``, its latent hierarchy is the same Gaussian
DDE used by the CopDDE CSP path.  This module therefore alternates between a
Gaussian CSP-MCEM fit and the standard SemiDDE update of ``g``.
"""

from __future__ import annotations

import numpy as np

from dde.utils.result_compression import compress_csp_result_components

from .normal_csp import fit_parametric_csp_copula
from .parametric_csp_mcem import fit_parametric_csp_mcem
from .parametric_families import NORMAL_FAMILY
from .semi_saem import _semiparametric_g_step


def get_SAEM_semiparametric_csp(
    Y,
    X_in,
    prop_in,
    B_list_in,
    gamma_in,
    A_list_in,
    *,
    C,
    t_spike,
    temp,
    tau,
    tol=0.005,
    max_outer_iter=10,
    max_inner_iter=50,
    max_enum_bits=20,
    mc_samples=50000,
    g_update_rho=1.0,
    delta=None,
    random_state=None,
    optimizer_maxiter=100,
    optimizer_tol=1e-6,
    convergence_window=7,
    convergence_mean_tol=0.01,
    burn=5,
    bound_val=10.0,
    estep_row_chunk_size=2048,
    estep_coordinate_chunk_size=1,
    keep_trace=False,
    verbose=True,
):
    """Fit maximal-width SemiDDE with layer-wise cumulative shrinkage.

    Each outer iteration fits the Gaussian latent hierarchy at its maximal
    width.  CSP stick weights, slab scales, allocations, and the annealed
    temperature are carried across outer iterations, matching the continuous
    CSP state used while CopDDE updates rank-likelihood pseudo-observations.
    The active hierarchy is compressed temporarily before the semiparametric
    ``g`` update, so CSP-inactive binary coordinates cannot create spurious
    pseudo-observation classes.  Full-width parameters remain available to
    the next CSP iteration and are compressed only on return by the public
    wrapper.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X_in, dtype=float).copy()
    prop = np.asarray(prop_in, dtype=float).reshape(-1).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list_in]
    gamma = np.asarray(gamma_in, dtype=float).reshape(-1).copy()
    A_list = [np.asarray(A, dtype=np.uint8).copy() for A in A_list_in]

    if Y.ndim != 2 or X.shape != Y.shape:
        raise ValueError("Y and X_in must be 2D arrays with the same shape.")
    if not 0.0 < float(g_update_rho) <= 1.0:
        raise ValueError("g_update_rho must be in (0, 1].")
    max_outer_iter = int(max_outer_iter)
    max_inner_iter = int(max_inner_iter)
    if max_outer_iter < 1 or max_inner_iter < 1:
        raise ValueError("max_outer_iter and max_inner_iter must be positive integers.")

    N = Y.shape[0]
    if delta is None:
        delta = min(
            1.0 / (4.0 * (N ** 0.25) * np.sqrt(np.pi * np.log(N))),
            1e-12,
        )
    delta = float(delta)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    total_iterations = 0
    inner_iterations = []
    g_relative_history = []
    active_width_history = []
    outer_trace = [] if keep_trace else None
    raw = None
    csp_state = None

    for outer_iteration in range(1, max_outer_iter + 1):
        if verbose:
            print(
                f"\n===== N {N} | SemiDDE-CSP outer iteration "
                f"{outer_iteration} ====="
            )

        raw = fit_parametric_csp_mcem(
            X,
            prop,
            B_list,
            gamma,
            A_list,
            family=NORMAL_FAMILY,
            C=C,
            max_iter=max_inner_iter,
            t_spike=t_spike,
            temperature=temp,
            threshold=tau,
            convergence_tol=tol,
            convergence_mean_tol=convergence_mean_tol,
            convergence_window=convergence_window,
            burn=burn,
            optimizer_maxiter=optimizer_maxiter,
            optimizer_tol=optimizer_tol,
            bound_val=bound_val,
            estep_row_chunk_size=estep_row_chunk_size,
            estep_coordinate_chunk_size=estep_coordinate_chunk_size,
            random_state=rng,
            keep_trace=keep_trace,
            verbose=verbose,
            initial_csp_state=csp_state,
        )
        csp_state = raw
        prop = raw["prop"]
        B_list = raw["B_list"]
        gamma = raw["gamma"]
        A_list = raw["A_list"]
        total_iterations += int(raw["iterations"])
        inner_iterations.append(int(raw["iterations"]))

        # Match the CopDDE public path: infer the effective hierarchy from the
        # thresholded coefficient graph and retain at least one node per layer.
        active = compress_csp_result_components(
            prop=prop,
            B_list=B_list,
            A_list=A_list,
            diagnostics={"original_K_list": tuple(B.shape[1] - 1 for B in B_list)},
            min_active=True,
        )
        active_widths = tuple(active["diagnostics"]["K_list"])
        active_width_history.append(active_widths)

        X_next, g_new, g_relative = _semiparametric_g_step(
            Y,
            X,
            active["prop"],
            active["B_list"],
            active["A_list"][0],
            delta=delta,
            max_enum_bits=max_enum_bits,
            mc_samples=mc_samples,
            g_update_rho=g_update_rho,
            g_mode="pattern",
        )
        g_relative_history.append(g_relative)

        if keep_trace:
            outer_trace.append(
                {
                    "outer_iteration": outer_iteration,
                    "inner_iterations": int(raw["iterations"]),
                    "active_widths": active_widths,
                    "g_relative_change": g_relative,
                    "inner_trace": raw["trace"],
                }
            )
        if verbose:
            print(
                f"SemiDDE-CSP active={list(active_widths)}, "
                f"diff g={g_relative:.3e}"
            )

        X = X_next
        if g_relative < 1e-2:
            break

    # ``max_outer_iter`` is validated above, so at least one inner fit exists.
    raw = dict(raw)
    raw.update(
        {
            "X": X,
            "total_iterations": total_iterations,
            "outer_iterations": len(g_relative_history),
            "inner_iterations_by_outer": tuple(inner_iterations),
            "g_relative_history": tuple(g_relative_history),
            "active_width_history": tuple(active_width_history),
            "outer_trace": outer_trace,
        }
    )
    return raw


def get_SAEM_semiparametric_csp_copula(
    Y,
    X_in,
    prop_in,
    B_list_in,
    gamma_in,
    A_list_in,
    *,
    C,
    t_spike,
    temp,
    tau,
    max_iter=50,
    tol=0.005,
    g_update_every=1,
    g_estimator="precision",
    g_mode="auto",
    max_enum_bits=20,
    mc_samples=50000,
    delta=None,
    random_state=None,
    optimizer_maxiter=15,
    keep_trace=False,
    verbose=True,
    **engine_kwargs,
):
    """SemiDDE CSP on the copula-structured CSP engine.

    The Normal-family engine (:func:`fit_parametric_csp_copula`) fits the
    latent hierarchy on the pseudo-observations ``X = g(Y)``, and after every
    ``g_update_every``-th committed iteration the semiparametric transformation
    step re-estimates ``X`` from the active part of the current hierarchy.  This
    is the CopDDE algorithm with its rank-likelihood resampling of ``Z``
    replaced by the SemiDDE g step; the Normal CSP defaults (synchronous
    mean-field transition, N-scaled lower-layer penalty, FISTA lower layers)
    apply unless overridden through ``engine_kwargs``.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X_in, dtype=float).copy()
    if Y.ndim != 2 or X.shape != Y.shape:
        raise ValueError("Y and X_in must be 2D arrays with the same shape.")
    N = Y.shape[0]
    if delta is None:
        delta = min(1.0 / (4.0 * (N ** 0.25) * np.sqrt(np.pi * np.log(N))), 1e-12)
    delta = float(delta)
    g_history = []

    def g_update(X_current, state):
        masks = state["active_masks"]
        keep = [np.flatnonzero(m) if np.any(m) else np.array([0]) for m in masks]
        B_active = []
        for d, B in enumerate(state["B_list"]):
            rows = B if d == 0 else B[keep[d - 1], :]
            B_active.append(np.column_stack((rows[:, 0], rows[:, keep[d] + 1])))
        A1 = np.asarray(state["A_list"][0])[:, keep[0]]
        prop = np.asarray(state["prop"], dtype=float)[keep[-1]]
        X_next, _, relative = _semiparametric_g_step(
            Y, X_current, prop, B_active, A1, delta=delta, max_enum_bits=max_enum_bits,
            mc_samples=mc_samples, g_mode=g_mode, g_estimator=g_estimator, gamma=state["gamma"],
        )
        g_history.append(float(relative))
        return X_next

    settings = dict(
        latent_transition="synchronous_mean_field",
        penalty_scaling_power=1.0,
        mstep_solver="fista_lower",
    )
    settings.update(engine_kwargs)
    raw = fit_parametric_csp_copula(
        X, prop_in, gamma_in, B_list_in, A_list_in, family="normal", C=C, it=max_iter,
        t_spike=t_spike, temp=temp, tau=tau, optimizer_maxiter=optimizer_maxiter,
        convergence_std=tol, random_state=random_state, keep_trace=keep_trace,
        verbose=verbose, observed_update=g_update, observed_update_every=g_update_every,
        **settings,
    )
    raw = dict(raw)
    raw["g_relative_history"] = tuple(g_history)
    raw["g_estimator"] = g_estimator
    return raw


__all__ = ["get_SAEM_semiparametric_csp", "get_SAEM_semiparametric_csp_copula"]
