"""CSP dimension selection for semiparametric DDEs.

The SemiDDE observation layer is a latent Gaussian pseudo-observation
``X = g(Y)``.  Conditional on ``X``, its latent hierarchy is the same Gaussian
DDE used by the CopDDE CSP path.  The copula-structured CSP engine fits that
hierarchy and applies the SemiDDE update of ``g`` after every committed
iteration.  (The former MCEM path, which alternated whole Gaussian CSP-MCEM
fits with ``g`` updates, was removed; git tag ``csp-mcem-last``.)
"""

from __future__ import annotations

import numpy as np

from .normal_csp import fit_parametric_csp_copula
from .semi_saem import _semiparametric_g_step


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


__all__ = ["get_SAEM_semiparametric_csp_copula"]
