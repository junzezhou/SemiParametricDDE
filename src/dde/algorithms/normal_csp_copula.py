"""Normal DDE CSP kernel ported from the CopDDE D-layer CSP algorithm.

This module adapts ``DDE-Copula-m/Algorithms/get_SAEM_RL_CSP_D.m`` (and its
package port :mod:`dde.algorithms.get_SAEM_RL_CSP_D`) to the parametric Normal
DDE.  The copula model treats the latent Gaussian matrix ``Z`` as unobserved
and resamples it from the rank likelihood at every iteration; the Normal DDE
observes ``Z = X`` directly, so that step is the only part of the algorithm
that is removed.  Every other component is reused unchanged:

1. synchronous mean-field conditional probabilities for all latent layers,
   computed from the previous latent point estimates;
2. ``C`` Bernoulli draws per layer, with the majority vote as the next state;
3. the cumulative-shrinkage-process (CUSP) MAP allocation, stick-breaking,
   and slab-scale update from :func:`dde.algorithms.csp.update_csp_layer`,
   swept from the deepest layer toward the data with ``num_rows`` equal to
   the previous iteration's active child count;
4. bounded L-BFGS-B weighted-L1 M-steps with a short iteration budget
   (``optimizer_maxiter=15`` as in the MATLAB ``optimset('MaxIter', 15)``)
   for the logistic layers; by default (``mstep_solver="hybrid"``) the
   observed Gaussian layer is instead solved by the Normal family's exact
   proximal (FISTA) weighted-lasso solver with ``fista_maxiter=300``
   iterations.  The inexact L-BFGS-B step cannot move loading mass between
   two identical latent columns (the likelihood is flat along that
   direction), which leaves duplicate columns of true factors active; the
   proximal solver resolves them.  ``mstep_solver="lbfgsb"`` restores the
   CopDDE behaviour in every layer;
5. the Gamma(1, 1) MAP update of the conditional variances;
6. hard thresholding of the shallow layers, hierarchical row gating of
   inactive children, the linear temperature schedule, and the rolling
   relative-Frobenius-change stopping rule.

The kernel keeps the ``K_list`` / ``B_list`` / ``A_list`` convention:
``B_list[0]`` has shape ``(J, K_1 + 1)`` and ``B_list[d]`` for ``d >= 1`` has
shape ``(K_d, K_{d+1} + 1)``.
"""

from __future__ import annotations

import numpy as np

from .csp import csp_active_mask, update_csp_layer
from .get_SAEM_RL_CSP_D import (
    _make_bounds,
    _optimize_gaussian_layer,
    _optimize_logistic_layer,
    _threshold,
    _update_bernoulli_probabilities_D,
)
from .normal_csp import _solve_logistic_weighted_l1
from .parametric_families import _solve_gaussian_weighted_l1


_THRESHOLD_RULES = ("first_two", "all_but_last", "none")
_MSTEP_SOLVERS = ("lbfgsb", "fista", "hybrid", "fista_lower")
_MSTEP_LATENTS = ("draw", "mean_field")
_PENALTY_SCALING_LAYERS = ("all", "deep", "lower")
_DEFAULT_OPTIMIZER_MAXITER = 15
_DEFAULT_FISTA_MAXITER = 300


def _resolve_random(random_state):
    """Return the random source used by the kernel.

    ``None`` selects NumPy's legacy global RNG so that, after
    ``np.random.seed(seed)``, the kernel consumes the same random stream as
    the package CopDDE kernel.  Any other value is converted to a
    :class:`numpy.random.Generator`.
    """
    if random_state is None:
        return np.random
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _sample_latent_layers(ber, C, random):
    """Draw ``C`` Bernoulli samples per layer with a leading intercept column."""
    N = ber[0].shape[0]
    samples = []
    for probs in ber:
        Kd = probs.shape[1]
        layer = np.zeros((N, Kd + 1, C), dtype=float)
        layer[:, 0, :] = 1.0
        for c in range(C):
            layer[:, 1:, c] = (random.random((N, Kd)) < probs).astype(float)
        samples.append(layer)
    return samples


def _validate(X, prop, gamma, B_list, A_list):
    D = len(B_list)
    if len(A_list) != D or D < 1:
        raise ValueError("B_list and A_list must be nonempty and of equal length.")
    N, J = X.shape
    K = [A.shape[1] for A in A_list]
    if B_list[0].shape != (J, K[0] + 1):
        raise ValueError(
            f"B_list[0] shape {B_list[0].shape} must be {(J, K[0] + 1)}."
        )
    for d in range(1, D):
        expected = (K[d - 1], K[d] + 1)
        if B_list[d].shape != expected:
            raise ValueError(
                f"B_list[{d}] shape {B_list[d].shape} must be {expected}."
            )
    if prop.shape != (K[-1],):
        raise ValueError("prop length must match the deepest latent width.")
    if gamma.shape != (J,):
        raise ValueError("gamma length must match the observed dimension J.")
    for d, A in enumerate(A_list):
        if A.shape != (N, K[d]):
            raise ValueError(f"A_list[{d}] shape {A.shape} must be {(N, K[d])}.")
    return D, N, J, K


def fit_normal_csp_copula(
    X,
    prop_in,
    gamma_in,
    B_list_in,
    A_list_in,
    *,
    C: int,
    it: int,
    t_spike,
    temp: float,
    tau: float,
    optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    optimizer_tol: float = 1e-6,
    bound_val: float = 10.0,
    n_jobs: int = 1,
    convergence_window: int = 7,
    convergence_std: float = 0.005,
    burn: int = 5,
    threshold_rule: str = "first_two",
    mstep_solver: str = "fista_lower",
    mstep_latent: str = "draw",
    observed_noise_scale: float = 0.0,
    fista_maxiter=None,
    slab_shape: float = 1.0,
    slab_rate: float = 5.0,
    penalty_scaling_power: float = 1.0,
    penalty_scaling_reference: float = 500.0,
    penalty_scaling_layers: str = "lower",
    penalty_scaling_allocation: bool = False,
    random_state=None,
    keep_trace: bool = False,
    verbose: bool = True,
):
    """Fit a maximal-width Normal DDE with the CopDDE CSP algorithm.

    Parameters
    ----------
    X : (N, J) observed Gaussian matrix (plays the role of ``Z``).
    prop_in, gamma_in, B_list_in, A_list_in : initial parameters and latent
        point estimates at the maximal widths.
    C : Monte Carlo draws per iteration.
    it : maximum number of outer iterations.
    t_spike : scalar or length-``D`` Laplace spike scales.
    temp : initial annealing temperature in ``(0, 1]``.
    tau : hard-threshold level for the shallow coefficient layers.
    optimizer_maxiter : L-BFGS-B iteration budget per coefficient row
        (``15`` reproduces the MATLAB ``fmincon`` budget).
    threshold_rule : ``"first_two"`` (package CopDDE rule), ``"all_but_last"``
        (MATLAB D-layer rule), or ``"none"``.
    mstep_solver : ``"fista_lower"`` (default: exact proximal FISTA
        weighted-lasso solves in layers 1..D-1 and the CopDDE L-BFGS-B step
        in the deepest layer), ``"hybrid"`` (FISTA for the observed Gaussian
        layer only), ``"lbfgsb"`` (CopDDE: L-BFGS-B in every layer),
        ``"fista"`` (FISTA in every layer; collapses the deepest layer at
        large ``N``), or a length-``D`` sequence of ``"fista"`` /
        ``"lbfgsb"`` giving the solver per layer.
    fista_maxiter : iteration budget of the FISTA solves; ``None`` selects
        ``300``.  Budgets below about 100 leave duplicate columns at large
        ``N`` because the per-iteration transfer of loading mass scales with
        ``gamma / N``.
    mstep_latent : ``"draw"`` (CopDDE: the ``C`` Bernoulli draws form the
        M-step design) or ``"mean_field"`` (the conditional probabilities
        replace the draws in the M-step design and responses; the committed
        latent state is still the majority vote of the draws).
    observed_noise_scale : nonnegative multiplier ``s``.  When positive, the
        observed-layer M-step sees ``X + s * sqrt(gamma / temp) * eps`` with
        standard Gaussian ``eps`` drawn each iteration, an SAEM-style
        surrogate for the CopDDE rank-likelihood resampling of ``Z``.  The
        conditional variances are still updated from the clean ``X``.
    slab_shape, slab_rate : Gamma hyperprior on the slab Laplace rate (CopDDE:
        1 and 5); see :func:`dde.algorithms.csp.update_csp_layer`.
    penalty_scaling_power, penalty_scaling_reference : the weighted-L1
        penalty of every M-step is multiplied by
        ``c_N = (N / penalty_scaling_reference) ** penalty_scaling_power``
        (power 0, the CopDDE default, leaves the O(1) prior against the O(N)
        likelihood; power 1, the default, makes the penalty proportional to
        N, equivalent to the CopDDE penalty on the average likelihood; power
        0.5 is the ``sqrt(N)`` rate).  ``penalty_scaling_power`` may be a
        length-``D`` sequence of per-layer powers.  The scaled penalty needs
        the proximal solver: the L-BFGS-B step stalls under large L1 weights.
    penalty_scaling_layers : ``"lower"`` (default: layers 1..D-1, leaving
        the deepest layer unscaled), ``"all"``, or ``"deep"`` (layers 2..D
        only); with a per-layer power sequence the excluded layers' powers
        are zeroed.
    penalty_scaling_allocation : when ``True`` the CUSP allocation evaluates
        the spike and slab densities on the scaled coefficients ``c_N * B``,
        which is the allocation implied by the scaled prior
        Laplace(0, scale / c_N); when ``False`` only the M-step penalty is
        scaled (a power prior) and the allocation sees the raw coefficients.
    random_state : ``None`` for the legacy global RNG, else a seed/Generator.

    Returns
    -------
    dict with the fitted ``prop``, ``B_list``, ``gamma``, ``A_list``, CUSP
    state (``allocation``, ``omega``, ``slab_scale``, ``penalty_scale``),
    ``active_masks``, ``num_active``, ``iterations`` and diagnostics.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix.")
    prop = np.asarray(prop_in, dtype=float).reshape(-1).copy()
    gamma = np.asarray(gamma_in, dtype=float).reshape(-1).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list_in]
    A_list = [np.asarray(A, dtype=float).copy() for A in A_list_in]
    D, N, J, K = _validate(X, prop, gamma, B_list, A_list)

    C = int(C)
    it = int(it)
    optimizer_maxiter = int(optimizer_maxiter)
    convergence_window = int(convergence_window)
    burn = int(burn)
    temp = float(temp)
    tau = float(tau)
    convergence_std = float(convergence_std)
    if C < 1 or it < 1 or optimizer_maxiter < 1 or convergence_window < 1:
        raise ValueError(
            "C, it, optimizer_maxiter, and convergence_window must be positive."
        )
    if burn < 0:
        raise ValueError("burn must be nonnegative.")
    if not np.isfinite(temp) or not 0.0 < temp <= 1.0:
        raise ValueError("temp must lie in (0, 1].")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative.")
    if not np.isfinite(convergence_std) or convergence_std < 0:
        raise ValueError("convergence_std must be finite and nonnegative.")
    threshold_rule = str(threshold_rule).strip().lower()
    if threshold_rule not in _THRESHOLD_RULES:
        raise ValueError(
            "threshold_rule must be one of " + ", ".join(_THRESHOLD_RULES) + "."
        )
    if isinstance(mstep_solver, str):
        mstep_solver = mstep_solver.strip().lower()
        if mstep_solver not in _MSTEP_SOLVERS:
            raise ValueError(
                "mstep_solver must be one of " + ", ".join(_MSTEP_SOLVERS)
                + " or a length-D sequence of 'fista'/'lbfgsb'."
            )
        if mstep_solver == "fista":
            layer_solvers = ("fista",) * D
        elif mstep_solver == "lbfgsb":
            layer_solvers = ("lbfgsb",) * D
        elif mstep_solver == "fista_lower":
            layer_solvers = ("fista",) * (D - 1) + ("lbfgsb",)
        else:
            layer_solvers = ("fista",) + ("lbfgsb",) * (D - 1)
    else:
        layer_solvers = tuple(str(value).strip().lower() for value in mstep_solver)
        if len(layer_solvers) != D or any(
            value not in ("fista", "lbfgsb") for value in layer_solvers
        ):
            raise ValueError(
                "A per-layer mstep_solver must be a length-D sequence of "
                "'fista'/'lbfgsb'."
            )
        if all(value == "fista" for value in layer_solvers):
            mstep_solver = "fista"
        elif all(value == "lbfgsb" for value in layer_solvers):
            mstep_solver = "lbfgsb"
        elif layer_solvers == ("fista",) + ("lbfgsb",) * (D - 1):
            mstep_solver = "hybrid"
        elif layer_solvers == ("fista",) * (D - 1) + ("lbfgsb",):
            mstep_solver = "fista_lower"
        else:
            mstep_solver = "+".join(layer_solvers)
    mstep_latent = str(mstep_latent).strip().lower()
    if mstep_latent not in _MSTEP_LATENTS:
        raise ValueError(
            "mstep_latent must be one of " + ", ".join(_MSTEP_LATENTS) + "."
        )
    observed_noise_scale = float(observed_noise_scale)
    if not np.isfinite(observed_noise_scale) or observed_noise_scale < 0:
        raise ValueError("observed_noise_scale must be finite and nonnegative.")
    fista_maxiter = _DEFAULT_FISTA_MAXITER if fista_maxiter is None else int(fista_maxiter)
    if fista_maxiter < 1:
        raise ValueError("fista_maxiter must be a positive integer.")
    observed_fista = layer_solvers[0] == "fista"
    slab_shape = float(slab_shape)
    slab_rate = float(slab_rate)
    penalty_scaling_power = np.asarray(penalty_scaling_power, dtype=float).reshape(-1)
    if penalty_scaling_power.size == 1:
        penalty_scaling_power = np.repeat(penalty_scaling_power.item(), D)
    if penalty_scaling_power.size != D:
        raise ValueError(
            "penalty_scaling_power must be a scalar or a length-D sequence."
        )
    penalty_scaling_reference = float(penalty_scaling_reference)
    penalty_scaling_layers = str(penalty_scaling_layers).strip().lower()
    penalty_scaling_allocation = bool(penalty_scaling_allocation)
    if not np.all(np.isfinite(penalty_scaling_power)) or np.any(
        penalty_scaling_power < 0
    ):
        raise ValueError("penalty_scaling_power must be finite and nonnegative.")
    if not np.isfinite(penalty_scaling_reference) or penalty_scaling_reference <= 0:
        raise ValueError("penalty_scaling_reference must be finite and positive.")
    if penalty_scaling_layers not in _PENALTY_SCALING_LAYERS:
        raise ValueError(
            "penalty_scaling_layers must be one of "
            + ", ".join(_PENALTY_SCALING_LAYERS) + "."
        )

    t_spike = np.asarray(t_spike, dtype=float).reshape(-1)
    if t_spike.size == 1:
        t_spike = np.repeat(t_spike.item(), D)
    if t_spike.size != D:
        raise ValueError("t_spike must be a scalar or a length-D sequence.")
    if not np.all(np.isfinite(t_spike)) or np.any(t_spike <= 0):
        raise ValueError("t_spike entries must be finite and positive.")

    random = _resolve_random(random_state)
    lb, ub = _make_bounds(K, D)
    if threshold_rule == "first_two":
        thresholded_layers = min(2, D)
    elif threshold_rule == "all_but_last":
        thresholded_layers = D - 1
    else:
        thresholded_layers = 0

    # CUSP state, initialized exactly as in the MATLAB kernel: every column
    # starts in the spike, labels sit at floor(K/2), sticks are uniform.
    theta_slab = [np.ones(Kd, dtype=float) for Kd in K]
    omega = [np.ones(Kd + 1, dtype=float) / (Kd + 1) for Kd in K]
    z_cusp = [np.floor(Kd / 2) * np.ones(Kd, dtype=int) for Kd in K]
    theta = [
        np.full((J if d == 0 else K[d - 1], K[d]), t_spike[d], dtype=float)
        for d in range(D)
    ]
    num_act = np.asarray(K, dtype=int)
    ind_inact = [np.zeros(0, dtype=int) for _ in range(D)]
    effective_power = penalty_scaling_power.copy()
    if penalty_scaling_layers == "deep":
        effective_power[0] = 0.0
    elif penalty_scaling_layers == "lower":
        effective_power[-1] = 0.0
    penalty_factor = (N / penalty_scaling_reference) ** effective_power
    theta_mstep = [value.copy() for value in theta]
    rel_hist = [[] for _ in range(D)]
    trace = [] if keep_trace else None
    x_sq_sum = np.sum(X**2, axis=0) if observed_fista else None

    it_num = 0
    stopped_by_convergence = False
    while it_num < it:
        # 1) Synchronous mean-field E-step from the previous point estimates.
        ber = _update_bernoulli_probabilities_D(X, A_list, B_list, gamma, prop, temp)
        A_sample_long = _sample_latent_layers(ber, C, random)
        if mstep_latent == "mean_field":
            A_mstep = [
                np.concatenate(
                    (np.ones((N, 1, 1)), probs[:, :, None]), axis=1
                )
                for probs in ber
            ]
        else:
            A_mstep = A_sample_long

        # 2) (CopDDE rank-likelihood resampling of Z is omitted: Z = X.)
        #    Optionally perturb the observed layer seen by the M-step.
        if observed_noise_scale > 0:
            X_mstep = X + observed_noise_scale * np.sqrt(gamma / temp)[None, :] * (
                random.standard_normal((N, J))
            )
        else:
            X_mstep = X

        # 3) CUSP allocation, deepest layer first.  With penalty scaling the
        #    M-step prior is Laplace(0, scale / c_N); the allocation either
        #    sees c_N * B (the matching densities) or the raw B (power prior).
        for d in range(D - 1, -1, -1):
            num_rows = J if d == 0 else int(num_act[d - 1])
            B_alloc = (
                B_list[d] * penalty_factor[d]
                if penalty_scaling_allocation and penalty_factor[d] != 1.0
                else B_list[d]
            )
            z_cusp[d], theta[d], theta_slab[d], omega[d] = update_csp_layer(
                B_alloc, omega[d], t_spike[d], theta_slab[d], num_rows,
                rng=random, slab_shape=slab_shape, slab_rate=slab_rate,
            )
            # The M-step prior scale on B is theta / c_N in both modes: with
            # a scaled allocation ``theta`` describes c_N * B, and with a
            # power prior the scaled penalty is the definition.
            theta_mstep[d] = theta[d] / penalty_factor[d]

        # 4) Top-layer Bernoulli MAP.
        prop_update = np.mean(ber[-1], axis=0)

        # 5) Observed Gaussian layer and conditional variances.
        B_update = [None] * D
        if not observed_fista:
            B_update[0], gamma_update = _optimize_gaussian_layer(
                X_mstep, A_mstep[0], B_list[0], gamma, theta_mstep[0], temp,
                lb[0], ub[0], optimizer_maxiter, n_jobs, "weighted_l1", tau,
            )
        else:
            B_update[0], gamma_update = _solve_gaussian_weighted_l1(
                X_mstep, A_mstep[0], B_list[0], gamma, theta_mstep[0], temp,
                x_sq_sum=None if X_mstep is not X else x_sq_sum,
                max_iter=fista_maxiter, tol=optimizer_tol, bound=bound_val,
            )
        if X_mstep is not X:
            # Conditional variances from the clean observations.
            eta = np.tensordot(A_mstep[0], B_update[0], axes=([1], [1]))
            C_mstep = A_mstep[0].shape[2]
            sse = np.sum((X[:, None, :] - eta) ** 2, axis=(0, 1))
            gamma_update = (1.0 + 0.5 * sse) / (1.0 + 0.5 * N * C_mstep + 1.0)

        # 6) Logistic layers.
        for d in range(1, D):
            if layer_solvers[d] != "fista":
                B_update[d] = _optimize_logistic_layer(
                    A_mstep[d - 1], A_mstep[d], B_list[d], theta_mstep[d],
                    temp, lb[d], ub[d], optimizer_maxiter, n_jobs,
                    "weighted_l1", tau,
                )
            else:
                B_update[d] = _solve_logistic_weighted_l1(
                    A_mstep[d - 1], A_mstep[d], B_list[d], theta_mstep[d],
                    temp, max_iter=fista_maxiter, tol=optimizer_tol,
                    bound=bound_val, bounds=list(zip(lb[d], ub[d])),
                )

        it_num += 1

        # 7) Active counts, thresholding, and hierarchical gating.
        for d in range(D):
            inactive = np.flatnonzero(~csp_active_mask(z_cusp[d]))
            ind_inact[d] = inactive
            num_act[d] = K[d] - inactive.size
        for d in range(thresholded_layers):
            B_update[d][:, 1:] = _threshold(B_update[d][:, 1:], tau)
        for d in range(D - 1):
            if ind_inact[d].size:
                B_update[d + 1][ind_inact[d], :] = 0.0

        # 8) Temperature schedule.
        if it_num > burn:
            temp = min(1.0, temp + 0.01 * ((it_num - burn) - 1))

        # 9) Rolling relative-change stopping rule.
        stop_now = True
        current_rel = []
        for d in range(D):
            rel = np.linalg.norm(B_update[d][:, 1:] - B_list[d][:, 1:], ord="fro") / (
                np.linalg.norm(B_list[d][:, 1:], ord="fro") + 1e-6
            )
            rel_hist[d].append(float(rel))
            current_rel.append(float(rel))
            if (
                len(rel_hist[d]) < convergence_window
                or np.std(rel_hist[d][-convergence_window:]) >= convergence_std
            ):
                stop_now = False

        if keep_trace:
            trace.append(
                {
                    "iteration": it_num,
                    "num_active": num_act.copy(),
                    "allocation": [value.copy() for value in z_cusp],
                    "omega": [value.copy() for value in omega],
                    "slab_scale": [value.copy() for value in theta_slab],
                    "updated_column_l1": [
                        np.sum(np.abs(B[:, 1:]), axis=0) for B in B_update
                    ],
                    "relative_change": tuple(current_rel),
                    "temperature": float(temp),
                    "will_stop": bool(stop_now),
                }
            )

        # 10) Commit the iteration (majority vote over the C draws).
        A_list = [
            (np.mean(A_sample_long[d][:, 1:, :], axis=2) > 0.5).astype(float)
            for d in range(D)
        ]
        B_list = [B.copy() for B in B_update]
        gamma = gamma_update.copy()
        prop = prop_update.copy()

        if verbose:
            print(
                f"Normal CSP (copula kernel) iteration {it_num}, "
                f"active={num_act.tolist()}, max relative change="
                f"{max(current_rel):.4g}"
            )
        if stop_now:
            stopped_by_convergence = True
            if verbose:
                print("ending algorithm")
            break

    if not stopped_by_convergence and verbose:
        print("Ending algorithm for iteration budget")

    active_masks = [csp_active_mask(value) for value in z_cusp]
    for d in range(D):
        inactive = np.flatnonzero(~active_masks[d])
        if inactive.size:
            B_list[d][:, inactive + 1] = 0.0

    return {
        "prop": prop,
        "B_list": B_list,
        "gamma": gamma,
        "A_list": [A.astype(np.uint8) for A in A_list],
        "allocation": [value.copy() for value in z_cusp],
        "omega": [value.copy() for value in omega],
        "slab_scale": [value.copy() for value in theta_slab],
        "penalty_scale": [value.copy() for value in theta],
        "active_masks": active_masks,
        "num_active": num_act.copy(),
        "iterations": it_num,
        "stopped_by_convergence": bool(stopped_by_convergence),
        "temperature": float(temp),
        "threshold_rule": threshold_rule,
        "mstep_solver": mstep_solver,
        "layer_solvers": layer_solvers,
        "mstep_latent": mstep_latent,
        "observed_noise_scale": observed_noise_scale,
        "optimizer_maxiter": optimizer_maxiter,
        "fista_maxiter": fista_maxiter,
        "slab_shape": slab_shape,
        "slab_rate": slab_rate,
        "penalty_scaling_power": tuple(float(value) for value in penalty_scaling_power),
        "penalty_scaling_reference": penalty_scaling_reference,
        "penalty_scaling_layers": penalty_scaling_layers,
        "penalty_scaling_allocation": penalty_scaling_allocation,
        "penalty_factor": tuple(float(value) for value in penalty_factor),
        "latent_transition": "synchronous_mean_field",
        "mc_state_update": "draw_majority",
        "relative_history": [tuple(values) for values in rel_hist],
        "trace": trace,
    }


__all__ = ["fit_normal_csp_copula"]
