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

Optional departures from the CopDDE kernel, all off by default
(see ``Results/normal_dde_csp_copula_kernel/REPORT_top_layer.md``):
``latent_transition="systematic_coordinate_gibbs"`` replaces the
synchronous mean-field draws by persistent coordinate-Gibbs sweeps of the
exact full conditionals (better deep-layer latent recovery, but split
latent columns are then never removed); ``merge_cosine`` merges active
logistic-layer columns with near-parallel coefficient columns (the split
factors that a weighted-L1 penalty cannot remove); ``mc_state_update``
selects the committed draw; ``reinit_top_after`` re-initializes the
deepest layer from the active children after a number of iterations.

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
from .parametric_csp_mcem import (
    _canonicalize_csp_layer_order,
    _solve_logistic_weighted_l1,
)
from .normal_init_cop import _binary_init_matlab
from .normal_init_semi import initialize_binary_parent
from .parametric_families import (
    NORMAL_FAMILY,
    _solve_gaussian_weighted_l1,
    resolve_parametric_family,
)
from .semi_saem import (
    _compute_bernoulli_probabilities_parametric,
    _sample_latent_layers_parametric,
)


_THRESHOLD_RULES = ("first_two", "all_but_last", "none")
_MSTEP_SOLVERS = ("lbfgsb", "fista", "hybrid", "fista_lower")
_MSTEP_LATENTS = ("draw", "mean_field")
_PENALTY_SCALING_LAYERS = ("all", "deep", "lower")
_LATENT_TRANSITIONS = ("synchronous_mean_field", "systematic_coordinate_gibbs")
_MC_STATE_UPDATES = ("draw_majority", "last_draw")
_REINIT_TOP_RULES = ("family", "copula")
_REINIT_TOP_SOURCES = ("state", "probability")
_MERGE_LAYERS = ("top", "logistic")
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


def _reinitialize_top_layer(children_all, B_list, z_cusp, K, rule, N):
    """Spectrally re-initialize the deepest layer from the active children.

    ``children_all`` is the ``N`` by ``K[D-2]`` child layer, binary or
    probabilities.  Returns ``None`` when fewer than two children of the top
    layer are active; otherwise a dict with the new top state ``A_top``
    (``N`` by ``K[-1]``, trailing columns zero when fewer parents than
    ``K[-1]`` can be initialized), the coefficient block ``B_top`` (inactive
    children's rows zero), the active child indices and the number of
    initialized parents.
    """
    D = len(B_list)
    active = np.flatnonzero(csp_active_mask(z_cusp[D - 2]))
    if active.size < 2:
        return None
    K_top = int(K[D - 1])
    n_parent = min(K_top, int(active.size))
    children = np.asarray(children_all, dtype=float)[:, active]
    if rule == "copula":
        A_new, B_new = _binary_init_matlab(N, active.size, n_parent, children)
    else:
        B_new, A_new = initialize_binary_parent(children, n_parent, binary=False)
    A_top = np.zeros((N, K_top), dtype=float)
    A_top[:, :n_parent] = np.asarray(A_new, dtype=float)
    B_top = np.zeros((int(K[D - 2]), K_top + 1), dtype=float)
    B_top[active, : n_parent + 1] = np.asarray(B_new, dtype=float)
    return {
        "A_top": A_top.astype(np.uint8),
        "B_top": B_top,
        "active_children": active.copy(),
        "num_parents": n_parent,
        "rule": rule,
    }


def _parallel_column_groups(B, allocation, cosine):
    """Greedy groups of active coefficient columns with cosine above ``cosine``.

    Columns are compared on their slopes (the intercept column of ``B`` is
    ignored); only active, nonzero columns take part.  Each group is listed
    in column order and has at least two members.
    """
    slopes = np.asarray(B, dtype=float)[:, 1:]
    active = np.flatnonzero(csp_active_mask(allocation))
    norms = np.linalg.norm(slopes[:, active], axis=0)
    active = active[norms > 0]
    if active.size < 2:
        return []
    unit = slopes[:, active] / np.linalg.norm(slopes[:, active], axis=0)
    similarity = unit.T @ unit
    groups = []
    used = set()
    for i in range(active.size):
        if i in used:
            continue
        members = [i] + [
            j for j in range(i + 1, active.size)
            if j not in used and similarity[i, j] > cosine
        ]
        used.update(members)
        if len(members) > 1:
            groups.append([int(active[m]) for m in members])
    return groups


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
    if gamma is not None and gamma.shape != (J,):
        raise ValueError("gamma length must match the observed dimension J.")
    for d, A in enumerate(A_list):
        if A.shape != (N, K[d]):
            raise ValueError(f"A_list[{d}] shape {A.shape} must be {(N, K[d])}.")
    return D, N, J, K


def fit_parametric_csp_copula(
    X,
    prop_in,
    gamma_in,
    B_list_in,
    A_list_in,
    *,
    family="normal",
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
    latent_transition: str = "synchronous_mean_field",
    mc_state_update=None,
    estep_row_chunk_size: int = 2048,
    reinit_top_after: int = 0,
    reinit_top_rule: str = "family",
    reinit_top_source: str = "state",
    merge_cosine: float = 0.0,
    merge_after: int = 10,
    merge_layers: str = "top",
    shrinkage_burn: int = 0,
    reorder_after_shrinkage_burn: bool = True,
    random_state=None,
    keep_trace: bool = False,
    verbose: bool = True,
):
    """Fit a maximal-width parametric DDE with the CopDDE CSP algorithm.

    ``family`` selects the observed layer: ``"normal"`` (the CopDDE Gaussian
    layer, with ``gamma_in`` the conditional variances) or ``"bernoulli"``
    (a logistic observed layer identical in form to the latent layers;
    ``gamma_in`` must be ``None``).  Everything below the observed layer is
    family independent.

    Parameters
    ----------
    X : (N, J) observed matrix (Gaussian ``Z`` for Normal, 0/1 for Bernoulli).
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
    latent_transition : ``"synchronous_mean_field"`` (CopDDE: every latent
        coordinate's conditional probability is computed from the previous
        iteration's point estimates, then all coordinates are drawn at once)
        or ``"systematic_coordinate_gibbs"`` (the ``C`` draws are
        consecutive systematic coordinate-Gibbs sweeps of the exact full
        conditionals, observed layer first, each coordinate conditioning on
        the freshly updated state; the persistent chain of the parametric
        MCEM kernel).  The Gibbs transition requires ``mstep_latent="draw"``.
    mc_state_update : ``"draw_majority"`` (CopDDE: the committed latent state
        is the majority vote of the ``C`` draws) or ``"last_draw"`` (the last
        draw, the persistent-chain rule).  ``None`` selects the majority vote
        for the mean-field transition and the last draw for Gibbs.
    estep_row_chunk_size : row block size of the Gibbs sweeps.
    reinit_top_after : when positive, after this many completed iterations
        the deepest layer is re-initialized once from the then-current
        active columns of layer ``D-1``: the latent top state, its
        coefficient block, ``prop`` and the layer's CUSP state are replaced
        by the spectral parent initialization, and the inactive children's
        rows stay zero.  Zero disables the re-initialization.
    reinit_top_rule : ``"family"`` (:func:`initialize_binary_parent`, the
        Normal family rule) or ``"copula"`` (the CopDDE ``binary_init``
        rule) for the re-initialization.
    reinit_top_source : ``"state"`` (the committed binary child state) or
        ``"probability"`` (the mean-field conditional probabilities of the
        children, computed from the current point estimates) as the input
        of the re-initialization.
    merge_cosine : when positive, from iteration ``merge_after`` on, active
        columns of a logistic layer whose coefficient columns have cosine
        similarity above this value (same sign pattern on the children) are
        merged: the coefficients are summed into the earliest column of the
        group, its latent state becomes the majority vote of the members,
        and the other members are zeroed (they then fall into the CUSP
        spike).  Such groups are "split" factors, several independent latent
        columns that share one factor's loadings; a weighted-L1 penalty is
        indifferent to the split and the coherent (Gibbs) latent chain keeps
        it alive.  Zero disables merging.
    merge_after : first iteration (completed iterations) at which merging
        is applied.
    merge_layers : ``"top"`` (deepest layer only) or ``"logistic"`` (every
        layer above the observed one).
    shrinkage_burn : number of initial iterations during which every
        column of every layer is held in the CUSP slab (allocation
        ``K_d + 1``, M-step prior scale equal to the slab scale, no hard
        thresholding, no gating, and no convergence check), so that the
        latent states and coefficients can settle before any column can
        fall into the spike; the CUSP updates start at iteration
        ``shrinkage_burn + 1``.  Zero (the CopDDE behaviour) starts the
        CUSP updates at iteration 1.  This is the slab-start warm-up of
        the parametric MCEM kernel (``shrinkage_burn`` there).
    reorder_after_shrinkage_burn : when ``shrinkage_burn`` is positive,
        permute each layer's columns by decreasing coefficient L1 norm
        once the burn-in ends (the model is permutation invariant, the
        ordered CUSP prior is not), as the MCEM kernel does.
    random_state : ``None`` for the legacy global RNG, else a seed/Generator.

    Returns
    -------
    dict with the fitted ``prop``, ``B_list``, ``gamma``, ``A_list``, CUSP
    state (``allocation``, ``omega``, ``slab_scale``, ``penalty_scale``),
    ``active_masks``, ``num_active``, ``iterations`` and diagnostics.
    """
    family = resolve_parametric_family(family)
    if family.key not in ("normal", "bernoulli"):
        raise NotImplementedError(
            "The copula CSP kernel supports the Normal and Bernoulli families."
        )
    is_normal = family.key == "normal"
    X = family.validate_y(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix.")
    prop = np.asarray(prop_in, dtype=float).reshape(-1).copy()
    if is_normal:
        gamma = np.asarray(gamma_in, dtype=float).reshape(-1).copy()
    elif gamma_in is not None:
        raise ValueError(f"{family.name} has no dispersion; gamma_in must be None.")
    else:
        gamma = None
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
    if observed_noise_scale > 0 and not is_normal:
        raise ValueError("observed_noise_scale applies to the Normal family only.")
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
    latent_transition = str(latent_transition).strip().lower()
    if latent_transition not in _LATENT_TRANSITIONS:
        raise ValueError(
            "latent_transition must be one of " + ", ".join(_LATENT_TRANSITIONS) + "."
        )
    gibbs = latent_transition == "systematic_coordinate_gibbs"
    if gibbs and mstep_latent != "draw":
        raise ValueError(
            "mstep_latent='mean_field' requires the synchronous mean-field "
            "transition."
        )
    if mc_state_update is None:
        mc_state_update = "last_draw" if gibbs else "draw_majority"
    mc_state_update = str(mc_state_update).strip().lower()
    if mc_state_update not in _MC_STATE_UPDATES:
        raise ValueError(
            "mc_state_update must be one of " + ", ".join(_MC_STATE_UPDATES) + "."
        )
    estep_row_chunk_size = int(estep_row_chunk_size)
    if estep_row_chunk_size < 1:
        raise ValueError("estep_row_chunk_size must be a positive integer.")
    reinit_top_after = int(reinit_top_after)
    if reinit_top_after < 0:
        raise ValueError("reinit_top_after must be a nonnegative integer.")
    reinit_top_rule = str(reinit_top_rule).strip().lower()
    if reinit_top_rule not in _REINIT_TOP_RULES:
        raise ValueError(
            "reinit_top_rule must be one of " + ", ".join(_REINIT_TOP_RULES) + "."
        )
    reinit_top_source = str(reinit_top_source).strip().lower()
    if reinit_top_source not in _REINIT_TOP_SOURCES:
        raise ValueError(
            "reinit_top_source must be one of " + ", ".join(_REINIT_TOP_SOURCES) + "."
        )
    merge_cosine = float(merge_cosine)
    if not np.isfinite(merge_cosine) or merge_cosine < 0 or merge_cosine >= 1:
        raise ValueError("merge_cosine must lie in [0, 1).")
    merge_after = int(merge_after)
    if merge_after < 0:
        raise ValueError("merge_after must be a nonnegative integer.")
    merge_layers = str(merge_layers).strip().lower()
    if merge_layers not in _MERGE_LAYERS:
        raise ValueError("merge_layers must be one of " + ", ".join(_MERGE_LAYERS) + ".")
    if reinit_top_after and D < 2:
        raise ValueError("reinit_top_after requires at least two latent layers.")
    shrinkage_burn = int(shrinkage_burn)
    if shrinkage_burn < 0:
        raise ValueError("shrinkage_burn must be a nonnegative integer.")
    if shrinkage_burn >= it:
        raise ValueError("shrinkage_burn must be smaller than it.")
    reorder_after_shrinkage_burn = bool(reorder_after_shrinkage_burn)

    t_spike = np.asarray(t_spike, dtype=float).reshape(-1)
    if t_spike.size == 1:
        t_spike = np.repeat(t_spike.item(), D)
    if t_spike.size != D:
        raise ValueError("t_spike must be a scalar or a length-D sequence.")
    if not np.all(np.isfinite(t_spike)) or np.any(t_spike <= 0):
        raise ValueError("t_spike entries must be finite and positive.")

    random = _resolve_random(random_state)
    if is_normal:
        lb, ub = _make_bounds(K, D)
    else:
        lb, ub = [], []
        for d in range(D):
            pairs = family.csp_coefficient_bounds(K[d] + 1, bound_val, observed=(d == 0))
            lb.append(np.asarray([pair[0] for pair in pairs], dtype=float))
            ub.append(np.asarray([pair[1] for pair in pairs], dtype=float))
        # Bernoulli observed layer: the data are the "child sample" of the
        # logistic M-step, with an intercept column and one copy per draw.
        X_child = np.concatenate((np.ones((N, 1)), X), axis=1)[:, :, None]

    def mean_field_probabilities(A_state, B_state, gamma_state, prop_state, temperature):
        if is_normal:
            return _update_bernoulli_probabilities_D(
                X, A_state, B_state, gamma_state, prop_state, temperature
            )
        return _compute_bernoulli_probabilities_parametric(
            X, prop_state, B_state, gamma_state, A_state, family=family,
            row_chunk_size=estep_row_chunk_size, temperature=temperature,
        )
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
    x_sq_sum = np.sum(X**2, axis=0) if (observed_fista and is_normal) else None
    top_reinit = None
    merge_events = []
    burn_permutations = None

    it_num = 0
    stopped_by_convergence = False
    while it_num < it:
        shrinkage_active = it_num >= shrinkage_burn
        # 0a) End of the slab-start burn-in: order the columns of every
        #     layer by strength.  The CUSP state is column symmetric at
        #     this point (all slab, unit slab scales), so only the model
        #     parameters and latent states need permuting.
        if (
            shrinkage_burn
            and shrinkage_active
            and reorder_after_shrinkage_burn
            and burn_permutations is None
        ):
            prop, B_list, A_list, burn_permutations = _canonicalize_csp_layer_order(
                prop, B_list, A_list
            )
            for d, permutation in enumerate(burn_permutations):
                theta_slab[d] = theta_slab[d][permutation]
                theta[d] = theta[d][:, permutation]
                theta_mstep[d] = theta_mstep[d][:, permutation]
                z_cusp[d] = z_cusp[d][permutation]
                if d >= 1 and burn_permutations[d - 1] is not None:
                    theta[d] = theta[d][burn_permutations[d - 1], :]
                    theta_mstep[d] = theta_mstep[d][burn_permutations[d - 1], :]

        # 0) Optional one-off re-initialization of the deepest layer from
        #    the current active children.
        if reinit_top_after and it_num == reinit_top_after and top_reinit is None:
            if reinit_top_source == "probability":
                children = mean_field_probabilities(A_list, B_list, gamma, prop, temp)[D - 2]
            else:
                children = A_list[D - 2]
            top_reinit = _reinitialize_top_layer(
                children, B_list, z_cusp, K, reinit_top_rule, N
            )
            if top_reinit is not None:
                top_reinit["iteration"] = it_num
                A_list[D - 1] = top_reinit["A_top"].astype(float)
                B_list[D - 1] = top_reinit["B_top"]
                prop = np.clip(np.mean(A_list[D - 1], axis=0), 1e-8, 1.0 - 1e-8)
                theta_slab[D - 1] = np.ones(K[D - 1], dtype=float)
                omega[D - 1] = np.ones(K[D - 1] + 1, dtype=float) / (K[D - 1] + 1)
                z_cusp[D - 1] = np.floor(K[D - 1] / 2) * np.ones(K[D - 1], dtype=int)
                theta[D - 1] = np.full((K[D - 2], K[D - 1]), t_spike[D - 1])
                theta_mstep[D - 1] = theta[D - 1].copy()

        # 0b) Optional merge of split (near-parallel) latent columns.
        if merge_cosine > 0 and it_num >= merge_after and D >= 2:
            merge_targets = range(D - 1, D) if merge_layers == "top" else range(1, D)
            for d in merge_targets:
                groups = _parallel_column_groups(B_list[d], z_cusp[d], merge_cosine)
                for group in groups:
                    keep, drop = group[0], group[1:]
                    B_list[d][:, keep + 1] = B_list[d][:, [g + 1 for g in group]].sum(axis=1)
                    A_list[d][:, keep] = (A_list[d][:, group].mean(axis=1) > 0.5).astype(float)
                    B_list[d][:, [g + 1 for g in drop]] = 0.0
                    A_list[d][:, drop] = 0.0
                    if d < D - 1:
                        B_list[d + 1][drop, :] = 0.0
                    merge_events.append((it_num, d, tuple(int(g) for g in group)))
                if groups and d == D - 1:
                    prop = np.clip(np.mean(A_list[d], axis=0), 1e-8, 1.0 - 1e-8)

        # 1) E-step: synchronous mean-field draws from the previous point
        #    estimates (CopDDE), or persistent systematic coordinate-Gibbs
        #    sweeps of the exact full conditionals.
        if gibbs:
            ber = None
            samples, ber_top = _sample_latent_layers_parametric(
                X, prop, B_list, gamma,
                [A.astype(np.uint8) for A in A_list], C,
                family=family, row_chunk_size=estep_row_chunk_size,
                rng=random, sparse_bernoulli=not is_normal, temperature=temp,
            )
            A_sample_long = [sample.astype(float) for sample in samples]
        else:
            ber = mean_field_probabilities(A_list, B_list, gamma, prop, temp)
            ber_top = ber[-1]
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
            if not shrinkage_active:
                # Slab-start burn-in: every column in the slab at the
                # current slab scale, as in the MCEM kernel's warm-up.
                z_cusp[d] = np.full(K[d], K[d] + 1, dtype=int)
                theta[d] = np.broadcast_to(
                    theta_slab[d][None, :], theta[d].shape
                ).copy()
                theta_mstep[d] = theta[d] / penalty_factor[d]
                continue
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
        prop_update = np.mean(ber_top, axis=0)

        # 5) Observed layer: Gaussian (with conditional variances) or logistic.
        B_update = [None] * D
        gamma_update = None
        if not is_normal:
            if not observed_fista:
                C_mstep = A_mstep[0].shape[2]
                B_update[0] = _optimize_logistic_layer(
                    np.repeat(X_child, C_mstep, axis=2), A_mstep[0], B_list[0],
                    theta_mstep[0], temp, lb[0], ub[0], optimizer_maxiter,
                    n_jobs, "weighted_l1", tau,
                )
            else:
                B_update[0] = _solve_logistic_weighted_l1(
                    X, A_mstep[0], B_list[0], theta_mstep[0], temp,
                    max_iter=fista_maxiter, tol=optimizer_tol,
                    bound=bound_val, bounds=list(zip(lb[0], ub[0])),
                )
        elif not observed_fista:
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
        if is_normal and X_mstep is not X:
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
        if shrinkage_active:
            for d in range(thresholded_layers):
                B_update[d][:, 1:] = _threshold(B_update[d][:, 1:], tau)
        for d in range(D - 1):
            if ind_inact[d].size:
                B_update[d + 1][ind_inact[d], :] = 0.0

        # 8) Temperature schedule.
        if it_num > burn:
            temp = min(1.0, temp + 0.01 * ((it_num - burn) - 1))

        # 9) Rolling relative-change stopping rule (never during the
        #    slab-start burn-in).
        stop_now = shrinkage_active
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
                    "shrinkage_active": bool(shrinkage_active),
                    "will_stop": bool(stop_now),
                }
            )

        # 10) Commit the iteration (majority vote over the C draws, or the
        #     last draw of the persistent chain).
        if mc_state_update == "last_draw":
            A_list = [A_sample_long[d][:, 1:, -1].copy() for d in range(D)]
        else:
            A_list = [
                (np.mean(A_sample_long[d][:, 1:, :], axis=2) > 0.5).astype(float)
                for d in range(D)
            ]
        B_list = [B.copy() for B in B_update]
        gamma = None if gamma_update is None else gamma_update.copy()
        prop = prop_update.copy()

        if verbose:
            print(
                f"{family.name} CSP (copula kernel) iteration {it_num}, "
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
        "family": family.name,
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
        "latent_transition": latent_transition,
        "mc_state_update": mc_state_update,
        "reinit_top_after": reinit_top_after,
        "reinit_top_rule": reinit_top_rule,
        "reinit_top_source": reinit_top_source,
        "top_reinit": top_reinit,
        "merge_cosine": merge_cosine,
        "merge_after": merge_after,
        "merge_layers": merge_layers,
        "merge_events": merge_events,
        "shrinkage_burn": shrinkage_burn,
        "reorder_after_shrinkage_burn": reorder_after_shrinkage_burn,
        "burn_permutations": burn_permutations,
        "relative_history": [tuple(values) for values in rel_hist],
        "trace": trace,
    }


def fit_normal_csp(X, prop_in, gamma_in, B_list_in, A_list_in, **kwargs):
    """Normal DDE CSP: the CopDDE kernel on an observed Gaussian layer."""
    return fit_parametric_csp_copula(
        X, prop_in, gamma_in, B_list_in, A_list_in, family="normal", **kwargs
    )


def fit_bernoulli_csp(X, prop_in, B_list_in, A_list_in, **kwargs):
    """Bernoulli DDE CSP: the CopDDE kernel with a logistic observed layer."""
    return fit_parametric_csp_copula(
        X, prop_in, None, B_list_in, A_list_in, family="bernoulli", **kwargs
    )


# Compatibility alias for the name used while the kernel was being developed.
fit_normal_csp_copula = fit_normal_csp

__all__ = [
    "fit_bernoulli_csp",
    "fit_normal_csp",
    "fit_normal_csp_copula",
    "fit_parametric_csp_copula",
]
