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

import itertools

import numpy as np
from scipy.special import expit, logsumexp

from .csp import _canonicalize_csp_layer_order, csp_active_mask, update_csp_layer
from .csp_solvers import _solve_logistic_weighted_l1, _solve_poisson_weighted_l1
from .get_SAEM_RL_CSP_D import (
    _make_bounds,
    _optimize_gaussian_layer,
    _optimize_logistic_layer,
    _threshold,
    _update_bernoulli_probabilities_D,
)
from .normal_init_cop import _binary_init_matlab
from .normal_init_semi import initialize_binary_parent
from .parametric_families import (
    NORMAL_FAMILY,
    _binary_child_conditional_logit,
    _solve_gaussian_weighted_l1,
    resolve_parametric_family,
)
from .semi_saem import (
    _compute_bernoulli_probabilities_parametric,
    _sample_latent_layers_parametric,
)


_THRESHOLD_RULES = ("first_two", "all_but_last", "all", "none")
_MSTEP_SOLVERS = ("lbfgsb", "fista", "hybrid", "fista_lower")
_MSTEP_LATENTS = ("draw", "mean_field")
_PENALTY_SCALING_LAYERS = ("all", "deep", "lower")
_LATENT_TRANSITIONS = (
    "synchronous_mean_field",
    "sequential_mean_field",
    "systematic_coordinate_gibbs",
)
_MC_STATE_UPDATES = ("draw_majority", "last_draw")
_REINIT_TOP_RULES = ("family", "copula")
_REINIT_TOP_SOURCES = ("state", "probability")
_MERGE_LAYERS = ("top", "logistic")
_TOP_PRUNE_RULES = ("none", "bic", "bic_dense")
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


def _binary_factor_em(children, B, prop, support, em_iter, ridge=1e-3):
    """EM for a deepest binary layer given its binary child states.

    The parent configurations are enumerated, so the returned log-likelihood
    is the exact marginal ``sum_n log sum_a p(a) p(children_n | a)``.  ``B``
    (children by parents + 1) is refit by damped Newton steps on its
    ``support`` (a boolean mask of the slopes; intercepts are always free).
    Returns ``(loglik, B, prop)`` after ``em_iter`` EM updates.
    """
    Y = np.asarray(children, dtype=float)
    B = np.asarray(B, dtype=float).copy()
    prop = np.clip(np.asarray(prop, dtype=float).copy(), 1e-6, 1.0 - 1e-6)
    N, J = Y.shape
    K = B.shape[1] - 1
    configs = np.array(list(itertools.product((0.0, 1.0), repeat=K)), dtype=float).reshape(-1, K)
    design = np.column_stack((np.ones(configs.shape[0]), configs))
    free = np.column_stack((np.ones((J, 1), dtype=bool), np.asarray(support, dtype=bool)))
    B[~free] = 0.0
    loglik = -np.inf
    for step in range(em_iter + 1):
        eta = design @ B.T
        log_child = Y @ eta.T - np.sum(np.logaddexp(0.0, eta), axis=1)[None, :]
        log_prior = configs @ np.log(prop) + (1.0 - configs) @ np.log1p(-prop)
        joint = log_child + log_prior[None, :]
        norm = logsumexp(joint, axis=1, keepdims=True)
        loglik = float(np.sum(norm))
        if step == em_iter:
            break
        weights = np.exp(joint - norm)
        count = weights.sum(axis=0)
        success = weights.T @ Y
        prop = np.clip((count @ configs) / N, 1e-6, 1.0 - 1e-6)
        for j in range(J):
            idx = np.flatnonzero(free[j])
            X_j = design[:, idx]
            b = B[j, idx]
            for _ in range(3):
                p = expit(X_j @ b)
                grad = X_j.T @ (success[:, j] - count * p) - ridge * b
                hess = (X_j * (count * p * (1.0 - p))[:, None]).T @ X_j + ridge * np.eye(idx.size)
                b = np.clip(b + np.linalg.solve(hess, grad), -10.0, 10.0)
            B[j, idx] = b
    return loglik, B, prop


def _bic_prune_top_column(children, B_top, prop, active, child_active, N, em_iter, dense=False):
    """Marginal-likelihood (BIC) deletion test for one deepest-layer column.

    Columns are tested from the weakest (smallest coefficient L1 norm) up;
    the first whose removal costs less than ``n_params log(N) / 2`` in the
    marginal log-likelihood of the child states is returned with its
    statistics.  ``None`` when every column passes or fewer than two are
    active.

    With ``dense=False`` both models are refit by ``em_iter`` EM updates on
    the current support and ``n_params`` is the column's nonzero count plus
    one.  The frozen support keeps the other columns from taking over the
    tested column's role, which overstates its value when the layer is
    redundant (more columns than factors).  With ``dense=True`` every active
    child loads on every active column: the full model is refit first, the
    reduced model starts from it without the column, ``n_params`` is the
    number of active children plus one, and the reduced refit (``B_rows``,
    rows of the active children with the intercept and the remaining
    columns, and ``prop_kept``) is returned for the caller to adopt.
    """
    cols = np.flatnonzero(active)
    rows = np.flatnonzero(child_active)
    if cols.size < 2 or rows.size == 0:
        return None
    Y = np.asarray(children, dtype=float)[:, rows]
    B_sub = np.asarray(B_top, dtype=float)[np.ix_(rows, np.concatenate(([0], cols + 1)))]
    prop_sub = np.asarray(prop, dtype=float)[cols]
    support = np.ones(B_sub[:, 1:].shape, dtype=bool) if dense else B_sub[:, 1:] != 0.0
    full_ll, B_full, prop_full = _binary_factor_em(Y, B_sub, prop_sub, support, em_iter)
    if dense:
        B_sub, prop_sub = B_full, prop_full
    strength = np.sum(np.abs(B_sub[:, 1:]), axis=0)
    for position in np.argsort(strength, kind="stable"):
        keep = np.delete(np.arange(cols.size), position)
        reduced_ll, B_reduced, prop_reduced = _binary_factor_em(
            Y, B_sub[:, np.concatenate(([0], keep + 1))], prop_sub[keep],
            support[:, keep], em_iter,
        )
        n_params = int(np.count_nonzero(support[:, position])) + 1
        gain = full_ll - reduced_ll
        if 2.0 * gain < n_params * np.log(N):
            removed = {
                "column": int(cols[position]),
                "loglik_gain": float(gain),
                "n_params": n_params,
                "bic_penalty": float(n_params * np.log(N) / 2.0),
            }
            if dense:
                removed.update(rows=rows, kept_columns=cols[keep], B_rows=B_reduced, prop_kept=prop_reduced)
            return removed
    return None


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
    top_prune: str = "none",
    top_prune_after: int = 10,
    top_prune_every: int = 5,
    top_prune_em_iter: int = 20,
    convergence_width_window: int = 0,
    staged_shrinkage_window: int = 0,
    random_state=None,
    keep_trace: bool = False,
    verbose: bool = True,
    observed_update=None,
    observed_update_every: int = 1,
):
    """Fit a maximal-width parametric DDE with the CopDDE CSP algorithm.

    ``family`` selects the observed layer: ``"normal"`` (the CopDDE Gaussian
    layer, with ``gamma_in`` the conditional variances), ``"bernoulli"``
    (a logistic observed layer identical in form to the latent layers) or
    ``"poisson"`` (log-link counts, solved by backtracking FISTA); for the
    last two ``gamma_in`` must be ``None``.  Everything below the observed
    layer is family independent.

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
        (MATLAB D-layer rule), ``"all"`` (every layer, the deepest included:
        the inexact L-BFGS-B step there otherwise leaves small nonzero
        coefficients that keep unused top columns in the slab), or
        ``"none"``.
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
        the freshly updated state; a persistent chain) or
        ``"sequential_mean_field"`` (the mean-field draws
        taken layer by layer from the data upward: each layer's conditional
        probabilities use the freshly committed draw of the layer below and
        the previous state of the layer above, so consecutive layers are
        drawn jointly instead of independently).  The Gibbs transition
        requires ``mstep_latent="draw"``.
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
        CUSP updates at iteration 1.
    reorder_after_shrinkage_burn : when ``shrinkage_burn`` is positive,
        permute each layer's columns by decreasing coefficient L1 norm
        once the burn-in ends (the model is permutation invariant, the
        ordered CUSP prior is not).
    top_prune : ``"none"`` (default), ``"bic"`` or ``"bic_dense"``.  With
        ``"bic"``, every
        ``top_prune_every`` iterations from ``top_prune_after`` on (after
        the slab burn-in), the deepest layer's active columns are tested
        from the weakest up: the parent configurations are enumerated, the
        model with and without the column is refit by ``top_prune_em_iter``
        EM updates on the committed child states (current support kept),
        and the first column whose marginal log-likelihood gain is below
        ``(nnz + 1) log(N) / 2`` is removed (coefficients and state zeroed,
        so it falls into the CUSP spike).  At most one column per test.
        The CUSP allocation compares the prior densities at the point
        estimate with a fixed coefficient scale, so a column that fits
        residual dependence with coefficients of order one stays in the
        slab at every ``N``; the likelihood test removes it consistently.
        ``"bic_dense"`` refits both models with every active child loading
        on every active column (penalty: active children plus one) and
        adopts the reduced refit for the remaining columns; with a frozen
        support the remaining columns cannot take over a redundant
        column's role, so ``"bic"`` overstates its value and keeps it.
    convergence_width_window : when positive, the stopping rule also
        requires the active widths to be unchanged over the last this many
        iterations (column pruning can pause for many iterations while the
        coefficient changes are already small).
    staged_shrinkage_window : when positive, the CUSP updates start layer by
        layer from the data upward: layer ``d + 1`` stays in the slab (as in
        the slab burn-in: no allocation, thresholding or deletion test)
        until layer ``d``'s active width has been unchanged for this many
        iterations.  Pruning a split lower layer zeroes the rows of its
        removed columns in the layer above (gating), so a parent fitted to
        a removed copy loses its support; holding the parent in the slab
        lets it re-fit to the surviving children first.  The stopping rule
        waits until every layer has started.
    random_state : ``None`` for the legacy global RNG, else a seed/Generator.
    observed_update : optional callable ``f(X, state) -> X_new`` (Normal family
        only) applied after every ``observed_update_every``-th committed
        iteration; ``state`` holds ``prop``, ``B_list``, ``gamma``, ``A_list``,
        ``active_masks`` and ``iteration``.  This is the slot of the CopDDE
        rank-likelihood resampling of ``Z``; the SemiDDE CSP uses it for its
        semiparametric transformation (g) step.  ``None`` keeps ``X`` fixed.

    Returns
    -------
    dict with the fitted ``prop``, ``B_list``, ``gamma``, ``A_list``, CUSP
    state (``allocation``, ``omega``, ``slab_scale``, ``penalty_scale``),
    ``active_masks``, ``num_active``, ``iterations`` and diagnostics.
    """
    family = resolve_parametric_family(family)
    if family.key not in ("normal", "bernoulli", "poisson"):
        raise NotImplementedError(
            "The copula CSP kernel supports the Normal, Bernoulli and Poisson families."
        )
    is_normal = family.key == "normal"
    is_poisson = family.key == "poisson"
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
            if D == 1 and is_poisson:
                # the single layer is the observed one, solved by FISTA
                layer_solvers = ("fista",)
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
    if observed_update is not None and not is_normal:
        raise ValueError("observed_update applies to the Normal family only.")
    observed_update_every = int(observed_update_every)
    if observed_update_every < 1:
        raise ValueError("observed_update_every must be a positive integer.")
    observed_update_count = 0
    fista_maxiter = _DEFAULT_FISTA_MAXITER if fista_maxiter is None else int(fista_maxiter)
    if fista_maxiter < 1:
        raise ValueError("fista_maxiter must be a positive integer.")
    observed_fista = layer_solvers[0] == "fista"
    if is_poisson and not observed_fista:
        raise ValueError(
            "The Poisson observed layer is solved by FISTA; use an mstep_solver "
            "whose first layer is 'fista'."
        )
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
    sequential = latent_transition == "sequential_mean_field"
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
    top_prune = str(top_prune).strip().lower()
    if top_prune not in _TOP_PRUNE_RULES:
        raise ValueError("top_prune must be one of " + ", ".join(_TOP_PRUNE_RULES) + ".")
    top_prune_after = int(top_prune_after)
    top_prune_every = int(top_prune_every)
    top_prune_em_iter = int(top_prune_em_iter)
    if top_prune_after < 0 or top_prune_every < 1 or top_prune_em_iter < 1:
        raise ValueError(
            "top_prune_after must be nonnegative and top_prune_every, "
            "top_prune_em_iter positive."
        )
    convergence_width_window = int(convergence_width_window)
    if convergence_width_window < 0:
        raise ValueError("convergence_width_window must be a nonnegative integer.")
    staged_shrinkage_window = int(staged_shrinkage_window)
    if staged_shrinkage_window < 0:
        raise ValueError("staged_shrinkage_window must be a nonnegative integer.")

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
        X_child = (
            None if is_poisson
            else np.concatenate((np.ones((N, 1)), X), axis=1)[:, :, None]
        )

    def mean_field_probabilities(A_state, B_state, gamma_state, prop_state, temperature):
        if is_normal:
            return _update_bernoulli_probabilities_D(
                X, A_state, B_state, gamma_state, prop_state, temperature
            )
        return _compute_bernoulli_probabilities_parametric(
            X, prop_state, B_state, gamma_state, A_state, family=family,
            row_chunk_size=estep_row_chunk_size, temperature=temperature,
        )

    def deep_layer_probabilities(d, A_state, B_state, prop_state, temperature):
        """Mean-field conditional probabilities of latent layer ``d >= 1``."""
        if d == D - 1:
            eps = 1e-12 if is_normal else 1e-15
            prop_clipped = np.clip(prop_state, eps, 1.0 - eps)
            prior_logit = np.log(prop_clipped / (1.0 - prop_clipped))[None, :]
        else:
            prior_logit = (
                B_state[d + 1][:, 0][None, :]
                + A_state[d + 1] @ B_state[d + 1][:, 1:].T
            )
        like_logit = _binary_child_conditional_logit(
            A_state[d - 1], B_state[d], A_state[d], parent_width=K[d],
            row_chunk_size=estep_row_chunk_size, sparse="auto",
        )
        return expit(temperature * (like_logit + prior_logit))

    if threshold_rule == "first_two":
        thresholded_layers = min(2, D)
    elif threshold_rule == "all_but_last":
        thresholded_layers = D - 1
    elif threshold_rule == "all":
        thresholded_layers = D
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
    prune_events = []
    width_hist = []
    stage_open = [True] + [not staged_shrinkage_window] * (D - 1)
    stage_iterations = [0] + [None] * (D - 1)
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

        # 0c) Optional marginal-likelihood (BIC) deletion test for one
        #     column of the deepest layer, on the committed child states.
        if (
            top_prune != "none"
            and D >= 2
            and shrinkage_active
            and stage_open[D - 1]
            and it_num >= top_prune_after
            and (it_num - top_prune_after) % top_prune_every == 0
        ):
            removed = _bic_prune_top_column(
                A_list[D - 2], B_list[D - 1], prop,
                csp_active_mask(z_cusp[D - 1]), csp_active_mask(z_cusp[D - 2]),
                N, top_prune_em_iter, dense=(top_prune == "bic_dense"),
            )
            if removed is not None:
                column = removed["column"]
                if "B_rows" in removed:
                    # adopt the reduced dense refit for the remaining columns
                    rows, kept = removed.pop("rows"), removed.pop("kept_columns")
                    B_list[D - 1][np.ix_(rows, np.concatenate(([0], kept + 1)))] = removed.pop("B_rows")
                    prop = prop.copy()
                    prop[kept] = removed.pop("prop_kept")
                B_list[D - 1][:, column + 1] = 0.0
                A_list[D - 1][:, column] = 0.0
                z_cusp[D - 1][column] = 1
                theta[D - 1][:, column] = t_spike[D - 1]
                theta_mstep[D - 1][:, column] = t_spike[D - 1] / penalty_factor[D - 1]
                removed["iteration"] = it_num
                prune_events.append(removed)

        # 1) E-step: synchronous mean-field draws from the previous point
        #    estimates (CopDDE), the same draws taken layer by layer from the
        #    data upward, or persistent systematic coordinate-Gibbs sweeps of
        #    the exact full conditionals.
        if gibbs:
            ber = None
            samples, ber_top = _sample_latent_layers_parametric(
                X, prop, B_list, gamma,
                [A.astype(np.uint8) for A in A_list], C,
                family=family, row_chunk_size=estep_row_chunk_size,
                rng=random, sparse_bernoulli=not is_normal, temperature=temp,
            )
            A_sample_long = [sample.astype(float) for sample in samples]
        elif sequential:
            ber = mean_field_probabilities(A_list, B_list, gamma, prop, temp)
            A_work = list(A_list)
            A_sample_long = [None] * D
            for d in range(D):
                if d > 0:
                    ber[d] = deep_layer_probabilities(d, A_work, B_list, prop, temp)
                A_sample_long[d] = _sample_latent_layers([ber[d]], C, random)[0]
                A_work[d] = (np.mean(A_sample_long[d][:, 1:, :], axis=2) > 0.5).astype(float)
            ber_top = ber[-1]
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
            if not shrinkage_active or not stage_open[d]:
                # Slab-start burn-in (or a layer whose stage has not
                # started): every column in the slab at the current slab
                # scale.
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
        if is_poisson:
            B_update[0] = _solve_poisson_weighted_l1(
                X, A_mstep[0], B_list[0], theta_mstep[0], temp,
                max_iter=fista_maxiter, tol=optimizer_tol,
                bound=bound_val, bounds=list(zip(lb[0], ub[0])),
            )
        elif not is_normal:
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
                if stage_open[d]:
                    B_update[d][:, 1:] = _threshold(B_update[d][:, 1:], tau)
        for d in range(D - 1):
            if ind_inact[d].size:
                B_update[d + 1][ind_inact[d], :] = 0.0

        # 8) Temperature schedule.
        if it_num > burn:
            temp = min(1.0, temp + 0.01 * ((it_num - burn) - 1))

        # 9) Rolling relative-change stopping rule (never during the
        #    slab-start burn-in).
        stop_now = shrinkage_active and all(stage_open)
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
        width_hist.append(tuple(int(value) for value in num_act))
        if staged_shrinkage_window and shrinkage_active:
            for d in range(1, D):
                if stage_open[d]:
                    continue
                recent = [widths[d - 1] for widths in width_hist[-staged_shrinkage_window:]]
                if (
                    stage_open[d - 1]
                    and it_num - stage_iterations[d - 1] >= staged_shrinkage_window
                    and len(set(recent)) == 1
                ):
                    stage_open[d] = True
                    stage_iterations[d] = it_num
                break
        if convergence_width_window and (
            len(width_hist) < convergence_width_window
            or len(set(width_hist[-convergence_width_window:])) > 1
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

        # 11) Optional observed-layer update (SemiDDE g step), from the
        #     committed state and the current CSP allocation.
        if (
            observed_update is not None
            and shrinkage_active
            and it_num % observed_update_every == 0
            and not stop_now
        ):
            X = np.asarray(
                observed_update(
                    X,
                    {
                        "prop": prop,
                        "B_list": B_list,
                        "gamma": gamma,
                        "A_list": A_list,
                        "active_masks": [csp_active_mask(value) for value in z_cusp],
                        "iteration": it_num,
                    },
                ),
                dtype=float,
            )
            if x_sq_sum is not None:
                x_sq_sum = np.sum(X**2, axis=0)
            observed_update_count += 1

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
        "top_prune": top_prune,
        "prune_events": prune_events,
        "convergence_width_window": convergence_width_window,
        "staged_shrinkage_window": staged_shrinkage_window,
        "stage_iterations": tuple(stage_iterations),
        "shrinkage_burn": shrinkage_burn,
        "reorder_after_shrinkage_burn": reorder_after_shrinkage_burn,
        "burn_permutations": burn_permutations,
        "relative_history": [tuple(values) for values in rel_hist],
        "trace": trace,
        "X": X,
        "observed_update_count": observed_update_count,
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
