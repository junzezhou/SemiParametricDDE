"""Parametric Deep Discrete Encoder fitting wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dde.results import DDEResult
from dde.utils import validate_K_list_identifiability, validate_dde_lists
from dde.utils.result_compression import compress_csp_result_components

from .csp import maximal_identifiable_widths
from .fixed_graph import (
    EXACT_EM_MAX_WIDTH,
    fit_fixed_graph_em_multistart,
    graph_anchored_initialize,
    resolve_graph_list,
)
from .normal_csp import fit_parametric_csp_copula
from .normal_init_cop import Normal_init_cop
from .parametric_families import (
    BERNOULLI_FAMILY,
    NORMAL_FAMILY,
    POISSON_FAMILY,
    resolve_parametric_family,
)
from .semi_saem import get_SAEM_parametric


DEFAULT_PARAMETRIC_CONVERGENCE_TOL = 0.005
# Copula-backend Bernoulli/Poisson CSP: full iteration budget, no early stop.
_DISCRETE_COPULA_CSP_MAX_IT = 100


def _standardize_observed_layer(B1, gamma, column_mean, column_sd):
    """Map observed-layer parameters from the original to the z-score scale."""
    B1 = np.asarray(B1, dtype=float).copy()
    gamma = np.asarray(gamma, dtype=float).copy()
    B1[:, 0] = (B1[:, 0] - column_mean) / column_sd
    B1[:, 1:] /= column_sd[:, None]
    gamma /= column_sd**2
    return B1, gamma


def _destandardize_observed_layer(B1, gamma, column_mean, column_sd):
    """Map observed-layer parameters from the z-score to the original scale."""
    B1 = np.asarray(B1, dtype=float).copy()
    gamma = np.asarray(gamma, dtype=float).copy()
    B1[:, 1:] *= column_sd[:, None]
    B1[:, 0] = B1[:, 0] * column_sd + column_mean
    gamma *= column_sd**2
    return B1, gamma


def _validate_pen_list(pen_list: Sequence[float], D: int) -> list[float]:
    if isinstance(pen_list, (str, bytes)):
        raise ValueError("pen_list must be a length-D sequence.")
    try:
        values = np.asarray(pen_list, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("pen_list must be a length-D numeric sequence.") from exc
    if values.ndim != 1:
        raise ValueError("pen_list must be a one-dimensional length-D sequence.")
    penalties = values.tolist()
    if len(penalties) != D:
        raise ValueError(f"pen_list must have length D={D}; got {len(penalties)}.")
    if not np.all(np.isfinite(penalties)) or any(value < 0 for value in penalties):
        raise ValueError("pen_list entries must be finite and nonnegative.")
    return penalties


def _layerwise_warm_start(
    X,
    prop,
    B_list,
    A_list,
    *,
    family_adapter,
    penalties,
    tau,
    C,
    tol,
    max_it,
    bound_val,
    epsilon,
    estep_row_chunk_size,
    sa_burn_in,
    oracle,
    B_list_true,
    G_list_true,
    verbose,
    observed_solver=None,
    dispersion=None,
    support_list=None,
):
    """Layer-wise warm start for fixed-width fits.

    Stage 1 runs the SAEM kernel on the observed layer alone (one latent
    layer with a flat Bernoulli prior) from the spectral initialization;
    ``dispersion`` is its starting dispersion (Normal variances, ``None``
    for Poisson) and the stage-1 estimate is returned with the state.
    Stage 2 fits the deeper layers as a fixed-width Bernoulli DDE on the
    stage-1 latent states, with the Bernoulli spectral initializer (anchor
    latent-class binarization, labelled by the true deep coefficients when
    ``oracle``) and the same burn-in.  The deep stage is not itself warm
    started layer by layer: binary children are far less informative per
    observation than counts or Gaussian responses, and a flat-prior fit of one binary layer loses
    the layers above it (D=4 design check: recursive warm start 0.89/0.67/0.54
    versus 0.99/0.92/0.92 per-layer graph accuracy for the deep layers).  The assembled state is
    the starting point of the full SAEM.  Identifiability of a DDE is
    layer-wise (the collapsed one-layer model identifies the observed layer;
    each deeper layer is identified from the layer below), so each stage
    solves an identified sub-problem from its own initialization instead of
    letting the deep layers drift while the observed layer is still far from
    its optimum.  With a fixed graph (``support_list``) stage 1 holds the
    observed-layer graph and stage 2 passes the deeper graphs on as its
    ``G_list``.
    """
    D = len(B_list)
    if support_list is None:
        support_list = [None] * D
    A1 = np.asarray(A_list[0], dtype=float)
    prop_stage1 = np.clip(np.mean(A1, axis=0), 1e-3, 1.0 - 1e-3)
    prop_stage1, B_stage1, dispersion, A_stage1, iterations_stage1 = get_SAEM_parametric(
        X,
        prop_stage1,
        [np.asarray(B_list[0], dtype=float)],
        dispersion,
        [penalties[0]],
        tau,
        [A1],
        C,
        tol,
        max_iter=max_it,
        bound_val=bound_val,
        verbose=False,
        estep_row_chunk_size=estep_row_chunk_size,
        family=family_adapter,
        sa_burn_in=sa_burn_in,
        observed_solver=observed_solver,
        support_list=[support_list[0]],
    )
    A1 = np.asarray(A_stage1[0], dtype=float)
    deep_widths = [int(np.asarray(B).shape[1] - 1) for B in B_list[1:]]
    deep_kwargs = dict(
        family="bernoulli",
        D=D - 1,
        K_list=deep_widths,
        dimension_method="fixed",
        pen_list=list(penalties[1:]),
        tau=tau,
        C=C,
        tol=tol,
        max_it=max_it,
        bound_val=bound_val,
        epsilon=epsilon,
        sa_burn_in=sa_burn_in,
        fixed_init_binarize="anchor_lca",
        fixed_warm_start="none",
        estep_row_chunk_size=estep_row_chunk_size,
        verbose=False,
    )
    if any(support is not None for support in support_list[1:]):
        deep_kwargs["G_list"] = list(support_list[1:])
    if oracle:
        deep_kwargs.update(
            initialize="oracle",
            B_list_true=list(B_list_true[1:]),
            G_list_true=list(G_list_true[1:]),
        )
    deep = fit_parametric_dde(A1, **deep_kwargs)
    if verbose:
        print(
            f"Layer-wise warm start: stage 1 {iterations_stage1} sweeps, "
            f"stage 2 {deep.diagnostics['iterations']} sweeps"
        )
    prop_out = np.asarray(deep.prop, dtype=float)
    B_out = [np.asarray(B_stage1[0], dtype=float)] + [
        np.asarray(B, dtype=float) for B in deep.B_list
    ]
    A_out = [A1] + [np.asarray(A, dtype=float) for A in deep.A_list]
    diagnostics = {
        "stage1_iterations": int(iterations_stage1),
        "stage2_iterations": int(deep.diagnostics["iterations"]),
        "stage2_initialization": deep.diagnostics.get("initialization"),
    }
    return prop_out, B_out, dispersion, A_out, diagnostics


def fit_parametric_dde(
    Y,
    *,
    family,
    D: int,
    K_list=None,
    dimension_method: str = "fixed",
    pen_list=None,
    t_spike=None,
    temp=None,
    tau=None,
    C=None,
    G_list=None,
    graph_algorithm=None,
    n_init=None,
    tol=None,
    max_it=None,
    initialize: str = "data",
    epsilon: float = 1e-4,
    prop_in=None,
    B_list_in=None,
    gamma_in=None,
    A_list_in=None,
    B_list_true=None,
    G_list_true=None,
    bound_val: float = 10.0,
    random_state=None,
    optimizer_maxiter=None,
    optimizer_tol: float = 1e-6,
    convergence_window: int = 7,
    convergence_mean_tol: float = 0.01,
    csp_backend=None,
    csp_initializer=None,
    csp_threshold_rule=None,
    csp_mstep_solver: str = "fista_lower",
    csp_mstep_latent: str = "draw",
    csp_observed_noise_scale: float = 0.0,
    csp_fista_maxiter=None,
    csp_slab_shape: float = 1.0,
    csp_slab_rate: float = 5.0,
    csp_penalty_scaling_power=None,
    csp_penalty_scaling_reference: float = 500.0,
    csp_penalty_scaling_layers=None,
    csp_penalty_scaling_allocation: bool = False,
    csp_latent_transition=None,
    csp_mc_state_update=None,
    csp_reinit_top_after: int = 0,
    csp_reinit_top_rule: str = "family",
    csp_reinit_top_source: str = "state",
    csp_merge_cosine: float = 0.0,
    csp_merge_after: int = 10,
    csp_merge_layers: str = "top",
    csp_shrinkage_burn=None,
    csp_reorder_after_shrinkage_burn: bool = True,
    csp_top_prune=None,
    csp_top_prune_after: int = 10,
    csp_top_prune_every: int = 5,
    csp_top_prune_em_iter: int = 20,
    csp_convergence_width_window: int = 0,
    csp_staged_shrinkage_window: int = 0,
    n_jobs: int = 1,
    standardize=None,
    burn: int = 5,
    sa_burn_in=None,
    fixed_init_binarize=None,
    fixed_warm_start=None,
    fixed_observed_solver=None,
    estep_row_chunk_size: int = 2048,
    estep_coordinate_chunk_size: int = 1,
    keep_trace: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> DDEResult:
    """Fit a fixed-width or CSP-shrunk parametric DDE.

    The family adapter validates the observed matrix, prepares initialization,
    and supplies observed-layer operations to the fixed-width SAEM kernel or
    the copula CSP kernel.
    In contrast, SemiDDE and CopDDE construct latent-Gaussian
    pseudo-observations before using their respective kernels.

    ``initialize="oracle"`` is intended only for simulation studies that need
    estimates aligned with known generating parameters.  Ordinary data fits
    should use the default ``initialize="data"``.

    For fixed-width fits, ``tol`` is the postprocessed parameter-update RMSE.
    Its default is ``0.005`` for a dimension-independent convergence scale.
    ``tol=None`` and ``max_it=None`` select ``0.005`` and 50 iterations,
    except for the copula-backend Bernoulli/Poisson CSP (their default CSP
    path), which runs 100 iterations with ``tol=0`` (no early stop).

    Normal CSP has a single kernel, ``csp_backend="copula"``: the CopDDE
    D-layer CSP algorithm with the rank-likelihood resampling removed because
    the Normal DDE observes its Gaussian layer.  It initializes with the Normal family's
    own data initializer (``csp_initializer="family"``; the CopDDE
    ``Normal_init_D`` rules remain available as ``csp_initializer="copula"``
    but leave more spurious deep columns active) and uses the CopDDE
    L-BFGS-B budget ``optimizer_maxiter=15`` for the deepest layer unless
    overridden.  Layers 1..D-1 are solved exactly by the proximal
    weighted-lasso solver (``csp_mstep_solver="fista_lower"``,
    ``csp_fista_maxiter=300``) with a penalty proportional to ``N``
    (``csp_penalty_scaling_power=1``, ``csp_penalty_scaling_layers="lower"``,
    reference 500): the exact solve removes duplicate latent columns and the
    scaled penalty removes weak deep columns that an O(1) prior cannot
    remove against an O(N) likelihood.  The deepest layer keeps the CopDDE
    inexact step and unscaled penalty, because an exact solve collapses it
    at large ``N``.  ``csp_mstep_solver="lbfgsb"`` with
    ``csp_penalty_scaling_power=0`` restores the CopDDE M-step in every
    layer; ``csp_slab_shape`` and ``csp_slab_rate`` change the CUSP slab
    hyperprior (CopDDE 1 and 5).
    Bernoulli and Poisson CSP also default to the copula kernel
    (``csp_backend="copula"``), with a logistic or log-link observed layer
    and these family defaults: the sequential mean-field transition
    (``csp_latent_transition="sequential_mean_field"``; the synchronous
    CopDDE draws attenuate the deep coefficients of discrete data until
    weak layers die), a ``sqrt(N)`` penalty in every layer
    (``csp_penalty_scaling_power=0.5``, ``csp_penalty_scaling_layers="all"``),
    hard thresholding of every layer (``csp_threshold_rule="all"``), the
    dense marginal-likelihood deletion test for the deepest layer
    (``csp_top_prune="bic_dense"``), 100 iterations without early stop,
    and for Bernoulli a slab-start burn-in over the first fifth of the
    budget (``min(20, max_it // 5)`` iterations in every layer, except the
    deepest layer of a model with three or more layers; Poisson: none).  The former MCEM CSP backends (``"mcem"``,
    ``"mcem_reinitialize"``, ``"mcem_cap_only"``) were removed; the git tag
    ``csp-mcem-last`` holds the last version with them.
    ``csp_shrinkage_burn`` holds every column in the CUSP slab (no
    allocation, thresholding or convergence check) for that many initial
    iterations before the CUSP updates start (a slab-start warm-up; a
    length-``D`` sequence sets it per layer);
    ``csp_reorder_after_shrinkage_burn`` then orders the
    columns by strength once.  ``csp_top_prune="bic"`` (or
    ``"bic_dense"``) adds the copula kernel's marginal-likelihood deletion
    test for the deepest layer (``csp_top_prune_after``,
    ``csp_top_prune_every``, ``csp_top_prune_em_iter``),
    ``csp_convergence_width_window`` makes its stopping rule wait for
    unchanged active widths, and ``csp_staged_shrinkage_window`` starts the
    CUSP updates layer by layer from the data upward; see
    :func:`fit_parametric_csp_copula`.  ``None`` for the copula options
    selects the family defaults above (Normal: synchronous transition,
    N-proportional penalty on layers 1..D-1, ``csp_threshold_rule=
    "first_two"``, no deletion test, no burn-in).

    Fixed-width fits take three further controls, with Poisson defaults
    chosen by the D=2..5 recovery study of ``Results/dde_poisson_fixed_*``,
    Bernoulli defaults by ``Results/bernoulli_fixed_recon/REPORT.md`` and
    Normal defaults by ``Results/dde_normal_fixed_recon/REPORT.md``:
    ``sa_burn_in`` (default 10) runs that many initial sweeps with step size
    one before the ``1/t`` stochastic approximation starts, so draws made
    from the initialization do not keep a lasting weight in the averaged
    objectives;
    ``fixed_init_binarize`` (default ``"anchor_mixture"`` for Poisson and
    Normal, ``"anchor_lca"`` for Bernoulli) reads the initial latent states
    off each column's anchor variables (two-component Poisson or Gaussian
    mixtures on the observed layer, a two-class latent class model over
    binary anchors in the deeper Normal layers and for Bernoulli) instead of
    thresholding the projected working response at zero, which otherwise
    assigns about half of the observations to every latent column and
    starts rare columns at the wrong prevalence (the copula-backend
    Bernoulli and Poisson CSP accept the anchor rules as an opt-in for
    their maximal-width start; every CSP default stays ``"projection"``);
    ``fixed_warm_start`` (default ``"layerwise"`` for Poisson and Normal
    with ``D >= 2``, ``"none"`` otherwise) estimates the observed layer
    first with a flat latent prior, then the deeper layers as a Bernoulli
    DDE on the resulting latent states, and only then runs the full SAEM
    from the assembled state; every stage burns in.  The warm start stays
    off for Bernoulli data, where a flat-prior fit of one binary layer loses
    the layers above it.  It is skipped when a full state (``prop_in``,
    ``B_list_in``, ``A_list_in``) is supplied; the burn-in still applies to
    a supplied state (pass ``sa_burn_in=0`` for the plain MATLAB-style
    schedule).  For Normal data with ``N <= 1000`` the unit-step sweeps of
    the full SAEM occasionally drive a deep row with a rare child into a
    separated (near-deterministic) link that the latent draws then
    reinforce, which raises the mean deep-layer RMSE at ``N = 500`` above
    that of the plain kernel while the graph accuracy is higher.
    ``fixed_observed_solver`` (Poisson, default ``"rowwise"``: the MATLAB
    per-row L-BFGS-B with ten steps on the stochastic-approximation
    objective) also offers ``"active_set"`` (exact solve on each row's
    support with KKT screening), ``"hybrid"`` (row-wise during the burn-in,
    active set afterwards) and ``"active_set_wide"``; on the D=4 design the
    exact variants lose observed-layer accuracy (layer-1 RMSE 0.12 versus
    0.07 at N=16000), so they are kept for wide layers where the dense
    solve is too slow.

    ``G_list`` fixes a user-specified graph (fixed-width fits only):
    ``G_list[0]`` is the ``J x K_1`` observed-layer graph and ``G_list[d]``
    the ``K_d x K_{d+1}`` graph of layer ``d + 1``, binary with a nonempty
    column for every latent variable.  A shorter list (or trailing ``None``
    entries) fixes only the shallowest layers; the deeper graphs are still
    estimated, and ``K_list`` gives their widths (the fixed widths follow
    from the graphs).  The fixed layers are fitted without penalty or
    threshold, their coefficients outside the graph held at exactly zero
    (the confirmatory EM of the paper's TIMSS analysis), so ``tau`` and the
    ``pen_list`` entries of fixed layers are not used; ``tau`` and
    ``pen_list`` may be omitted when every layer is fixed.  The data
    initialization is then graph-anchored (see
    :mod:`dde.algorithms.fixed_graph`): each latent column is read off its
    children with the fewest parents, oriented so that the value one raises
    their means, which fixes the column labels to the graph's.  The
    layer-wise warm start passes the graphs to its stages.
    ``graph_algorithm`` chooses the estimator: ``"em"`` is the exact EM of
    :func:`dde.algorithms.fixed_graph.fit_fixed_graph_em` (every layer
    fixed; it enumerates the ``2**K_d`` states of each layer and reaches
    the maximum-likelihood estimate on the graph without Monte Carlo
    error), ``"saem"`` the fixed-width SAEM kernel above (any width, and
    the only choice when deeper graphs are estimated), and the default
    ``"auto"`` takes ``"em"`` when every layer is fixed and no layer is
    wider than ``EXACT_EM_MAX_WIDTH`` (10).  For ``"em"`` ``C`` is not
    used, ``max_it`` defaults to 500 M-steps and ``tol`` (default
    ``1e-6``) bounds the log-likelihood gain per observation.  The EM
    likelihood on a fixed graph can be multimodal on real data (TIMSS
    response times, ``K_list=[7, 1]``: 20 random starts reach 9 local
    optima spanning 140 log-likelihood units, and the graph-anchored start
    is not the best), so ``"em"`` runs ``n_init`` starts (default 10): the
    graph-anchored start (or the supplied state) and random starts on the
    graph (:func:`dde.algorithms.fixed_graph.random_graph_start`, drawn
    from ``random_state`` or else from the global NumPy generator), and
    keeps the largest log-likelihood.

    ``standardize`` (default ``True`` for the copula backend, ``False``
    otherwise) runs a Normal CSP fit on column-standardized observations, the
    scale on which the CopDDE latent Gaussian layer lives and on which the
    CopDDE ``t_spike`` and ``tau`` values were calibrated.  The returned
    observed-layer coefficients and variances are mapped back to the original
    scale; the hard threshold therefore acts on standardized loadings.
    """
    family_adapter = resolve_parametric_family(family)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(
            f"Unexpected {family_adapter.name} DDE argument(s): {unknown}."
        )

    X = family_adapter.validate_y(Y)

    D = int(D)
    if D < 1:
        raise ValueError("D must be a positive integer.")
    method = str(dimension_method).strip().lower()
    if method not in {"fixed", "csp"}:
        raise ValueError(
            f"{family_adapter.name} DDE dimension_method must be 'fixed' or 'csp'."
        )
    if method not in family_adapter.supported_dimension_methods:
        supported = " or ".join(
            repr(value) for value in family_adapter.supported_dimension_methods
        )
        raise NotImplementedError(
            f"{family_adapter.name} DDE currently supports "
            f"dimension_method={supported} only."
        )
    support_list = [None] * D
    if method == "fixed":
        if t_spike is not None or temp is not None:
            raise ValueError(
                "t_spike and temp are only used by dimension_method='csp'."
            )
        if G_list is not None:
            K_tuple, support_list = resolve_graph_list(
                G_list, D=D, J=X.shape[1], K_list=K_list
            )
        else:
            if K_list is None:
                raise ValueError("K_list is required when dimension_method='fixed'.")
            K_tuple = validate_K_list_identifiability(K_list, D=D, J=X.shape[1])
        all_fixed = all(support is not None for support in support_list)
        if pen_list is None and all_fixed:
            penalties = [0.0] * D
        else:
            penalties = _validate_pen_list(pen_list, D)
    elif method == "csp":
        if G_list is not None:
            raise ValueError(
                "G_list fixes the graph and the widths; use dimension_method='fixed'."
            )
        if pen_list is not None:
            raise ValueError("pen_list is only used by dimension_method='fixed'.")
        if t_spike is None or temp is None:
            raise ValueError(
                f"{family_adapter.name} DDE with dimension_method='csp' "
                "requires t_spike and temp."
            )
        # Optional maximal widths; by default the recursive maximal
        # identifiable widths floor(J/3), floor(K_1/3), ...
        K_tuple = (
            maximal_identifiable_widths(X.shape[1], D)
            if K_list is None
            else validate_K_list_identifiability(K_list, D=D, J=X.shape[1])
        )
        penalties = None

    if graph_algorithm is None:
        graph_algorithm = "auto"
    graph_algorithm = str(graph_algorithm).strip().lower()
    if graph_algorithm not in {"auto", "em", "saem"}:
        raise ValueError("graph_algorithm must be 'auto', 'em' or 'saem'.")
    every_layer_fixed = all(support is not None for support in support_list)
    if graph_algorithm == "auto":
        graph_algorithm = (
            "em"
            if every_layer_fixed and max(K_tuple) <= EXACT_EM_MAX_WIDTH
            else "saem"
        )
    if graph_algorithm == "em" and not every_layer_fixed:
        raise ValueError(
            "graph_algorithm='em' needs G_list to fix every layer; partially fixed "
            "graphs use 'saem'."
        )
    if graph_algorithm == "em" and max(K_tuple) > 16:
        raise ValueError(
            "graph_algorithm='em' enumerates 2**K_d states per layer; widths above 16 "
            "need graph_algorithm='saem'."
        )
    exact_em = graph_algorithm == "em"
    if n_init is not None and not exact_em:
        raise ValueError("n_init applies to the exact EM on a fixed graph only.")
    n_init_value = 10 if n_init is None else int(n_init)
    if n_init_value < 1:
        raise ValueError("n_init must be a positive integer.")
    # The copula-backend Bernoulli/Poisson CSP (the default CSP backend of
    # those families) runs its full iteration budget: column pruning pauses
    # for 15-25 iterations while the relative coefficient change is already
    # small, and the relative-change rule stops earlier at larger N, leaving
    # over-selected layers.  ``tol=0`` disables that rule.
    discrete_copula_csp = (
        method == "csp"
        and family_adapter.key in ("bernoulli", "poisson")
        and (csp_backend is None or str(csp_backend).strip().lower() == "copula")
    )
    if max_it is None:
        if exact_em:
            max_it = 500
        else:
            max_it = _DISCRETE_COPULA_CSP_MAX_IT if discrete_copula_csp else 50
    if tol is None:
        if exact_em:
            tol = 1e-6
        else:
            tol = 0.0 if discrete_copula_csp else DEFAULT_PARAMETRIC_CONVERGENCE_TOL
    graph_fixed = any(support is not None for support in support_list)
    if tau is None:
        if not all(support is not None for support in support_list):
            raise ValueError("tau is required unless G_list fixes every layer.")
        # Unused: fixed layers are neither penalized nor thresholded.
        tau = 1.0
        tau_used = None
    else:
        tau_used = float(tau)
    tau = float(tau)
    tol = float(tol)
    epsilon = float(epsilon)
    bound_val = float(bound_val)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    if not np.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly between 0 and 0.5.")
    if not np.isfinite(bound_val) or bound_val <= 0:
        raise ValueError("bound_val must be finite and positive.")
    # One latent draw per stochastic E-step is the manuscript's default.
    C = 1 if C is None else int(C)
    if C < 1:
        raise ValueError("C must be a positive integer.")
    max_it = int(max_it)
    if max_it < 1:
        raise ValueError("max_it must be a positive integer.")

    is_poisson = family_adapter.key == "poisson"
    is_bernoulli = family_adapter.key == "bernoulli"
    is_normal = family_adapter.key == "normal"
    if sa_burn_in is None:
        # A graph-anchored Normal start already pins the layer-1 states; the
        # unit-step sweeps then only fit single draws, which separate the
        # rare-child rows of layer 2 (fixed-graph SAEM, (54, 18, 6) B_s
        # design, N = 500: RMSE 0.33 with 10 sweeps, 0.20 without).
        if method != "fixed" or (is_normal and graph_fixed):
            sa_burn_in = 0
        else:
            sa_burn_in = 10
    sa_burn_in = int(sa_burn_in)
    if sa_burn_in < 0:
        raise ValueError("sa_burn_in must be a nonnegative integer.")
    if sa_burn_in and method != "fixed":
        raise ValueError("sa_burn_in applies to dimension_method='fixed' only.")
    if fixed_init_binarize is None:
        if method == "fixed" and (is_poisson or is_normal):
            fixed_init_binarize = "anchor_mixture"
        elif method == "fixed" and is_bernoulli:
            fixed_init_binarize = "anchor_lca"
        else:
            fixed_init_binarize = "projection"
    fixed_init_binarize = str(fixed_init_binarize).strip().lower()
    if fixed_init_binarize not in {"projection", "anchor_mixture", "anchor_lca"}:
        raise ValueError(
            "fixed_init_binarize must be 'projection', 'anchor_mixture' (Poisson, Normal) or 'anchor_lca' (Bernoulli)."
        )
    # The anchor initializers are opt-in for the copula-backend Bernoulli and
    # Poisson CSP (the maximal-width start of that kernel).
    anchor_csp = discrete_copula_csp
    if fixed_init_binarize == "anchor_mixture" and (
        not (is_poisson or is_normal) or (method != "fixed" and not anchor_csp)
    ):
        raise ValueError(
            "fixed_init_binarize='anchor_mixture' is implemented for fixed-width Poisson and Normal fits "
            "and the copula-backend Poisson CSP only."
        )
    if fixed_init_binarize == "anchor_lca" and (
        not is_bernoulli or (method != "fixed" and not anchor_csp)
    ):
        raise ValueError(
            "fixed_init_binarize='anchor_lca' is implemented for fixed-width Bernoulli fits "
            "and the copula-backend Bernoulli CSP only."
        )
    if fixed_observed_solver is not None:
        fixed_observed_solver = str(fixed_observed_solver).strip().lower()
        if not is_poisson or method != "fixed":
            raise ValueError("fixed_observed_solver is a fixed-width Poisson option.")
        if fixed_observed_solver not in {"active_set", "rowwise", "hybrid", "active_set_wide"}:
            raise ValueError("fixed_observed_solver must be 'active_set', 'rowwise', 'hybrid' or 'active_set_wide'.")
    if fixed_warm_start is None:
        fixed_warm_start = (
            "layerwise"
            if (is_poisson or is_normal) and method == "fixed" and D >= 2
            else "none"
        )
    fixed_warm_start = str(fixed_warm_start).strip().lower()
    if fixed_warm_start not in {"none", "layerwise"}:
        raise ValueError("fixed_warm_start must be 'none' or 'layerwise'.")
    if fixed_warm_start == "layerwise" and (method != "fixed" or D < 2):
        raise ValueError(
            "fixed_warm_start='layerwise' needs dimension_method='fixed' and D >= 2."
        )

    if csp_backend is None:
        backend = "copula" if method == "csp" else None
    else:
        backend = str(csp_backend).strip().lower()
        if backend != "copula":
            raise ValueError(
                f"csp_backend={csp_backend!r} is not available: CSP runs on the copula "
                "kernel (csp_backend='copula', the default).  The MCEM backends "
                "('mcem', 'mcem_reinitialize', 'mcem_cap_only') were removed; the git "
                "tag csp-mcem-last holds the last version with them."
            )
    if method == "csp" and family_adapter.key not in ("normal", "bernoulli", "poisson"):
        raise NotImplementedError(
            "CSP is implemented for the Normal, Bernoulli and Poisson families only."
        )
    if csp_initializer is None:
        csp_initializer = "family"
    csp_initializer = str(csp_initializer).strip().lower()
    if csp_initializer not in {"copula", "family"}:
        raise ValueError("csp_initializer must be 'copula' or 'family'.")
    if csp_initializer == "copula" and family_adapter.key != "normal":
        raise ValueError(
            "csp_initializer='copula' is available for the Normal family only."
        )
    if optimizer_maxiter is None:
        optimizer_maxiter = 15 if backend == "copula" else 100
    optimizer_maxiter = int(optimizer_maxiter)
    if optimizer_maxiter < 1:
        raise ValueError("optimizer_maxiter must be a positive integer.")
    if standardize is None:
        standardize = (
            method == "csp" and backend == "copula" and family_adapter.key == "normal"
        )
    standardize = bool(standardize)
    if standardize and (method != "csp" or family_adapter.key != "normal"):
        raise ValueError("standardize is only supported by Normal CSP fits.")
    X_raw = X
    column_mean = None
    column_sd = None
    if standardize:
        column_mean = np.mean(X_raw, axis=0)
        column_sd = np.std(X_raw, axis=0)
        column_sd = np.where(column_sd > 0, column_sd, 1.0)
        X = (X_raw - column_mean[None, :]) / column_sd[None, :]

    supplied_names = ["prop_in", "B_list_in"]
    supplied = [prop_in, B_list_in]
    if family_adapter.has_dispersion:
        supplied_names.append("gamma_in")
        supplied.append(gamma_in)
    elif gamma_in is not None:
        raise ValueError(
            f"{family_adapter.name} has no dispersion parameter; gamma_in must be omitted."
        )
    supplied_names.append("A_list_in")
    supplied.append(A_list_in)
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        joined_names = ", ".join(supplied_names[:-1])
        raise ValueError(
            f"{joined_names}, and {supplied_names[-1]} must be supplied together."
        )

    if all(value is not None for value in supplied):
        initialization = "provided"
        validate_dde_lists(
            X,
            B_list_in,
            A_list=A_list_in,
            prop=prop_in,
            gamma=None,
            K_list=K_tuple,
            observed_name="Y",
            require_binary_A=True,
        )
        prop = np.asarray(prop_in, dtype=float)
        B_list = [np.asarray(B, dtype=float) for B in B_list_in]
        gamma = (
            family_adapter.validate_dispersion(gamma_in, X.shape[1])
            if family_adapter.has_dispersion
            else None
        )
        A_list = [np.asarray(A, dtype=float) for A in A_list_in]
        if standardize:
            B_list[0], gamma = _standardize_observed_layer(
                B_list[0], gamma, column_mean, column_sd
            )
    else:
        initialization = str(initialize).lower()
        if graph_fixed and initialization != "data":
            raise ValueError(
                "G_list labels the latent columns itself; use initialize='data' "
                "(the graph-anchored initialization) or supply a full state."
            )
        if initialization == "data":
            if B_list_true is not None or G_list_true is not None:
                raise ValueError(
                    "B_list_true and G_list_true are only used with initialize='oracle'."
                )
            if graph_fixed:
                prop, B_list, gamma, A_list, graph_init = graph_anchored_initialize(
                    X,
                    support_list,
                    K_tuple,
                    family=family_adapter,
                    epsilon=epsilon,
                    bound_val=bound_val,
                )
                initialization = "data_graph"
            elif method == "csp" and csp_initializer == "copula":
                prop, B_list, gamma, A_list = Normal_init_cop(
                    X, K_tuple, epsilon=epsilon
                )
                initialization = "data_copula"
            else:
                prop, B_list, gamma, A_list, _ = family_adapter.initialize(
                    X,
                    K_tuple,
                    epsilon=epsilon,
                    mode="data",
                    random_state=random_state if method == "csp" else None,
                    binarize=fixed_init_binarize,
                )
        elif initialization == "oracle":
            if B_list_true is None or G_list_true is None:
                raise ValueError(
                    "initialize='oracle' requires B_list_true and G_list_true."
                )
            prop, B_list, gamma, A_list, _ = family_adapter.initialize(
                X,
                K_tuple,
                B_list_true=B_list_true,
                G_list_true=G_list_true,
                epsilon=epsilon,
                mode="oracle",
                random_state=random_state if method == "csp" else None,
                binarize=fixed_init_binarize,
            )
        else:
            raise ValueError("initialize must be 'data' or 'oracle'.")

    if method == "fixed":
        if random_state is not None and not exact_em:
            raise ValueError(
                "random_state is supported by the CSP path and the exact EM on a "
                "fixed graph only."
            )
        if exact_em:
            if random_state is None:
                # Tie the random starts to the global generator, like the
                # fixed kernels' draws.
                start_rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
            else:
                start_rng = np.random.default_rng(random_state)
            prop, B_list, gamma, A_list, iterations, em_diagnostics = (
                fit_fixed_graph_em_multistart(
                    X,
                    prop,
                    B_list,
                    gamma,
                    support_list,
                    family=family_adapter,
                    n_init=n_init_value,
                    rng=start_rng,
                    max_iter=max_it,
                    tol=tol,
                    bound_val=bound_val,
                    row_chunk_size=estep_row_chunk_size,
                    verbose=bool(verbose),
                )
            )
            diagnostics = {
                "iterations": iterations,
                "D": D,
                "K_list": list(K_tuple),
                "family": family_adapter.name,
                "dimension_method": "fixed",
                "initialization": initialization,
                "graph_algorithm": "em",
                "convergence_metric": "loglik_gain_per_observation",
                "convergence_tol": tol,
                "converged": em_diagnostics["converged"],
                "loglik": em_diagnostics["loglik"],
                "n_init": n_init_value,
                "start_logliks": em_diagnostics["start_logliks"],
                "best_start": em_diagnostics["best_start"],
                "loglik_history": em_diagnostics["loglik_history"],
                "posterior_marginals": em_diagnostics["posterior_marginals"],
                "fixed_graph_layers": [True] * D,
                "G_list": [support.astype(np.uint8) for support in support_list],
            }
            if initialization == "data_graph":
                diagnostics["graph_init_anchors"] = graph_init["anchors"]
            return DDEResult(
                model="DDE",
                K_list=list(K_tuple),
                prop=prop,
                B_list=B_list,
                gamma=gamma,
                A_list=A_list,
                transformed=X.copy(),
                diagnostics=diagnostics,
                raw={
                    "X": X.copy(), "prop": prop, "B_list": B_list, "gamma": gamma,
                    "A_list": A_list, "iterations": iterations,
                },
            )
        fixed_trace = [] if keep_trace else None
        warm_start_diagnostics = None
        if fixed_warm_start == "layerwise" and initialization != "provided":
            prop, B_list, gamma, A_list, warm_start_diagnostics = _layerwise_warm_start(
                X,
                prop,
                B_list,
                A_list,
                family_adapter=family_adapter,
                penalties=penalties,
                tau=tau,
                C=C,
                tol=tol,
                max_it=max_it,
                bound_val=bound_val,
                epsilon=epsilon,
                estep_row_chunk_size=estep_row_chunk_size,
                sa_burn_in=sa_burn_in,
                oracle=(initialization == "oracle"),
                B_list_true=B_list_true,
                G_list_true=G_list_true,
                verbose=bool(verbose),
                observed_solver=fixed_observed_solver,
                dispersion=gamma,
                support_list=support_list,
            )
        prop, B_list, gamma, A_list, iterations = get_SAEM_parametric(
            X,
            prop,
            B_list,
            gamma,
            penalties,
            tau,
            A_list,
            C,
            tol,
            max_iter=max_it,
            bound_val=bound_val,
            verbose=bool(verbose),
            estep_row_chunk_size=estep_row_chunk_size,
            trace=fixed_trace,
            family=family_adapter,
            sa_burn_in=sa_burn_in,
            observed_solver=fixed_observed_solver,
            support_list=support_list,
        )

        X_result = X.copy()
        diagnostics = {
            "iterations": iterations,
            "D": D,
            "K_list": list(K_tuple),
            "family": family_adapter.name,
            "dimension_method": "fixed",
            "initialization": initialization,
            "latent_transition": "systematic_coordinate_gibbs",
            "mc_state_update": "last_draw",
            "convergence_metric": "postprocessed_parameter_rmse",
            "convergence_tol": tol,
            "sa_burn_in": sa_burn_in,
            "fixed_init_binarize": fixed_init_binarize,
            "fixed_warm_start": (
                fixed_warm_start if initialization != "provided" else "none"
            ),
            "fixed_observed_solver": (
                (fixed_observed_solver or "rowwise") if is_poisson else "family"
            ),
            "tau": tau_used,
            "fixed_graph_layers": [support is not None for support in support_list],
        }
        if graph_fixed:
            diagnostics["graph_algorithm"] = "saem"
            diagnostics["G_list"] = [
                None if support is None else support.astype(np.uint8)
                for support in support_list
            ]
            if initialization == "data_graph":
                diagnostics["graph_init_anchors"] = graph_init["anchors"]
        if warm_start_diagnostics is not None:
            diagnostics["warm_start"] = warm_start_diagnostics
        if fixed_trace is not None:
            diagnostics["trace"] = fixed_trace
        return DDEResult(
            model="DDE",
            K_list=list(K_tuple),
            prop=prop,
            B_list=B_list,
            gamma=gamma,
            A_list=A_list,
            transformed=X_result,
            diagnostics=diagnostics,
            raw={
                "X": X_result, "prop": prop, "B_list": B_list, "gamma": gamma,
                "A_list": A_list, "iterations": iterations,
            },
        )

    # Family defaults for the copula kernel.  The Gaussian observed layer
    # identifies the latent states well enough for the CopDDE single
    # synchronous mean-field draw and benefits from the N-scaled
    # lower-layer penalty.  On Bernoulli/Poisson data the synchronous
    # draws of consecutive layers are independent given the previous
    # state, which attenuates the deep coefficients every iteration (weak
    # deep layers die); the sequential transition draws each layer given
    # the fresh draw below.  The deepest layer is thresholded and tested
    # by the dense BIC deletion test (the CUSP allocation keeps any column
    # above a fixed coefficient scale at every N), and the sqrt(N)
    # penalty applies to every layer
    # (Results/csp_redesign/bench_v2, arm cop_seqPD100).
    discrete = family_adapter.key in ("bernoulli", "poisson")
    if csp_latent_transition is None:
        csp_latent_transition = (
            "sequential_mean_field" if discrete else "synchronous_mean_field"
        )
    if csp_penalty_scaling_power is None:
        csp_penalty_scaling_power = 0.5 if discrete else 1.0
    if csp_penalty_scaling_layers is None:
        csp_penalty_scaling_layers = "all" if discrete else "lower"
    if csp_threshold_rule is None:
        csp_threshold_rule = "all" if discrete else "first_two"
    if csp_top_prune is None:
        csp_top_prune = "bic_dense" if discrete else "none"
    if csp_shrinkage_burn is None:
        # Bernoulli: hold the columns in the slab for the first fifth of the
        # budget (20 of the default 100 iterations).  Without it the CUSP
        # prunes the rarest latents within the first ten iterations, before
        # their states have settled, and they never return (B_s design
        # (18, 6, 2), J=64: width error 3.1-3.6 at every N; CopDDE design
        # (10, 3), J=50: layer 2 one column short at every N).  The deepest
        # layer of a D >= 3 model starts its CUSP updates at once: its
        # children are the noisiest states, and a burn-in there leaves an
        # extra column that the deletion test keeps (CopDDE design
        # (18, 6, 2), J=108).  Results/dde_modification.
        if family_adapter.key == "bernoulli":
            burn_length = min(20, max_it // 5)
            csp_shrinkage_burn = (
                burn_length if D <= 2 else [burn_length] * (D - 1) + [0]
            )
        else:
            csp_shrinkage_burn = 0
    raw = fit_parametric_csp_copula(
        X,
        prop,
        gamma,
        B_list,
        A_list,
        family=family_adapter,
        C=C,
        it=max_it,
        t_spike=t_spike,
        temp=temp,
        tau=tau,
        optimizer_maxiter=optimizer_maxiter,
        optimizer_tol=optimizer_tol,
        bound_val=bound_val,
        n_jobs=n_jobs,
        convergence_window=convergence_window,
        convergence_std=tol,
        burn=burn,
        threshold_rule=csp_threshold_rule,
        mstep_solver=csp_mstep_solver,
        mstep_latent=csp_mstep_latent,
        observed_noise_scale=csp_observed_noise_scale,
        fista_maxiter=csp_fista_maxiter,
        slab_shape=csp_slab_shape,
        slab_rate=csp_slab_rate,
        penalty_scaling_power=csp_penalty_scaling_power,
        penalty_scaling_reference=csp_penalty_scaling_reference,
        penalty_scaling_layers=csp_penalty_scaling_layers,
        penalty_scaling_allocation=csp_penalty_scaling_allocation,
        latent_transition=csp_latent_transition,
        mc_state_update=csp_mc_state_update,
        estep_row_chunk_size=estep_row_chunk_size,
        reinit_top_after=csp_reinit_top_after,
        reinit_top_rule=csp_reinit_top_rule,
        reinit_top_source=csp_reinit_top_source,
        merge_cosine=csp_merge_cosine,
        merge_after=csp_merge_after,
        merge_layers=csp_merge_layers,
        shrinkage_burn=csp_shrinkage_burn,
        reorder_after_shrinkage_burn=csp_reorder_after_shrinkage_burn,
        top_prune=csp_top_prune,
        top_prune_after=csp_top_prune_after,
        top_prune_every=csp_top_prune_every,
        top_prune_em_iter=csp_top_prune_em_iter,
        convergence_width_window=csp_convergence_width_window,
        staged_shrinkage_window=csp_staged_shrinkage_window,
        random_state=random_state,
        keep_trace=keep_trace,
        verbose=bool(verbose),
    )
    X_result = X_raw.copy()
    if standardize:
        raw["B_list"][0], raw["gamma"] = _destandardize_observed_layer(
            raw["B_list"][0], raw["gamma"], column_mean, column_sd
        )
    diagnostics = {
        "iterations": raw["iterations"],
        "D": D,
        "K_list": tuple(raw["num_active"].tolist()),
        "K_max_list": K_tuple,
        "original_K_list": K_tuple,
        "family": family_adapter.name,
        "dimension_method": "csp",
        "initialization": initialization,
        "initializer": csp_initializer,
        "init_binarize": fixed_init_binarize,
        "max_iterations": max_it,
        "tol": tol,
        "standardized": standardize,
        "column_mean": None if column_mean is None else column_mean.copy(),
        "column_sd": None if column_sd is None else column_sd.copy(),
        "backend": "copula",
        "kernel": "copdde_csp",
        "latent_transition": raw["latent_transition"],
        "mc_state_update": raw["mc_state_update"],
        "threshold_rule": raw["threshold_rule"],
        "mstep_solver": raw["mstep_solver"],
        "mstep_latent": raw["mstep_latent"],
        "observed_noise_scale": raw["observed_noise_scale"],
        "optimizer_maxiter": raw["optimizer_maxiter"],
        "fista_maxiter": raw["fista_maxiter"],
        "slab_shape": raw["slab_shape"],
        "slab_rate": raw["slab_rate"],
        "penalty_scaling_power": raw["penalty_scaling_power"],
        "penalty_scaling_reference": raw["penalty_scaling_reference"],
        "penalty_scaling_layers": raw["penalty_scaling_layers"],
        "penalty_scaling_allocation": raw["penalty_scaling_allocation"],
        "penalty_factor": raw["penalty_factor"],
        "reinit_top_after": raw["reinit_top_after"],
        "reinit_top_rule": raw["reinit_top_rule"],
        "reinit_top_source": raw["reinit_top_source"],
        "merge_cosine": raw["merge_cosine"],
        "merge_after": raw["merge_after"],
        "merge_layers": raw["merge_layers"],
        "merge_events": raw["merge_events"],
        "top_prune": raw["top_prune"],
        "prune_events": raw["prune_events"],
        "convergence_width_window": raw["convergence_width_window"],
        "staged_shrinkage_window": raw["staged_shrinkage_window"],
        "stage_iterations": raw["stage_iterations"],
        "top_reinit": raw["top_reinit"],
        "shrinkage_burn": raw["shrinkage_burn"],
        "reorder_after_shrinkage_burn": raw["reorder_after_shrinkage_burn"],
        "burn_permutations": raw["burn_permutations"],
        "convergence_metric": "rolling_relative_frobenius_std",
        "convergence_tol": tol,
        "convergence_window": int(convergence_window),
        "num_active": raw["num_active"].copy(),
        "allocation": [value.copy() for value in raw["allocation"]],
        "omega": [value.copy() for value in raw["omega"]],
        "slab_scale": [value.copy() for value in raw["slab_scale"]],
        "relative_history": raw["relative_history"],
        "temperature": raw["temperature"],
        "stopped_by_convergence": raw["stopped_by_convergence"],
    }
    if keep_trace:
        diagnostics["trace"] = raw["trace"]
    compressed = compress_csp_result_components(
        prop=raw["prop"],
        B_list=raw["B_list"],
        A_list=raw["A_list"],
        diagnostics=diagnostics,
        raw=raw,
        active_masks=raw["active_masks"],
        min_active=False,
    )
    return DDEResult(
        model="DDE",
        K_list=list(compressed["diagnostics"]["K_list"]),
        prop=compressed["prop"],
        B_list=compressed["B_list"],
        gamma=raw["gamma"],
        A_list=compressed["A_list"],
        transformed=X_result,
        diagnostics=compressed["diagnostics"],
        raw=compressed["raw"],
    )


def fit_dde_normal(Y, **kwargs) -> DDEResult:
    """Compatibility wrapper for the public Normal DDE fitting function."""
    if "family" in kwargs:
        raise TypeError("Unexpected Normal DDE argument(s): family.")
    return fit_parametric_dde(Y, family=NORMAL_FAMILY, **kwargs)


def fit_dde_bernoulli(Y, **kwargs) -> DDEResult:
    """Compatibility wrapper for the public Bernoulli DDE fitting function."""
    if "family" in kwargs:
        raise TypeError("Unexpected Bernoulli DDE argument(s): family.")
    return fit_parametric_dde(Y, family=BERNOULLI_FAMILY, **kwargs)


def fit_dde_poisson(Y, **kwargs) -> DDEResult:
    """Compatibility wrapper for the public Poisson DDE fitting function."""
    if "family" in kwargs:
        raise TypeError("Unexpected Poisson DDE argument(s): family.")
    return fit_parametric_dde(Y, family=POISSON_FAMILY, **kwargs)


__all__ = [
    "fit_dde_bernoulli",
    "fit_dde_normal",
    "fit_dde_poisson",
    "fit_parametric_dde",
]
