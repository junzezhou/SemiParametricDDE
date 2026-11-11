"""CopDDE workflow interface."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import norm, rankdata

from dde.copula import simulate_gaussian_mixture
from dde.results import DDEResult
from dde.utils import validate_dde_lists
from dde.utils.result_compression import compress_csp_result_components

from .fixed_graph import (
    EXACT_EM_MAX_WIDTH,
    fit_fixed_graph_em_multistart,
    resolve_graph_list,
    spectral_graph_initialize,
)
from .get_SAEM_RL_CSP_D import get_SAEM_RL_CSP_D
from .get_SAEM_RL_D import get_SAEM_RL_D
from .normal_init_cop import Normal_init_cop


def prepare_rank_likelihood_inputs(X):
    """Create ``Z``, ``R``, and ``Rlevels`` for D-layer CopDDE fitting."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {X.shape}.")
    N, J = X.shape
    R = np.full((N, J), np.nan, dtype=float)
    Z = np.zeros((N, J), dtype=float)
    Rlevels = np.zeros(J, dtype=int)

    for j in range(J):
        xj = X[:, j]
        obs = ~np.isnan(xj)
        if not np.any(obs):
            Rlevels[j] = 0
            Z[:, j] = 0.0
            continue

        values, inverse = np.unique(xj[obs], return_inverse=True)
        Rlevels[j] = values.size
        R[obs, j] = inverse + 1

        ranks = rankdata(xj[obs], method="average")
        u = np.clip((ranks - 0.5) / np.sum(obs), 1e-6, 1.0 - 1e-6)
        Z[obs, j] = norm.ppf(u)
        if np.any(~obs):
            Z[~obs, j] = 0.0

    return Z, R, Rlevels


def normal_initialize_copula_dde(
    X,
    K_list: Sequence[int],
    *,
    Z=None,
    K1_max: Optional[int] = None,
    B_list_oracle=None,
    G_list_oracle=None,
    random_state: Optional[object] = None,
    epsilon: float = 1e-4,
):
    """Initialize D-layer CopDDE with mixture ``Z`` and MATLAB CopDDE rules."""
    if K1_max is not None and int(K1_max) != int(K_list[0]):
        raise ValueError("CopDDE Normal_init_D rules require K1_max == K_list[0].")
    if B_list_oracle is not None or G_list_oracle is not None:
        raise ValueError("CopDDE normal initialization does not use oracle B/G lists.")
    if Z is None:
        Z = simulate_gaussian_mixture(X, np.asarray(X).shape[1], random_state=random_state)

    prop, B_list, gamma, A_list = Normal_init_cop(Z, K_list, epsilon=epsilon)
    return {
        "prop": prop,
        "B_list": B_list,
        "gamma": gamma,
        "A_list": A_list,
        "used_oracle": False,
    }


def _graph_start(X, Z, R, Rlevels, K_tuple, support_list, *, epsilon, n_init, warmup_it, rng, kernel_kwargs):
    """Start of a CopDDE fit on a given graph.

    The spectral estimate is matched to the graph (``spectral_graph_initialize``)
    and ``warmup_it`` kernel iterations settle the first latent layer, which the
    latent Gaussian ``Z`` determines closely.  When every layer has a fixed
    graph and the latent layers above the first are at most
    ``EXACT_EM_MAX_WIDTH`` wide, those layers are then refitted as a Bernoulli
    DDE on the first layer's states by exact EM from the warm-up estimate and
    ``n_init - 1`` random starts on the graph, keeping the largest likelihood:
    from the spectral start alone the kernel leaves deep layers in poor modes
    (``D = 3`` paper design, ``N = 2000``: layer-2/3 MSE 0.1-0.9 / ~2 versus
    0.02 / 0.04-0.13 from the truth).
    """
    prop, B_list, gamma, A_list, init_info = spectral_graph_initialize(
        Z, support_list, K_tuple, epsilon=epsilon
    )
    info = {"spectral_matches": init_info["spectral_matches"], "warmup_it": 0}
    D = len(K_tuple)
    deep_em = (
        D >= 2
        and all(support is not None for support in support_list)
        and max(K_tuple[1:]) <= EXACT_EM_MAX_WIDTH
        and int(n_init) >= 1
    )
    if not deep_em or int(warmup_it) < 1:
        return prop, B_list, gamma, A_list, Z, info
    warm = get_SAEM_RL_D(
        X, Z, R, Rlevels, prop, gamma, B_list, A_list,
        **{**kernel_kwargs, "it": int(warmup_it), "tol": 0.0, "verbose": False},
    )
    info["warmup_it"] = int(warm["it_num"])
    children = np.asarray(warm["A_list"][0], dtype=float)
    fit = fit_fixed_graph_em_multistart(
        children, warm["prop"], warm["B_list"][1:], None, support_list[1:],
        family="bernoulli", n_init=int(n_init), rng=rng, max_iter=500, tol=1e-6,
        criterion="loglik_gain",
    )
    info["deep_em_start_logliks"] = fit[5]["start_logliks"]
    info["deep_em_best_start"] = fit[5]["best_start"]
    B_list = [warm["B_list"][0]] + [np.asarray(B, dtype=float) for B in fit[1]]
    A_list = [children] + [np.asarray(A, dtype=float) for A in fit[3]]
    return fit[0], B_list, warm["gamma"], A_list, warm["Z"], info


def fit_copula_dde(
    X,
    *,
    K_list: Optional[Sequence[int]] = None,
    dimension_method: Optional[str] = None,
    G_list=None,
    Z=None,
    R=None,
    Rlevels=None,
    prop_in=None,
    gamma_in=None,
    B_list_in=None,
    A_list_in=None,
    initialize: str = "data",
    random_state: Optional[object] = None,
    K1_max: Optional[int] = None,
    B_list_oracle=None,
    G_list_oracle=None,
    epsilon: float = 1e-4,
    n_init: Optional[int] = None,
    graph_warmup_it: Optional[int] = None,
    **kwargs: Any,
) -> DDEResult:
    """Fit CopDDE through the D-layer rank-likelihood kernel.

    ``dimension_method`` is ``"csp"`` (default) or ``"fixed"``; it defaults
    to ``"fixed"`` when ``G_list`` is given.  ``C`` defaults to 1 and the
    iteration limit ``it`` to 50.  Fixed-width fits run at ``temp=1``
    unless ``temp`` is given; for CSP fits ``tol`` bounds every layer's
    relative coefficient change averaged over the last seven iterations
    (``convergence_tol``).

    ``G_list`` gives the allowable connections, as in
    :func:`dde.algorithms.dde.fit_parametric_dde`: binary matrices
    ``G_list[0]`` (``J x K_1``) and ``G_list[d]`` (``K_d x K_{d+1}``), a
    prefix of the layers.  The coefficients outside the graph are held at
    zero; the allowable ones get the fixed-width kernel's TLP penalty
    (``pen_list``) and ``tau`` threshold.  A graph that allows every
    connection restricts nothing (all-ones ``G_list`` give the fit without
    a graph).  With a restricting graph the latent draws are systematic
    coordinate-Gibbs sweeps (``latent_transition``) as in the MATLAB
    ``get_SAEM_RL_G.m``, and the defaults are ``C=1``, ``it=50``, ``tol=0``
    (run all iterations) and ``temp=1``.  The start matches the spectral
    estimate to the graph and, when every layer is fixed and the deep widths
    are at most 10, refits the deep layers by exact EM after
    ``graph_warmup_it`` (default 10) kernel iterations, keeping the best of
    ``n_init`` (default 20) starts.  A binary graph identifies each latent
    variable only up to its 0/1 coding, and latent variables whose graph
    columns and rows coincide only up to their order.
    """
    X = np.asarray(X)
    verbose = bool(kwargs.get("verbose", True))
    method = ("fixed" if G_list is not None else "csp") if dimension_method is None else str(dimension_method).lower()
    support_list = None
    if G_list is not None:
        if method != "fixed":
            raise ValueError("G_list requires dimension_method='fixed'; CSP estimates the graph.")
        if B_list_oracle is not None or G_list_oracle is not None:
            raise ValueError("G_list cannot be combined with oracle initialization lists.")
        D_graph = len(K_list) if K_list is not None else len(G_list)
        K_tuple, support_list = resolve_graph_list(G_list, D=D_graph, J=X.shape[1], K_list=K_list)
        K_list = list(K_tuple)
        n_fixed = sum(support is not None for support in support_list)
        if n_fixed == 0:
            # Every graph allows every connection: the fit without a graph.
            support_list = None
        else:
            kwargs.setdefault("C", 1)
            kwargs.setdefault("it", 50)
            kwargs.setdefault("tol", 0.0)
            kwargs.setdefault("temp", 1.0)
            kwargs.setdefault("latent_transition", "systematic_coordinate_gibbs")
            kwargs["support_list"] = support_list
    elif n_init is not None or graph_warmup_it is not None:
        raise ValueError("n_init and graph_warmup_it apply only to fits with G_list.")
    if method not in {"fixed", "csp"}:
        raise ValueError("dimension_method must be 'fixed' or 'csp'.")
    # Manuscript defaults: one latent draw per E-step and 50 iterations.  The
    # temperature is a CSP argument; fixed-width fits run untempered unless
    # ``temp`` is given.
    if kwargs.get("C") is None:
        kwargs["C"] = 1
    if kwargs.get("it") is None:
        kwargs["it"] = 50
    if method == "fixed":
        kwargs.setdefault("temp", 1.0)
    elif "tol" in kwargs:
        # The CSP kernel stops when every layer's relative coefficient
        # change, averaged over ``convergence_window`` iterations, is at
        # most ``tol``, as in the DDE and SemiDDE CSP paths.
        tol = kwargs.pop("tol")
        if tol is not None:
            if "convergence_tol" in kwargs and float(kwargs["convergence_tol"]) != float(tol):
                raise ValueError("tol conflicts with convergence_tol.")
            kwargs["convergence_tol"] = float(tol)
    graph_info = None
    if Z is None or R is None or Rlevels is None:
        _, R0, Rlevels0 = prepare_rank_likelihood_inputs(X)
        if Z is None:
            Z = simulate_gaussian_mixture(X, X.shape[1], random_state=random_state)
        if R is None:
            R = R0
        if Rlevels is None:
            Rlevels = Rlevels0

    if B_list_in is None or A_list_in is None or prop_in is None or gamma_in is None:
        if initialize is True:
            init_method = "data"
        elif initialize in {False, None}:
            init_method = "none"
        else:
            init_method = str(initialize).lower()
        if init_method == "normal":
            init_method = "data"
        if init_method != "data":
            raise ValueError(
                "initialize must be 'data'. Pass prop_in, gamma_in, B_list_in, "
                "and A_list_in to bypass initialization."
            )
        if K_list is None:
            raise ValueError("K_list is required when initialized arguments are incomplete.")
        if verbose:
            print("CopDDE initialization: triggered using initialize='data'.")
        if support_list is not None:
            missing = [name for name in ("C", "temp", "pen_list", "tau") if name not in kwargs]
            if missing:
                raise ValueError("fit_copula_dde with G_list needs: " + ", ".join(missing))
            if isinstance(random_state, np.random.Generator):
                rng = random_state
            else:
                rng = np.random.default_rng(
                    np.random.randint(0, 2**31 - 1) if random_state is None else random_state
                )
            kernel_kwargs = {
                name: kwargs[name]
                for name in ("C", "temp", "pen_list", "tau", "optimizer_maxiter", "n_jobs",
                             "support_list", "latent_transition")
                if name in kwargs
            }
            prop0, B0, gamma0, A0, Z, graph_info = _graph_start(
                X, Z, R, Rlevels, tuple(K_list), support_list,
                epsilon=epsilon, n_init=20 if n_init is None else n_init,
                warmup_it=10 if graph_warmup_it is None else graph_warmup_it,
                rng=rng, kernel_kwargs=kernel_kwargs,
            )
            init = {"prop": prop0, "B_list": B0, "gamma": gamma0, "A_list": A0}
        else:
            init = normal_initialize_copula_dde(
                X,
                K_list,
                Z=Z,
                K1_max=K1_max,
                B_list_oracle=B_list_oracle,
                G_list_oracle=G_list_oracle,
                random_state=random_state,
                epsilon=epsilon,
            )
        prop_in = init["prop"] if prop_in is None else prop_in
        gamma_in = init["gamma"] if gamma_in is None else gamma_in
        B_list_in = init["B_list"] if B_list_in is None else B_list_in
        A_list_in = init["A_list"] if A_list_in is None else A_list_in
    else:
        if verbose:
            print("CopDDE initialization: not triggered; using supplied initialized arguments.")

    shape = validate_dde_lists(
        Z,
        B_list_in,
        A_list=A_list_in,
        prop=prop_in,
        gamma=gamma_in,
        K_list=K_list,
    )

    common = dict(
        X=X,
        Z=Z,
        R=R,
        Rlevels=Rlevels,
        prop_in=prop_in,
        gamma_in=gamma_in,
        B_list_in=B_list_in,
        A_list_in=A_list_in,
    )

    if method == "fixed":
        required = ["C", "tol", "it", "temp", "pen_list", "tau"]
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise ValueError(
                "fit_copula_dde fixed mode needs initialized arguments and algorithm controls: "
                + ", ".join(missing)
            )
        raw = get_SAEM_RL_D(**common, **kwargs)
        diagnostics = {
            "iterations": raw["it_num"],
            "err": raw["err"],
            "p": raw["p"],
            "A_sample_long": raw["A_sample_long"],
            "K_list": shape.K_list,
            "latent_transition": raw["latent_transition"],
        }
        if support_list is not None:
            diagnostics["G_list"] = [
                support.astype(int) for support in support_list if support is not None
            ]
            diagnostics["fixed_graph_layers"] = n_fixed
            diagnostics["graph_start"] = graph_info
    elif method == "csp":
        required = ["C", "it", "t_spike", "temp", "tau"]
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise ValueError(
                "fit_copula_dde csp mode needs initialized arguments and algorithm controls: "
                + ", ".join(missing)
            )
        raw = get_SAEM_RL_CSP_D(**common, **kwargs)
        diagnostics = {
            "iterations": raw["it_num"],
            "num_active": raw["num_act"],
            "theta": raw["theta"],
            "theta_slab": raw["theta_slab"],
            "z_cusp": raw["z_cusp"],
            "omega": raw["omega"],
            "p": raw["p"],
            "A_sample_long": raw["A_sample_long"],
            "K_list": shape.K_list,
            "original_K_list": shape.K_list,
            "backend": "csp",
            "threshold_rule": "first_two_layers",
        }
        compressed = compress_csp_result_components(
            prop=raw["prop"],
            B_list=raw["B_list"],
            A_list=raw["A_list"],
            diagnostics=diagnostics,
            raw=raw,
        )
        raw = compressed["raw"]
        diagnostics = compressed["diagnostics"]
    else:
        raise ValueError("dimension_method must be 'fixed' or 'csp'.")

    diagnostics["dimension_method"] = method
    diagnostics["D"] = len(shape.K_list)
    return DDEResult(
        model="CopDDE",
        K_list=list(diagnostics["K_list"]),
        prop=raw["prop"],
        B_list=raw["B_list"],
        gamma=raw["gamma"],
        A_list=raw["A_list"],
        transformed=raw["Z"],
        diagnostics=diagnostics,
        raw=raw,
    )


__all__ = [
    "Normal_init_cop",
    "fit_copula_dde",
    "normal_initialize_copula_dde",
    "prepare_rank_likelihood_inputs",
]
