"""User-specified graphs for fixed-width DDE fits.

A DDE whose graphical matrices are known (a Q-matrix in cognitive
diagnosis, a designed measurement structure) is fitted by holding the
graph fixed and estimating only the coefficients on it, the confirmatory
analysis of the TIMSS data in Lee and Gu (2025, Supplement S.5.1;
``get_EM_multimodal_confirm.m`` in the Deep-Discrete-Encoders code).
``G_list`` gives the graph of the shallowest ``m`` layers (``G_list[0]``
is ``J x K_1``, ``G_list[d]`` is ``K_d x K_{d+1}``); deeper layers, if
any, keep the penalized graph estimation.  Only a prefix can be fixed: the
rows of ``G_list[d]`` name the latent variables of layer ``d``, and those
are labelled only when the graph below them is fixed.

The fixed layers are initialized from the graph instead of the spectral
estimate, whose columns come in an arbitrary order.  Each latent column is
read off its anchor children (the children with the fewest parents, pure
children first) by the families' two-class anchor models, oriented so that
the value one raises the anchors' means; every row's coefficients are then
fitted on its graph with the parents' class probabilities as weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit

from .bernoulli_init import (
    _initialize_binary_parent_layer,
    _two_class_latent_class_model,
)
from .normal_init_cop import Normal_init_cop
from .normal_init_semi import _two_class_gaussian_mixture
from .parametric_families import resolve_parametric_family
from .poisson_init import _two_component_poisson_mixture


def resolve_graph_list(G_list, *, D: int, J: int, K_list=None):
    """Validate a user graph and return ``(K_tuple, support_list)``.

    ``G_list`` has at most ``D`` entries; missing trailing entries and
    ``None`` entries mark layers whose graph is estimated.  The fixed layers
    must form a prefix, each graph must be binary with a nonempty column
    for every latent variable, and consecutive shapes must chain from the
    ``J`` observed variables.  Widths of fixed layers come from the graph;
    ``K_list`` must agree with them and supplies the free widths, which keep
    the ``K_d >= 3 K_{d+1}`` identifiability ordering below them.
    """
    D = int(D)
    if isinstance(G_list, (str, bytes)) or not isinstance(G_list, Sequence):
        raise ValueError("G_list must be a sequence of graphical matrices.")
    entries = list(G_list)
    if not 1 <= len(entries) <= D:
        raise ValueError(f"G_list must have between 1 and D={D} entries.")
    entries += [None] * (D - len(entries))
    if entries[0] is None:
        raise ValueError("G_list[0], the observed-layer graph, must be given.")
    n_fixed = next((d for d, G in enumerate(entries) if G is None), D)
    if any(G is not None for G in entries[n_fixed:]):
        raise ValueError(
            "The fixed graphs must be the shallowest layers: G_list[d] names "
            "the latent variables of layer d through its rows, which are only "
            "labelled when the graph below them is fixed."
        )

    support_list = []
    widths = []
    n_rows = int(J)
    for d, G in enumerate(entries[:n_fixed]):
        array = np.asarray(G)
        if array.ndim != 2:
            raise ValueError(f"G_list[{d}] must be a 2D matrix; got shape {array.shape}.")
        try:
            finite = np.all(np.isfinite(array.astype(float)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"G_list[{d}] must be numeric 0/1.") from exc
        if not finite or not np.all((array == 0) | (array == 1)):
            raise ValueError(f"G_list[{d}] must contain only 0/1 entries.")
        if array.shape[0] != n_rows:
            expected = "J" if d == 0 else f"K_{d}"
            raise ValueError(
                f"G_list[{d}] has {array.shape[0]} rows; expected {expected}={n_rows}."
            )
        support = array.astype(bool)
        empty = np.flatnonzero(~support.any(axis=0))
        if array.shape[1] < 1 or empty.size:
            raise ValueError(
                f"G_list[{d}] columns {empty.tolist()} have no children; every "
                "latent variable needs at least one child."
            )
        support_list.append(support)
        widths.append(int(array.shape[1]))
        n_rows = int(array.shape[1])

    if K_list is not None:
        K_given = tuple(int(width) for width in K_list)
        if len(K_given) != D:
            raise ValueError(f"K_list length {len(K_given)} must match D={D}.")
        if K_given[:n_fixed] != tuple(widths):
            raise ValueError(
                f"K_list {K_given} disagrees with the widths {tuple(widths)} of the "
                "fixed graphs."
            )
        widths = list(K_given)
    elif n_fixed < D:
        raise ValueError("K_list is required when G_list leaves deeper layers free.")
    if any(width < 1 for width in widths):
        raise ValueError("K_list entries must be positive integers.")
    for d in range(n_fixed, D):
        if widths[d - 1] < 3 * widths[d]:
            raise ValueError(
                "Layers with an estimated graph need K_list[d-1] >= 3 * K_list[d]; "
                f"got {tuple(widths)}."
            )
    support_list += [None] * (D - n_fixed)
    return tuple(widths), support_list


def _graph_anchors(support, k, *, min_anchors, max_anchors=8):
    """Children of column ``k`` with the fewest parents, pure children first."""
    children = np.flatnonzero(support[:, k])
    n_parents = support[children].sum(axis=1)
    order = children[np.lexsort((children, n_parents))]
    n_pure = int(np.sum(n_parents == 1))
    n_take = min(max(n_pure, int(min_anchors)), int(max_anchors), children.size)
    return order[:n_take]


def _classify_column(children, anchors, kind):
    """States and class probabilities of one latent column from its anchors."""
    block = children[:, anchors]
    if kind == "poisson":
        logit = np.zeros(children.shape[0], dtype=float)
        pis = []
        for j in anchors:
            x = children[:, j]
            pi, lam0, lam1 = _two_component_poisson_mixture(x)
            logit += x * np.log(lam1 / lam0) - (lam1 - lam0)
            pis.append(pi)
        pi = float(np.clip(np.mean(pis), 1e-3, 1.0 - 1e-3))
        prob = expit(logit + np.log(pi / (1.0 - pi)))
    elif kind == "normal":
        prob, _ = _two_class_gaussian_mixture(block)
    else:
        prob, _ = _two_class_latent_class_model(block)
    state = (prob > 0.5).astype(np.uint8)
    if state.min() == state.max():
        # Degenerate two-class fit: split at the median standardized score.
        scale = np.maximum(block.std(axis=0), 1e-12)
        score = ((block - block.mean(axis=0)) / scale).sum(axis=1)
        state = (score > np.median(score)).astype(np.uint8)
        if state.min() == state.max():
            state[np.argsort(score)[score.size // 2 :]] = 1
        prob = state.astype(float)
    return state, np.asarray(prob, dtype=float)


# Rows with more parents than this are initialized from the hard states.
_SOFT_INIT_MAX_PARENTS = 10


def _row_statistics(y, probabilities, columns):
    """Parent-state design, weights, and response sums of one row.

    The parents' class probabilities are combined as independent Bernoulli
    weights over the ``2**|S|`` parent states, so the fit is one EM step
    from the anchor models; fitting the hard classifications instead
    separates the anchors (each anchor nearly determines its parent's
    class) and starts their coefficients at the bounds.
    """
    if columns.size > _SOFT_INIT_MAX_PARENTS:
        states = (probabilities[:, columns] > 0.5).astype(np.uint8)
        patterns, inverse = np.unique(states, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        weights = np.bincount(inverse, minlength=patterns.shape[0]).astype(float)
        sums = np.bincount(inverse, weights=y, minlength=patterns.shape[0])
        squares = np.bincount(inverse, weights=y**2, minlength=patterns.shape[0])
        configs = patterns.astype(float)
    else:
        configs = _layer_configs(columns.size)
        prob = np.clip(probabilities[:, columns], 1e-6, 1.0 - 1e-6)
        log_weights = np.log(prob) @ configs.T + np.log1p(-prob) @ (1.0 - configs).T
        point_weights = np.exp(log_weights)
        weights = point_weights.sum(axis=0)
        sums = point_weights.T @ y
        squares = point_weights.T @ (y**2)
    design = np.column_stack([np.ones(configs.shape[0]), configs])
    return design, weights, sums, squares


def _fit_glm_aggregated(design, weights, sums, kind, bounds, *, epsilon):
    """Weighted maximum-likelihood logistic or log-link coefficients of one row."""
    mean = float(np.clip(sums.sum() / max(weights.sum(), 1e-12), epsilon, None))
    if kind == "poisson":
        start = np.log(mean)

        def objective(x):
            eta = design @ x
            rate = np.exp(np.minimum(eta, 50.0))
            value = float(weights @ rate - sums @ eta)
            return value, design.T @ (weights * rate - sums)

    else:
        mean = min(mean, 1.0 - epsilon)
        start = np.log(mean / (1.0 - mean))

        def objective(x):
            eta = design @ x
            value = float(weights @ np.logaddexp(0.0, eta) - sums @ eta)
            prob = 0.5 * (1.0 + np.tanh(0.5 * eta))
            return value, design.T @ (weights * prob - sums)

    x0 = np.zeros(design.shape[1])
    x0[0] = float(np.clip(start, bounds[0][0], bounds[0][1]))
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 200},
    )
    return result.x


def _fit_layer_on_graph(children, probabilities, support, kind, bounds, *, epsilon):
    """Coefficients (and Normal variances) of one layer given its parents' class probabilities."""
    children = np.asarray(children, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    n_rows, K = support.shape
    N = children.shape[0]
    B = np.zeros((n_rows, K + 1), dtype=float)
    gamma = None
    if kind == "normal":
        gamma = np.zeros(n_rows, dtype=float)
    for j in range(n_rows):
        columns = np.flatnonzero(support[j])
        y = children[:, j]
        design, weights, sums, squares = _row_statistics(y, probabilities, columns)
        if kind == "normal":
            gram = design.T @ (weights[:, None] * design)
            coefficients = np.linalg.lstsq(gram, design.T @ sums, rcond=None)[0]
            eta = design @ coefficients
            residual = float(np.sum(squares - 2.0 * eta * sums + weights * eta**2)) / N
            gamma[j] = max(residual, 1e-6 * max(float(np.var(y)), 1e-12))
        else:
            row_bounds = [bounds[0]] + [bounds[1 + k] for k in columns]
            coefficients = _fit_glm_aggregated(
                design, weights, sums, kind, row_bounds, epsilon=epsilon
            )
        B[j, 0] = coefficients[0]
        B[j, 1 + columns] = coefficients[1:]
    return B, gamma


def graph_anchored_initialize(
    X,
    support_list,
    K_tuple,
    *,
    family,
    epsilon: float = 1e-4,
    bound_val: float = 10.0,
):
    """Initialize a fixed-width DDE whose shallowest graphs are given.

    Returns ``(prop, B_list, gamma, A_list, diagnostics)`` in the format of
    the family initializers.  Free deeper layers start from the Bernoulli
    spectral initializer with anchor latent-class binarization on the last
    fixed layer's states.
    """
    family = resolve_parametric_family(family)
    X = np.asarray(X, dtype=float)
    D = len(K_tuple)
    B_list, A_list, anchors_list = [], [], []
    gamma = None
    children = X
    for d, support in enumerate(support_list):
        if support is None:
            break
        kind = family.key if d == 0 else "binary"
        min_anchors = 3 if kind in {"bernoulli", "binary"} else 2
        K = support.shape[1]
        parents = np.zeros((children.shape[0], K), dtype=np.uint8)
        probabilities = np.zeros((children.shape[0], K), dtype=float)
        anchors_d = []
        for k in range(K):
            anchors = _graph_anchors(support, k, min_anchors=min_anchors)
            parents[:, k], probabilities[:, k] = _classify_column(children, anchors, kind)
            anchors_d.append(anchors.tolist())
        if d == 0 and family.key == "normal":
            B_d, gamma = _fit_layer_on_graph(
                children, probabilities, support, "normal", None, epsilon=epsilon
            )
        else:
            bounds = family.fixed_latent_bounds(K + 1, bound_val)
            glm_kind = "poisson" if kind == "poisson" else "logistic"
            B_d, _ = _fit_layer_on_graph(
                children, probabilities, support, glm_kind, bounds, epsilon=epsilon
            )
        B_list.append(B_d)
        A_list.append(parents)
        anchors_list.append(anchors_d)
        children = parents
    for d in range(len(B_list), D):
        B_d, parents, _, _ = _initialize_binary_parent_layer(
            children,
            K_tuple[d],
            epsilon=epsilon,
            threshold_constant=3.5,
            binarize="anchor_lca",
        )
        B_list.append(B_d)
        A_list.append(np.asarray(parents, dtype=np.uint8))
        children = A_list[-1]
    prop = np.clip(np.mean(A_list[-1], axis=0), 1e-3, 1.0 - 1e-3)
    diagnostics = {"anchors": anchors_list}
    return prop, B_list, gamma, A_list, diagnostics


def _match_columns_to_graph(children, states, support):
    """Estimated latent column matched to each graph column (Hungarian assignment).

    The assignment maximizes the variance of the graph's children explained
    by their matched parents: estimated column ``c`` scores
    ``sum_{j in S_k} corr(child_j, state_c)**2`` for graph column ``k``.
    The score is read from the data, not from the spectral loadings, whose
    small entries are too noisy to tell apart graph columns that share their
    strongly loading children; it ignores signs, as a binary graph does.
    """
    children = np.asarray(children, dtype=float)
    states = np.asarray(states, dtype=float)
    centred_children = children - children.mean(axis=0)
    centred_states = states - states.mean(axis=0)
    scale = np.outer(
        np.sqrt(np.sum(centred_children**2, axis=0)),
        np.sqrt(np.sum(centred_states**2, axis=0)),
    )
    correlation = np.divide(
        centred_children.T @ centred_states, scale,
        out=np.zeros(scale.shape), where=scale > 0,
    )
    score = support.T.astype(float) @ correlation**2
    _, columns = linear_sum_assignment(-score)
    return columns


def spectral_graph_initialize(
    X,
    support_list,
    K_tuple,
    *,
    epsilon: float = 1e-4,
    bound_val: float = 10.0,
    layer_em_starts: int = 0,
    rng=None,
):
    """Initialize a Normal-observed DDE on a given graph from the spectral estimate.

    The CopDDE spectral initializer (``Normal_init_cop``: SVD, varimax and
    the binary-layer double SVD at the given widths) supplies unlabelled
    latent columns; each fixed layer's columns are matched to the graph's
    (``_match_columns_to_graph``), the matching carried into the rows of the
    layer above, and the fixed layers' coefficients refitted on the graph
    from the matched states.  This follows ``Normal_init_G`` in the CopDDE
    code (``Sim_SampleSplit.m``), which aligns the spectral columns to a
    graph before the unpenalized refit; unlike the anchor initializer it
    needs no children with few parents, so it suits designs in which every
    child loads on several latent variables.

    Returns ``(prop, B_list, gamma, A_list, diagnostics)``.
    """
    X = np.asarray(X, dtype=float)
    prop, B_list, gamma, A_list = Normal_init_cop(X, K_tuple, epsilon=epsilon)
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list]
    A_list = [np.asarray(A, dtype=float).copy() for A in A_list]
    D = len(K_tuple)
    matches = []
    for d, support in enumerate(support_list):
        if support is None:
            break
        columns = _match_columns_to_graph(X if d == 0 else A_list[d - 1], A_list[d], support)
        matches.append(columns.tolist())
        B_list[d] = np.column_stack((B_list[d][:, 0], B_list[d][:, 1:][:, columns]))
        A_list[d] = A_list[d][:, columns]
        if d + 1 < D:
            B_list[d + 1] = B_list[d + 1][columns, :]
        else:
            prop = prop[columns]
    latent_bounds = resolve_parametric_family("normal").fixed_latent_bounds
    layer_em_logliks = []
    for d, support in enumerate(support_list[: len(matches)]):
        if d == 0:
            B_list[0], gamma = _fit_layer_on_graph(
                X, A_list[0], support, "normal", None, epsilon=epsilon
            )
            continue
        B_list[d], _ = _fit_layer_on_graph(
            A_list[d - 1], A_list[d], support, "logistic",
            latent_bounds(support.shape[1] + 1, bound_val), epsilon=epsilon,
        )
        if layer_em_starts and support.shape[1] <= EXACT_EM_MAX_WIDTH:
            # The layer's children are the matched states below it; refit the
            # layer as a one-layer Bernoulli DDE on its graph by exact EM.
            fit = fit_fixed_graph_em_multistart(
                A_list[d - 1].astype(float),
                np.clip(A_list[d].mean(axis=0), 0.05, 0.95),
                [B_list[d]], None, [support], family="bernoulli",
                n_init=layer_em_starts, rng=rng, max_iter=300, tol=1e-6, criterion="loglik_gain",
            )
            B_list[d] = fit[1][0]
            A_list[d] = np.asarray(fit[3][0], dtype=float)
            layer_em_logliks.append(fit[5]["start_logliks"])
    prop = np.clip(np.mean(A_list[-1], axis=0), 1e-3, 1.0 - 1e-3)
    A_list = [A.astype(np.uint8) for A in A_list]
    return prop, B_list, gamma, A_list, {"spectral_matches": matches, "layer_em_logliks": layer_em_logliks}


# ---------------------------------------------------------------------------
# Exact EM on a fixed graph
# ---------------------------------------------------------------------------

# Exact EM enumerates the 2**K_d states of every layer; above this width the
# SAEM kernel is used.
EXACT_EM_MAX_WIDTH = 10


def _layer_configs(K):
    """All ``2**K`` binary states of a layer, bit ``k`` in column ``k``."""
    codes = np.arange(1 << int(K), dtype=np.int64)
    return ((codes[:, None] >> np.arange(int(K))[None, :]) & 1).astype(float)


def _observed_loglik(X, eta, family, dispersion):
    """``log p(x_i | a_1)`` for every observation and layer-1 state (no constants)."""
    if family.key == "bernoulli":
        return X @ eta.T - np.logaddexp(0.0, eta).sum(axis=1)[None, :]
    if family.key == "poisson":
        return X @ eta.T - np.exp(np.minimum(eta, 50.0)).sum(axis=1)[None, :]
    weight = 1.0 / np.asarray(dispersion, dtype=float)
    return (
        -0.5 * ((X**2) @ weight)[:, None]
        + (X * weight[None, :]) @ eta.T
        - 0.5 * ((eta**2) @ weight)[None, :]
    )


def _transition_log(configs_child, configs_parent, B):
    """``log p(a_child | a_parent)`` as a ``M_child x M_parent`` matrix."""
    design = np.column_stack([np.ones(configs_parent.shape[0]), configs_parent])
    eta = design @ B.T
    return configs_child @ eta.T - np.logaddexp(0.0, eta).sum(axis=1)[None, :]


def _weighted_glm_rows(B, support, design, weights, sums, kind, bounds, max_iter):
    """Weighted maximum-likelihood update of every row on its graph."""
    B = np.asarray(B, dtype=float).copy()
    for j in range(B.shape[0]):
        columns = np.concatenate(([0], 1 + np.flatnonzero(support[j])))
        design_j = design[:, columns]
        sums_j = sums[:, j]

        if kind == "poisson":

            def objective(x):
                eta = design_j @ x
                rate = np.exp(np.minimum(eta, 50.0))
                return (
                    float(weights @ rate - sums_j @ eta),
                    design_j.T @ (weights * rate - sums_j),
                )

        else:

            def objective(x):
                eta = design_j @ x
                prob = 0.5 * (1.0 + np.tanh(0.5 * eta))
                return (
                    float(weights @ np.logaddexp(0.0, eta) - sums_j @ eta),
                    design_j.T @ (weights * prob - sums_j),
                )

        row_bounds = [bounds[c] for c in columns]
        x0 = np.clip(B[j, columns], [b[0] for b in row_bounds], [b[1] for b in row_bounds])
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=row_bounds,
            options={"maxiter": int(max_iter)},
        )
        B[j] = 0.0
        B[j, columns] = result.x
    return B


def _fixed_graph_estep(X, prop, B_list, dispersion, configs, designs, family, row_chunk_size, collect):
    """One exact E-step: expected statistics, log-likelihood, and optionally posteriors."""
    N, J = X.shape
    D = len(B_list)
    is_normal = family.key == "normal"
    log_priors = [None] * D
    transitions = [None] * D
    log_priors[-1] = configs[-1] @ np.log(prop) + (1.0 - configs[-1]) @ np.log1p(-prop)
    for d in range(D - 1, 0, -1):
        transitions[d] = _transition_log(configs[d - 1], configs[d], B_list[d])
        log_priors[d - 1] = _logsumexp_rows(transitions[d] + log_priors[d][None, :])
    eta_obs = designs[0] @ B_list[0].T
    stats = {
        "weights_obs": np.zeros(configs[0].shape[0]),
        "sums_obs": np.zeros((configs[0].shape[0], J)),
        "sq_obs": np.zeros((configs[0].shape[0], J)) if is_normal else None,
        "message_sums": [np.zeros(configs[d].shape[0]) for d in range(D)],
        "transitions": transitions,
        "log_priors": log_priors,
        "loglik": 0.0,
        "posteriors": [[] for _ in range(D)] if collect else None,
    }
    prior_obs = np.exp(log_priors[0])
    for start in range(0, N, int(row_chunk_size)):
        stop = min(start + int(row_chunk_size), N)
        X_chunk = X[start:stop]
        upward = [None] * D
        upward[0] = _observed_loglik(X_chunk, eta_obs, family, dispersion)
        for d in range(1, D):
            upward[d] = _logsumexp_cols(upward[d - 1], transitions[d])
        log_z = _logsumexp_rows(upward[-1] + log_priors[-1][None, :])
        stats["loglik"] += float(log_z.sum())
        for d in range(D):
            scaled = np.exp(upward[d] - log_z[:, None])
            stats["message_sums"][d] += scaled.sum(axis=0)
            if d == 0:
                post_obs = scaled * prior_obs[None, :]
            if collect:
                stats["posteriors"][d].append(scaled * np.exp(log_priors[d])[None, :])
        stats["weights_obs"] += post_obs.sum(axis=0)
        stats["sums_obs"] += post_obs.T @ X_chunk
        if is_normal:
            stats["sq_obs"] += post_obs.T @ (X_chunk**2)
    if is_normal:
        stats["loglik"] -= 0.5 * N * float(np.sum(np.log(2.0 * np.pi * dispersion)))
    return stats


def fit_fixed_graph_em(
    X,
    prop,
    B_list,
    dispersion,
    support_list,
    *,
    family,
    max_iter: int = 500,
    tol: float = 1e-4,
    criterion: str = "relative_parameter_change",
    bound_val: float = 10.0,
    mstep_maxiter: int = 50,
    row_chunk_size: int = 2048,
    verbose: bool = False,
):
    """Exact EM for a DDE whose graph is fixed in every layer.

    The E-step enumerates each layer's ``2**K_d`` states.  Only the
    observed layer sees data, so the posterior of a layer pair is the
    upward message from the data times the prior marginal from above:
    with ``u_1 = log p(x | a_1)`` and ``u_{d+1}(a_{d+1}) = log sum_{a_d}
    exp(u_d(a_d)) p(a_d | a_{d+1})``,

        p(a_d, a_{d+1} | x) = exp(u_d(a_d)) p(a_d | a_{d+1}) pi_{d+1}(a_{d+1}) / Z,

    where ``pi`` are the prior marginals.  The M-step maximizes the
    expected complete-data log-likelihood row by row on the graph: least
    squares and variances for Normal responses, bounded logistic or
    log-link fits (the fixed kernel's coefficient bounds) otherwise.  This
    is the confirmatory EM of the paper's TIMSS analysis
    (``get_EM_multimodal_confirm.m``) for any depth.  With
    ``criterion="relative_parameter_change"`` (default) EM stops when an
    M-step changes the parameters (top-layer probabilities, every
    coefficient and the variances) by at most ``tol`` relative to their
    norm, ``||theta_new - theta_old|| / ||theta_old||``, the stopping rule
    of the package's other fitting routines; ``criterion="loglik_gain"``
    stops when the log-likelihood gain per observation falls below ``tol``.
    Either way EM ends after ``max_iter`` M-steps.

    Returns ``(prop, B_list, dispersion, A_list, iterations, diagnostics)``;
    ``A_list`` holds each layer's most probable state given the data, and
    the diagnostics hold the log-likelihood path and the per-layer marginal
    posterior probabilities of the latent variables.
    """
    family = resolve_parametric_family(family)
    X = np.asarray(X, dtype=float)
    N, J = X.shape
    D = len(B_list)
    K = [int(np.asarray(B).shape[1]) - 1 for B in B_list]
    support_list = [np.asarray(support, dtype=bool) for support in support_list]
    prop = np.clip(np.asarray(prop, dtype=float).copy(), 1e-6, 1.0 - 1e-6)
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list]
    for B, support in zip(B_list, support_list):
        B[:, 1:] *= support
    dispersion = family.copy_dispersion(dispersion)
    configs = [_layer_configs(width) for width in K]
    designs = [np.column_stack([np.ones(c.shape[0]), c]) for c in configs]
    latent_bounds = [family.fixed_latent_bounds(width + 1, bound_val) for width in K]
    constant = 0.0
    if family.key == "poisson":
        from scipy.special import gammaln

        constant = -float(gammaln(X + 1.0).sum())
    if criterion not in ("relative_parameter_change", "loglik_gain"):
        raise ValueError("criterion must be 'relative_parameter_change' or 'loglik_gain'.")
    loglik_history = []
    change_history = []
    converged = False
    iterations = 0
    while True:
        stats = _fixed_graph_estep(
            X, prop, B_list, dispersion, configs, designs, family, row_chunk_size, False
        )
        loglik_history.append(stats["loglik"] + constant)
        if verbose:
            print(f"Fixed-graph EM iteration {iterations}, loglik {loglik_history[-1]:.4f}")
        if criterion == "loglik_gain":
            stop = len(loglik_history) > 1 and abs(loglik_history[-1] - loglik_history[-2]) <= tol * N
        else:
            stop = bool(change_history) and change_history[-1] <= tol
        if stop:
            converged = True
            break
        if iterations >= int(max_iter):
            break
        theta_old = _em_parameter_vector(prop, B_list, dispersion)

        log_priors = stats["log_priors"]
        message_sums = stats["message_sums"]
        top_weights = message_sums[-1] * np.exp(log_priors[-1])
        prop = np.clip(top_weights @ configs[-1] / N, 1e-6, 1.0 - 1e-6)
        if family.key == "normal":
            weights, sums, squares = stats["weights_obs"], stats["sums_obs"], stats["sq_obs"]
            B_new = np.zeros_like(B_list[0])
            for j in range(J):
                columns = np.concatenate(([0], 1 + np.flatnonzero(support_list[0][j])))
                design_j = designs[0][:, columns]
                gram = design_j.T @ (weights[:, None] * design_j)
                rhs = design_j.T @ sums[:, j]
                try:
                    coefficients = np.linalg.solve(gram, rhs)
                except np.linalg.LinAlgError:
                    coefficients = np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]
                B_new[j, columns] = coefficients
                eta_j = design_j @ coefficients
                residual = squares[:, j] - 2.0 * eta_j * sums[:, j] + weights * eta_j**2
                dispersion[j] = max(float(residual.sum()) / N, 1e-8)
            B_list[0] = B_new
        else:
            kind = "poisson" if family.key == "poisson" else "logistic"
            B_list[0] = _weighted_glm_rows(
                B_list[0], support_list[0], designs[0], stats["weights_obs"],
                stats["sums_obs"], kind, latent_bounds[0], mstep_maxiter,
            )
        for d in range(1, D):
            pair = (
                message_sums[d - 1][:, None]
                * np.exp(stats["transitions"][d])
                * np.exp(log_priors[d])[None, :]
            )
            B_list[d] = _weighted_glm_rows(
                B_list[d], support_list[d], designs[d], pair.sum(axis=0),
                pair.T @ configs[d - 1], "logistic", latent_bounds[d], mstep_maxiter,
            )
        theta_new = _em_parameter_vector(prop, B_list, dispersion)
        change_history.append(
            float(np.linalg.norm(theta_new - theta_old) / max(np.linalg.norm(theta_old), 1e-12))
        )
        iterations += 1

    final = _fixed_graph_estep(
        X, prop, B_list, dispersion, configs, designs, family, row_chunk_size, True
    )
    posteriors = [np.vstack(chunks) for chunks in final["posteriors"]]
    A_list = [
        configs[d][np.argmax(posteriors[d], axis=1)].astype(np.uint8) for d in range(D)
    ]
    diagnostics = {
        "loglik": loglik_history[-1],
        "loglik_history": loglik_history,
        "relative_change_history": change_history,
        "convergence_metric": criterion,
        "converged": converged,
        "posterior_marginals": [posteriors[d] @ configs[d] for d in range(D)],
    }
    return prop, B_list, dispersion, A_list, iterations, diagnostics


def _em_parameter_vector(prop, B_list, dispersion):
    """The parameters whose relative change stops the exact EM."""
    parts = [np.ravel(prop)] + [np.ravel(B) for B in B_list]
    if dispersion is not None:
        parts.append(np.ravel(dispersion))
    return np.concatenate(parts).astype(float)


def random_graph_start(X, support_list, *, family, rng):
    """A random start on the graph in the style of the paper's TIMSS code.

    Slopes on the graph are drawn uniformly (positive, so that the value one
    of a latent variable raises its children's means, the orientation of the
    graph-anchored start) on the scale of each layer's link: ``U(0.5, 1.5)``
    response standard deviations for Normal responses, ``U(0.5, 1.5)`` on
    the log scale for counts and ``U(0.5, 2.5)`` on the logit scale for
    binary children.  Intercepts place each child's mean at its sample mean
    when every parent has prevalence one half; the top prevalences are drawn
    from ``U(0.3, 0.7)`` and Normal variances start at half the sample
    variances.  Returns ``(prop, B_list, dispersion)``.
    """
    family = resolve_parametric_family(family)
    X = np.asarray(X, dtype=float)
    B_list = []
    dispersion = None
    for d, support in enumerate(support_list):
        support = np.asarray(support, dtype=bool)
        n_rows, K = support.shape
        if d == 0 and family.key == "normal":
            scale = np.maximum(X.std(axis=0), 1e-6)
            slopes = support * rng.uniform(0.5, 1.5, size=(n_rows, K)) * scale[:, None]
            intercept = X.mean(axis=0) - 0.5 * slopes.sum(axis=1)
            dispersion = np.maximum(0.5 * X.var(axis=0), 1e-6)
        elif d == 0 and family.key == "poisson":
            slopes = support * rng.uniform(0.5, 1.5, size=(n_rows, K))
            intercept = np.log(np.maximum(X.mean(axis=0), 1e-3)) - 0.5 * slopes.sum(axis=1)
        else:
            slopes = support * rng.uniform(0.5, 2.5, size=(n_rows, K))
            if d == 0:
                mean = np.clip(X.mean(axis=0), 1e-3, 1.0 - 1e-3)
                intercept = np.log(mean / (1.0 - mean)) - 0.5 * slopes.sum(axis=1)
            else:
                intercept = -0.5 * slopes.sum(axis=1)
        bounds = (
            None
            if d == 0 and family.key == "normal"
            else family.fixed_latent_bounds(K + 1, np.inf)
        )
        B = np.column_stack([intercept, slopes])
        if bounds is not None:
            B = np.clip(B, [b[0] for b in bounds], [b[1] for b in bounds]) * np.column_stack(
                [np.ones(n_rows), support]
            )
        B_list.append(B)
    prop = rng.uniform(0.3, 0.7, size=np.asarray(support_list[-1]).shape[1])
    return prop, B_list, dispersion


def fit_fixed_graph_em_multistart(
    X,
    prop,
    B_list,
    dispersion,
    support_list,
    *,
    family,
    n_init: int = 1,
    rng=None,
    **em_kwargs,
):
    """Exact EM from the given start and ``n_init - 1`` random starts.

    Returns the fit with the largest log-likelihood in the format of
    :func:`fit_fixed_graph_em`; its diagnostics add every start's final
    log-likelihood and the index of the kept start (0 is the given start).
    """
    n_init = int(n_init)
    if n_init < 1:
        raise ValueError("n_init must be a positive integer.")
    if rng is None:
        rng = np.random.default_rng()
    best = fit_fixed_graph_em(X, prop, B_list, dispersion, support_list, family=family, **em_kwargs)
    start_logliks = [best[5]["loglik"]]
    best_index = 0
    for start in range(1, n_init):
        prop_r, B_r, dispersion_r = random_graph_start(X, support_list, family=family, rng=rng)
        candidate = fit_fixed_graph_em(
            X, prop_r, B_r, dispersion_r, support_list, family=family, **em_kwargs
        )
        start_logliks.append(candidate[5]["loglik"])
        if candidate[5]["loglik"] > best[5]["loglik"]:
            best, best_index = candidate, start
    best[5]["start_logliks"] = start_logliks
    best[5]["best_start"] = best_index
    return best


def _logsumexp_rows(values):
    """Row-wise ``log sum exp`` of a matrix (or of a vector's entries per row)."""
    peak = np.max(values, axis=-1, keepdims=True)
    return np.squeeze(peak, axis=-1) + np.log(np.sum(np.exp(values - peak), axis=-1))


def _logsumexp_cols(log_message, log_transition):
    """``log sum_a exp(m_i(a)) T(a, b)`` for messages ``m`` and log-transitions ``T``."""
    peak_m = np.max(log_message, axis=1, keepdims=True)
    peak_t = np.max(log_transition, axis=0, keepdims=True)
    product = np.exp(log_message - peak_m) @ np.exp(log_transition - peak_t)
    return np.log(np.maximum(product, 1e-300)) + peak_m + peak_t


__all__ = [
    "EXACT_EM_MAX_WIDTH",
    "fit_fixed_graph_em",
    "fit_fixed_graph_em_multistart",
    "graph_anchored_initialize",
    "random_graph_start",
    "resolve_graph_list",
]
