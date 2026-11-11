"""User-specified graphs (``G_list``) for fixed-width DDE and SemiDDE fits."""

import itertools

import numpy as np
import pytest
from scipy.special import expit, gammaln

from dde import fit
from dde.algorithms.dde import fit_fixed_graph_reference
from dde.algorithms.fixed_graph import (
    _fixed_graph_estep,
    _layer_configs,
    fit_fixed_graph_em,
    graph_anchored_initialize,
    resolve_graph_list,
)
from dde.algorithms.parametric_families import (
    NORMAL_FAMILY,
    _solve_normal_on_support,
    resolve_parametric_family,
)
from dde.algorithms.semi_saem import get_SAEM_parametric


def _bs_layer(K):
    """MATLAB ``Sim_sid_*.m`` B_s coefficients for one 3K x K layer."""
    slopes = np.zeros((3 * K, K))
    half = K // 2
    for k in range(K):
        slopes[k, k] = 4.0
        slopes[k + K, k] = 4.0
    for k in range(half):
        slopes[k + 2 * K, k] = 4.0
        slopes[k + 2 * K + half, k] = -4.0 / 3.0
        slopes[k + 2 * K, k + half] = -4.0 / 3.0
        slopes[k + 2 * K + half, k + half] = 4.0
    intercept = np.concatenate([np.full(K, -2.0), np.full(K, -4.0), np.full(K, -2.0)])
    return np.column_stack([intercept, slopes])


def _simulate(family, N, K_list, seed):
    rng = np.random.default_rng(seed)
    B_list = [_bs_layer(K) for K in K_list]
    A_list = [None] * len(K_list)
    A_list[-1] = rng.binomial(1, 0.5, size=(N, K_list[-1]))
    for d in range(len(K_list) - 2, -1, -1):
        eta = B_list[d + 1][:, 0] + A_list[d + 1] @ B_list[d + 1][:, 1:].T
        A_list[d] = rng.binomial(1, expit(eta))
    eta = B_list[0][:, 0] + A_list[0] @ B_list[0][:, 1:].T
    if family == "bernoulli":
        X = rng.binomial(1, expit(eta))
    elif family == "poisson":
        X = rng.poisson(np.exp(eta))
    else:
        X = eta + rng.normal(size=eta.shape)
    return X, B_list, A_list


def _graph(B_list):
    return [B[:, 1:] != 0 for B in B_list]


# ---------------------------------------------------------------------------
# Graph validation
# ---------------------------------------------------------------------------


def test_resolve_graph_list_derives_widths_and_pads_free_layers():
    G1 = np.zeros((9, 3), dtype=int)
    G1[np.arange(9), np.arange(9) % 3] = 1
    K, support = resolve_graph_list([G1], D=2, J=9, K_list=[3, 1])
    assert K == (3, 1)
    assert support[0].dtype == bool and support[1] is None
    # a graph that allows every connection restricts nothing
    K, support = resolve_graph_list([G1, np.ones((3, 1))], D=2, J=9)
    assert K == (3, 1) and support[0].shape == (9, 3) and support[1] is None
    K, support = resolve_graph_list([np.ones((9, 3)), np.ones((3, 1))], D=2, J=9)
    assert K == (3, 1) and support == [None, None]


def test_exact_em_keeps_an_all_ones_layer_of_the_graph():
    X, B_true, _ = _simulate("normal", 600, (6, 2), seed=16)
    G = [_graph(B_true)[0], np.ones((6, 2), int)]
    result = fit_fixed_graph_reference(X, family="normal", G_list=G, n_init=2)
    assert result.diagnostics["graph_algorithm"] == "em"
    assert result.diagnostics["fixed_graph_layers"] == [True, True]
    K, support = resolve_graph_list(G, D=2, J=18, drop_unrestricted=False)
    assert support[1].all()


def test_resolve_graph_list_rejects_restricted_layers_above_an_all_ones_graph():
    G2 = np.array([[1], [1], [0]])
    with pytest.raises(ValueError, match="does not label its latent variables"):
        resolve_graph_list([np.ones((9, 3)), G2], D=2, J=9)
    with pytest.raises(ValueError, match="J >= 3"):
        resolve_graph_list([np.ones((9, 4))], D=1, J=9)


@pytest.mark.parametrize(
    "G_list, kwargs, message",
    [
        ([None, np.ones((3, 1))], {}, "observed-layer graph"),
        ([np.ones((9, 3)), None, np.ones((1, 1))], {"K_list": [3, 1, 1]}, "shallowest"),
        ([np.ones((8, 3))], {"K_list": [3]}, "rows"),
        ([np.full((9, 3), 0.5)], {}, "0/1"),
        ([np.column_stack([np.ones(9), np.ones(9), np.zeros(9)])], {}, "no children"),
        ([np.ones((9, 3))], {"K_list": [2]}, "disagrees"),
    ],
)
def test_resolve_graph_list_rejects_invalid_graphs(G_list, kwargs, message):
    D = kwargs.get("K_list", [None] * len(G_list))
    with pytest.raises(ValueError, match=message):
        resolve_graph_list(G_list, D=len(D), J=9, **kwargs)


def test_resolve_graph_list_requires_widths_and_ladder_for_free_layers():
    G1 = np.ones((9, 3))
    with pytest.raises(ValueError, match="K_list is required"):
        resolve_graph_list([G1], D=2, J=9)
    with pytest.raises(ValueError, match=">= 3"):
        resolve_graph_list([G1], D=2, J=9, K_list=[3, 2])


# ---------------------------------------------------------------------------
# M-step pieces
# ---------------------------------------------------------------------------


def test_normal_support_solve_matches_full_solve_and_restricted_least_squares():
    rng = np.random.default_rng(0)
    design = np.column_stack([np.ones(200), rng.binomial(1, 0.5, size=(200, 3))])
    Q = design.T @ design
    q_all = rng.normal(size=(4, 4)) @ Q
    full = _solve_normal_on_support(Q, q_all, np.ones((4, 3), dtype=bool))
    np.testing.assert_allclose(full, np.linalg.solve(Q, q_all.T).T)
    support = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]], dtype=bool)
    restricted = _solve_normal_on_support(Q, q_all, support)
    assert np.all(restricted[:, 1:][~support] == 0.0)
    for j in range(4):
        columns = np.concatenate(([0], 1 + np.flatnonzero(support[j])))
        expected = np.linalg.solve(Q[np.ix_(columns, columns)], q_all[j, columns])
        np.testing.assert_allclose(restricted[j, columns], expected)


@pytest.mark.parametrize("family", ["bernoulli", "poisson", "normal"])
def test_saem_kernel_penalizes_and_thresholds_the_allowable_slopes(family):
    X, B_true, _ = _simulate(family, 600, (6, 2), seed=1)
    support = _graph(B_true)
    adapter = resolve_parametric_family(family)
    prop, B_init, gamma, A_init, _ = graph_anchored_initialize(
        X, support, (6, 2), family=adapter
    )
    args = (X.astype(float), prop, B_init, gamma)
    # The graph only marks allowable connections: tau thresholds them as
    # without a graph, so a huge tau zeroes every slope.
    _, B_huge, _, _, _ = get_SAEM_parametric(
        *args, [0.0, 0.0], 50.0, A_init, 1, 0.005,
        max_iter=3, verbose=False, family=adapter, support_list=support,
    )
    assert all(np.all(B[:, 1:] == 0.0) for B in B_huge)
    _, B_list, _, _, _ = get_SAEM_parametric(
        *args, [1.0, 1.0], 0.3, A_init, 1, 0.005,
        max_iter=8, verbose=False, family=adapter, support_list=support,
    )
    for B, G in zip(B_list, support):
        assert np.all(B[:, 1:][~G] == 0.0)
        slopes = B[:, 1:][G]
        assert np.all((slopes == 0.0) | (np.abs(slopes) >= 0.3))
        assert np.mean(slopes != 0.0) > 0.9


# ---------------------------------------------------------------------------
# Exact EM
# ---------------------------------------------------------------------------


def _brute_force_loglik(X, prop, B_list, gamma, family):
    """log p(X) by summing the joint over every state of every layer."""
    K = [B.shape[1] - 1 for B in B_list]
    total = np.zeros(X.shape[0])
    terms = []
    for states in itertools.product(*[range(1 << k) for k in K]):
        a = [_layer_configs(k)[s] for k, s in zip(K, states)]
        log_joint = np.sum(a[-1] * np.log(prop) + (1 - a[-1]) * np.log1p(-prop))
        for d in range(len(K) - 1, 0, -1):
            eta = B_list[d][:, 0] + B_list[d][:, 1:] @ a[d]
            log_joint += np.sum(a[d - 1] * eta - np.logaddexp(0.0, eta))
        eta = B_list[0][:, 0] + B_list[0][:, 1:] @ a[0]
        if family == "bernoulli":
            obs = X @ eta - np.logaddexp(0.0, eta).sum()
        elif family == "poisson":
            obs = X @ eta - np.exp(eta).sum() - gammaln(X + 1).sum(axis=1)
        else:
            obs = -0.5 * (((X - eta) ** 2) / gamma).sum(axis=1) - 0.5 * np.log(2 * np.pi * gamma).sum()
        terms.append(obs + log_joint)
    terms = np.vstack(terms)
    peak = terms.max(axis=0)
    total = peak + np.log(np.exp(terms - peak).sum(axis=0))
    return float(total.sum())


@pytest.mark.parametrize("family", ["bernoulli", "poisson", "normal"])
def test_exact_em_estep_matches_brute_force_three_layers(family):
    rng = np.random.default_rng(3)
    K = (3, 2, 1)
    J = 7
    B_list = [
        np.column_stack([rng.normal(size=J), rng.normal(size=(J, 3))]),
        np.column_stack([rng.normal(size=3), rng.normal(size=(3, 2))]),
        np.column_stack([rng.normal(size=2), rng.normal(size=(2, 1))]),
    ]
    prop = np.array([0.35])
    gamma = rng.uniform(0.5, 1.5, size=J) if family == "normal" else None
    if family == "bernoulli":
        X = rng.binomial(1, 0.4, size=(40, J)).astype(float)
    elif family == "poisson":
        X = rng.poisson(1.5, size=(40, J)).astype(float)
    else:
        X = rng.normal(size=(40, J))
    adapter = resolve_parametric_family(family)
    configs = [_layer_configs(k) for k in K]
    designs = [np.column_stack([np.ones(c.shape[0]), c]) for c in configs]
    stats = _fixed_graph_estep(X, prop, B_list, gamma, configs, designs, adapter, 16, True)
    expected = _brute_force_loglik(X, prop, B_list, gamma, family)
    constant = -float(gammaln(X + 1).sum()) if family == "poisson" else 0.0
    assert stats["loglik"] + constant == pytest.approx(expected, rel=1e-10)
    for posterior in stats["posteriors"]:
        np.testing.assert_allclose(np.vstack(posterior).sum(axis=1), 1.0)


@pytest.mark.parametrize("family", ["bernoulli", "poisson", "normal"])
def test_exact_em_increases_loglik_and_recovers_parameters(family):
    X, B_true, A_true = _simulate(family, 4000, (6, 2), seed=11)
    support = _graph(B_true)
    adapter = resolve_parametric_family(family)
    prop, B_init, gamma, _, _ = graph_anchored_initialize(X, support, (6, 2), family=adapter)
    prop, B_list, gamma, A_list, iterations, diagnostics = fit_fixed_graph_em(
        X, prop, B_init, gamma, support, family=adapter
    )
    history = np.asarray(diagnostics["loglik_history"])
    assert np.all(np.diff(history) >= -1e-6 * X.shape[0])
    assert diagnostics["converged"]
    for B, G in zip(B_list, support):
        assert np.all(B[:, 1:][~G] == 0.0)
    assert np.sqrt(np.mean((B_list[0] - B_true[0]) ** 2)) < 0.25
    assert np.sqrt(np.mean((B_list[1] - B_true[1]) ** 2)) < 0.6
    assert np.mean(A_list[0] == A_true[0]) > 0.9
    marginals = diagnostics["posterior_marginals"]
    assert marginals[0].shape == (4000, 6) and marginals[1].shape == (4000, 2)


# ---------------------------------------------------------------------------
# Public fit paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["bernoulli", "poisson", "normal"])
def test_exact_em_is_a_reference_implementation_outside_fit(family):
    X, B_true, _ = _simulate(family, 1500, (6, 2), seed=5)
    G = _graph(B_true)
    with pytest.raises(ValueError, match="pen_list"):
        fit(X, model="DDE", dimension_method="fixed", family=family, G_list=G, verbose=False)
    for removed in ({"graph_algorithm": "em"}, {"n_init": 3}):
        with pytest.raises(TypeError, match="fit_fixed_graph_reference"):
            fit(X, model="DDE", dimension_method="fixed", family=family, G_list=G,
                pen_list=[1.0, 1.0], tau=0.3, verbose=False, **removed)
    result = fit_fixed_graph_reference(X, family=family, G_list=G)
    assert result.K_list == [6, 2]
    assert result.diagnostics["graph_algorithm"] == "em"
    assert result.diagnostics["initialization"] == "data_graph"
    for B, support in zip(result.B_list, G):
        assert np.all(B[:, 1:][~support] == 0.0)
    np.testing.assert_array_equal(result.diagnostics["G_list"][0], G[0].astype(np.uint8))


@pytest.mark.parametrize("family", ["bernoulli", "poisson", "normal"])
def test_dde_fit_with_graph_saem_path(family):
    X, B_true, _ = _simulate(family, 1500, (6, 2), seed=6)
    G = _graph(B_true)
    result = fit(
        X, model="DDE", dimension_method="fixed", family=family, D=2, G_list=G,
        pen_list=[1.0, 1.0], tau=0.3, max_it=15, verbose=False,
    )
    diagnostics = result.diagnostics
    assert diagnostics["graph_algorithm"] == "saem"
    assert diagnostics["fixed_graph_layers"] == [True, True]
    assert diagnostics["tau"] == 0.3
    assert diagnostics["sa_burn_in"] == (0 if family == "normal" else 10)
    for B, support in zip(result.B_list, G):
        assert np.all(B[:, 1:][~support] == 0.0)


def test_dde_fit_with_partial_graph_estimates_deeper_layer():
    X, B_true, _ = _simulate("poisson", 1500, (6, 2), seed=7)
    G1 = _graph(B_true)[0]
    result = fit(
        X, model="DDE", dimension_method="fixed", family="poisson", D=2, K_list=[6, 2], G_list=[G1],
        pen_list=[0.0, 2.0], tau=0.3, C=1, max_it=15, verbose=False,
    )
    diagnostics = result.diagnostics
    assert diagnostics["graph_algorithm"] == "saem"
    assert diagnostics["fixed_graph_layers"] == [True, False]
    assert np.all(result.B_list[0][:, 1:][~G1] == 0.0)
    slopes = result.B_list[1][:, 1:]
    assert np.all((slopes == 0.0) | (np.abs(slopes) >= 0.3))


def test_dde_fit_graph_argument_errors():
    X, B_true, _ = _simulate("bernoulli", 300, (6, 2), seed=8)
    G = _graph(B_true)
    with pytest.raises(ValueError, match="dimension_method='fixed'"):
        fit(X, model="DDE", family="bernoulli", D=2, G_list=G, dimension_method="csp",
            t_spike=[0.1, 0.1], temp=0.9, tau=0.3, C=1, verbose=False)
    with pytest.raises(ValueError, match="restrict every layer"):
        fit_fixed_graph_reference(X, family="bernoulli", G_list=[G[0], None], K_list=[6, 2])
    with pytest.raises(ValueError, match="tau is required"):
        fit(X, model="DDE", dimension_method="fixed", family="bernoulli", D=2, K_list=[6, 2], G_list=[G[0]],
            pen_list=[0.0, 1.0], C=1, verbose=False)
    with pytest.raises(ValueError, match="labels the latent columns"):
        fit(X, model="DDE", dimension_method="fixed", family="bernoulli", G_list=G, initialize="oracle",
            B_list_true=B_true, G_list_true=G, pen_list=[1.0, 1.0], tau=0.3, verbose=False)
    with pytest.raises(ValueError, match="C must be a positive integer"):
        fit(X, model="DDE", dimension_method="fixed", family="bernoulli", G_list=G,
            pen_list=[1.0, 1.0], tau=0.3, C=0, verbose=False)
    with pytest.raises(ValueError, match="pen_list"):
        fit(X, model="DDE", dimension_method="fixed", family="bernoulli", G_list=G, tau=0.3, verbose=False)


def test_semidde_fit_with_graph_holds_support():
    X, B_true, A_true = _simulate("normal", 1000, (6, 2), seed=9)
    Y = np.exp(X / 2.0)
    G = _graph(B_true)
    result = fit(Y, model="SemiDDE", dimension_method="fixed", G_list=G, pen_list=[1.0, 1.0], tau=0.1,
                 C=1, tol=0.005, verbose=False)
    assert result.diagnostics["initialization"] == "data_graph"
    assert result.diagnostics["fixed_graph_layers"] == [True, True]
    assert result.diagnostics["sa_burn_in"] == 0
    for B, support in zip(result.B_list, G):
        assert np.all(B[:, 1:][~support] == 0.0)
    assert np.mean(result.A_list[0] == A_true[0]) > 0.95


def test_exact_em_multistart_keeps_best_start_and_is_reproducible():
    X, B_true, _ = _simulate("normal", 600, (6, 2), seed=12)
    G = _graph(B_true)
    em = dict(family="normal", G_list=G)
    first = fit_fixed_graph_reference(X, n_init=4, random_state=3, **em)
    again = fit_fixed_graph_reference(X, n_init=4, random_state=3, **em)
    logliks = first.diagnostics["start_logliks"]
    assert len(logliks) == 4 and first.diagnostics["n_init"] == 4
    assert first.diagnostics["loglik"] == pytest.approx(max(logliks))
    assert logliks[first.diagnostics["best_start"]] == pytest.approx(max(logliks))
    np.testing.assert_array_equal(first.B_list[1], again.B_list[1])
    single = fit_fixed_graph_reference(X, n_init=1, **em)
    assert single.diagnostics["loglik"] <= first.diagnostics["loglik"] + 1e-8
    with pytest.raises(ValueError, match="random_state"):
        fit(X, model="DDE", dimension_method="fixed", family="normal", D=2, G_list=G, C=1,
            pen_list=[1.0, 1.0], tau=0.3, random_state=1, verbose=False)


@pytest.mark.parametrize("family", ["normal", "bernoulli"])
def test_all_ones_graph_gives_the_fit_without_a_graph(family):
    X, B_true, _ = _simulate(family, 400, (6, 2), seed=14)
    args = dict(model="DDE", dimension_method="fixed", family=family, D=2,
                pen_list=[1.0, 1.0], tau=0.3, max_it=6, verbose=False)
    np.random.seed(2)
    plain = fit(X, K_list=[6, 2], **args)
    np.random.seed(2)
    ones = fit(X, G_list=[np.ones((18, 6), int), np.ones((6, 2), int)], **args)
    assert ones.K_list == plain.K_list == [6, 2]
    for B_ones, B_plain in zip(ones.B_list, plain.B_list):
        np.testing.assert_array_equal(B_ones, B_plain)
    assert ones.diagnostics["initialization"] == plain.diagnostics["initialization"] == "data"


def test_graph_penalty_and_threshold_prune_allowable_edges():
    # The true graph plus every cross edge of layer 1 is allowed; the threshold
    # removes most of the extra edges while the graph keeps G = 0 entries at zero.
    X, B_true, _ = _simulate("normal", 3000, (6, 2), seed=15)
    G = _graph(B_true)
    G_super = [G[0].copy(), G[1]]
    G_super[0][2 * 6:, :] = True
    np.random.seed(3)
    result = fit(X, model="DDE", dimension_method="fixed", family="normal", G_list=G_super,
                 pen_list=[3000 ** 0.25] * 2, tau=0.3, max_it=30, verbose=False)
    slopes = result.B_list[0][:, 1:]
    extra = G_super[0] & ~G[0]
    assert np.all(slopes[~G_super[0]] == 0.0)
    assert np.mean(slopes[extra] == 0.0) > 0.8
    assert np.all(slopes[G[0]] != 0.0)


@pytest.mark.parametrize("family", ["bernoulli", "poisson", "normal"])
def test_random_graph_start_respects_graph_and_bounds(family):
    from dde.algorithms.fixed_graph import random_graph_start

    X, B_true, _ = _simulate(family, 300, (6, 2), seed=13)
    G = _graph(B_true)
    adapter = resolve_parametric_family(family)
    prop, B_list, dispersion = random_graph_start(X, G, family=adapter, rng=np.random.default_rng(0))
    assert prop.shape == (2,) and np.all((prop > 0.3) & (prop < 0.7))
    for d, (B, support) in enumerate(zip(B_list, G)):
        assert np.all(B[:, 1:][~support] == 0.0)
        assert np.all(B[:, 1:][support] > 0.0)
        if not (d == 0 and family == "normal"):
            bounds = adapter.fixed_latent_bounds(B.shape[1], np.inf)
            assert np.all(B >= [b[0] for b in bounds]) and np.all(B <= [b[1] for b in bounds])
    assert (dispersion is None) == (family != "normal")
