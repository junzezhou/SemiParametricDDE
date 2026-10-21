"""Fixed-width Poisson DDE: anchor-mixture initialization, SA burn-in, layer-wise warm start."""
from __future__ import annotations

import numpy as np
import pytest

from dde import fit
from dde.algorithms.poisson_init import (
    Poisson_init,
    _anchor_mixture_binarize,
    _two_component_poisson_mixture,
)
from dde.identifiability import build_true_params
from dde.simulation import generate_X_poisson


def _design(K_list, seed, N):
    J = 3 * K_list[0]
    prop, B_list, G_list, _ = build_true_params(list(K_list), J, 4.0)
    B_list = [np.asarray(B, float) for B in B_list]
    G_list = [np.asarray(G, bool) for G in G_list]
    X, A_true = generate_X_poisson(N, np.asarray(prop, float), B_list, random_state=seed)
    return X, [np.asarray(A, float) for A in A_true], B_list, G_list


def test_two_component_poisson_mixture_recovers_prevalence_and_rates():
    rng = np.random.default_rng(0)
    z = rng.random(20000) < 0.15
    x = rng.poisson(np.where(z, 8.0, 0.7))
    pi, lam0, lam1 = _two_component_poisson_mixture(x)
    assert abs(pi - 0.15) < 0.02
    assert abs(lam0 - 0.7) < 0.1 and abs(lam1 - 8.0) < 0.4


def test_anchor_mixture_binarize_recovers_rare_column_prevalence():
    rng = np.random.default_rng(1)
    N = 6000
    A = np.column_stack([rng.random(N) < 0.5, rng.random(N) < 0.12]).astype(float)
    # two anchors per latent column, log-rate -1 + 2.5 * A
    rates = np.exp(-1.0 + 2.5 * A[:, [0, 0, 1, 1]])
    X = rng.poisson(rates)
    B_est = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    parent, prevalence = _anchor_mixture_binarize(X, B_est)
    assert abs(prevalence[1] - 0.12) < 0.03
    assert np.mean(parent[:, 1] == A[:, 1]) > 0.97
    assert np.mean(parent[:, 0] == A[:, 0]) > 0.97


def test_anchor_mixture_initializer_starts_rare_columns_near_true_prevalence():
    X, A_true, B_true, G_true = _design((18, 6, 2), seed=3, N=4000)
    truth = A_true[0].mean(axis=0)
    _, _, _, A_proj, _ = Poisson_init(X, (18, 6, 2), mode="oracle", B_list=B_true, G_list=G_true, binarize="projection")
    _, _, _, A_mix, _ = Poisson_init(X, (18, 6, 2), mode="oracle", B_list=B_true, G_list=G_true, binarize="anchor_mixture")
    rare = truth < 0.2
    assert rare.any()
    # projection at zero puts about half of the observations in every column
    assert np.all(np.abs(A_proj[0].mean(axis=0)[rare] - truth[rare]) > 0.25)
    assert np.all(np.abs(A_mix[0].mean(axis=0)[rare] - truth[rare]) < 0.05)
    assert np.mean(A_mix[0] == A_true[0]) > np.mean(A_proj[0] == A_true[0]) + 0.05


def test_poisson_fixed_defaults_record_burn_in_and_warm_start():
    X, _, B_true, G_true = _design((6, 2), seed=5, N=600)
    result = fit(X, method="DDE", family="Poisson", D=2, dimension_method="fixed", K_list=[6, 2],
                 pen_list=[2.0, 2.0], tau=0.3, C=1, tol=0.01, max_it=12, verbose=False)
    d = result.diagnostics
    assert d["sa_burn_in"] == 10 and d["fixed_init_binarize"] == "anchor_mixture"
    assert d["fixed_warm_start"] == "layerwise"
    assert d["warm_start"]["stage1_iterations"] >= 1 and d["warm_start"]["stage2_iterations"] >= 1
    # the convergence test starts after the burn-in
    assert d["iterations"] > 10


def test_poisson_one_layer_and_provided_state_skip_the_warm_start():
    X, A_true, B_true, G_true = _design((6, 2), seed=6, N=500)
    one = fit(X, method="DDE", family="Poisson", D=1, dimension_method="fixed", K_list=[6],
              pen_list=[2.0], tau=0.3, C=1, tol=0.01, max_it=12, verbose=False)
    assert one.diagnostics["fixed_warm_start"] == "none" and "warm_start" not in one.diagnostics
    provided = fit(X, method="DDE", family="Poisson", D=2, dimension_method="fixed", K_list=[6, 2],
                   pen_list=[2.0, 2.0], tau=0.3, C=1, tol=0.01, max_it=12, verbose=False,
                   prop_in=np.full(2, 0.5), B_list_in=B_true, A_list_in=A_true)
    assert provided.diagnostics["fixed_warm_start"] == "none" and "warm_start" not in provided.diagnostics


def test_other_families_keep_their_fixed_defaults():
    rng = np.random.default_rng(7)
    Y = (rng.random((200, 18)) < 0.4).astype(float)
    bernoulli = fit(Y, method="DDE", family="Bernoulli", D=2, dimension_method="fixed", K_list=[6, 2],
                    pen_list=[1.0, 1.0], tau=0.3, C=1, tol=0.5, max_it=3, verbose=False)
    assert bernoulli.diagnostics["sa_burn_in"] == 10
    assert bernoulli.diagnostics["fixed_init_binarize"] == "anchor_lca"
    assert bernoulli.diagnostics["fixed_warm_start"] == "none"
    # the Normal defaults are covered by tests/test_normal_fixed_defaults.py
    with pytest.raises(ValueError):
        fit(Y, method="DDE", family="Bernoulli", D=2, dimension_method="fixed", K_list=[6, 2],
            pen_list=[1.0, 1.0], tau=0.3, C=1, max_it=2, verbose=False, fixed_init_binarize="anchor_mixture")


def test_poisson_two_layer_recovery_with_oracle_labels():
    X, A_true, B_true, G_true = _design((6, 2), seed=11, N=2000)
    np.random.seed(11)
    result = fit(X, method="DDE", family="Poisson", D=2, dimension_method="fixed", K_list=[6, 2],
                 pen_list=[2000 ** 0.25] * 2, tau=0.3, C=1, tol=0.01, max_it=50,
                 initialize="oracle", B_list_true=B_true, G_list_true=G_true, verbose=False)
    for B_hat, B_t, G_t in zip(result.B_list, B_true, G_true):
        assert np.mean((np.asarray(B_hat)[:, 1:] != 0) == G_t) >= 0.95
    assert np.sqrt(np.mean((np.asarray(result.B_list[0]) - B_true[0]) ** 2)) < 0.1
    assert np.mean(np.asarray(result.A_list[0]) == A_true[0]) > 0.9


def test_accumulators_reset_history_on_unit_step_size():
    from dde.algorithms.parametric_families import _ObservedPoissonAccumulator
    from dde.algorithms.semi_saem import (
        _BernoulliLayerAccumulator,
        _BinaryHistoryBernoulliLayerAccumulator,
    )
    rng = np.random.default_rng(0)
    N, K = 300, 25   # K > direct-address bits: grouped and binary-history backends
    X = rng.poisson(2.0, size=(N, 4)).astype(float)
    draws = [np.concatenate([np.ones((N, 1, 1)), rng.random((N, K, 1)) < 0.5], axis=1).astype(np.uint8) for _ in range(3)]
    acc = _ObservedPoissonAccumulator(4, K + 1)
    acc.update(X, draws[0], 1); acc.update(X, draws[1], 1)        # two burn-in sweeps: history replaced
    assert acc.pattern_count == np.unique(np.packbits(draws[1][:, 1:, 0], axis=1, bitorder="little"), axis=0).shape[0]
    acc.update(X, draws[2], 2)                                     # averaging starts: union of two sweeps
    assert acc.pattern_count > N // 2 and abs(acc.counts.sum() - N) < 1e-9
    grouped = _BernoulliLayerAccumulator(K, 4)
    children = (rng.random((N, 4)) < 0.5).astype(np.uint8)
    grouped.update(children, draws[0], 1); grouped.update(children, draws[1], 1)
    assert grouped.codes.size <= N and abs(grouped.counts.sum() - N) < 1e-9
    history = _BinaryHistoryBernoulliLayerAccumulator(K, 4)
    history.update(children, draws[0], 1); history.update(children, draws[1], 1)
    assert history._size == N                                      # history dropped, then one sweep stored
    history.update(children, draws[2], 2)                          # sequential 1/t continues from the reset
    assert history._size == 2 * N


def test_poisson_support_tables_match_dense_design_and_active_set_matches_rowwise():
    from dde.algorithms.parametric_families import (
        _ObservedPoissonAccumulator,
        _solve_poisson_tlp_active_set,
        _solve_poisson_tlp_rowwise,
    )
    rng = np.random.default_rng(3)
    N, K, J = 400, 7, 5
    A = (rng.random((N, K)) < 0.4).astype(np.uint8)
    B_true = np.zeros((J, K + 1)); B_true[:, 0] = -0.5
    for j in range(J):
        B_true[j, 1 + (j % K)] = 2.0
    X = rng.poisson(np.exp(np.column_stack([np.ones(N), A]) @ B_true.T)).astype(float)
    acc = _ObservedPoissonAccumulator(J, K + 1)
    draws = np.concatenate([np.ones((N, 1, 1)), A[:, :, None]], axis=1).astype(np.uint8)
    acc.update(X, draws, 1)
    support = np.array([1, 4])
    weights, cross = acc.support_tables(support)
    assert abs(weights.sum() - N) < 1e-9
    np.testing.assert_allclose(cross.sum(axis=0), acc.counts @ acc.design, rtol=1e-12)
    # cross-table for sub-pattern 0 only counts patterns with both support bits off
    design = acc.design
    mask = (design[:, 2] == 0) & (design[:, 5] == 0)
    np.testing.assert_allclose(cross[0], acc.counts[mask] @ design[mask], rtol=1e-12)
    bounds = [(-5.0, 5.0)] + [(-2.5, 5.0)] * K
    start = B_true + rng.normal(0, 0.1, B_true.shape)
    active = _solve_poisson_tlp_active_set(acc, start, 1.0, 0.3, bounds)
    dense = _solve_poisson_tlp_rowwise(acc, start, 1.0, 0.3, bounds, max_iter=200)
    def penalized(B):
        return sum(acc.value(j, B[j]) + 1.0 * np.sum(np.minimum(np.abs(B[j, 1:]), 0.3)) for j in range(J))
    assert penalized(active) <= penalized(dense) + 1e-3
    assert np.max(np.abs(active - B_true)) < 0.35
