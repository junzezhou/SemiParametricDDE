"""User-specified graphs (``G_list``) for CopDDE fits."""

import numpy as np
import pytest
from scipy.optimize import minimize

from dde import fit
from dde.algorithms.copula import fit_copula_dde, prepare_rank_likelihood_inputs
from dde.algorithms.fixed_graph import spectral_graph_initialize
from dde.algorithms.get_SAEM_RL_D import (
    _gaussian_layer_on_support,
    _logistic_layer_on_support,
    get_SAEM_RL_D,
)
from dde.copula import simulate_gaussian_mixture
from dde.simulation import generate_X_Cop_marg_emp_D


def _two_parent_design(J=30, K1=6, K2=2):
    """Every child loads strongly on one parent and weakly negatively on the next,
    the pattern of the CopDDE paper's designs (no child has a single parent)."""
    B1 = np.zeros((J, K1 + 1))
    B1[:, 0] = -1.0
    for j in range(J):
        k = j % K1
        B1[j, 1 + k] = 2.0
        B1[j, 1 + (k + 1) % K1] = -0.7
    B2 = np.zeros((K1, K2 + 1))
    B2[:, 0] = -2.0
    for k in range(K1):
        B2[k, 1 + k % K2] = 4.0
    return np.full(K2, 0.5), [B1, B2], np.full(J, 0.3)


def _simulate(N, seed):
    prop, B_list, gamma = _two_parent_design()
    sim = generate_X_Cop_marg_emp_D(
        N, prop, lambdas=np.full(B_list[0].shape[0], 4.0), gamma=gamma, B_list=B_list,
        random_state=seed,
    )
    return sim, [B[:, 1:] != 0 for B in B_list]


def _state_agreement(fitted, true):
    """Per-column agreement of binary states up to the column's 0/1 coding."""
    same = np.mean(np.asarray(fitted) == np.asarray(true), axis=0)
    return np.maximum(same, 1.0 - same)


def test_gaussian_layer_on_support_is_restricted_least_squares():
    rng = np.random.default_rng(0)
    N, J, K, C = 200, 5, 3, 2
    A = np.concatenate([np.ones((N, 1, C)), rng.integers(0, 2, size=(N, K, C)).astype(float)], axis=1)
    Z = rng.normal(size=(N, J))
    support = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=bool)
    lb, ub = np.full(K + 1, -10.0), np.full(K + 1, 10.0)
    B, gamma = _gaussian_layer_on_support(Z, A, np.zeros((J, K + 1)), support, lb, ub, 50)
    design = np.transpose(A, (0, 2, 1)).reshape(N * C, K + 1)
    for j in range(J):
        columns = np.concatenate(([0], 1 + np.flatnonzero(support[j])))
        expected = np.linalg.lstsq(design[:, columns], np.repeat(Z[:, j], C), rcond=None)[0]
        np.testing.assert_allclose(B[j, columns], expected, atol=1e-8)
        assert np.all(B[j, 1:][~support[j]] == 0.0)
    assert gamma.shape == (J,) and np.all(gamma > 0)


def test_gaussian_layer_on_support_respects_bounds():
    rng = np.random.default_rng(1)
    N, C = 300, 1
    A = np.concatenate([np.ones((N, 1, C)), rng.integers(0, 2, size=(N, 1, C)).astype(float)], axis=1)
    Z = (30.0 * A[:, 1, 0])[:, None] + rng.normal(size=(N, 1))
    lb, ub = np.array([-10.0, -10.0]), np.array([10.0, 10.0])
    B, _ = _gaussian_layer_on_support(Z, A, np.zeros((1, 2)), np.ones((1, 1), bool), lb, ub, 200)
    assert B[0, 1] == pytest.approx(10.0)


def test_logistic_layer_on_support_matches_unaggregated_fit():
    rng = np.random.default_rng(2)
    N, K_child, K_parent, C = 400, 4, 3, 2
    parents = np.concatenate(
        [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, K_parent, C)).astype(float)], axis=1
    )
    children = np.concatenate(
        [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, K_child, C)).astype(float)], axis=1
    )
    support = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool)
    lb, ub = np.array([-5.0, -10.0, -10.0, -10.0]), np.full(K_parent + 1, 10.0)
    temp = 0.8
    B = _logistic_layer_on_support(
        children, parents, np.zeros((K_child, K_parent + 1)), support, temp, lb, ub, 500, 1
    )
    design = np.transpose(parents, (0, 2, 1)).reshape(N * C, K_parent + 1)
    for k in range(K_child):
        columns = np.concatenate(([0], 1 + np.flatnonzero(support[k])))
        y = children[:, k + 1, :].reshape(N * C)
        X = design[:, columns]

        def objective(x):
            eta = X @ x
            return temp / C * np.sum(np.logaddexp(0.0, eta) - y * eta)

        expected = minimize(objective, np.zeros(columns.size), method="L-BFGS-B",
                            bounds=list(zip(lb[columns], ub[columns]))).x
        np.testing.assert_allclose(B[k, columns], expected, atol=1e-4)
        assert np.all(B[k, 1:][~support[k]] == 0.0)


def test_kernel_holds_graph_and_skips_threshold():
    sim, G = _simulate(400, seed=3)
    X = sim["X"]
    Z, R, Rlevels = prepare_rank_likelihood_inputs(X)
    prop, B, gamma, A, _ = spectral_graph_initialize(Z, G, (6, 2))
    np.random.seed(0)
    raw = get_SAEM_RL_D(
        X, Z, R, Rlevels, prop, gamma, B, A, 1, 0.0, 3, 1.0, 0.0, 50.0,
        verbose=False, support_list=G, latent_transition="systematic_coordinate_gibbs",
    )
    for B_d, G_d in zip(raw["B_list"], G):
        assert np.all(B_d[:, 1:][~G_d] == 0.0)
        # tau = 50 would zero every slope of a thresholded layer
        assert np.count_nonzero(B_d[:, 1:][G_d]) == np.count_nonzero(G_d)
    assert raw["latent_transition"] == "systematic_coordinate_gibbs"
    assert raw["p"] is None and raw["top_probabilities"].shape == (400, 2)


def test_kernel_validates_support_and_transition():
    sim, G = _simulate(60, seed=4)
    Z, R, Rlevels = prepare_rank_likelihood_inputs(sim["X"])
    prop, B, gamma, A, _ = spectral_graph_initialize(Z, G, (6, 2))
    args = (sim["X"], Z, R, Rlevels, prop, gamma, B, A, 1, 0.0, 1, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="support_list\\[0\\] shape"):
        get_SAEM_RL_D(*args, verbose=False, support_list=[G[0][:, :5]])
    with pytest.raises(ValueError, match="latent_transition"):
        get_SAEM_RL_D(*args, verbose=False, latent_transition="gibbs")


def test_spectral_graph_initialize_labels_columns_by_the_graph():
    sim, G = _simulate(1500, seed=5)
    Z = simulate_gaussian_mixture(sim["X"], sim["X"].shape[1], random_state=5)
    prop, B, gamma, A, info = spectral_graph_initialize(Z, G, (6, 2))
    assert sorted(info["spectral_matches"][0]) == list(range(6))
    assert np.min(_state_agreement(A[0], sim["A_list"][0])) > 0.9
    for B_d, G_d in zip(B, G):
        assert np.all(B_d[:, 1:][~G_d] == 0.0)


def test_fit_copdde_on_a_given_graph():
    sim, G = _simulate(1500, seed=6)
    np.random.seed(6)
    result = fit(sim["X"], model="CopDDE", dimension_method="fixed", G_list=[g.astype(int) for g in G],
                 random_state=6, verbose=False, it=20, n_init=5)
    assert result.K_list == [6, 2]
    assert result.diagnostics["dimension_method"] == "fixed"
    assert result.diagnostics["fixed_graph_layers"] == 2
    assert result.diagnostics["latent_transition"] == "systematic_coordinate_gibbs"
    start = result.diagnostics["graph_start"]
    assert start["warmup_it"] == 10 and len(start["deep_em_start_logliks"]) == 5
    for B_d, G_d in zip(result.B_list, G):
        assert np.all(B_d[:, 1:][~G_d] == 0.0)
    assert np.min(_state_agreement(result.A_list[0], sim["A_list"][0])) > 0.95
    assert np.min(_state_agreement(result.A_list[1], sim["A_list"][1])) > 0.8
    # standardized observed-layer slopes against the truth, up to each latent's coding
    sd = result.transformed.std(axis=0, ddof=1)
    true_sd = sim["Z"].std(axis=0, ddof=1)
    for k in range(6):
        rows = G[0][:, k]
        fitted = result.B_list[0][rows, k + 1] / sd[rows]
        true = sim["B_list"][0][rows, k + 1] / true_sd[rows]
        assert abs(np.corrcoef(fitted, true)[0, 1]) > 0.95


def test_graph_argument_errors():
    sim, G = _simulate(80, seed=7)
    X = sim["X"]
    with pytest.raises(ValueError, match="G_list requires dimension_method='fixed'"):
        fit_copula_dde(X, G_list=G, dimension_method="csp", verbose=False)
    with pytest.raises(ValueError, match="only to fits with G_list"):
        fit_copula_dde(X, K_list=[6, 2], dimension_method="fixed", n_init=3, verbose=False)
    with pytest.raises(ValueError, match="partial G_list needs"):
        fit_copula_dde(X, K_list=[6, 2], G_list=[G[0]], verbose=False)
    with pytest.raises(ValueError, match="rows"):
        fit_copula_dde(X, G_list=[G[0][:-1]], verbose=False)


def test_partial_graph_keeps_free_layer_penalized():
    sim, G = _simulate(300, seed=8)
    np.random.seed(8)
    result = fit_copula_dde(
        sim["X"], K_list=[6, 2], G_list=[G[0]], verbose=False,
        pen_list=[0.0, 1.0], tau=0.3, it=3,
    )
    assert result.diagnostics["fixed_graph_layers"] == 1
    assert np.all(result.B_list[0][:, 1:][~G[0]] == 0.0)
    free_slopes = result.B_list[1][:, 1:]
    assert np.all((free_slopes == 0.0) | (np.abs(free_slopes) >= 0.3))
