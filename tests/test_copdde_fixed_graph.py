"""User-specified graphs (``G_list``) for CopDDE fits."""

import numpy as np
import pytest

from dde import fit
from dde.algorithms.copula import fit_copula_dde, prepare_rank_likelihood_inputs
from dde.algorithms.fixed_graph import spectral_graph_initialize
from dde.algorithms.get_SAEM_RL_D import get_SAEM_RL_D
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


def test_kernel_holds_graph_and_thresholds_allowable_slopes():
    sim, G = _simulate(400, seed=3)
    X = sim["X"]
    Z, R, Rlevels = prepare_rank_likelihood_inputs(X)
    prop, B, gamma, A, _ = spectral_graph_initialize(Z, G, (6, 2))
    args = (X, Z, R, Rlevels, prop, gamma, B, A, 1, 0.0, 3, 1.0)
    # The graph only marks allowable connections: a huge tau zeroes them all.
    np.random.seed(0)
    huge = get_SAEM_RL_D(*args, 0.0, 50.0, verbose=False, support_list=G,
                         latent_transition="systematic_coordinate_gibbs")
    assert all(np.all(B_d[:, 1:] == 0.0) for B_d in huge["B_list"])
    np.random.seed(0)
    raw = get_SAEM_RL_D(*args, 1.0, 0.1, verbose=False, support_list=G,
                        latent_transition="systematic_coordinate_gibbs")
    for B_d, G_d in zip(raw["B_list"], G):
        assert np.all(B_d[:, 1:][~G_d] == 0.0)
        slopes = B_d[:, 1:][G_d]
        assert np.all((slopes == 0.0) | (np.abs(slopes) >= 0.1))
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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_graph_matching_separates_overlapping_parents(seed):
    """Three correlated latent variables whose children pair them cyclically,
    (strong, weak) = (0, 2), (1, 0), (2, 1), as latent variables 4, 6, 8 of the
    CopDDE paper's J = 100 design.  Given the true states, summed squared
    marginal correlations prefer a cyclic relabelling; the joint
    least-squares criterion returns the truth, and is invariant to a state's
    0/1 coding."""
    from scipy.special import expit

    from dde.algorithms.fixed_graph import _marginal_graph_matching, _match_columns_to_graph

    rng = np.random.default_rng(seed)
    N = 2000
    top = rng.binomial(1, 0.5, N)
    A = np.column_stack([rng.binomial(1, expit(-2.0 + 4.0 * top)) for _ in range(3)]).astype(float)
    children, G = [], []
    for strong, weak in [(0, 2), (1, 0), (2, 1)]:
        for _ in range(8):
            children.append(-2.0 + 4.0 * A[:, strong] - 1.5 * A[:, weak] + rng.normal(size=N))
            row = np.zeros(3, dtype=bool)
            row[[strong, weak]] = True
            G.append(row)
    children, G = np.column_stack(children), np.array(G)
    assert not np.array_equal(_marginal_graph_matching(children, A, G), np.arange(3))
    order = np.array([2, 0, 1])
    states = A[:, order]
    states[:, 1] = 1.0 - states[:, 1]
    assert np.array_equal(_match_columns_to_graph(children, states, G), np.argsort(order))


def test_fit_copdde_on_a_given_graph():
    sim, G = _simulate(1500, seed=6)
    np.random.seed(6)
    result = fit(sim["X"], model="CopDDE", dimension_method="fixed", G_list=[g.astype(int) for g in G],
                 pen_list=0.0, tau=0.05, random_state=6, verbose=False, it=20, n_init=5)
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
    with pytest.raises(ValueError, match="G_list needs: pen_list, tau"):
        fit_copula_dde(X, G_list=G, verbose=False)
    with pytest.raises(ValueError, match="G_list needs"):
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
    # the free layer above the graph is refitted by the start's exact EM too
    assert "deep_em_best_start" in result.diagnostics["graph_start"]
    assert np.all(result.B_list[0][:, 1:][~G[0]] == 0.0)
    free_slopes = result.B_list[1][:, 1:]
    assert np.all((free_slopes == 0.0) | (np.abs(free_slopes) >= 0.3))


def test_all_ones_graph_gives_the_fit_without_a_graph():
    sim, _ = _simulate(300, seed=9)
    args = dict(dimension_method="fixed", pen_list=[1.0, 1.0], tau=0.3, tol=0.0, it=3,
                random_state=9, verbose=False)
    np.random.seed(9)
    plain = fit_copula_dde(sim["X"], K_list=[6, 2], **args)
    np.random.seed(9)
    ones = fit_copula_dde(sim["X"], G_list=[np.ones((30, 6), int), np.ones((6, 2), int)], **args)
    assert ones.K_list == plain.K_list == [6, 2]
    assert "G_list" not in ones.diagnostics
    for B_ones, B_plain in zip(ones.B_list, plain.B_list):
        np.testing.assert_array_equal(B_ones, B_plain)


def test_fixed_fits_draw_latent_states_by_gibbs_sweeps_by_default():
    """Fixed-width CopDDE fits use systematic Gibbs sweeps with or without a
    graph; the synchronous mean-field draw is still available."""
    sim, _ = _simulate(200, seed=10)
    args = dict(K_list=[6, 2], dimension_method="fixed", pen_list=[1.0, 1.0], tau=0.3, tol=0.0, it=2,
                random_state=10, verbose=False)
    np.random.seed(10)
    default = fit_copula_dde(sim["X"], **args)
    assert default.diagnostics["latent_transition"] == "systematic_coordinate_gibbs"
    assert default.raw["p"] is None
    np.random.seed(10)
    synchronous = fit_copula_dde(sim["X"], latent_transition="synchronous_mean_field", **args)
    assert synchronous.diagnostics["latent_transition"] == "synchronous_mean_field"
    assert synchronous.raw["p"][0].shape == (200, 6)
