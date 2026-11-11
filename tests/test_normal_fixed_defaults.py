"""Fixed-width Normal DDE: anchor-mixture initialization, SA burn-in and layer-wise warm start.

The study behind the defaults is ``Results/dde_normal_fixed_recon/REPORT.md``
(paper designs D=2..5).  The double-SVD projection rule starts every latent
column near 50 % prevalence; rare columns then collapse and neighbouring
columns absorb their signal through false edges.
"""
from __future__ import annotations

import numpy as np
import pytest

from dde import fit
from dde.algorithms.normal_init_semi import (
    Normal_init_semi,
    _anchor_gaussian_mixture_binarize,
    _two_class_gaussian_mixture,
)
from dde.identifiability import build_true_params
from dde.simulation import generate_X_normal


def _design(K_list, seed, N):
    prop, B_list, G_list, gamma = build_true_params(list(K_list), 3 * K_list[0], 4.0)
    B_list = [np.asarray(B, float) for B in B_list]
    G_list = [np.asarray(G, bool) for G in G_list]
    np.random.seed(seed)
    X, A_true = generate_X_normal(N, np.asarray(prop, float), B_list, np.asarray(gamma, float))
    return X, [np.asarray(A, float) for A in A_true], B_list, G_list


def test_two_class_gaussian_mixture_recovers_a_rare_class():
    rng = np.random.default_rng(0)
    N = 8000
    z = rng.random(N) < 0.06
    Y = rng.normal(np.where(z[:, None], 1.5, 0.0), 0.5, size=(N, 3))
    resp, pi = _two_class_gaussian_mixture(Y)
    assert abs(pi - 0.06) < 0.01
    assert np.mean((resp > 0.5) == z) > 0.99


def test_anchor_gaussian_mixture_binarize_recovers_rare_column_prevalence():
    rng = np.random.default_rng(1)
    N = 6000
    A = np.column_stack([rng.random(N) < 0.5, rng.random(N) < 0.08]).astype(float)
    # two strong anchors per latent column and one weak cross-loading
    X = -0.5 + 1.8 * A[:, [0, 0, 1, 1]] + rng.normal(0.0, 0.5, size=(N, 4))
    B_est = np.array([[0.7, 0.0], [0.7, 0.0], [0.1, 0.7], [0.0, 0.7]])
    parent, prevalence = _anchor_gaussian_mixture_binarize(X, B_est)
    assert abs(prevalence[1] - 0.08) < 0.01
    assert np.mean(parent[:, 1] == A[:, 1]) > 0.99
    assert np.mean(parent[:, 0] == A[:, 0]) > 0.99


def test_anchor_mixture_initializer_starts_rare_columns_near_true_prevalence():
    X, A_true, B_true, G_true = _design((18, 6, 2), seed=3, N=4000)
    kwargs = dict(B_list=B_true, G_list=G_true, mode="oracle")
    _, _, _, A_proj, _ = Normal_init_semi(X, 18, (18, 6, 2), binarize="projection", **kwargs)
    _, _, _, A_mix, _ = Normal_init_semi(X, 18, (18, 6, 2), binarize="anchor_mixture", **kwargs)
    truth = A_true[0].mean(axis=0)
    rare = truth < 0.2
    assert rare.any()
    # projection at zero overstates every rare column's prevalence
    assert np.all(A_proj[0].mean(axis=0)[rare] - truth[rare] > 0.05)
    assert np.all(np.abs(A_mix[0].mean(axis=0)[rare] - truth[rare]) < 0.01)
    # deeper layers use the anchor latent class model on the layer-1 states
    for layer in range(3):
        assert np.mean(A_mix[layer] == A_true[layer]) > np.mean(A_proj[layer] == A_true[layer])


def test_normal_init_rejects_unknown_binarize_rule():
    X, _, _, _ = _design((6, 2), seed=4, N=300)
    with pytest.raises(ValueError, match="binarize"):
        Normal_init_semi(X, 6, (6, 2), mode="data", binarize="anchor_lca")


def test_normal_fixed_defaults_record_burn_in_anchor_mixture_and_warm_start():
    X, _, _, _ = _design((6, 2), seed=5, N=600)
    np.random.seed(5)
    result = fit(X, model="DDE", family="Normal", D=2, dimension_method="fixed", K_list=[6, 2],
                 pen_list=[2.0, 2.0], tau=0.3, C=1, tol=0.01, max_it=14, verbose=False)
    d = result.diagnostics
    assert d["sa_burn_in"] == 10 and d["fixed_init_binarize"] == "anchor_mixture"
    assert d["fixed_warm_start"] == "layerwise"
    assert d["warm_start"]["stage1_iterations"] > 10 and d["warm_start"]["stage2_iterations"] > 10
    assert d["iterations"] > 10  # every stage burns in; the convergence test starts after it
    assert all(np.all(np.isfinite(B)) for B in result.B_list)
    assert result.gamma.shape == (18,) and np.all(result.gamma > 0)


def test_normal_one_layer_and_provided_state_skip_the_warm_start():
    X, A_true, B_true, _ = _design((6, 2), seed=6, N=500)
    one = fit(X, model="DDE", family="Normal", D=1, dimension_method="fixed", K_list=[6],
              pen_list=[2.0], tau=0.3, C=1, tol=0.01, max_it=12, verbose=False)
    assert one.diagnostics["fixed_warm_start"] == "none" and "warm_start" not in one.diagnostics
    provided = fit(X, model="DDE", family="Normal", D=2, dimension_method="fixed", K_list=[6, 2],
                   pen_list=[2.0, 2.0], tau=0.3, C=1, tol=0.01, max_it=12, verbose=False,
                   prop_in=np.full(2, 0.5), B_list_in=B_true, gamma_in=np.full(18, 0.25), A_list_in=A_true)
    assert provided.diagnostics["fixed_warm_start"] == "none" and "warm_start" not in provided.diagnostics


def test_normal_fixed_plain_kernel_and_csp_path_remain_available():
    X, _, _, _ = _design((6, 2), seed=6, N=400)
    np.random.seed(6)
    plain = fit(X, model="DDE", family="Normal", D=2, dimension_method="fixed", K_list=[6, 2],
                pen_list=[2.0, 2.0], tau=0.3, C=1, tol=0.01, max_it=4, verbose=False,
                sa_burn_in=0, fixed_init_binarize="projection", fixed_warm_start="none")
    d = plain.diagnostics
    assert d["sa_burn_in"] == 0 and d["fixed_init_binarize"] == "projection"
    assert d["fixed_warm_start"] == "none" and "warm_start" not in d
    with pytest.raises(ValueError):
        fit(X, model="DDE", family="Normal", D=2, dimension_method="fixed", K_list=[6, 2],
            pen_list=[2.0, 2.0], tau=0.3, C=1, max_it=2, verbose=False, fixed_init_binarize="anchor_lca")
    # the CSP path never burns in or re-binarizes
    with pytest.raises(ValueError):
        fit(X, model="DDE", family="Normal", D=2, dimension_method="csp", t_spike=[0.03, 0.05], temp=0.9,
            tau=0.3, C=1, max_it=2, verbose=False, sa_burn_in=10)
    with pytest.raises(ValueError):
        fit(X, model="DDE", family="Normal", D=2, dimension_method="csp", t_spike=[0.03, 0.05], temp=0.9,
            tau=0.3, C=1, max_it=2, verbose=False, fixed_init_binarize="anchor_mixture")


def test_normal_two_layer_recovery_with_oracle_labels_under_defaults():
    X, A_true, B_true, G_true = _design((6, 2), seed=11, N=4000)
    np.random.seed(11)
    result = fit(X, model="DDE", family="Normal", D=2, dimension_method="fixed", K_list=[6, 2],
                 pen_list=[4000 ** 0.25] * 2, tau=0.3, C=1,
                 initialize="oracle", B_list_true=B_true, G_list_true=G_true, verbose=False)
    for B_hat, G_t in zip(result.B_list, G_true):
        assert np.mean((np.asarray(B_hat)[:, 1:] != 0) == G_t) == 1.0
    assert np.sqrt(np.mean((np.asarray(result.B_list[0]) - B_true[0]) ** 2)) < 0.02
    assert np.sqrt(np.mean((np.asarray(result.B_list[1]) - B_true[1]) ** 2)) < 0.5
    assert np.mean(np.asarray(result.A_list[0]) == A_true[0]) > 0.99
