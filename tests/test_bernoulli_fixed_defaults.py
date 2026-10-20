"""Fixed-width Bernoulli DDE: anchor latent-class initialization and SA burn-in defaults.

The study behind the defaults is ``Results/bernoulli_fixed_recon/REPORT.md``
(D=2..4 ablation: burn-in 10 + anchor_lca versus burn-in alone versus the
plain kernel).  A layer-wise warm start is not a Bernoulli default: a
flat-prior fit of one binary layer loses the layers above it.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from dde import fit
from dde.algorithms.bernoulli_init import (
    Bernoulli_init,
    _anchor_lca_binarize,
    _two_class_latent_class_model,
)


def _paper_layer(n_parent):
    """Equation (S.36) 3K-by-K block with intercepts (-2, -4, -2)."""
    slopes = np.zeros((3 * n_parent, n_parent))
    slopes[:n_parent] = 4.0 * np.eye(n_parent)
    slopes[n_parent:2 * n_parent] = 4.0 * np.eye(n_parent)
    third = 4.0 * np.eye(n_parent)
    for row in range(n_parent):
        third[row, (row + n_parent // 2) % n_parent] = 4.0 / 3.0
    slopes[2 * n_parent:] = third
    intercept = np.concatenate([np.full(n_parent, -2.0), np.full(n_parent, -4.0), np.full(n_parent, -2.0)])
    return np.column_stack([intercept, slopes])


def _design(K_list, seed, N):
    rng = np.random.default_rng(seed)
    B_list = [_paper_layer(k) for k in K_list]
    A = [None] * len(K_list)
    A[-1] = rng.binomial(1, 0.5, size=(N, K_list[-1])).astype(np.uint8)
    for layer in range(len(K_list) - 2, -1, -1):
        B = B_list[layer + 1]
        A[layer] = rng.binomial(1, expit(B[:, 0] + A[layer + 1] @ B[:, 1:].T)).astype(np.uint8)
    X = rng.binomial(1, expit(B_list[0][:, 0] + A[0] @ B_list[0][:, 1:].T)).astype(np.uint8)
    return X, A, B_list, [B[:, 1:] != 0 for B in B_list]


def test_two_class_latent_class_model_recovers_a_rare_class():
    rng = np.random.default_rng(0)
    N = 8000
    z = rng.random(N) < 0.15
    Y = rng.binomial(1, np.where(z[:, None], 0.85, 0.1), size=(N, 4))
    resp, pi = _two_class_latent_class_model(Y)
    assert abs(pi - 0.15) < 0.03
    assert np.mean((resp > 0.5) == z) > 0.9


def test_anchor_lca_binarize_recovers_rare_column_prevalence():
    rng = np.random.default_rng(1)
    N = 6000
    A = np.column_stack([rng.random(N) < 0.5, rng.random(N) < 0.12]).astype(float)
    # three anchors per latent column, logit -2 + 4 A
    X = rng.binomial(1, expit(-2.0 + 4.0 * A[:, [0, 0, 0, 1, 1, 1]]))
    B_est = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    parent, prevalence = _anchor_lca_binarize(X, B_est)
    assert abs(prevalence[1] - 0.12) < 0.04
    assert np.mean(parent[:, 1] == A[:, 1]) > 0.9
    assert np.mean(parent[:, 0] == A[:, 0]) > 0.85


def test_anchor_lca_initializer_improves_deep_latent_states():
    X, A_true, B_true, G_true = _design((18, 6, 2), seed=3, N=3000)
    _, _, _, A_proj, _ = Bernoulli_init(X, (18, 6, 2), mode="oracle", B_list=B_true, G_list=G_true, binarize="projection")
    _, _, _, A_lca, _ = Bernoulli_init(X, (18, 6, 2), mode="oracle", B_list=B_true, G_list=G_true, binarize="anchor_lca")
    truth = A_true[1].mean(axis=0)
    # the paper design's layer-2 columns are rare (~30 %); projection at zero
    # starts them near 50 %, the latent class model near the truth
    assert np.mean(np.abs(A_lca[1].mean(axis=0) - truth)) < np.mean(np.abs(A_proj[1].mean(axis=0) - truth))
    assert np.mean(A_lca[1] == A_true[1]) > np.mean(A_proj[1] == A_true[1])


def test_bernoulli_fixed_defaults_record_burn_in_and_anchor_lca():
    X, _, _, _ = _design((6, 2), seed=5, N=600)
    np.random.seed(5)
    result = fit(X, method="DDE", family="Bernoulli", D=2, dimension_method="fixed", K_list=[6, 2],
                 pen_list=[2.0, 2.0], tau=0.3, C=1, tol=0.01, max_it=14, verbose=False)
    d = result.diagnostics
    assert d["sa_burn_in"] == 10 and d["fixed_init_binarize"] == "anchor_lca"
    assert d["fixed_warm_start"] == "none" and "warm_start" not in d
    assert d["iterations"] > 10  # the convergence test starts after the burn-in
    assert all(np.all(np.isfinite(B)) for B in result.B_list)


def test_bernoulli_fixed_plain_kernel_remains_available():
    X, _, _, _ = _design((6, 2), seed=6, N=400)
    np.random.seed(6)
    result = fit(X, method="DDE", family="Bernoulli", D=2, dimension_method="fixed", K_list=[6, 2],
                 pen_list=[2.0, 2.0], tau=0.3, C=1, tol=0.01, max_it=4, verbose=False,
                 sa_burn_in=0, fixed_init_binarize="projection")
    d = result.diagnostics
    assert d["sa_burn_in"] == 0 and d["fixed_init_binarize"] == "projection"
    with pytest.raises(ValueError):
        fit(X, method="DDE", family="Bernoulli", D=2, dimension_method="fixed", K_list=[6, 2],
            pen_list=[2.0, 2.0], tau=0.3, C=1, max_it=2, verbose=False, fixed_init_binarize="anchor_mixture")
    # the CSP path never burns in or re-binarizes
    with pytest.raises(ValueError):
        fit(X, method="DDE", family="Bernoulli", D=2, dimension_method="csp", t_spike=[0.03, 0.05], temp=0.9,
            tau=0.3, C=1, max_it=2, verbose=False, sa_burn_in=10)


def test_bernoulli_two_layer_recovery_with_oracle_labels_under_defaults():
    X, A_true, B_true, G_true = _design((6, 2), seed=11, N=2000)
    np.random.seed(11)
    result = fit(X, method="DDE", family="Bernoulli", D=2, dimension_method="fixed", K_list=[6, 2],
                 pen_list=[2000 ** 0.25] * 2, tau=0.3, C=1, tol=0.005, max_it=50,
                 initialize="oracle", B_list_true=B_true, G_list_true=G_true, verbose=False)
    for B_hat, G_t in zip(result.B_list, G_true):
        assert np.mean((np.asarray(B_hat)[:, 1:] != 0) == G_t) >= 0.9
    assert np.mean(np.asarray(result.A_list[0]) == A_true[0]) > 0.85
