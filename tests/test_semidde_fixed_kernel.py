"""Fixed-width SemiDDE: precision-weighted transformation estimate and staged schedule.

The study behind the defaults is ``Results/semidde_fixed_recon/REPORT.md``.
The averaged estimator ``sum_a pi(a) Phi^{-1}(F_hat_a(y))`` needs every
class's CDF over the whole range of ``Y_j``; its clipped tails leave an error
floor that does not shrink with N, and with full first-layer patterns it
degenerates once ``2**K_1`` exceeds N (every class holds about one row).
"""
from __future__ import annotations

import numpy as np
import pytest

import dde.algorithms.semi as semi_module
import dde.algorithms.semi_saem as semi_saem_module
from dde import fit
from dde.algorithms.normal_init_semi import Normal_init_semi, rank_gaussianize_columns
from dde.algorithms.semi_saem import (
    _precision_weighted_g_estimate,
    _semiparametric_g_step,
    compute_marginal_pmf_A1,
    get_SAEM_normal,
    get_SAEM_semiparametricv1,
)
from dde.identifiability import build_true_params
from dde.simulation import generate_X_normal
from dde.utils import g_est_rversion


def _design(K_list, seed, N):
    prop, B_list, G_list, gamma = build_true_params(list(K_list), 3 * K_list[0], 4.0)
    B_list = [np.asarray(B, float) for B in B_list]
    G_list = [np.asarray(G, bool) for G in G_list]
    prop = np.asarray(prop, float)
    gamma = np.asarray(gamma, float)
    np.random.seed(seed)
    X, A_true = generate_X_normal(N, prop, B_list, gamma)
    return X, [np.asarray(A, float) for A in A_true], prop, B_list, G_list, gamma


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_precision_estimate_with_true_labels_beats_the_averaged_estimator():
    X, A_true, prop, B_list, _, gamma = _design((6, 2), 151, 4000)
    Y = np.exp(X)
    A1 = A_true[0]
    precision = _precision_weighted_g_estimate(Y, X, B_list[0], gamma, A1)
    labels = (A1.astype(np.int64) @ (1 << np.arange(5, -1, -1))).astype(int)
    delta_N = 1.0 / (4.0 * 4000 ** 0.25 * np.sqrt(np.pi * np.log(4000)))
    averaged = g_est_rversion(
        Y, labels, 4000, Y.shape[1], 64, compute_marginal_pmf_A1(prop, B_list), delta_N
    )
    assert _rmse(precision, X) < 0.04
    assert _rmse(averaged, X) > 0.1
    np.testing.assert_allclose(precision.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(precision.std(axis=0), 1.0, atol=1e-8)


def test_precision_estimate_is_rank_based_and_monotone_within_classes():
    X, A_true, _, B_list, _, gamma = _design((6, 2), 3, 600)
    Y = np.exp(X)
    first = _precision_weighted_g_estimate(Y, X, B_list[0], gamma, A_true[0])
    # Any strictly increasing map of Y leaves the estimate unchanged.
    second = _precision_weighted_g_estimate(
        np.log1p(Y) ** 3, X, B_list[0], gamma, A_true[0]
    )
    np.testing.assert_allclose(first, second, atol=1e-12)
    for j in range(4):
        parent = np.flatnonzero(B_list[0][j, 1:])[0]
        for value in (0.0, 1.0):
            mask = A_true[0][:, parent] == value
            order = np.argsort(Y[mask, j])
            assert np.all(np.diff(first[mask, j][order]) >= -1e-10)


def test_precision_g_step_requires_gamma_and_validates_estimator():
    X, A_true, prop, B_list, _, gamma = _design((6, 2), 5, 200)
    Y = np.exp(X)
    with pytest.raises(ValueError, match="requires gamma"):
        _semiparametric_g_step(
            Y, X, prop, B_list, A_true[0], delta=1e-3, g_estimator="precision"
        )
    with pytest.raises(ValueError, match="g_estimator"):
        _semiparametric_g_step(
            Y, X, prop, B_list, A_true[0], delta=1e-3, g_estimator="bogus"
        )
    X_next, g_new, diff = _semiparametric_g_step(
        Y, X, prop, B_list, A_true[0], delta=1e-3,
        g_estimator="precision", gamma=gamma, g_update_rho=0.5,
    )
    np.testing.assert_allclose(X_next, 0.5 * X + 0.5 * g_new)
    assert diff > 0.0


def test_default_truncation_is_the_manuscript_delta(monkeypatch):
    X, _, _, B_list, G_list, _ = _design((6, 2), 7, 300)
    Y = np.exp(X)
    X0 = rank_gaussianize_columns(Y)
    prop, B_init, gamma, A_init, _ = Normal_init_semi(X0, 6, [6, 2], B_list, G_list, mode="oracle")
    seen = []

    def fake_g_step(Y, X, prop, B_list, A1, *, delta, **kwargs):
        seen.append(delta)
        return X, X, 0.0

    monkeypatch.setattr(semi_saem_module, "_semiparametric_g_step", fake_g_step)
    get_SAEM_semiparametricv1(
        Y, X0, prop, B_init, gamma, [1.0, 1.0], 0.3, A_init, 0.01, 1,
        max_outer_iter=1, max_inner_iter=1, schedule="nested",
        g_estimator="average", sa_burn_in=0, verbose=False,
    )
    N = 300
    assert seen == [pytest.approx(1.0 / (4.0 * N ** 0.25 * np.sqrt(np.pi * np.log(N))))]


def test_nested_average_schedule_reproduces_the_earlier_outer_iteration():
    X, _, _, B_list, G_list, _ = _design((6, 2), 11, 400)
    Y = np.exp(X)
    X0 = rank_gaussianize_columns(Y)
    prop, B_init, gamma, A_init, _ = Normal_init_semi(X0, 6, [6, 2], B_list, G_list, mode="oracle")
    pen = [400 ** 0.25] * 2

    np.random.seed(0)
    expected = get_SAEM_normal(X0, prop, B_init, gamma, pen, 0.3, A_init, 1, 0.01, max_iter=50, verbose=False)
    X_expected, _, _ = _semiparametric_g_step(
        Y, X0, expected[0], expected[1], expected[3][0], delta=1e-3
    )

    np.random.seed(0)
    X_out, prop_out, B_out, gamma_out, _, _, diagnostics = get_SAEM_semiparametricv1(
        Y, X0, prop, B_init, gamma, pen, 0.3, A_init, 0.01, 1,
        delta=1e-3, max_outer_iter=1, schedule="nested", g_estimator="average",
        sa_burn_in=0, return_diagnostics=True, verbose=False,
    )
    np.testing.assert_allclose(X_out, X_expected)
    np.testing.assert_allclose(prop_out, expected[0])
    for B, B_ref in zip(B_out, expected[1]):
        np.testing.assert_allclose(B, B_ref)
    assert diagnostics["schedule"] == "nested"
    assert diagnostics["final_iterations"] is None


def test_staged_schedule_recovers_the_d2_design():
    X, A_true, _, B_list, G_list, _ = _design((6, 2), 151, 2000)
    result = fit(
        np.exp(X), model="SemiDDE", D=2, K_list=[6, 2], dimension_method="fixed",
        initialize="oracle", B_list_true=B_list, G_list_true=G_list,
        pen_list=[2000 ** 0.25] * 2, tau=0.3, tol=0.01, C=1, verbose=False,
    )
    diagnostics = result.diagnostics
    assert diagnostics["schedule"] == "staged"
    assert diagnostics["g_estimator"] == "precision"
    assert diagnostics["g_converged"]
    assert diagnostics["warm_start"]["stage2_iterations"] > 0
    assert diagnostics["final_iterations"] > diagnostics["sa_burn_in"]
    assert _rmse(result.transformed, X) < 0.06
    for B, G in zip(result.B_list, G_list):
        assert np.array_equal(B[:, 1:] != 0.0, G)
    assert np.mean(result.A_list[0] == A_true[0]) > 0.99


def test_fixed_semidde_initializes_with_anchor_mixture(monkeypatch):
    X, _, _, B_list, G_list, _ = _design((6, 2), 2, 120)
    calls = []
    original = semi_module.Normal_init_semi

    def recording_init(*args, **kwargs):
        calls.append(kwargs.get("binarize"))
        return original(*args, **kwargs)

    monkeypatch.setattr(semi_module, "Normal_init_semi", recording_init)
    common = dict(
        model="SemiDDE", D=2, K_list=[6, 2], dimension_method="fixed",
        pen_list=[1.0, 1.0], tau=0.3, tol=0.01, C=1,
        max_outer_iter=1, max_inner_iter=2, verbose=False,
    )
    fit(np.exp(X), initialize="data", **common)
    fit(np.exp(X), initialize="oracle", B_list_true=B_list, G_list_true=G_list, **common)
    fit(np.exp(X), initialize="data", init_binarize="projection", **common)
    assert calls == ["anchor_mixture", "anchor_mixture", "projection"]


def test_staged_schedule_rejects_unpaired_oracle_lists():
    X, _, _, B_list, G_list, _ = _design((6, 2), 4, 100)
    Y = np.exp(X)
    X0 = rank_gaussianize_columns(Y)
    prop, B_init, gamma, A_init, _ = Normal_init_semi(X0, 6, [6, 2], B_list, G_list, mode="oracle")
    with pytest.raises(ValueError, match="together"):
        get_SAEM_semiparametricv1(
            Y, X0, prop, B_init, gamma, [1.0, 1.0], 0.3, A_init, 0.01, 1,
            B_list_true=B_list, verbose=False,
        )
    with pytest.raises(ValueError, match="schedule"):
        get_SAEM_semiparametricv1(
            Y, X0, prop, B_init, gamma, [1.0, 1.0], 0.3, A_init, 0.01, 1,
            schedule="bogus", verbose=False,
        )
