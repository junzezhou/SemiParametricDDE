"""Tests for the Poisson observed layer of the copula CSP kernel."""

import numpy as np
import pytest
from scipy.special import expit

from dde import fit
from dde.algorithms.normal_csp import fit_parametric_csp_copula


def _simulate(seed, N=300, J=18, K_true=(6, 2)):
    rng = np.random.default_rng(seed)
    B1 = np.zeros((J, K_true[0]))
    for k in range(K_true[0]):
        B1[k, k] = 1.5
        B1[K_true[0] + k, k] = 1.5
        B1[2 * K_true[0] + k, k] = 1.5
    B2 = np.zeros((K_true[0], K_true[1]))
    B2[:3, 0] = 4.0
    B2[3:, 1] = 4.0
    A2 = rng.binomial(1, 0.5, (N, K_true[1]))
    A1 = rng.binomial(1, expit(-2.0 + A2 @ B2.T))
    Y = rng.poisson(np.exp(0.0 + A1 @ B1.T))
    return Y.astype(float)


def test_fit_routes_poisson_to_copula_backend():
    Y = _simulate(3)
    res = fit(
        Y, method="DDE", family="Poisson", D=2, dimension_method="csp",
        t_spike=(0.02, 0.04), temp=0.7, tau=0.4, C=1, max_iter=4, verbose=False,
        csp_backend="copula", random_state=2,
    )
    assert res.diagnostics["backend"] == "copula"
    assert len(res.B_list) == 2
    assert all(np.all(np.isfinite(B)) for B in res.B_list)


def test_poisson_kernel_requires_fista_observed_layer_and_no_gamma():
    Y = _simulate(4, N=80)
    rng = np.random.default_rng(1)
    widths = (6, 2)
    B_list = [rng.normal(scale=0.2, size=(Y.shape[1], widths[0] + 1)),
              rng.normal(scale=0.2, size=(widths[0], widths[1] + 1))]
    A_list = [rng.integers(0, 2, size=(Y.shape[0], w)).astype(float) for w in widths]
    prop = np.full(widths[-1], 0.5)
    with pytest.raises(ValueError, match="FISTA"):
        fit_parametric_csp_copula(Y, prop, None, B_list, A_list, family="poisson", C=1, it=2,
                                  t_spike=0.03, temp=0.9, tau=0.3, mstep_solver="lbfgsb", verbose=False)
    with pytest.raises(ValueError, match="gamma_in"):
        fit_parametric_csp_copula(Y, prop, np.ones(Y.shape[1]), B_list, A_list, family="poisson", C=1, it=2,
                                  t_spike=0.03, temp=0.9, tau=0.3, verbose=False)
    out = fit_parametric_csp_copula(Y, prop, None, B_list, A_list, family="poisson", C=1, it=2,
                                    t_spike=0.03, temp=0.9, tau=0.3, verbose=False)
    assert out["B_list"][0].shape == B_list[0].shape
    assert out["gamma"] is None
