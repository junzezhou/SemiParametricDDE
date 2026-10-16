"""Tests for the Bernoulli observed layer of the copula CSP kernel."""

import numpy as np
import pytest
from scipy.special import expit

from dde import fit
from dde.algorithms.normal_csp import fit_bernoulli_csp, fit_parametric_csp_copula


def _simulate(seed, N=400, J=27, K_true=(6, 2)):
    rng = np.random.default_rng(seed)
    B1 = np.zeros((J, K_true[0]))
    for k in range(K_true[0]):
        B1[k, k] = 4.0
        B1[K_true[0] + k, k] = 4.0
        B1[2 * K_true[0] + k, k] = 4.0
    B2 = np.zeros((K_true[0], K_true[1]))
    B2[:3, 0] = 4.0
    B2[3:, 1] = 4.0
    A2 = rng.binomial(1, 0.5, (N, K_true[1]))
    A1 = rng.binomial(1, expit(-2.0 + A2 @ B2.T))
    X = rng.binomial(1, expit(-2.0 + A1 @ B1.T))
    return X.astype(float), (A1, A2)


def _inputs(seed=11, N=50, J=18, widths=(6, 2)):
    rng = np.random.default_rng(seed)
    return {
        "X": rng.integers(0, 2, size=(N, J)).astype(float),
        "prop": np.full(widths[-1], 0.5),
        "B_list": [
            rng.normal(scale=0.3, size=(J, widths[0] + 1)),
            rng.normal(scale=0.3, size=(widths[0], widths[1] + 1)),
        ],
        "A_list": [rng.integers(0, 2, size=(N, w)).astype(float) for w in widths],
    }


def _copy(values):
    return [np.asarray(v).copy() for v in values]


@pytest.mark.parametrize("solver", ["lbfgsb", "fista_lower", "fista"])
def test_bernoulli_kernel_runs_with_each_solver(solver):
    inputs = _inputs()
    raw = fit_bernoulli_csp(
        inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        C=1, it=3, t_spike=[0.03, 0.05], temp=0.9, tau=0.3, verbose=False,
        mstep_solver=solver, random_state=2,
    )
    assert raw["family"] == "Bernoulli"
    assert raw["gamma"] is None
    assert raw["iterations"] == 3
    assert all(np.all(np.isfinite(B)) for B in raw["B_list"])
    assert raw["B_list"][0].shape == (18, 7)
    # Bernoulli bounds: intercept within [-5, 5], slopes within [-2.5, 5].
    assert np.all(np.abs(raw["B_list"][0][:, 0]) <= 5.0 + 1e-9)
    assert np.all(raw["B_list"][0][:, 1:] >= -2.5 - 1e-9)


def test_bernoulli_kernel_rejects_gamma_and_normal_only_options():
    inputs = _inputs()
    base = dict(C=1, it=1, t_spike=0.03, temp=0.9, tau=0.3, verbose=False)
    with pytest.raises(ValueError, match="no dispersion"):
        fit_parametric_csp_copula(
            inputs["X"], inputs["prop"], np.ones(18), _copy(inputs["B_list"]),
            _copy(inputs["A_list"]), family="bernoulli", **base,
        )
    with pytest.raises(ValueError, match="Normal family only"):
        fit_bernoulli_csp(
            inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
            observed_noise_scale=0.5, **base,
        )
    with pytest.raises(NotImplementedError):
        fit_parametric_csp_copula(
            inputs["X"], inputs["prop"], None, _copy(inputs["B_list"]),
            _copy(inputs["A_list"]), family="poisson", **base,
        )


def test_bernoulli_gibbs_transition_runs():
    inputs = _inputs()
    raw = fit_bernoulli_csp(
        inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        C=2, it=2, t_spike=0.03, temp=0.9, tau=0.3, verbose=False,
        latent_transition="systematic_coordinate_gibbs", random_state=3,
    )
    assert raw["latent_transition"] == "systematic_coordinate_gibbs"
    assert all(A.shape[0] == 50 for A in raw["A_list"])


def test_fit_routes_bernoulli_to_copula_backend_explicitly():
    X, _ = _simulate(5)
    result = fit(
        X, method="DDE", family="Bernoulli", D=2, dimension_method="csp",
        t_spike=[0.03, 0.05], temp=0.9, tau=0.3, C=1, max_iter=30,
        random_state=7, verbose=False, csp_backend="copula",
    )
    d = result.diagnostics
    assert d["backend"] == "copula"
    assert d["family"] == "Bernoulli"
    assert d["standardized"] is False
    assert d["initializer"] == "family"
    assert d["mstep_solver"] == "fista_lower"
    assert d["latent_transition"] == "systematic_coordinate_gibbs"
    assert d["penalty_scaling_power"] == (0.0, 0.0)
    assert result.gamma is None
    assert tuple(d["K_max_list"]) == (9, 3)
    K1, K2 = result.K_list
    assert 1 <= K1 <= 9 and 1 <= K2 <= 3
    assert result.B_list[0].shape == (27, K1 + 1)
    default = fit(
        X, method="DDE", family="Bernoulli", D=2, dimension_method="csp",
        t_spike=[0.03, 0.05], temp=0.9, tau=0.3, C=1, max_iter=1,
        random_state=7, verbose=False,
    )
    assert default.diagnostics["backend"] == "mcem"


def test_bernoulli_copula_backend_rejects_standardize_and_poisson():
    X, _ = _simulate(6, N=120)
    with pytest.raises(ValueError, match="standardize"):
        fit(
            X, method="DDE", family="Bernoulli", D=2, dimension_method="csp",
            t_spike=0.03, temp=0.9, tau=0.3, C=1, max_iter=1, verbose=False,
            csp_backend="copula", standardize=True,
        )
    rng = np.random.default_rng(0)
    Y = rng.poisson(2.0, size=(60, 18)).astype(float)
    with pytest.raises(NotImplementedError):
        fit(
            Y, method="DDE", family="Poisson", D=2, dimension_method="csp",
            t_spike=0.03, temp=0.9, tau=0.3, C=1, max_iter=1, verbose=False,
            csp_backend="copula",
        )
