import numpy as np
from scipy.special import expit

import semidde.algorithms.get_SAEM_RL_CSP_D as copula_csp_D


def test_copula_d_gaussian_objective_value_and_gradient():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(24, 4))
    y = rng.normal(size=24)
    x = rng.normal(size=4)
    weights = np.array([0.0, 0.2, 0.4, 0.6])
    scale = 0.73

    value, grad = copula_csp_D._gaussian_objective_value_grad_scaled(
        x, X, y, weights, scale
    )

    resid = X @ x - y
    expected_value = 0.5 * scale * np.dot(resid, resid) + np.sum(weights[1:] * np.abs(x[1:]))
    expected_grad = scale * (X.T @ resid)
    expected_grad[1:] += weights[1:] * np.sign(x[1:])

    assert np.allclose(value, expected_value)
    assert np.allclose(grad, expected_grad)


def test_copula_d_logistic_objective_value_and_gradient():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(18, 5))
    y = rng.integers(0, 2, size=18).astype(float)
    x = rng.normal(size=5)
    weights = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    temp = 0.9
    sample_count = 3

    value, grad = copula_csp_D._logistic_objective_value_grad(
        x, X, y, weights, temp, sample_count=sample_count
    )

    eta = X @ x
    scale = temp / sample_count
    expected_value = scale * np.sum(np.logaddexp(0.0, eta) - y * eta)
    expected_value += np.sum(weights[1:] * np.abs(x[1:]))
    expected_grad = scale * (X.T @ (expit(eta) - y))
    expected_grad[1:] += weights[1:] * np.sign(x[1:])

    assert np.allclose(value, expected_value)
    assert np.allclose(grad, expected_grad)
