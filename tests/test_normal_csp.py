import numpy as np
from scipy.special import expit

from dde import fit
from dde.algorithms.normal_csp import (
    _solve_gaussian_weighted_l1,
    _solve_logistic_weighted_l1,
)
from dde.algorithms.semi_saem import _normal_sample_statistics


def _assert_l1_kkt(gradient, coefficients, weights, atol=2e-4):
    assert np.max(np.abs(gradient[:, 0])) <= atol
    slope = coefficients[:, 1:]
    slope_grad = gradient[:, 1:]
    nonzero = np.abs(slope) > 1e-7
    if np.any(nonzero):
        residual = slope_grad[nonzero] + weights[nonzero] * np.sign(slope[nonzero])
        assert np.max(np.abs(residual)) <= atol
    if np.any(~nonzero):
        assert np.max(np.abs(slope_grad[~nonzero]) - weights[~nonzero]) <= atol


def test_batched_gaussian_weighted_l1_solver_satisfies_kkt_conditions():
    rng = np.random.default_rng(14)
    N, J, K, C = 35, 4, 3, 2
    latent = rng.integers(0, 2, size=(N, K, C), dtype=np.uint8)
    sample = np.empty((N, K + 1, C), dtype=np.uint8)
    sample[:, 0, :] = 1
    sample[:, 1:, :] = latent
    X = rng.normal(size=(N, J))
    gamma = rng.uniform(0.7, 1.3, size=J)
    scale = np.full((J, K), 4.0)
    temperature = 0.8

    B, gamma_update = _solve_gaussian_weighted_l1(
        X,
        sample,
        np.zeros((J, K + 1)),
        gamma,
        scale,
        temperature,
        max_iter=4000,
        tol=1e-11,
    )

    Q, q = _normal_sample_statistics(X, sample)
    gradient = (temperature / gamma[:, None]) * (B @ Q - q)
    _assert_l1_kkt(gradient, B, 1.0 / scale)
    assert np.all(np.isfinite(gamma_update))
    assert np.all(gamma_update > 0)


def test_batched_logistic_weighted_l1_solver_satisfies_kkt_conditions():
    rng = np.random.default_rng(91)
    N, n_children, K_parent, C = 45, 5, 3, 2
    parent = np.empty((N, K_parent + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(
        0, 2, size=(N, K_parent, C), dtype=np.uint8
    )
    child = np.empty((N, n_children + 1, C), dtype=np.uint8)
    child[:, 0, :] = 1
    child[:, 1:, :] = rng.integers(
        0, 2, size=(N, n_children, C), dtype=np.uint8
    )
    scale = np.full((n_children, K_parent), 5.0)
    temperature = 0.9

    B = _solve_logistic_weighted_l1(
        child,
        parent,
        np.zeros((n_children, K_parent + 1)),
        scale,
        temperature,
        max_iter=4000,
        tol=1e-11,
    )

    design = np.transpose(parent, (0, 2, 1)).reshape(N * C, K_parent + 1)
    response = np.transpose(child[:, 1:, :], (0, 2, 1)).reshape(
        N * C, n_children
    )
    gradient = (temperature / C) * (
        (expit(design @ B.T) - response).T @ design
    )
    _assert_l1_kkt(gradient, B, 1.0 / scale)


def test_normal_csp_d5_maximal_width_one_iteration_smoke():
    rng = np.random.default_rng(505)
    N, J = 4, 486
    K_list = [162, 54, 18, 6, 2]
    X = rng.normal(size=(N, J))
    B_list = [rng.normal(scale=0.02, size=(J, K_list[0] + 1))]
    B_list.extend(
        rng.normal(scale=0.02, size=(K_list[d - 1], K_list[d] + 1))
        for d in range(1, len(K_list))
    )
    A_list = [
        rng.integers(0, 2, size=(N, width), dtype=np.uint8)
        for width in K_list
    ]

    result = fit(
        X,
        method="DDE",
        D=5,
        family="Normal",
        dimension_method="csp",
        t_spike=[0.02] * 5,
        temp=0.9,
        tau=0.3,
        C=1,
        tol=0.0,
        max_it=1,
        prop_in=np.full(2, 0.5),
        B_list_in=B_list,
        gamma_in=np.ones(J),
        A_list_in=A_list,
        optimizer_maxiter=1,
        random_state=808,
        verbose=False,
    )

    assert result.diagnostics["K_max_list"] == tuple(K_list)
    assert len(result.B_list) == 5
    assert len(result.diagnostics["allocation"]) == 5
    assert "A_sample_long" not in result.raw
