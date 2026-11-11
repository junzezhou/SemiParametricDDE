import numpy as np
import pytest
from scipy.linalg import svd
from scipy.special import gammaln

import dde.algorithms.parametric_families as parametric_families
import dde.algorithms.semi_saem as semi_saem_module
from dde import fit
from dde.algorithms.csp_solvers import (
    _poisson_mc_design_and_response_statistic,
    _poisson_mc_value_and_grad,
    _solve_poisson_weighted_l1,
)
from dde.algorithms.parametric_families import (
    POISSON_FAMILY,
    _ObservedPoissonAccumulator,
    _solve_bernoulli_tlp_rowwise,
    _solve_poisson_tlp_rowwise,
    resolve_parametric_family,
)
from dde.algorithms.poisson_init import (
    Poisson_init,
    _poisson_working_response,
)
from dde.algorithms.semi_saem import (
    _latent_coordinate_probability_parametric,
    _observed_coordinate_loglik_delta,
    _sample_latent_layers_parametric,
    _sparse_poisson_coordinate_probability,
    get_SAEM_poisson,
)
from dde.simulation import generate_X_poisson


def _central_difference(function, point, step=1e-6):
    point = np.asarray(point, dtype=float)
    gradient = np.empty_like(point)
    for index in range(point.size):
        direction = np.zeros_like(point)
        direction[index] = step
        gradient[index] = (
            function(point + direction) - function(point - direction)
        ) / (2.0 * step)
    return gradient


def _matlab_poisson_parameters():
    """Return the identifiable setting from MATLAB ``Sim_sid_Poisson.m``."""
    K2 = 2
    K1 = 3 * K2
    J = 3 * K1
    max_value = 4.0

    B2_slopes = np.zeros((K1, K2), dtype=float)
    half_K2 = K2 // 2
    for k in range(K2):
        if k + half_K2 < K2:
            B2_slopes[k + 2 * K2, k] = max_value
            B2_slopes[k + 2 * K2 + half_K2, k] = -max_value / 3.0
            B2_slopes[k + 2 * K2, k + half_K2] = -max_value / 3.0
            B2_slopes[k + 2 * K2 + half_K2, k + half_K2] = max_value
        B2_slopes[[k, k + K2], k] = max_value
    B2 = np.column_stack(
        [
            np.concatenate(
                [
                    np.full(K2, -max_value / 2.0),
                    np.full(K2, -max_value),
                    np.full(K2, -max_value / 2.0),
                ]
            ),
            B2_slopes,
        ]
    )

    B1_slopes = np.zeros((J, K1), dtype=float)
    half_K1 = K1 // 2
    for k in range(K1):
        if k + half_K1 < K1:
            B1_slopes[k + 2 * K1, k] = 4.0
            B1_slopes[k + 2 * K1 + half_K1, k] = -4.0 / 3.0
            B1_slopes[k + 2 * K1, k + half_K1] = -4.0 / 3.0
            B1_slopes[k + 2 * K1 + half_K1, k + half_K1] = 4.0
        B1_slopes[[k, k + K1], k] = 4.0
    B1 = np.column_stack(
        [
            np.concatenate(
                [
                    np.full(K1, -2.0),
                    np.full(K1, -4.0),
                    np.full(K1, -2.0),
                ]
            ),
            B1_slopes,
        ]
    )
    prop = np.full(K2, 0.5)
    return prop, [B1, B2]


def _small_poisson_state(seed=20260903):
    rng = np.random.default_rng(seed)
    N, J = 36, 9
    K_list = [3, 1]
    prop = np.array([0.5])
    B0 = rng.normal(0.0, 0.18, size=(J, K_list[0] + 1))
    B0[:, 0] = -0.3
    B1 = rng.normal(0.0, 0.25, size=(K_list[0], K_list[1] + 1))
    B1[:, 0] = -0.2
    B_list = [B0, B1]
    X, A_list = generate_X_poisson(N, prop, B_list, random_state=seed + 1)
    return X, prop, B_list, A_list


def test_poisson_family_registry_validation_and_dispersion_semantics():
    assert resolve_parametric_family("Poisson") is POISSON_FAMILY
    assert POISSON_FAMILY.supported_dimension_methods == ("fixed", "csp")
    assert POISSON_FAMILY.has_dispersion is False
    assert POISSON_FAMILY.validate_dispersion(None, 4) is None

    valid = np.array([[0, 1], [2, 7]])
    np.testing.assert_array_equal(POISSON_FAMILY.validate_y(valid), valid)
    for invalid in (
        np.array([[0.0, 0.5]]),
        np.array([[0.0, np.nan]]),
        np.array([[0, -1]]),
    ):
        with pytest.raises(ValueError, match="nonnegative integer counts"):
            POISSON_FAMILY.validate_y(invalid)
    with pytest.raises(ValueError, match="no dispersion"):
        POISSON_FAMILY.validate_dispersion(np.ones(2), 2)


def test_poisson_working_response_matches_matlab_full_svd_formula():
    rng = np.random.default_rng(41)
    X = rng.poisson(2.5, size=(28, 12)).astype(float)
    K1 = 3
    epsilon = 1e-4
    U, singular_values, Vt = svd(X, full_matrices=False)
    rank = min(
        max(K1 + 1, int(np.sum(singular_values > 1.01 * np.sqrt(X.shape[0])))),
        singular_values.size,
    )
    reconstructed = (U[:, :rank] * singular_values[:rank][None, :]) @ Vt[:rank]
    expected = np.log(np.maximum(reconstructed, epsilon))
    expected -= np.mean(expected, axis=0, keepdims=True)

    actual = _poisson_working_response(X, K1, epsilon)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        POISSON_FAMILY.inverse_mean_link([0.0, 1.0, 3.0], epsilon=epsilon),
        np.log([epsilon, 1.0, 3.0]),
    )


def test_poisson_initialization_supports_arbitrary_depth():
    rng = np.random.default_rng(42)
    K_list = [9, 3, 1]
    B_list = [
        rng.normal(0.0, 0.08, size=(27, 10)),
        rng.normal(0.0, 0.18, size=(9, 4)),
        rng.normal(0.0, 0.18, size=(3, 2)),
    ]
    B_list[0][:, 0] = -0.2
    X, _ = generate_X_poisson(48, np.array([0.5]), B_list, random_state=43)

    prop, B_initial, dispersion, A_initial, graph_errors = Poisson_init(
        X, K_list, mode="data"
    )

    assert dispersion is None
    assert prop.shape == (1,)
    assert [B.shape for B in B_initial] == [(27, 10), (9, 4), (3, 2)]
    assert [A.shape for A in A_initial] == [(48, 9), (48, 3), (48, 1)]
    assert all(np.all(np.isfinite(B)) for B in B_initial)
    assert all(set(np.unique(A)).issubset({0, 1}) for A in A_initial)
    assert graph_errors == [None, None, None]


def test_poisson_closed_coordinate_delta_matches_log_probability_difference():
    rng = np.random.default_rng(44)
    X = rng.poisson(2.0, size=(17, 9)).astype(float)
    eta_zero = rng.normal(scale=0.7, size=X.shape)
    slope = rng.normal(scale=0.5, size=X.shape[1])
    expected = np.sum(
        POISSON_FAMILY.conditional_log_prob(X, eta_zero + slope[None, :], None)
        - POISSON_FAMILY.conditional_log_prob(X, eta_zero, None),
        axis=1,
    )

    direct = POISSON_FAMILY.observed_coordinate_loglik_delta(
        X, eta_zero, slope, None
    )
    chunked = _observed_coordinate_loglik_delta(
        X,
        eta_zero,
        slope,
        None,
        family=POISSON_FAMILY,
        row_chunk_size=4,
    )
    np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chunked, direct, rtol=0.0, atol=0.0)


def test_poisson_synchronous_observed_logits_match_coordinate_differences():
    rng = np.random.default_rng(440)
    N, J, K = 11, 9, 3
    X = rng.poisson(1.7, size=(N, J)).astype(float)
    B = rng.normal(scale=0.35, size=(J, K + 1))
    A = rng.integers(0, 2, size=(N, K), dtype=np.uint8)
    actual = POISSON_FAMILY.observed_conditional_logit(
        X,
        B,
        None,
        A,
        width=K,
        row_chunk_size=4,
        coordinate_chunk_size=2,
    )

    expected = np.empty_like(actual)
    for coordinate in range(K):
        A_zero = A.copy()
        A_zero[:, coordinate] = 0
        eta_zero = B[:, 0][None, :] + A_zero @ B[:, 1:].T
        expected[:, coordinate] = np.sum(
            POISSON_FAMILY.conditional_log_prob(
                X, eta_zero + B[:, coordinate + 1][None, :], None
            )
            - POISSON_FAMILY.conditional_log_prob(X, eta_zero, None),
            axis=1,
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_poisson_fixed_accumulator_matches_matlab_objective_and_gradient():
    rng = np.random.default_rng(45)
    N, J, K, C = 13, 5, 3, 2
    X = rng.poisson(2.0, size=(N, J)).astype(float)
    parent = np.empty((N, K + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(0, 2, size=(N, K, C), dtype=np.uint8)
    accumulator = _ObservedPoissonAccumulator(J, K + 1)
    accumulator.update(X, parent, 1)

    design = np.moveaxis(parent, 1, 2).reshape(N * C, K + 1).astype(float)
    response = np.repeat(X, C, axis=0)
    point = rng.normal(scale=0.2, size=K + 1)
    for response_index in range(J):
        eta = design @ point
        expected = (
            np.sum(np.exp(eta)) - response[:, response_index] @ eta
        ) / C
        expected_gradient = (
            design.T @ (np.exp(eta) - response[:, response_index]) / C
        )
        value, gradient = accumulator.value_and_grad(response_index, point)
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=1e-12, atol=1e-12
        )
        numerical = _central_difference(
            lambda candidate: accumulator.value(response_index, candidate),
            point,
        )
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-6)

    parent_second = np.empty_like(parent)
    parent_second[:, 0, :] = 1
    parent_second[:, 1:, :] = rng.integers(
        0, 2, size=(N, K, C), dtype=np.uint8
    )
    second_design = (
        np.moveaxis(parent_second, 1, 2).reshape(N * C, K + 1).astype(float)
    )
    accumulator.update(X, parent_second, 2)
    eta_first = design @ point
    eta_second = second_design @ point
    expected_average = 0.5 * (
        (np.sum(np.exp(eta_first)) - response[:, 0] @ eta_first) / C
        + (np.sum(np.exp(eta_second)) - response[:, 0] @ eta_second) / C
    )
    assert accumulator.value(0, point) == pytest.approx(
        expected_average, rel=1e-12, abs=1e-12
    )


def test_poisson_csp_mc_objective_and_gradient_match_direct_calculation():
    rng = np.random.default_rng(901)
    N, J, K, C = 7, 4, 3, 2
    X = rng.poisson(1.7, size=(N, J)).astype(float)
    parent_sample = np.empty((N, K + 1, C), dtype=np.uint8)
    parent_sample[:, 0, :] = 1
    parent_sample[:, 1:, :] = rng.binomial(1, 0.45, size=(N, K, C))
    coefficients = rng.normal(0.0, 0.2, size=(J, K + 1))
    temperature = 0.8
    design, response_statistic, *_ = _poisson_mc_design_and_response_statistic(
        X, parent_sample
    )

    value, gradient = _poisson_mc_value_and_grad(
        coefficients,
        design,
        response_statistic,
        temperature=temperature,
        mc_samples=C,
        row_chunk_size=3,
        coordinate_chunk_size=2,
    )
    eta = design @ coefficients.T
    expected_value = (
        temperature
        / C
        * (np.sum(np.exp(eta)) - np.sum(response_statistic * coefficients))
    )
    expected_gradient = (
        temperature / C * (np.exp(eta).T @ design - response_statistic)
    )
    np.testing.assert_allclose(value, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=1e-12, atol=1e-12
    )

    numerical = _central_difference(
        lambda flat: _poisson_mc_value_and_grad(
            flat.reshape(coefficients.shape),
            design,
            response_statistic,
            temperature=temperature,
            mc_samples=C,
            row_chunk_size=design.shape[0],
            coordinate_chunk_size=J,
        )[0],
        coefficients.ravel(),
    ).reshape(coefficients.shape)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-6)


def test_poisson_csp_weighted_l1_solver_reduces_objective_and_satisfies_kkt():
    rng = np.random.default_rng(902)
    N, J, K, C = 80, 4, 2, 2
    X = rng.poisson(1.4, size=(N, J)).astype(float)
    parent_sample = np.empty((N, K + 1, C), dtype=np.uint8)
    parent_sample[:, 0, :] = 1
    parent_sample[:, 1:, :] = rng.binomial(1, 0.5, size=(N, K, C))
    start = np.zeros((J, K + 1), dtype=float)
    penalty_scale = np.full((J, K), 0.8)
    temperature = 0.9
    design, response_statistic, *_ = _poisson_mc_design_and_response_statistic(
        X, parent_sample
    )

    def objective(coefficients):
        smooth, _ = _poisson_mc_value_and_grad(
            coefficients,
            design,
            response_statistic,
            temperature=temperature,
            mc_samples=C,
            row_chunk_size=31,
            coordinate_chunk_size=2,
        )
        return smooth + np.sum(np.abs(coefficients[:, 1:]) / penalty_scale)

    fitted = _solve_poisson_weighted_l1(
        X,
        parent_sample,
        start,
        penalty_scale,
        temperature,
        max_iter=1000,
        tol=1e-10,
        bounds=[(-5.0, 5.0), (-2.5, 5.0), (-2.5, 5.0)],
        row_chunk_size=31,
        coordinate_chunk_size=2,
    )
    assert objective(fitted) < objective(start)
    _, gradient = _poisson_mc_value_and_grad(
        fitted,
        design,
        response_statistic,
        temperature=temperature,
        mc_samples=C,
        row_chunk_size=31,
        coordinate_chunk_size=2,
    )
    assert np.max(np.abs(gradient[:, 0])) < 3e-4
    weights = 1.0 / penalty_scale
    nonzero = np.abs(fitted[:, 1:]) > 1e-7
    residual = gradient[:, 1:] + np.sign(fitted[:, 1:]) * weights
    assert np.max(np.abs(residual[nonzero]), initial=0.0) < 3e-4
    assert (
        np.max(
            np.abs(gradient[:, 1:][~nonzero]) - weights[~nonzero],
            initial=-np.inf,
        )
        < 3e-4
    )


def test_poisson_accumulator_reuses_rate_in_value_and_gradient(monkeypatch):
    X, _, _, A_list = _small_poisson_state(451)
    samples = np.empty((X.shape[0], A_list[0].shape[1] + 1, 1), dtype=np.uint8)
    samples[:, 0, 0] = 1
    samples[:, 1:, 0] = A_list[0]
    accumulator = _ObservedPoissonAccumulator(X.shape[1], samples.shape[1])
    accumulator.update(X, samples, 1)

    original = parametric_families._poisson_optimization_rate
    calls = 0

    def counted_rate(eta):
        nonlocal calls
        calls += 1
        return original(eta)

    monkeypatch.setattr(
        parametric_families, "_poisson_optimization_rate", counted_rate
    )
    value, gradient = accumulator.value_and_grad(0, np.zeros(samples.shape[1]))

    assert calls == 1
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))


def test_poisson_accumulator_bit_packs_wide_parent_patterns_exactly():
    rng = np.random.default_rng(4521)
    N, J, K = 83, 5, 162
    X = rng.poisson(1.5, size=(N, J)).astype(float)
    samples = np.empty((N, K + 1, 1), dtype=np.uint8)
    samples[:, 0, 0] = 1
    samples[:, 1:, 0] = rng.integers(0, 2, size=(N, K), dtype=np.uint8)
    accumulator = _ObservedPoissonAccumulator(J, K + 1)
    accumulator.update(X, samples, 1)

    expected = np.unique(samples[:, 1:, 0], axis=0)
    np.testing.assert_array_equal(
        np.unique(accumulator.patterns, axis=0), expected
    )
    assert accumulator.pattern_count == expected.shape[0]
    assert accumulator._packed_patterns.shape[1] == (K + 7) // 8
    assert accumulator._packed_patterns.nbytes < expected.nbytes


def test_poisson_fixed_mstep_preserves_responsewise_optimizer(monkeypatch):
    rng = np.random.default_rng(453)
    N, J, K = 31, 7, 4
    X = rng.poisson(1.8, size=(N, J)).astype(float)
    samples = np.empty((N, K + 1, 1), dtype=np.uint8)
    samples[:, 0, 0] = 1
    samples[:, 1:, 0] = rng.integers(0, 2, size=(N, K), dtype=np.uint8)
    accumulator = _ObservedPoissonAccumulator(J, K + 1)
    B_start = rng.normal(scale=0.08, size=(J, K + 1))
    expected = B_start + 0.125
    calls = []

    def responsewise(accumulator_arg, B_arg, penalty, tau, bounds, **kwargs):
        calls.append((accumulator_arg, B_arg, penalty, tau, bounds, kwargs))
        return expected

    monkeypatch.setattr(
        parametric_families, "_solve_poisson_tlp_rowwise", responsewise
    )
    actual, dispersion = POISSON_FAMILY.fixed_mstep(
        X,
        samples,
        B_start,
        None,
        accumulator,
        iteration=1,
        depth=2,
        penalty=1.2,
        tau=0.25,
        bound_val=10.0,
        response_cache=None,
        observed_solver="rowwise",
    )

    np.testing.assert_array_equal(actual, expected)
    assert dispersion is None
    assert len(calls) == 1
    assert calls[0][0] is accumulator
    assert calls[0][2:4] == (1.2, 0.25)


def test_poisson_batched_latent_mstep_matches_rowwise_recovery(monkeypatch):
    X, prop, B_list, A_list = _small_poisson_state(454)
    common = dict(
        X=X,
        prop_in=prop,
        B_list_in=B_list,
        pen_list=[0.01, 0.01],
        tau=0.1,
        A_list_in=A_list,
        C=2,
        tol=0.0,
        max_iter=4,
        verbose=False,
    )

    np.random.seed(455)
    optimized = get_SAEM_poisson(**common)

    monkeypatch.setattr(
        semi_saem_module,
        "_solve_bernoulli_tlp_batched",
        _solve_bernoulli_tlp_rowwise,
    )
    np.random.seed(455)
    rowwise = get_SAEM_poisson(**common)

    np.testing.assert_allclose(optimized[0], rowwise[0], rtol=0.0, atol=0.02)
    difference = np.concatenate(
        [
            (left - right).ravel()
            for left, right in zip(optimized[1], rowwise[1])
        ]
    )
    assert np.sqrt(np.mean(difference**2)) <= 0.2
    support_agreement = np.concatenate(
        [
            ((left[:, 1:] != 0.0) == (right[:, 1:] != 0.0)).ravel()
            for left, right in zip(optimized[1], rowwise[1])
        ]
    )
    assert np.mean(support_agreement) >= 0.9
    latent_agreement = np.concatenate(
        [
            (left == right).ravel()
            for left, right in zip(optimized[3], rowwise[3])
        ]
    )
    assert np.mean(latent_agreement) >= 0.9
    assert optimized[4] == rowwise[4]


def test_sparse_poisson_observed_coordinate_matches_dense_probability():
    prop, B_list = _matlab_poisson_parameters()
    X, A_list = generate_X_poisson(45, prop, B_list, random_state=456)
    child_eta = B_list[0][:, 0][None, :] + A_list[0] @ B_list[0][:, 1:].T
    for coordinate in range(B_list[0].shape[1] - 1):
        active = np.flatnonzero(B_list[0][:, coordinate + 1] != 0.0)
        dense_probability, dense_eta_zero = (
            _latent_coordinate_probability_parametric(
                X,
                prop,
                B_list,
                None,
                A_list,
                0,
                coordinate,
                family=POISSON_FAMILY,
                child_eta=child_eta,
                sparse_bernoulli=False,
                row_chunk_size=7,
            )
        )
        sparse_probability, sparse_eta_zero = (
            _sparse_poisson_coordinate_probability(
                X,
                prop,
                B_list,
                A_list,
                coordinate,
                child_eta,
                active,
                row_chunk_size=7,
            )
        )
        np.testing.assert_allclose(
            sparse_probability, dense_probability, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            sparse_eta_zero,
            dense_eta_zero[:, active],
            rtol=0.0,
            atol=0.0,
        )


def test_sparse_poisson_full_gibbs_sweep_matches_dense_sweep():
    prop, B_list = _matlab_poisson_parameters()
    X, A_list = generate_X_poisson(55, prop, B_list, random_state=457)
    common = dict(
        X=X,
        prop=prop,
        B_list=B_list,
        dispersion=None,
        A_list=A_list,
        C=2,
        family=POISSON_FAMILY,
        row_chunk_size=9,
    )

    np.random.seed(458)
    dense_samples, dense_top = _sample_latent_layers_parametric(
        **common, sparse_bernoulli=False
    )
    np.random.seed(458)
    sparse_samples, sparse_top = _sample_latent_layers_parametric(
        **common, sparse_bernoulli=True
    )

    for sparse, dense in zip(sparse_samples, dense_samples):
        np.testing.assert_array_equal(sparse, dense)
    np.testing.assert_allclose(sparse_top, dense_top, rtol=1e-12, atol=1e-12)


def test_public_fixed_poisson_matches_shared_kernel_with_provided_state():
    X, prop0, B0, A0 = _small_poisson_state()
    common = dict(
        X=X,
        prop_in=prop0,
        B_list_in=B0,
        pen_list=[0.0, 0.0],
        tau=0.3,
        A_list_in=A0,
        C=1,
        tol=0.0,
        max_iter=2,
        verbose=False,
    )
    np.random.seed(46)
    expected = get_SAEM_poisson(**common)
    np.random.seed(46)
    actual = fit(
        X,
        model="DDE",
        family="Poisson",
        D=2,
        dimension_method="fixed",
        K_list=[3, 1],
        prop_in=prop0,
        B_list_in=B0,
        A_list_in=A0,
        pen_list=[0.0, 0.0],
        tau=0.3,
        C=1,
        tol=0.0,
        max_it=2,
        verbose=False,
    )

    np.testing.assert_allclose(actual.prop, expected[0])
    for fitted, direct in zip(actual.B_list, expected[1]):
        np.testing.assert_allclose(fitted, direct)
    assert actual.gamma is expected[2] is None
    for fitted, direct in zip(actual.A_list, expected[3]):
        np.testing.assert_array_equal(fitted, direct)
    assert actual.diagnostics["family"] == "Poisson"
    assert actual.diagnostics["iterations"] == expected[4]


def test_matlab_poisson_setting_runs_through_fixed_family_kernel():
    prop, B_list = _matlab_poisson_parameters()
    G_list = [(B[:, 1:] != 0).astype(int) for B in B_list]
    N = 500
    X, _ = generate_X_poisson(N, prop, B_list, random_state=47)
    penalty = float(N ** (2.0 / 8.0))
    tau = float(3.0 * N ** (-0.3))

    np.random.seed(48)
    result = fit(
        X,
        model="DDE",
        family="Poisson",
        D=2,
        dimension_method="fixed",
        K_list=[6, 2],
        pen_list=[penalty, penalty],
        tau=tau,
        C=1,
        # MATLAB uses K2/2 with an unnormalised Frobenius metric. The shared
        # Python kernel uses postprocessed parameter RMSE instead.
        tol=0.005,
        max_it=50,
        initialize="oracle",
        B_list_true=B_list,
        G_list_true=G_list,
        verbose=False,
    )

    assert result.K_list == [6, 2]
    assert result.gamma is None
    assert [B.shape for B in result.B_list] == [(18, 7), (6, 3)]
    assert [A.shape for A in result.A_list] == [(N, 6), (N, 2)]
    assert np.all(np.isfinite(result.prop))
    assert np.all((result.prop >= 0.0) & (result.prop <= 1.0))
    for B in result.B_list:
        assert np.all(np.isfinite(B))
        assert np.all(B[:, 0] >= -5.0)
        assert np.all(B[:, 0] <= 5.0)
        assert np.all(B[:, 1:] >= -2.5)
        assert np.all(B[:, 1:] <= 5.0)
    recovered_support = sum(
        np.sum((B[:, 1:] != 0) == G) for B, G in zip(result.B_list, G_list)
    ) / sum(G.size for G in G_list)
    squared_error = np.sum((result.prop - prop) ** 2) + sum(
        np.sum((fitted - truth) ** 2)
        for fitted, truth in zip(result.B_list, B_list)
    )
    parameter_count = prop.size + sum(B.size for B in B_list)
    assert recovered_support >= 0.9
    assert np.sqrt(squared_error / parameter_count) < 0.6


def test_public_fixed_poisson_supports_three_latent_layers():
    rng = np.random.default_rng(53)
    K_list = [9, 3, 1]
    prop = np.array([0.5])
    B_list = [
        rng.normal(0.0, 0.08, size=(27, 10)),
        rng.normal(0.0, 0.15, size=(9, 4)),
        rng.normal(0.0, 0.15, size=(3, 2)),
    ]
    B_list[0][:, 0] = -0.3
    X, A_list = generate_X_poisson(30, prop, B_list, random_state=54)

    np.random.seed(55)
    result = fit(
        X,
        model="DDE",
        family="Poisson",
        D=3,
        dimension_method="fixed",
        K_list=K_list,
        prop_in=prop,
        B_list_in=B_list,
        A_list_in=A_list,
        pen_list=[0.0, 0.0, 0.0],
        tau=0.2,
        C=1,
        tol=0.0,
        max_it=2,
        verbose=False,
    )

    assert result.K_list == K_list
    assert result.gamma is None
    assert [B.shape for B in result.B_list] == [(27, 10), (9, 4), (3, 2)]
    assert [A.shape for A in result.A_list] == [(30, 9), (30, 3), (30, 1)]


@pytest.mark.parametrize("K_list", ([27, 9, 3, 1], [81, 27, 9, 3, 1]))
def test_public_fixed_poisson_smoke_tests_depth_four_and_five(K_list):
    rng = np.random.default_rng(560 + len(K_list))
    prop = np.full(K_list[-1], 0.5)
    child_widths = [3 * K_list[0], *K_list[:-1]]
    B_list = [
        rng.normal(0.0, 0.02, size=(child_width, parent_width + 1))
        for child_width, parent_width in zip(child_widths, K_list)
    ]
    B_list[0][:, 0] = -0.3
    X, A_list = generate_X_poisson(
        8, prop, B_list, random_state=570 + len(K_list)
    )

    np.random.seed(580 + len(K_list))
    result = fit(
        X,
        model="DDE",
        family="Poisson",
        D=len(K_list),
        dimension_method="fixed",
        K_list=K_list,
        prop_in=prop,
        B_list_in=B_list,
        A_list_in=A_list,
        pen_list=[0.0] * len(K_list),
        tau=0.2,
        C=1,
        tol=0.0,
        max_it=2,
        verbose=False,
    )

    assert result.K_list == K_list
    assert result.gamma is None
    assert [B.shape for B in result.B_list] == [
        (child_width, parent_width + 1)
        for child_width, parent_width in zip(child_widths, K_list)
    ]
    assert [A.shape for A in result.A_list] == [(8, width) for width in K_list]
    assert np.all(np.isfinite(result.prop))
    assert all(np.all(np.isfinite(B)) for B in result.B_list)


def test_poisson_fixed_lightweight_trace_is_output_equivalent_and_scalar_only():
    X, prop, B_list, A_list = _small_poisson_state(530)
    common = dict(
        model="DDE",
        family="Poisson",
        D=2,
        dimension_method="fixed",
        K_list=[3, 1],
        prop_in=prop,
        B_list_in=B_list,
        A_list_in=A_list,
        pen_list=[0.0, 0.0],
        tau=0.2,
        C=1,
        tol=0.0,
        max_it=3,
        verbose=False,
    )

    np.random.seed(531)
    untraced = fit(X, keep_trace=False, **common)
    np.random.seed(531)
    traced = fit(X, keep_trace=True, **common)

    np.testing.assert_allclose(traced.prop, untraced.prop, rtol=0.0, atol=0.0)
    for actual, expected in zip(traced.B_list, untraced.B_list):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    for actual, expected in zip(traced.A_list, untraced.A_list):
        np.testing.assert_array_equal(actual, expected)

    trace = traced.diagnostics["trace"]
    assert len(trace) == traced.diagnostics["iterations"] - 1
    for iteration, entry in enumerate(trace, start=1):
        assert entry["iteration"] == iteration
        assert np.isfinite(entry["error"])
        assert np.isfinite(entry["prop_update_rmse"])
        assert len(entry["layer_update_rmse"]) == 2
        assert len(entry["m_step_seconds"]) == 2
        assert entry["e_step_seconds"] >= 0.0
        assert entry["iteration_seconds"] >= 0.0
        summaries = [
            entry["accumulators"]["observed"],
            *entry["accumulators"]["latent"],
        ]
        for summary in summaries:
            assert isinstance(summary["backend"], str)
            assert isinstance(summary["pattern_count"], int)
            assert isinstance(summary["storage_bytes"], int)
            assert summary["pattern_count"] >= 0
            assert summary["storage_bytes"] >= 0


def test_poisson_conditional_scoring_and_d_layer_simulation():
    Y = np.array([[0.0, 1.0, 4.0]])
    eta = np.array([[-1.0, 0.2, 1.1]])
    expected = Y * eta - np.exp(eta) - gammaln(Y + 1.0)
    np.testing.assert_allclose(
        POISSON_FAMILY.conditional_log_prob(Y, eta, None), expected
    )
    first = POISSON_FAMILY.sample_observed(
        np.tile(eta, (20, 1)), None, np.random.default_rng(49)
    )
    second = POISSON_FAMILY.sample_observed(
        np.tile(eta, (20, 1)), None, np.random.default_rng(49)
    )
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0)

    rng = np.random.default_rng(50)
    B_list = [
        rng.normal(0.0, 0.1, size=(27, 10)),
        rng.normal(0.0, 0.1, size=(9, 4)),
        rng.normal(0.0, 0.1, size=(3, 2)),
    ]
    B_list[0][:, 0] = -0.3
    X_first, A_first = generate_X_poisson(
        25, np.array([0.5]), B_list, random_state=51
    )
    X_second, A_second = generate_X_poisson(
        25, np.array([0.5]), B_list, random_state=51
    )
    np.testing.assert_array_equal(X_first, X_second)
    assert [A.shape for A in A_first] == [(25, 9), (25, 3), (25, 1)]
    for left, right in zip(A_first, A_second):
        np.testing.assert_array_equal(left, right)
    assert np.all(X_first == np.floor(X_first))


def test_public_poisson_csp_uses_maximum_width_and_is_reproducible():
    X, _, _, _ = _small_poisson_state(52)
    arguments = dict(
        model="DDE",
        family="Poisson",
        D=2,
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.1,
        C=2,
        tol=0.0,
        max_it=2,
        optimizer_maxiter=20,
        random_state=53,
        verbose=False,
    )
    first = fit(X, **arguments)
    second = fit(X, **arguments)

    assert first.gamma is None
    assert first.diagnostics["K_max_list"] == (3, 1)
    assert first.diagnostics["original_K_list"] == (3, 1)
    assert first.diagnostics["family"] == "Poisson"
    assert first.diagnostics["backend"] == "copula"
    assert first.diagnostics["kernel"] == "copdde_csp"
    assert first.diagnostics["latent_transition"] == "sequential_mean_field"
    assert first.diagnostics["mc_state_update"] == "draw_majority"
    assert len(first.K_list) == 2
    assert all(
        fitted <= maximum for fitted, maximum in zip(first.K_list, (3, 1))
    )
    np.testing.assert_array_equal(first.prop, second.prop)
    for first_B, second_B in zip(first.B_list, second.B_list):
        np.testing.assert_array_equal(first_B, second_B)
    for first_A, second_A in zip(first.A_list, second.A_list):
        np.testing.assert_array_equal(first_A, second_A)
    assert first.K_list == second.K_list


def test_public_poisson_csp_supports_arbitrary_depth():
    rng = np.random.default_rng(55)
    B_list = [
        rng.normal(0.0, 0.08, size=(27, 10)),
        rng.normal(0.0, 0.15, size=(9, 4)),
        rng.normal(0.0, 0.15, size=(3, 2)),
    ]
    B_list[0][:, 0] = -0.3
    X, _ = generate_X_poisson(32, np.array([0.5]), B_list, random_state=56)

    result = fit(
        X,
        model="DDE",
        family="Poisson",
        D=3,
        dimension_method="csp",
        t_spike=[0.01, 0.03, 0.005],
        temp=0.9,
        tau=0.1,
        C=1,
        tol=0.0,
        max_it=1,
        optimizer_maxiter=3,
        random_state=57,
        verbose=False,
    )

    assert result.diagnostics["K_max_list"] == (9, 3, 1)
    assert len(result.K_list) == 3
    assert all(
        fitted <= maximum for fitted, maximum in zip(result.K_list, (9, 3, 1))
    )
    assert [B.shape[0] for B in result.B_list] == [
        27,
        result.K_list[0],
        result.K_list[1],
    ]
    assert all(np.all(np.isfinite(B)) for B in result.B_list)


def test_poisson_csp_recovers_matlab_dimensions_from_maximum_width():
    data_seed = 40_460_917
    fit_seed = 540_460_917
    prop_true, B_true = _matlab_poisson_parameters()
    B_true[0] = np.row_stack([B_true[0], np.zeros((9, 7), dtype=float)])
    X, A_true = generate_X_poisson(
        200, prop_true, B_true, random_state=data_seed
    )

    B_start = [np.zeros((27, 10), dtype=float), np.zeros((9, 4), dtype=float)]
    B_start[0][:, 0] = B_true[0][:, 0]
    B_start[0][:, 1:7] = B_true[0][:, 1:]
    B_start[1][:6, 0] = B_true[1][:, 0]
    B_start[1][:6, 1:3] = B_true[1][:, 1:]
    rng = np.random.default_rng(data_seed + 101)
    A_start = [
        rng.integers(0, 2, size=(200, 9), dtype=np.uint8),
        rng.integers(0, 2, size=(200, 3), dtype=np.uint8),
    ]
    A_start[0][:, :6] = A_true[0]
    A_start[1][:, :2] = A_true[1]
    prop_start = np.full(3, 0.5)

    result = fit(
        X,
        model="DDE",
        family="Poisson",
        D=2,
        dimension_method="csp",
        prop_in=prop_start,
        B_list_in=B_start,
        A_list_in=A_start,
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        random_state=fit_seed,
        verbose=False,
    )

    true_dimensions = np.array([6, 2])
    fitted_dimensions = np.asarray(result.K_list)
    assert result.diagnostics["K_max_list"] == (9, 3)
    assert int(np.array_equal(fitted_dimensions, true_dimensions)) == 1
    assert int(np.sum(np.abs(fitted_dimensions - true_dimensions))) == 0


def test_poisson_csp_data_initialization_recovers_matlab_dimensions():
    data_seed = 40_760_918
    fit_seed = 540_760_918
    prop_true, B_true = _matlab_poisson_parameters()
    B_true[0] = np.row_stack([B_true[0], np.zeros((9, 7), dtype=float)])
    X, _ = generate_X_poisson(500, prop_true, B_true, random_state=data_seed)

    result = fit(
        X,
        model="DDE",
        family="Poisson",
        D=2,
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        max_it=50,
        optimizer_maxiter=100,
        random_state=fit_seed,
        verbose=False,
    )

    assert result.diagnostics["K_max_list"] == (9, 3)
    assert result.K_list == [6, 2]
    assert all(np.all(np.isfinite(B)) for B in result.B_list)


def test_public_poisson_rejects_gamma():
    X, _, _, _ = _small_poisson_state(54)

    with pytest.raises(ValueError, match="no dispersion"):
        fit(
            X,
            model="DDE",
            family="Poisson",
            D=2,
            dimension_method="fixed",
            K_list=[3, 1],
            pen_list=[0.0, 0.0],
            tau=0.3,
            C=1,
            max_it=1,
            gamma_in=np.ones(X.shape[1]),
            verbose=False,
        )
