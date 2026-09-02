import itertools

import numpy as np
import pytest
from scipy.linalg import svd
from scipy.special import expit

import dde.algorithms.parametric_families as family_module
import dde.algorithms.semi_saem as semi_saem_module
from dde import fit
from dde.algorithms.bernoulli_init import (
    Bernoulli_init,
    _bernoulli_working_response,
)
from dde.algorithms.normal_csp import (
    _logistic_mc_design_and_response_statistic,
    _logistic_mc_value_and_grad,
    _solve_logistic_weighted_l1,
)
from dde.algorithms.parametric_families import (
    BERNOULLI_FAMILY,
    _binary_child_conditional_logit,
    _solve_bernoulli_tlp_batched,
    _solve_bernoulli_tlp_rowwise,
    _tlp_penalty,
    resolve_parametric_family,
)
from dde.algorithms.semi_saem import (
    _compute_bernoulli_probabilities_parametric,
    _latent_coordinate_probability_parametric,
    _make_bernoulli_layer_accumulator,
    _observed_coordinate_loglik_delta,
    _postprocess_and_measure_fixed_update,
    _sample_latent_layers_parametric,
    get_SAEM_bernoulli,
)
from dde.simulation import generate_X_bernoulli


def _bernoulli_state(seed=910):
    rng = np.random.default_rng(seed)
    N, J = 24, 9
    widths = (3, 1)
    prop = np.array([0.43])
    B_list = [
        rng.normal(scale=0.45, size=(J, widths[0] + 1)),
        rng.normal(scale=0.45, size=(widths[0], widths[1] + 1)),
    ]
    np.random.seed(seed + 1)
    X, A1, A2 = generate_X_bernoulli(
        N, prop, B_list[0], B_list[1]
    )
    return X, prop, B_list, [A1.astype(np.uint8), A2.astype(np.uint8)]


def _assert_nested_equal(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
        return
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def _assert_result_equal(left, right):
    np.testing.assert_array_equal(left.prop, right.prop)
    for left_B, right_B in zip(left.B_list, right.B_list):
        np.testing.assert_array_equal(left_B, right_B)
    for left_A, right_A in zip(left.A_list, right.A_list):
        np.testing.assert_array_equal(left_A, right_A)
    assert left.gamma is right.gamma is None
    _assert_nested_equal(left.diagnostics, right.diagnostics)


def _central_difference(fun, point, step=1e-6):
    gradient = np.empty_like(point, dtype=float)
    for index in range(point.size):
        delta = np.zeros_like(point, dtype=float)
        delta[index] = step
        gradient[index] = (
            fun(point + delta) - fun(point - delta)
        ) / (2.0 * step)
    return gradient


class _PathRng:
    """Force one selected branch at each coordinate of a Gibbs sweep."""

    def __init__(self, target):
        self._target = iter(target)

    def random(self, size):
        value = next(self._target)
        return np.zeros(size) if value else np.ones(size)


@pytest.mark.parametrize("depth", (2, 3))
def test_systematic_fixed_gibbs_transition_preserves_exact_posterior(depth):
    X = np.ones((1, 1), dtype=float)
    prop = np.array([0.45])
    B_list = [np.array([[-1.2, 2.4]])] + [
        np.array([[-1.0, 2.0]]) for _ in range(depth - 1)
    ]
    states = np.asarray(
        list(itertools.product((0, 1), repeat=depth)), dtype=np.uint8
    )
    state_index = {tuple(state): index for index, state in enumerate(states)}

    posterior = np.empty(states.shape[0], dtype=float)
    for index, state in enumerate(states):
        probability = prop[0] if state[-1] else 1.0 - prop[0]
        for layer in range(depth - 1, 0, -1):
            success = expit(
                B_list[layer][0, 0]
                + B_list[layer][0, 1] * state[layer]
            )
            probability *= success if state[layer - 1] else 1.0 - success
        observed_success = expit(
            B_list[0][0, 0] + B_list[0][0, 1] * state[0]
        )
        posterior[index] = probability * observed_success
    posterior /= np.sum(posterior)

    transition = np.zeros((states.shape[0], states.shape[0]), dtype=float)
    for source_index, source in enumerate(states):
        for target in states:
            working = [
                np.array([[value]], dtype=np.uint8) for value in source
            ]
            path_probability = 1.0
            for layer, value in enumerate(target):
                conditional, _ = _latent_coordinate_probability_parametric(
                    X,
                    prop,
                    B_list,
                    None,
                    working,
                    layer,
                    0,
                    family=BERNOULLI_FAMILY,
                )
                path_probability *= (
                    conditional[0] if value else 1.0 - conditional[0]
                )
                working[layer][0, 0] = value

            sampled, _ = _sample_latent_layers_parametric(
                X,
                prop,
                B_list,
                None,
                [np.array([[value]], dtype=np.uint8) for value in source],
                1,
                family=BERNOULLI_FAMILY,
                rng=_PathRng(target),
            )
            sampled_state = tuple(int(layer[0, 1, 0]) for layer in sampled)
            assert sampled_state == tuple(target)
            transition[source_index, state_index[tuple(target)]] = path_probability

    np.testing.assert_allclose(np.sum(transition, axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(
        posterior @ transition, posterior, rtol=0.0, atol=1e-14
    )


def test_fixed_convergence_error_uses_postprocessed_parameter_scale():
    prop = np.array([0.4])
    prop_update = np.array([0.5])
    B_list = [np.zeros((1, 2)), np.zeros((1, 2))]
    B_updates = [np.array([[1.0, 0.05]]), np.zeros((1, 2))]

    processed, error = _postprocess_and_measure_fixed_update(
        prop,
        prop_update,
        B_list,
        B_updates,
        None,
        None,
        family=BERNOULLI_FAMILY,
        tau=0.1,
        depth=2,
    )

    assert processed[0][0, 1] == 0.0
    assert error == pytest.approx(np.sqrt((0.1**2 + 1.0**2) / 5.0))


def test_bernoulli_family_registry_validation_and_dispersion_semantics():
    assert resolve_parametric_family("Bernoulli") is BERNOULLI_FAMILY
    assert resolve_parametric_family("binary") is BERNOULLI_FAMILY
    assert BERNOULLI_FAMILY.has_dispersion is False
    assert BERNOULLI_FAMILY.validate_dispersion(None, 4) is None

    valid = np.array([[0, 1], [1, 0]])
    np.testing.assert_array_equal(BERNOULLI_FAMILY.validate_y(valid), valid)
    for invalid in (
        np.array([[0.0, 0.5]]),
        np.array([[0.0, np.nan]]),
        np.array([[0, 2]]),
    ):
        with pytest.raises(ValueError, match="binary 0/1"):
            BERNOULLI_FAMILY.validate_y(invalid)
    with pytest.raises(ValueError, match="no dispersion"):
        BERNOULLI_FAMILY.validate_dispersion(np.ones(2), 2)


def test_bernoulli_double_svd_initialization_supports_deep_models():
    rng = np.random.default_rng(913)
    X = rng.integers(0, 2, size=(42, 27), dtype=np.uint8)

    prop, B_list, dispersion, A_list, graph_errors = Bernoulli_init(
        X, [9, 3, 1], epsilon=1e-4, mode="data"
    )

    assert dispersion is None
    assert prop.shape == (1,)
    assert [B.shape for B in B_list] == [(27, 10), (9, 4), (3, 2)]
    assert [A.shape for A in A_list] == [(42, 9), (42, 3), (42, 1)]
    assert all(np.all(np.isfinite(B)) for B in B_list)
    assert all(set(np.unique(A)).issubset({0, 1}) for A in A_list)
    assert graph_errors == [None, None, None]

    working = BERNOULLI_FAMILY.initial_working_response(
        X, epsilon=1e-4, K1=9
    )
    assert working.shape == X.shape
    np.testing.assert_allclose(np.mean(working, axis=0), 0.0, atol=1e-12)


def test_bernoulli_working_response_matches_matlab_full_svd_formula():
    rng = np.random.default_rng(929)
    X = rng.integers(0, 2, size=(25, 12), dtype=np.uint8)
    K1 = 3
    epsilon = 1e-4
    U, singular_values, Vt = svd(X.astype(float), full_matrices=False)
    rank = min(
        max(K1 + 1, int(np.sum(singular_values > 1.01 * np.sqrt(X.shape[0])))),
        singular_values.size,
    )
    reconstructed = (
        U[:, :rank] * singular_values[:rank][None, :]
    ) @ Vt[:rank]
    reconstructed = np.clip(reconstructed, epsilon, 1.0 - epsilon)
    expected = np.log(reconstructed / (1.0 - reconstructed))
    expected -= np.mean(expected, axis=0, keepdims=True)

    actual = _bernoulli_working_response(X, K1, epsilon)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_bernoulli_initialization_and_estep_handle_degenerate_extremes():
    X = np.zeros((30, 9), dtype=np.uint8)
    prop, B_list, dispersion, A_list, _ = Bernoulli_init(X, [3, 1])
    assert dispersion is None
    assert np.all(np.isfinite(prop))
    assert all(np.all(np.isfinite(B)) for B in B_list)
    assert all(np.all(A == 0) for A in A_list)

    extreme_B = [np.full((9, 4), 100.0), np.full((3, 2), -100.0)]
    probabilities = _compute_bernoulli_probabilities_parametric(
        X,
        np.array([0.5]),
        extreme_B,
        None,
        A_list,
        family=BERNOULLI_FAMILY,
    )
    assert all(np.all(np.isfinite(layer)) for layer in probabilities)
    assert all(np.all((layer >= 0.0) & (layer <= 1.0)) for layer in probabilities)


def test_bernoulli_oracle_initialization_returns_alignment_diagnostics():
    X, _, B_reference, _ = _bernoulli_state(931)
    G_reference = [(B[:, 1:] != 0).astype(int) for B in B_reference]
    prop, B_list, dispersion, A_list, graph_errors = Bernoulli_init(
        X,
        [3, 1],
        mode="oracle",
        B_list=B_reference,
        G_list=G_reference,
    )

    assert prop.shape == (1,)
    assert dispersion is None
    assert [B.shape for B in B_list] == [(9, 4), (3, 2)]
    assert [A.shape for A in A_list] == [(24, 3), (24, 1)]
    assert all(error is not None and 0.0 <= error <= 1.0 for error in graph_errors)


def test_bernoulli_estep_matches_matlab_coordinate_formula():
    X, prop, B_list, A_list = _bernoulli_state(914)
    temperature = 0.87
    actual = _compute_bernoulli_probabilities_parametric(
        X,
        prop,
        B_list,
        None,
        A_list,
        family=BERNOULLI_FAMILY,
        row_chunk_size=5,
        coordinate_chunk_size=2,
        temperature=temperature,
    )

    expected_first = np.empty_like(actual[0])
    prior_first = (
        B_list[1][:, 0][None, :]
        + A_list[1] @ B_list[1][:, 1:].T
    )
    for coordinate in range(A_list[0].shape[1]):
        parent_one = A_list[0].copy()
        parent_zero = A_list[0].copy()
        parent_one[:, coordinate] = 1
        parent_zero[:, coordinate] = 0
        eta_one = (
            B_list[0][:, 0][None, :]
            + parent_one @ B_list[0][:, 1:].T
        )
        eta_zero = (
            B_list[0][:, 0][None, :]
            + parent_zero @ B_list[0][:, 1:].T
        )
        likelihood = X @ B_list[0][:, coordinate + 1]
        likelihood -= np.sum(
            np.logaddexp(0.0, eta_one)
            - np.logaddexp(0.0, eta_zero),
            axis=1,
        )
        expected_first[:, coordinate] = expit(
            temperature * (prior_first[:, coordinate] + likelihood)
        )

    expected_top = np.empty_like(actual[1])
    clipped_prop = np.clip(prop, 1e-15, 1.0 - 1e-15)
    for coordinate in range(A_list[1].shape[1]):
        parent_one = A_list[1].copy()
        parent_zero = A_list[1].copy()
        parent_one[:, coordinate] = 1
        parent_zero[:, coordinate] = 0
        eta_one = B_list[1][:, 0][None, :] + parent_one @ B_list[1][:, 1:].T
        eta_zero = B_list[1][:, 0][None, :] + parent_zero @ B_list[1][:, 1:].T
        likelihood = A_list[0] @ B_list[1][:, coordinate + 1]
        likelihood -= np.sum(
            np.logaddexp(0.0, eta_one)
            - np.logaddexp(0.0, eta_zero),
            axis=1,
        )
        prior = np.log(clipped_prop[coordinate] / (1.0 - clipped_prop[coordinate]))
        expected_top[:, coordinate] = expit(
            temperature * (prior + likelihood)
        )

    np.testing.assert_allclose(actual[0], expected_first, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(actual[1], expected_top, rtol=0.0, atol=2e-16)
    unit_temperature = _compute_bernoulli_probabilities_parametric(
        X,
        prop,
        B_list,
        None,
        A_list,
        family=BERNOULLI_FAMILY,
        row_chunk_size=5,
        coordinate_chunk_size=2,
    )
    for layer, probabilities in enumerate(unit_temperature):
        for coordinate in range(probabilities.shape[1]):
            sequential_probability, _ = (
                _latent_coordinate_probability_parametric(
                    X,
                    prop,
                    B_list,
                    None,
                    A_list,
                    layer,
                    coordinate,
                    family=BERNOULLI_FAMILY,
                    row_chunk_size=5,
                )
            )
            np.testing.assert_allclose(
                sequential_probability,
                probabilities[:, coordinate],
                rtol=1e-13,
                atol=1e-13,
            )


@pytest.mark.parametrize("zero_fraction", (0.0, 0.75, 1.0))
def test_bernoulli_csp_sparse_synchronous_logit_matches_dense(
    zero_fraction,
):
    rng = np.random.default_rng(946)
    N, n_children, n_parents = 31, 17, 5
    children = rng.integers(0, 2, size=(N, n_children), dtype=np.uint8)
    parents = rng.integers(0, 2, size=(N, n_parents), dtype=np.uint8)
    coefficients = rng.normal(scale=0.6, size=(n_children, n_parents + 1))
    mask = rng.random(coefficients[:, 1:].shape) < zero_fraction
    coefficients[:, 1:][mask] = 0.0

    dense = _binary_child_conditional_logit(
        children,
        coefficients,
        parents,
        parent_width=n_parents,
        row_chunk_size=None,
        coordinate_chunk_size=n_parents,
        sparse=False,
    )
    sparse = _binary_child_conditional_logit(
        children,
        coefficients,
        parents,
        parent_width=n_parents,
        row_chunk_size=7,
        coordinate_chunk_size=2,
        sparse=True,
    )

    np.testing.assert_allclose(sparse, dense, rtol=1e-13, atol=1e-13)


def test_bernoulli_csp_sparse_synchronous_logit_is_stable_at_extremes():
    children = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    parents = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    coefficients = np.array(
        [
            [1000.0, -1000.0, 0.0],
            [-1000.0, 1000.0, -1000.0],
            [0.0, 0.0, 1000.0],
        ]
    )
    result = _binary_child_conditional_logit(
        children,
        coefficients,
        parents,
        parent_width=2,
        row_chunk_size=1,
        coordinate_chunk_size=1,
        sparse=True,
    )
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("zero_fraction", (0.0, 0.75, 1.0))
def test_sparse_bernoulli_coordinate_probability_matches_dense_reference(
    zero_fraction,
):
    X, prop, B_list, A_list = _bernoulli_state(940)
    rng = np.random.default_rng(941)
    B_sparse = [coefficient.copy() for coefficient in B_list]
    for coefficient in B_sparse:
        mask = rng.random(coefficient[:, 1:].shape) < zero_fraction
        coefficient[:, 1:][mask] = 0.0

    for layer, state in enumerate(A_list):
        child_eta = (
            B_sparse[layer][:, 0][None, :]
            + state @ B_sparse[layer][:, 1:].T
        )
        for coordinate in range(state.shape[1]):
            sparse_probability, sparse_eta_zero = (
                _latent_coordinate_probability_parametric(
                    X,
                    prop,
                    B_sparse,
                    None,
                    A_list,
                    layer,
                    coordinate,
                    family=BERNOULLI_FAMILY,
                    child_eta=child_eta,
                    sparse_bernoulli=True,
                )
            )
            dense_probability, dense_eta_zero = (
                _latent_coordinate_probability_parametric(
                    X,
                    prop,
                    B_sparse,
                    None,
                    A_list,
                    layer,
                    coordinate,
                    family=BERNOULLI_FAMILY,
                    child_eta=child_eta,
                    sparse_bernoulli=False,
                )
            )
            np.testing.assert_allclose(
                sparse_probability,
                dense_probability,
                rtol=0.0,
                atol=2e-16,
            )
            np.testing.assert_array_equal(sparse_eta_zero, dense_eta_zero)


def test_sparse_bernoulli_gibbs_matches_dense_fixed_seed_sweep():
    X, prop, B_list, A_list = _bernoulli_state(942)
    rng = np.random.default_rng(943)
    B_sparse = [coefficient.copy() for coefficient in B_list]
    for coefficient in B_sparse:
        coefficient[:, 1:][
            rng.random(coefficient[:, 1:].shape) < 0.75
        ] = 0.0

    sparse_samples, sparse_top_probability = (
        _sample_latent_layers_parametric(
            X,
            prop,
            B_sparse,
            None,
            A_list,
            3,
            family=BERNOULLI_FAMILY,
            rng=np.random.default_rng(944),
            sparse_bernoulli=True,
        )
    )
    dense_samples, dense_top_probability = (
        _sample_latent_layers_parametric(
            X,
            prop,
            B_sparse,
            None,
            A_list,
            3,
            family=BERNOULLI_FAMILY,
            rng=np.random.default_rng(944),
            sparse_bernoulli=False,
        )
    )
    for sparse_sample, dense_sample in zip(sparse_samples, dense_samples):
        np.testing.assert_array_equal(sparse_sample, dense_sample)
    np.testing.assert_allclose(
        sparse_top_probability,
        dense_top_probability,
        rtol=0.0,
        atol=2e-16,
    )


def test_sparse_bernoulli_full_kernel_matches_dense_reference(monkeypatch):
    X, prop, B_list, A_list = _bernoulli_state(945)
    B_sparse = [coefficient.copy() for coefficient in B_list]
    for coefficient in B_sparse:
        coefficient[:, 1:][np.abs(coefficient[:, 1:]) < 0.4] = 0.0
    B_sparse[0][:, 1] = 0.0
    common = dict(
        X=X,
        prop_in=prop,
        B_list_in=B_sparse,
        pen_list=[0.01, 0.01],
        tau=0.1,
        A_list_in=A_list,
        C=2,
        tol=0.0,
        max_iter=4,
        verbose=False,
    )

    np.random.seed(946)
    sparse = get_SAEM_bernoulli(**common)
    production_sampler = semi_saem_module._sample_latent_layers_parametric

    def dense_reference_sampler(*args, **kwargs):
        kwargs["sparse_bernoulli"] = False
        return production_sampler(*args, **kwargs)

    monkeypatch.setattr(
        semi_saem_module,
        "_sample_latent_layers_parametric",
        dense_reference_sampler,
    )
    np.random.seed(946)
    dense = get_SAEM_bernoulli(**common)

    np.testing.assert_array_equal(sparse[0], dense[0])
    for sparse_B, dense_B in zip(sparse[1], dense[1]):
        np.testing.assert_array_equal(sparse_B, dense_B)
    assert sparse[2] is dense[2] is None
    for sparse_A, dense_A in zip(sparse[3], dense[3]):
        np.testing.assert_array_equal(sparse_A, dense_A)
    assert sparse[4] == dense[4]


def test_bernoulli_closed_coordinate_delta_matches_log_probability_difference():
    rng = np.random.default_rng(930)
    X = rng.integers(0, 2, size=(17, 9)).astype(float)
    eta_zero = rng.normal(scale=2.0, size=X.shape)
    eta_zero[0, :4] = np.array([-100.0, -25.0, 25.0, 100.0])
    slope = rng.normal(scale=1.5, size=X.shape[1])
    expected = np.sum(
        BERNOULLI_FAMILY.conditional_log_prob(
            X, eta_zero + slope[None, :], None
        )
        - BERNOULLI_FAMILY.conditional_log_prob(X, eta_zero, None),
        axis=1,
    )

    direct = BERNOULLI_FAMILY.observed_coordinate_loglik_delta(
        X, eta_zero, slope, None
    )
    chunked = _observed_coordinate_loglik_delta(
        X,
        eta_zero,
        slope,
        None,
        family=BERNOULLI_FAMILY,
        row_chunk_size=4,
    )
    np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=2e-14)
    np.testing.assert_allclose(chunked, direct, rtol=0.0, atol=0.0)


def test_bernoulli_fixed_accumulator_matches_matlab_objective_and_gradient():
    rng = np.random.default_rng(915)
    N, J, K, C = 13, 5, 3, 2
    X = rng.integers(0, 2, size=(N, J), dtype=np.uint8)
    parent = np.empty((N, K + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(
        0, 2, size=(N, K, C), dtype=np.uint8
    )
    accumulator = BERNOULLI_FAMILY.make_fixed_accumulator(J, K + 1)
    accumulator.update(X, parent, 1)

    design = np.transpose(parent, (0, 2, 1)).reshape(N * C, K + 1)
    response = np.repeat(X, C, axis=0).astype(float)
    point = rng.normal(scale=0.2, size=K + 1)
    for response_index in range(J):
        eta = design @ point
        expected = np.sum(
            -response[:, response_index] * eta + np.logaddexp(0.0, eta)
        ) / C
        value, gradient = accumulator.value_and_grad(response_index, point)
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-12)
        numerical = _central_difference(
            lambda candidate: accumulator.value(response_index, candidate),
            point,
        )
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-6)


def test_batched_bernoulli_tlp_solver_preserves_objective_and_bound_kkt():
    rng = np.random.default_rng(934)
    N, J, K, C = 160, 7, 4, 2
    X = rng.integers(0, 2, size=(N, J), dtype=np.uint8)
    parent = np.empty((N, K + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(
        0, 2, size=(N, K, C), dtype=np.uint8
    )
    accumulator = _make_bernoulli_layer_accumulator(K, J)
    accumulator.update(X, parent, 1)
    B_start = rng.normal(scale=0.08, size=(J, K + 1))
    bounds = BERNOULLI_FAMILY.fixed_latent_bounds(K + 1, 10.0)
    penalty, tau = 1.3, 0.25

    def penalized_objective(coefficients):
        value, _ = accumulator.batched_value_and_grad(
            coefficients, row_chunk_size=13
        )
        return value + _tlp_penalty(coefficients[:, 1:], penalty, tau)

    rowwise = _solve_bernoulli_tlp_rowwise(
        accumulator,
        B_start,
        penalty,
        tau,
        bounds,
        max_iter=20,
    )
    batched = _solve_bernoulli_tlp_batched(
        accumulator,
        B_start,
        penalty,
        tau,
        bounds,
        max_iter=20,
        row_chunk_size=13,
    )
    start_value = penalized_objective(B_start)
    rowwise_value = penalized_objective(rowwise)
    batched_value = penalized_objective(batched)
    assert batched_value < start_value
    assert batched_value <= rowwise_value + 2e-3 * abs(rowwise_value)

    smooth_solution = _solve_bernoulli_tlp_batched(
        accumulator,
        B_start,
        0.0,
        tau,
        bounds,
        max_iter=200,
        row_chunk_size=13,
    )
    _, smooth_gradient = accumulator.batched_value_and_grad(
        smooth_solution, row_chunk_size=13
    )
    lower = np.asarray([bound[0] for bound in bounds])[None, :]
    upper = np.asarray([bound[1] for bound in bounds])[None, :]
    projected_gradient = smooth_solution - np.clip(
        smooth_solution - smooth_gradient, lower, upper
    )
    assert np.max(np.abs(projected_gradient)) <= 5e-3


def test_batched_bernoulli_full_kernel_matches_rowwise_recovery(monkeypatch):
    X, prop, B_list, A_list = _bernoulli_state(935)
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

    np.random.seed(936)
    batched = get_SAEM_bernoulli(**common)
    monkeypatch.setattr(
        family_module,
        "_solve_bernoulli_tlp_batched",
        _solve_bernoulli_tlp_rowwise,
    )
    monkeypatch.setattr(
        semi_saem_module,
        "_solve_bernoulli_tlp_batched",
        _solve_bernoulli_tlp_rowwise,
    )
    np.random.seed(936)
    rowwise = get_SAEM_bernoulli(**common)

    np.testing.assert_allclose(batched[0], rowwise[0], rtol=0.0, atol=0.01)
    coefficient_difference = np.concatenate(
        [
            (batch_coefficient - row_coefficient).ravel()
            for batch_coefficient, row_coefficient in zip(
                batched[1], rowwise[1]
            )
        ]
    )
    assert np.sqrt(np.mean(coefficient_difference**2)) <= 0.2
    support_agreement = np.concatenate(
        [
            (
                (batch_coefficient[:, 1:] != 0.0)
                == (row_coefficient[:, 1:] != 0.0)
            ).ravel()
            for batch_coefficient, row_coefficient in zip(
                batched[1], rowwise[1]
            )
        ]
    )
    assert np.mean(support_agreement) >= 0.9
    latent_agreement = np.concatenate(
        [
            (batch_state == row_state).ravel()
            for batch_state, row_state in zip(batched[3], rowwise[3])
        ]
    )
    assert np.mean(latent_agreement) >= 0.9
    assert batched[4] == rowwise[4]


def test_binary_history_full_bernoulli_kernel_matches_grouped(monkeypatch):
    rng = np.random.default_rng(947)
    N, J = 17, 30
    K_list = [21, 3, 1]
    X = rng.integers(0, 2, size=(N, J), dtype=np.uint8)
    B_list = [rng.normal(scale=0.08, size=(J, K_list[0] + 1))]
    B_list.extend(
        rng.normal(
            scale=0.08,
            size=(K_list[layer - 1], K_list[layer] + 1),
        )
        for layer in range(1, len(K_list))
    )
    for coefficient in B_list:
        coefficient[:, 1:][np.abs(coefficient[:, 1:]) < 0.04] = 0.0
    A_list = [
        rng.integers(0, 2, size=(N, width), dtype=np.uint8)
        for width in K_list
    ]
    common = dict(
        X=X,
        prop_in=np.array([0.45]),
        B_list_in=B_list,
        pen_list=[0.01] * len(K_list),
        tau=0.1,
        A_list_in=A_list,
        C=1,
        tol=-1.0,
        max_iter=3,
        verbose=False,
    )

    np.random.seed(948)
    history = get_SAEM_bernoulli(**common)
    monkeypatch.setattr(
        semi_saem_module,
        "_make_bernoulli_layer_accumulator",
        lambda K_upper, n_children: (
            semi_saem_module._BernoulliLayerAccumulator(
                K_upper, n_children
            )
        ),
    )
    np.random.seed(948)
    grouped = get_SAEM_bernoulli(**common)

    np.testing.assert_allclose(history[0], grouped[0], rtol=1e-12, atol=1e-12)
    for history_B, grouped_B in zip(history[1], grouped[1]):
        np.testing.assert_allclose(
            history_B, grouped_B, rtol=1e-10, atol=1e-10
        )
    assert history[2] is grouped[2] is None
    for history_A, grouped_A in zip(history[3], grouped[3]):
        np.testing.assert_array_equal(history_A, grouped_A)
    assert history[4] == grouped[4]


@pytest.mark.parametrize("K", (3, 12, 21))
def test_all_bernoulli_accumulator_backends_accept_fixed_observations(K):
    rng = np.random.default_rng(930 + K)
    N, J, C = 17, 4, 2
    X = rng.integers(0, 2, size=(N, J), dtype=np.uint8)
    parent = np.empty((N, K + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(
        0, 2, size=(N, K, C), dtype=np.uint8
    )
    expanded_child = np.empty((N, J + 1, C), dtype=np.uint8)
    expanded_child[:, 0, :] = 1
    expanded_child[:, 1:, :] = X[:, :, None]
    fixed = _make_bernoulli_layer_accumulator(K, J)
    expanded = _make_bernoulli_layer_accumulator(K, J)
    fixed.update(X, parent, 1)
    expanded.update(expanded_child, parent, 1)

    point = rng.normal(scale=0.05, size=K + 1)
    for response_index in range(J):
        fixed_value, fixed_gradient = fixed.value_and_grad(
            response_index, point
        )
        expanded_value, expanded_gradient = expanded.value_and_grad(
            response_index, point
        )
        assert fixed_value == pytest.approx(
            expanded_value, rel=1e-12, abs=1e-12
        )
        np.testing.assert_allclose(
            fixed_gradient, expanded_gradient, rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("fixed_children", (True, False))
def test_bernoulli_csp_chunked_logistic_value_and_gradient_match_dense(
    fixed_children,
):
    rng = np.random.default_rng(947)
    N, J, K, C = 13, 7, 3, 2
    parent = np.empty((N, K + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(
        0, 2, size=(N, K, C), dtype=np.uint8
    )
    if fixed_children:
        response = rng.integers(0, 2, size=(N, J), dtype=np.uint8)
    else:
        response = np.empty((N, J + 1, C), dtype=np.uint8)
        response[:, 0, :] = 1
        response[:, 1:, :] = rng.integers(
            0, 2, size=(N, J, C), dtype=np.uint8
        )
    coefficients = rng.normal(scale=0.3, size=(J, K + 1))
    design, response_statistic, *_ = (
        _logistic_mc_design_and_response_statistic(response, parent)
    )
    temperature = 0.83

    value, gradient = _logistic_mc_value_and_grad(
        coefficients,
        design,
        response_statistic,
        temperature=temperature,
        mc_samples=C,
        row_chunk_size=6,
        coordinate_chunk_size=3,
    )
    logits = design @ coefficients.T
    if fixed_children:
        expanded_response = np.repeat(response, C, axis=0)
    else:
        expanded_response = np.transpose(
            response[:, 1:, :], (0, 2, 1)
        ).reshape(N * C, J)
    expected_value = (temperature / C) * np.sum(
        np.logaddexp(0.0, logits) - expanded_response * logits
    )
    expected_gradient = (temperature / C) * (
        (expit(logits) - expanded_response).T @ design
    )

    assert value == pytest.approx(expected_value, rel=1e-14, abs=1e-14)
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=1e-14, atol=1e-14
    )


def test_bernoulli_csp_solver_enforces_fixed_kernel_bounds():
    rng = np.random.default_rng(948)
    N, J, K, C = 25, 5, 3, 2
    response = rng.integers(0, 2, size=(N, J), dtype=np.uint8)
    parent = np.empty((N, K + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(
        0, 2, size=(N, K, C), dtype=np.uint8
    )
    bounds = BERNOULLI_FAMILY.csp_coefficient_bounds(
        K + 1, 10.0, observed=True
    )
    coefficients = _solve_logistic_weighted_l1(
        response,
        parent,
        np.full((J, K + 1), 100.0),
        np.full((J, K), 1e6),
        1.0,
        max_iter=2,
        bounds=bounds,
        row_chunk_size=8,
        coordinate_chunk_size=2,
    )
    lower = np.asarray([pair[0] for pair in bounds])
    upper = np.asarray([pair[1] for pair in bounds])
    assert np.all(coefficients >= lower[None, :])
    assert np.all(coefficients <= upper[None, :])


def test_bernoulli_csp_observed_solver_satisfies_weighted_l1_kkt():
    rng = np.random.default_rng(916)
    N, J, K, C = 50, 6, 3, 2
    X = rng.integers(0, 2, size=(N, J), dtype=np.uint8)
    parent = np.empty((N, K + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(
        0, 2, size=(N, K, C), dtype=np.uint8
    )
    penalty_scale = np.full((J, K), 5.0)
    temperature = 0.9
    B = _solve_logistic_weighted_l1(
        X,
        parent,
        np.zeros((J, K + 1)),
        penalty_scale,
        temperature,
        max_iter=4000,
        tol=1e-11,
    )

    design = np.transpose(parent, (0, 2, 1)).reshape(N * C, K + 1)
    response = np.repeat(X, C, axis=0)
    gradient = (
        temperature / C
    ) * ((expit(design @ B.T) - response).T @ design)
    assert np.max(np.abs(gradient[:, 0])) <= 2e-4
    slopes = B[:, 1:]
    slope_gradient = gradient[:, 1:]
    nonzero = np.abs(slopes) > 1e-7
    if np.any(nonzero):
        residual = slope_gradient[nonzero] + (
            np.sign(slopes[nonzero]) / penalty_scale[nonzero]
        )
        assert np.max(np.abs(residual)) <= 2e-4
    if np.any(~nonzero):
        assert np.max(
            np.abs(slope_gradient[~nonzero])
            - 1.0 / penalty_scale[~nonzero]
        ) <= 2e-4


def test_public_fixed_bernoulli_matches_direct_kernel_with_provided_state():
    X, prop, B_list, A_list = _bernoulli_state(917)
    common = dict(
        pen_list=[0.01, 0.01],
        tau=0.1,
        C=2,
        tol=0.0,
        max_iter=3,
        verbose=False,
    )
    np.random.seed(918)
    expected = get_SAEM_bernoulli(
        X,
        prop,
        B_list,
        common["pen_list"],
        common["tau"],
        A_list,
        common["C"],
        common["tol"],
        max_iter=common["max_iter"],
        verbose=False,
    )
    np.random.seed(918)
    actual = fit(
        X,
        method="DDE",
        family="Bernoulli",
        D=2,
        dimension_method="fixed",
        K_list=[3, 1],
        prop_in=prop,
        B_list_in=B_list,
        A_list_in=A_list,
        pen_list=common["pen_list"],
        tau=common["tau"],
        C=common["C"],
        tol=common["tol"],
        max_it=common["max_iter"],
        verbose=False,
    )

    np.testing.assert_array_equal(actual.prop, expected[0])
    for fitted, direct in zip(actual.B_list, expected[1]):
        np.testing.assert_array_equal(fitted, direct)
    assert actual.gamma is expected[2] is None
    for fitted, direct in zip(actual.A_list, expected[3]):
        np.testing.assert_array_equal(fitted, direct)
    assert actual.diagnostics["family"] == "Bernoulli"
    assert actual.diagnostics["iterations"] == expected[4]
    assert (
        actual.diagnostics["latent_transition"]
        == "systematic_coordinate_gibbs"
    )
    assert actual.diagnostics["mc_state_update"] == "last_draw"
    assert (
        actual.diagnostics["convergence_metric"]
        == "postprocessed_parameter_rmse"
    )
    assert actual.diagnostics["convergence_tol"] == common["tol"]


def test_public_fixed_bernoulli_data_initialization_smoke():
    X, _, _, _ = _bernoulli_state(919)
    np.random.seed(920)
    result = fit(
        X,
        method="DDE",
        family="binary",
        D=2,
        dimension_method="fixed",
        K_list=[3, 1],
        pen_list=[0.0, 0.0],
        tau=0.1,
        C=1,
        max_it=2,
        verbose=False,
    )

    assert result.model == "DDE"
    assert result.gamma is None
    assert result.diagnostics["family"] == "Bernoulli"
    assert result.diagnostics["initialization"] == "data"
    assert result.diagnostics["convergence_tol"] == 0.005
    assert [B.shape for B in result.B_list] == [(9, 4), (3, 2)]
    assert all(np.all(np.isfinite(B)) for B in result.B_list)


def test_public_fixed_bernoulli_supports_three_latent_layers():
    rng = np.random.default_rng(925)
    N, J = 36, 27
    widths = (9, 3, 1)
    prop = np.array([0.47])
    B_list = [
        rng.normal(scale=0.2, size=(J, widths[0] + 1)),
        rng.normal(scale=0.2, size=(widths[0], widths[1] + 1)),
        rng.normal(scale=0.2, size=(widths[1], widths[2] + 1)),
    ]
    A3 = rng.binomial(1, prop, size=(N, widths[2])).astype(np.uint8)
    A2 = rng.binomial(
        1,
        expit(B_list[2][:, 0][None, :] + A3 @ B_list[2][:, 1:].T),
    ).astype(np.uint8)
    A1 = rng.binomial(
        1,
        expit(B_list[1][:, 0][None, :] + A2 @ B_list[1][:, 1:].T),
    ).astype(np.uint8)
    X = rng.binomial(
        1,
        expit(B_list[0][:, 0][None, :] + A1 @ B_list[0][:, 1:].T),
    )

    np.random.seed(926)
    result = fit(
        X,
        method="DDE",
        family="Bernoulli",
        D=3,
        dimension_method="fixed",
        K_list=widths,
        prop_in=prop,
        B_list_in=B_list,
        A_list_in=[A1, A2, A3],
        pen_list=[0.0, 0.0, 0.0],
        tau=0.1,
        C=1,
        tol=0.0,
        max_it=2,
        verbose=False,
    )

    assert result.K_list == list(widths)
    assert result.gamma is None
    assert [B.shape for B in result.B_list] == [(27, 10), (9, 4), (3, 2)]
    assert all(np.all(np.isfinite(B)) for B in result.B_list)


@pytest.mark.parametrize(
    "backend", ("mcem", "mcem_reinitialize", "mcem_cap_only")
)
def test_public_bernoulli_csp_routes_all_backends_reproducibly(backend):
    X, prop, B_list, A_list = _bernoulli_state(921)
    arguments = dict(
        method="DDE",
        family="Bernoulli",
        D=2,
        dimension_method="csp",
        prop_in=prop,
        B_list_in=B_list,
        A_list_in=A_list,
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.1,
        C=2,
        tol=0.0,
        max_it=2,
        optimizer_maxiter=20,
        mcem_stage_min_iter=1,
        mcem_stage_width_window=1,
        csp_backend=backend,
        random_state=922,
        verbose=False,
    )

    first = fit(X, **arguments)
    second = fit(X, **arguments)
    _assert_result_equal(first, second)
    assert first.diagnostics["family"] == "Bernoulli"
    assert first.diagnostics["backend"] == backend


def test_public_bernoulli_csp_uses_data_initialization():
    X, _, _, _ = _bernoulli_state(927)
    result = fit(
        X,
        method="DDE",
        family="Bernoulli",
        D=2,
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.1,
        C=1,
        tol=0.0,
        max_it=1,
        optimizer_maxiter=5,
        random_state=928,
        verbose=False,
    )

    assert result.gamma is None
    assert result.diagnostics["family"] == "Bernoulli"
    assert result.diagnostics["initialization"] == "data"
    assert result.diagnostics["K_max_list"] == (3, 1)


def test_bernoulli_conditional_scoring_and_simulation_are_stable():
    eta = np.array([[-1000.0, -0.2, 0.7, 1000.0]])
    Y = np.array([[0.0, 1.0, 0.0, 1.0]])
    log_probability = BERNOULLI_FAMILY.conditional_log_prob(Y, eta, None)
    expected = np.where(
        Y == 1.0,
        -np.logaddexp(0.0, -eta),
        -np.logaddexp(0.0, eta),
    )
    np.testing.assert_array_equal(log_probability, expected)
    assert np.all(np.isfinite(log_probability))

    sample = BERNOULLI_FAMILY.sample_observed(
        np.tile(eta, (20, 1)), None, np.random.default_rng(923)
    )
    assert sample.shape == (20, 4)
    assert set(np.unique(sample)).issubset({0, 1})


def test_public_bernoulli_rejects_nonbinary_data_and_gamma():
    X, _, _, _ = _bernoulli_state(924)
    common = dict(
        method="DDE",
        family="Bernoulli",
        D=2,
        dimension_method="fixed",
        K_list=[3, 1],
        pen_list=[0.0, 0.0],
        tau=0.1,
        C=1,
        tol=0.0,
        max_it=1,
        verbose=False,
    )
    invalid = X.astype(float)
    invalid[0, 0] = 0.25
    with pytest.raises(ValueError, match="binary 0/1"):
        fit(invalid, **common)
    with pytest.raises(ValueError, match="no dispersion"):
        fit(X, gamma_in=np.ones(X.shape[1]), **common)
