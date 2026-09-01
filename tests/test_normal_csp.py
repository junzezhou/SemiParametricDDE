import numpy as np
from scipy.special import expit

from dde import fit
from dde.utils import logistic
from dde.algorithms.normal_csp import (
    _apply_staged_parent_cap_only,
    _canonicalize_csp_layer_order,
    _solve_gaussian_weighted_l1,
    _solve_gaussian_weighted_l1_from_statistics,
    _solve_logistic_weighted_l1,
    _spectral_parent_width,
    fit_normal_csp_mcem,
)
from dde.algorithms.normal_init_semi import initialize_binary_parent
from dde.algorithms.semi_saem import (
    _ObservedNormalAccumulator,
    _compute_bernoulli_probabilities_normal,
    _normal_sample_statistics,
)


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


def _scalar_probability_reference(X, prop, B_list, gamma, A_list, temperature):
    """Pre-optimization scalar-coordinate Normal E-step for equivalence tests."""
    N = X.shape[0]
    widths = [B.shape[1] - 1 for B in B_list]
    probabilities = []
    for d, width in enumerate(widths):
        if d == len(widths) - 1:
            clipped = np.clip(prop, 1e-15, 1.0 - 1e-15)
            prior = np.broadcast_to(np.log(clipped / (1.0 - clipped)), (N, width))
        else:
            B_above = B_list[d + 1]
            prior = B_above[:, 0][None, :] + A_list[d + 1] @ B_above[:, 1:].T

        if d == 0:
            slopes = B_list[0][:, 1:]
            weighted_slopes = slopes / gamma[:, None]
            gram = slopes.T @ weighted_slopes
            diagonal = np.diag(gram)
            likelihood = ((X - B_list[0][:, 0][None, :]) / gamma[None, :]) @ slopes
            likelihood -= A_list[0] @ gram
            likelihood += (A_list[0] - 0.5) * diagonal[None, :]
        else:
            B = B_list[d]
            base_eta = B[:, 0][None, :] + A_list[d] @ B[:, 1:].T
            base_softplus = np.logaddexp(0.0, np.clip(base_eta, -500.0, 500.0))
            child_linear = A_list[d - 1] @ B[:, 1:]
            likelihood = np.empty((N, width), dtype=float)
            for coordinate in range(width):
                direction = 1.0 - 2.0 * A_list[d][:, coordinate].astype(float)
                shifted = np.clip(
                    base_eta + direction[:, None] * B[:, coordinate + 1][None, :],
                    -500.0,
                    500.0,
                )
                normalizer_delta = direction * np.sum(
                    np.logaddexp(0.0, shifted) - base_softplus,
                    axis=1,
                )
                likelihood[:, coordinate] = (
                    child_linear[:, coordinate] - normalizer_delta
                )
        probabilities.append(logistic(temperature * (prior + likelihood)))
    return probabilities


def test_blocked_normal_estep_matches_scalar_coordinate_reference():
    rng = np.random.default_rng(822)
    N, J = 17, 15
    widths = (6, 2, 1)
    X = rng.normal(size=(N, J))
    prop = rng.uniform(0.2, 0.8, size=widths[-1])
    B_list = [rng.normal(scale=0.3, size=(J, widths[0] + 1))]
    B_list.extend(
        rng.normal(scale=0.3, size=(widths[d - 1], widths[d] + 1))
        for d in range(1, len(widths))
    )
    A_list = [rng.integers(0, 2, size=(N, width), dtype=np.uint8) for width in widths]
    gamma = rng.uniform(0.7, 1.3, size=J)
    expected = _scalar_probability_reference(X, prop, B_list, gamma, A_list, 0.87)

    for coordinate_chunk_size in (1, 2, 4):
        actual = _compute_bernoulli_probabilities_normal(
            X,
            prop,
            B_list,
            gamma,
            A_list,
            row_chunk_size=5,
            coordinate_chunk_size=coordinate_chunk_size,
            temperature=0.87,
        )
        for actual_layer, expected_layer in zip(actual, expected):
            np.testing.assert_allclose(
                actual_layer, expected_layer, rtol=0.0, atol=2e-16
            )


def test_cap_aware_normal_estep_skips_only_zero_ineligible_tails():
    rng = np.random.default_rng(823)
    N, J = 19, 18
    widths = (6, 3, 2)
    eligible = (4, 2, 1)
    X = rng.normal(size=(N, J))
    prop = rng.uniform(0.2, 0.8, size=widths[-1])
    B_list = [rng.normal(scale=0.2, size=(J, widths[0] + 1))]
    B_list.extend(
        rng.normal(scale=0.2, size=(widths[d - 1], widths[d] + 1))
        for d in range(1, len(widths))
    )
    A_list = [rng.integers(0, 2, size=(N, width), dtype=np.uint8) for width in widths]
    for d, cap in enumerate(eligible):
        A_list[d][:, cap:] = 0
        B_list[d][:, cap + 1 :] = 0.0
        if d + 1 < len(widths):
            B_list[d + 1][cap:, :] = 0.0
    gamma = rng.uniform(0.8, 1.2, size=J)
    expected = _scalar_probability_reference(X, prop, B_list, gamma, A_list, 0.91)
    for d, cap in enumerate(eligible):
        expected[d][:, cap:] = 0.0

    actual = _compute_bernoulli_probabilities_normal(
        X,
        prop,
        B_list,
        gamma,
        A_list,
        row_chunk_size=7,
        coordinate_chunk_size=2,
        eligible_widths=eligible,
        temperature=0.91,
    )

    for actual_layer, expected_layer in zip(actual, expected):
        np.testing.assert_allclose(actual_layer, expected_layer, rtol=0.0, atol=2e-16)


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


def test_precomputed_observed_sum_of_squares_is_numerically_equivalent():
    rng = np.random.default_rng(821)
    N, J, K, C = 31, 6, 4, 2
    sample = np.empty((N, K + 1, C), dtype=np.uint8)
    sample[:, 0, :] = 1
    sample[:, 1:, :] = rng.integers(0, 2, size=(N, K, C), dtype=np.uint8)
    X = rng.normal(size=(N, J))
    Q, q_all = _normal_sample_statistics(X, sample)
    arguments = (
        X,
        Q,
        q_all,
        rng.normal(scale=0.1, size=(J, K + 1)),
        rng.uniform(0.8, 1.2, size=J),
        rng.uniform(0.5, 1.5, size=(J, K)),
        0.9,
    )

    baseline = _solve_gaussian_weighted_l1_from_statistics(
        *arguments, mc_samples=C, max_iter=50
    )
    precomputed = _solve_gaussian_weighted_l1_from_statistics(
        *arguments,
        mc_samples=C,
        x_sq_sum=np.sum(X**2, axis=0),
        max_iter=50,
    )

    np.testing.assert_array_equal(precomputed[0], baseline[0])
    np.testing.assert_array_equal(precomputed[1], baseline[1])


def test_batched_logistic_weighted_l1_solver_satisfies_kkt_conditions():
    rng = np.random.default_rng(91)
    N, n_children, K_parent, C = 45, 5, 3, 2
    parent = np.empty((N, K_parent + 1, C), dtype=np.uint8)
    parent[:, 0, :] = 1
    parent[:, 1:, :] = rng.integers(0, 2, size=(N, K_parent, C), dtype=np.uint8)
    child = np.empty((N, n_children + 1, C), dtype=np.uint8)
    child[:, 0, :] = 1
    child[:, 1:, :] = rng.integers(0, 2, size=(N, n_children, C), dtype=np.uint8)
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
    response = np.transpose(child[:, 1:, :], (0, 2, 1)).reshape(N * C, n_children)
    gradient = (temperature / C) * ((expit(design @ B.T) - response).T @ design)
    _assert_l1_kkt(gradient, B, 1.0 / scale)


def test_gaussian_one_step_sa_statistics_match_current_sample_solver():
    rng = np.random.default_rng(811)
    N, J, K, C = 24, 5, 3, 2
    sample = np.empty((N, K + 1, C), dtype=np.uint8)
    sample[:, 0, :] = 1
    sample[:, 1:, :] = rng.integers(0, 2, size=(N, K, C), dtype=np.uint8)
    X = rng.normal(size=(N, J))
    B_start = rng.normal(scale=0.1, size=(J, K + 1))
    gamma = rng.uniform(0.8, 1.2, size=J)
    scale = rng.uniform(0.5, 1.5, size=(J, K))

    current = _solve_gaussian_weighted_l1(
        X,
        sample,
        B_start,
        gamma,
        scale,
        1.0,
        max_iter=250,
        tol=1e-10,
    )
    accumulator = _ObservedNormalAccumulator(J, K + 1)
    accumulator.update(X, sample, 1)
    accumulated = _solve_gaussian_weighted_l1_from_statistics(
        X,
        accumulator.Q,
        accumulator.q_all,
        B_start,
        gamma,
        scale,
        1.0,
        mc_samples=C,
        max_iter=250,
        tol=1e-10,
    )

    np.testing.assert_allclose(accumulated[0], current[0], atol=1e-12)
    np.testing.assert_allclose(accumulated[1], current[1], atol=1e-12)


def test_two_step_sa_statistics_are_the_arithmetic_mean_of_draws():
    rng = np.random.default_rng(813)
    N, J, K, C = 12, 3, 4, 2
    X = rng.normal(size=(N, J))
    samples = []
    for _ in range(2):
        sample = np.empty((N, K + 1, C), dtype=np.uint8)
        sample[:, 0, :] = 1
        sample[:, 1:, :] = rng.integers(0, 2, size=(N, K, C), dtype=np.uint8)
        samples.append(sample)

    accumulator = _ObservedNormalAccumulator(J, K + 1)
    accumulator.update(X, samples[0], 1)
    accumulator.update(X, samples[1], 2)
    first_Q, first_q = _normal_sample_statistics(X, samples[0])
    second_Q, second_q = _normal_sample_statistics(X, samples[1])

    np.testing.assert_allclose(accumulator.Q, 0.5 * (first_Q + second_Q))
    np.testing.assert_allclose(accumulator.q_all, 0.5 * (first_q + second_q))


def test_binary_parent_reinitializer_returns_finite_aligned_shapes():
    rng = np.random.default_rng(817)
    child = rng.integers(0, 2, size=(80, 9), dtype=np.uint8)

    B_parent, A_parent = initialize_binary_parent(child, 3)

    assert B_parent.shape == (9, 4)
    assert A_parent.shape == (80, 3)
    assert np.all(np.isfinite(B_parent))
    assert set(np.unique(A_parent)).issubset({0, 1})


def test_spectral_parent_width_uses_only_a_clear_gap():
    rng = np.random.default_rng(820)
    factors = rng.integers(0, 2, size=(200, 2), dtype=np.uint8)
    child = np.column_stack([factors[:, 0]] * 5 + [factors[:, 1]] * 4)

    selected, singular_values, largest_gap = _spectral_parent_width(child, 3, 1.5)
    conservative, _, no_test_gap = _spectral_parent_width(child, 3, None)

    assert selected == 2
    assert conservative == 3
    assert singular_values.shape == (4,)
    assert largest_gap > 1.5
    assert no_test_gap is None


def test_mcem_staged_reinitialization_enforces_child_implied_cap():
    rng = np.random.default_rng(818)
    N, J = 30, 27
    widths = (9, 4)
    B1 = np.column_stack([np.zeros(J), np.full((J, widths[0]), 2.0)])
    raw = fit_normal_csp_mcem(
        rng.normal(size=(N, J)),
        np.full(widths[-1], 0.5),
        [B1, rng.normal(scale=0.2, size=(widths[0], widths[1] + 1))],
        np.ones(J),
        [rng.integers(0, 2, size=(N, width), dtype=np.uint8) for width in widths],
        C=1,
        max_iter=2,
        t_spike=[0.01, 0.01],
        temperature=0.9,
        threshold=0.3,
        convergence_tol=0.0,
        stagewise=True,
        stage_min_iter=1,
        stage_width_window=1,
        stage_transition_mode="reinitialize",
        optimizer_maxiter=2,
        random_state=819,
        verbose=False,
    )

    assert raw["stage_caps"] == (9, 3)
    assert raw["num_active"][1] <= 3
    np.testing.assert_allclose(raw["B_list"][1][:, 4:], 0.0)
    np.testing.assert_array_equal(raw["A_list"][1][:, 3:], 0)
    assert raw["stage_transition_events"][0]["mode"] == "reinitialize"


def test_cap_only_transition_preserves_eligible_parent_state():
    rng = np.random.default_rng(826)
    N = 12
    widths = (9, 4)
    prop = rng.uniform(0.2, 0.8, size=widths[-1])
    B_list = [
        rng.normal(size=(27, widths[0] + 1)),
        rng.normal(size=(widths[0], widths[1] + 1)),
    ]
    A_list = [rng.integers(0, 2, size=(N, width), dtype=np.uint8) for width in widths]
    active_masks = [
        np.ones(widths[0], dtype=bool),
        np.ones(widths[1], dtype=bool),
    ]
    allocation = [np.full(width, width + 1, dtype=int) for width in widths]
    penalty_scale = [
        np.ones((27, widths[0])),
        np.ones((widths[0], widths[1])),
    ]
    slab_scale = [np.ones(width) for width in widths]
    omega = [np.ones(width + 1) / (width + 1) for width in widths]
    stage_caps = np.asarray(widths, dtype=int)
    parent_B_before = B_list[1].copy()
    parent_A_before = A_list[1].copy()
    prop_before = prop.copy()

    prop_after, event = _apply_staged_parent_cap_only(
        1,
        prop,
        B_list,
        A_list,
        active_masks,
        allocation,
        penalty_scale,
        slab_scale,
        omega,
        stage_caps,
        np.array([0.01, 0.02]),
        None,
    )

    np.testing.assert_allclose(B_list[1][:, :4], parent_B_before[:, :4])
    np.testing.assert_array_equal(A_list[1][:, :3], parent_A_before[:, :3])
    np.testing.assert_allclose(prop_after[:3], prop_before[:3])
    np.testing.assert_allclose(B_list[1][:, 4:], 0.0)
    np.testing.assert_array_equal(A_list[1][:, 3:], 0)
    np.testing.assert_allclose(prop_after[3:], 0.5)
    assert stage_caps.tolist() == [9, 3]
    assert event == {
        "layer": 2,
        "mode": "cap_only",
        "child_active": 9,
        "structural_cap": 3,
        "initial_width": 3,
        "singular_values": (),
        "largest_gap": None,
    }


def test_csp_layer_ordering_is_likelihood_invariant_and_strength_sorted():
    rng = np.random.default_rng(816)
    N, J = 13, 27
    widths = (9, 3)
    X = rng.normal(size=(N, J))
    prop = rng.uniform(0.2, 0.8, size=widths[-1])
    B_list = [
        rng.normal(scale=0.3, size=(J, widths[0] + 1)),
        rng.normal(scale=0.3, size=(widths[0], widths[1] + 1)),
    ]
    A_list = [rng.integers(0, 2, size=(N, width), dtype=np.uint8) for width in widths]
    before = _compute_bernoulli_probabilities_normal(
        X, prop, B_list, np.ones(J), A_list
    )

    ordered_prop, ordered_B, ordered_A, permutations = _canonicalize_csp_layer_order(
        prop, B_list, A_list
    )
    after = _compute_bernoulli_probabilities_normal(
        X, ordered_prop, ordered_B, np.ones(J), ordered_A
    )

    for d, permutation in enumerate(permutations):
        np.testing.assert_allclose(after[d], before[d][:, permutation])
        strength = np.sum(np.abs(ordered_B[d][:, 1:]), axis=0)
        assert np.all(strength[:-1] >= strength[1:])


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
    A_list = [rng.integers(0, 2, size=(N, width), dtype=np.uint8) for width in K_list]

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
