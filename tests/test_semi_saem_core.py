import numpy as np
import pytest

import dde.algorithms.semi_saem as semi_saem_module
from dde.algorithms.parametric_families import NORMAL_FAMILY
from dde.algorithms.semi_saem import (
    _BinaryHistoryBernoulliLayerAccumulator,
    _BernoulliLayerAccumulator,
    _DirectAddressBernoulliLayerAccumulator,
    _ObservedNormalAccumulator,
    _compute_bernoulli_probabilities_normal,
    _latent_coordinate_probability_parametric,
    _matlab_debias,
    _make_bernoulli_layer_accumulator,
    _normal_sample_statistics,
    _observed_coordinate_loglik_delta,
    _tlp_penalty,
    _tlp_penalty_value_grad,
    _update_deep_normal_observed,
    get_SAEM_normal,
)
from dde.utils import logistic


def test_tlp_penalty_matches_paper_and_matlab_parameterization():
    slopes = np.array([0.05, -0.1, 0.4])

    assert _tlp_penalty(slopes, pen=2.0, tau=0.1) == pytest.approx(
        2.0 * (0.05 + 0.1 + 0.1)
    )


def _legacy_bernoulli_probabilities(X, prop, B_list, gamma, A_list):
    N = X.shape[0]
    D = len(B_list)
    K = [B.shape[1] - 1 for B in B_list]
    ber = [np.zeros((N, width)) for width in K]

    for d_idx in range(D):
        for l in range(K[d_idx]):
            if d_idx == D - 1:
                p_l = np.clip(prop[l], 1e-15, 1.0 - 1e-15)
                prior_logit = np.full(N, np.log(p_l / (1.0 - p_l)))
            else:
                A_above_aug = np.column_stack([np.ones(N), A_list[d_idx + 1]])
                prior_logit = A_above_aug @ B_list[d_idx + 1][l, :]

            if d_idx == 0:
                cols_keep = [c for c in range(K[0] + 1) if c != l + 1]
                A1_minus_l = np.delete(A_list[0], l, axis=1)
                A1_aug_minus_l = np.column_stack([np.ones(N), A1_minus_l])
                residual = X - A1_aug_minus_l @ B_list[0][:, cols_keep].T
                like_logit = -np.sum(
                    (
                        B_list[0][:, l + 1] ** 2
                        - 2.0 * residual * B_list[0][None, :, l + 1]
                    )
                    / gamma[None, :],
                    axis=1,
                ) / 2.0
            else:
                A_cur_1 = A_list[d_idx].copy()
                A_cur_1[:, l] = 1.0
                A_cur_0 = A_list[d_idx].copy()
                A_cur_0[:, l] = 0.0
                eta_1 = np.clip(
                    np.column_stack([np.ones(N), A_cur_1]) @ B_list[d_idx].T,
                    -500,
                    500,
                )
                eta_0 = np.clip(
                    np.column_stack([np.ones(N), A_cur_0]) @ B_list[d_idx].T,
                    -500,
                    500,
                )
                like_logit = (
                    np.sum(
                        A_list[d_idx - 1]
                        * B_list[d_idx][:, l + 1][None, :],
                        axis=1,
                    )
                    - np.sum(
                        np.log((1.0 + np.exp(eta_1)) / (1.0 + np.exp(eta_0))),
                        axis=1,
                    )
                )

            ber[d_idx][:, l] = logistic(prior_logit + like_logit)

    return ber


@pytest.mark.parametrize("K_list", ([3], [6, 2], [8, 4, 2, 1]))
def test_vectorized_normal_estep_matches_legacy_coordinate_sweep(K_list):
    rng = np.random.default_rng(20260827 + len(K_list))
    N = 7
    J = max(12, 2 * K_list[0])
    X = rng.normal(size=(N, J))
    B_list = [rng.normal(scale=0.2, size=(J, K_list[0] + 1))]
    B_list.extend(
        rng.normal(scale=0.2, size=(K_list[d - 1], K_list[d] + 1))
        for d in range(1, len(K_list))
    )
    A_list = [
        rng.integers(0, 2, size=(N, width)).astype(float)
        for width in K_list
    ]
    prop = rng.uniform(0.2, 0.8, size=K_list[-1])
    gamma = rng.uniform(0.5, 1.5, size=J)

    expected = _legacy_bernoulli_probabilities(
        X, prop, B_list, gamma, A_list
    )
    actual = _compute_bernoulli_probabilities_normal(
        X, prop, B_list, gamma, A_list
    )

    for expected_layer, actual_layer in zip(expected, actual):
        np.testing.assert_allclose(actual_layer, expected_layer, rtol=1e-12, atol=1e-12)


def test_chunked_normal_estep_matches_single_block():
    rng = np.random.default_rng(1010)
    N, J = 23, 24
    K_list = [8, 4, 2]
    X = rng.normal(size=(N, J))
    B_list = [rng.normal(scale=0.15, size=(J, K_list[0] + 1))]
    B_list.extend(
        rng.normal(scale=0.15, size=(K_list[d - 1], K_list[d] + 1))
        for d in range(1, len(K_list))
    )
    A_list = [rng.integers(0, 2, size=(N, k), dtype=np.uint8) for k in K_list]
    prop = np.array([0.4, 0.6])
    gamma = rng.uniform(0.7, 1.3, size=J)

    expected = _compute_bernoulli_probabilities_normal(
        X, prop, B_list, gamma, A_list, row_chunk_size=None
    )
    actual = _compute_bernoulli_probabilities_normal(
        X, prop, B_list, gamma, A_list, row_chunk_size=7
    )

    for expected_layer, actual_layer in zip(expected, actual):
        np.testing.assert_allclose(actual_layer, expected_layer, rtol=1e-12, atol=1e-12)


def test_normal_coordinate_conditionals_match_vectorized_static_state():
    rng = np.random.default_rng(1016)
    N, J = 9, 12
    K_list = [4, 2]
    X = rng.normal(size=(N, J))
    B_list = [rng.normal(scale=0.2, size=(J, K_list[0] + 1))]
    B_list.append(rng.normal(scale=0.2, size=(K_list[0], K_list[1] + 1)))
    A_list = [
        rng.integers(0, 2, size=(N, width), dtype=np.uint8)
        for width in K_list
    ]
    prop = rng.uniform(0.2, 0.8, size=K_list[-1])
    gamma = rng.uniform(0.5, 1.5, size=J)
    expected = _compute_bernoulli_probabilities_normal(
        X, prop, B_list, gamma, A_list
    )

    for layer, width in enumerate(K_list):
        for coordinate in range(width):
            actual, _ = _latent_coordinate_probability_parametric(
                X,
                prop,
                B_list,
                gamma,
                A_list,
                layer,
                coordinate,
                family=NORMAL_FAMILY,
                row_chunk_size=3,
            )
            np.testing.assert_allclose(
                actual,
                expected[layer][:, coordinate],
                rtol=1e-12,
                atol=1e-12,
            )


def test_normal_closed_coordinate_delta_matches_log_probability_difference():
    rng = np.random.default_rng(1017)
    X = rng.normal(size=(17, 12))
    eta_zero = rng.normal(scale=2.0, size=X.shape)
    eta_zero[0, :4] = np.array([-100.0, -25.0, 25.0, 100.0])
    slope = rng.normal(scale=1.5, size=X.shape[1])
    gamma = rng.uniform(0.4, 1.7, size=X.shape[1])
    expected = np.sum(
        NORMAL_FAMILY.conditional_log_prob(
            X, eta_zero + slope[None, :], gamma
        )
        - NORMAL_FAMILY.conditional_log_prob(X, eta_zero, gamma),
        axis=1,
    )

    direct = NORMAL_FAMILY.observed_coordinate_loglik_delta(
        X, eta_zero, slope, gamma
    )
    chunked = _observed_coordinate_loglik_delta(
        X,
        eta_zero,
        slope,
        gamma,
        family=NORMAL_FAMILY,
        row_chunk_size=4,
    )
    np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chunked, direct, rtol=0.0, atol=0.0)


def test_chunked_normal_estep_preserves_fixed_seed_full_kernel():
    rng = np.random.default_rng(1011)
    N, J = 31, 18
    K_list = [6, 2, 1]
    X = rng.normal(size=(N, J))
    B_list = [rng.normal(scale=0.08, size=(J, K_list[0] + 1))]
    B_list.extend(
        rng.normal(scale=0.08, size=(K_list[d - 1], K_list[d] + 1))
        for d in range(1, len(K_list))
    )
    A_list = [rng.integers(0, 2, size=(N, k)) for k in K_list]
    common = dict(
        X=X,
        prop_in=np.array([0.5]),
        B_list_in=B_list,
        gamma_in=np.ones(J),
        pen_list=[0.1] * len(K_list),
        tau=0.3,
        A_list_in=A_list,
        C=1,
        tol=-1.0,
        max_iter=3,
        verbose=False,
    )

    np.random.seed(1012)
    expected = get_SAEM_normal(**common, estep_row_chunk_size=None)
    np.random.seed(1012)
    actual = get_SAEM_normal(**common, estep_row_chunk_size=7)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    for actual_B, expected_B in zip(actual[1], expected[1]):
        np.testing.assert_allclose(actual_B, expected_B, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-11, atol=1e-11)
    for actual_A, expected_A in zip(actual[3], expected[3]):
        np.testing.assert_array_equal(actual_A, expected_A)
    assert actual[4] == expected[4]


def test_shared_observed_accumulator_matches_per_response_statistics():
    rng = np.random.default_rng(1001)
    N, J, P, C = 9, 5, 4, 3
    X = rng.normal(size=(N, J))
    accumulator = _ObservedNormalAccumulator(J, P)

    expected_Q = np.zeros((P, P))
    expected_q = np.zeros((J, P))
    for t in (1, 2):
        samples = rng.integers(0, 2, size=(N, P, C)).astype(float)
        alpha = 1.0 / t
        Q_new = np.einsum("nkc,nlc->kl", samples, samples) / C
        q_new = np.vstack(
            [np.einsum("nkc,n->k", samples, X[:, j]) / C for j in range(J)]
        )
        expected_Q = (1.0 - alpha) * expected_Q + alpha * Q_new
        expected_q = (1.0 - alpha) * expected_q + alpha * q_new
        accumulator.update(X, samples, t)

    np.testing.assert_allclose(accumulator.Q, expected_Q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(accumulator.q_all, expected_q, rtol=1e-12, atol=1e-12)
    probe = rng.normal(size=P)
    for j in range(J):
        expected_value = probe @ expected_Q @ probe - 2.0 * expected_q[j] @ probe
        assert accumulator.value(j, probe) == pytest.approx(expected_value)


def test_shared_bernoulli_accumulator_matches_per_child_statistics():
    rng = np.random.default_rng(1002)
    N, R, U, C = 11, 6, 3, 2
    accumulator = _BernoulliLayerAccumulator(U, R)
    legacy = [dict(codes=np.empty(0, dtype=np.int64), counts={}, sums={}) for _ in range(R)]

    for t in (1, 2):
        upper = np.concatenate(
            [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, U, C))],
            axis=1,
        ).astype(float)
        lower = np.concatenate(
            [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, R, C))],
            axis=1,
        ).astype(float)
        accumulator.update(lower, upper, t)

        parents = np.moveaxis(upper[:, 1:, :], 1, 2).reshape(-1, U)
        weights = np.int64(1) << np.arange(U - 1, -1, -1, dtype=np.int64)
        codes = parents.astype(np.int64) @ weights
        children = np.moveaxis(lower[:, 1:, :], 1, 2).reshape(-1, R)
        alpha = 1.0 / t
        all_codes = np.union1d(legacy[0]["codes"], np.unique(codes))
        for k in range(R):
            old_counts = legacy[k]["counts"]
            old_sums = legacy[k]["sums"]
            new_counts = {int(code): 0.0 for code in all_codes}
            new_sums = {int(code): 0.0 for code in all_codes}
            for code, y in zip(codes, children[:, k]):
                new_counts[int(code)] += 1.0 / C
                new_sums[int(code)] += float(y) / C
            legacy[k]["counts"] = {
                int(code): (1.0 - alpha) * old_counts.get(int(code), 0.0)
                + alpha * new_counts[int(code)]
                for code in all_codes
            }
            legacy[k]["sums"] = {
                int(code): (1.0 - alpha) * old_sums.get(int(code), 0.0)
                + alpha * new_sums[int(code)]
                for code in all_codes
            }
            legacy[k]["codes"] = all_codes

    np.testing.assert_array_equal(accumulator.codes, legacy[0]["codes"])
    for k in range(R):
        expected_counts = np.array([legacy[k]["counts"][int(c)] for c in accumulator.codes])
        expected_sums = np.array([legacy[k]["sums"][int(c)] for c in accumulator.codes])
        np.testing.assert_allclose(accumulator.counts, expected_counts, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(accumulator.sum_y[:, k], expected_sums, rtol=1e-12, atol=1e-12)
        probe = rng.normal(size=U + 1)
        eta = probe[0] + accumulator.parent_states @ probe[1:]
        expected_value = -(expected_sums @ eta) + expected_counts @ np.logaddexp(0.0, eta)
        assert accumulator.value(k, probe) == pytest.approx(expected_value)


def test_binary_history_accumulator_matches_grouped_objective_and_gradient():
    rng = np.random.default_rng(1008)
    N, R, U, C = 23, 9, 21, 2
    grouped = _BernoulliLayerAccumulator(U, R)
    history = _BinaryHistoryBernoulliLayerAccumulator(U, R)

    for t in (1, 2, 3):
        upper = np.concatenate(
            [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, U, C))],
            axis=1,
        ).astype(float)
        lower = np.concatenate(
            [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, R, C))],
            axis=1,
        ).astype(float)
        grouped.update(lower, upper, t)
        history.update(lower, upper, t)

    for child_index in range(R):
        probe = rng.normal(scale=0.2, size=U + 1)
        expected_value, expected_grad = grouped.value_and_grad(child_index, probe)
        actual_value, actual_grad = history.value_and_grad(child_index, probe)
        assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
        assert history.value(child_index, probe) == pytest.approx(
            grouped.value(child_index, probe), rel=1e-12, abs=1e-12
        )
    np.testing.assert_allclose(
        history.weighted_design_gram(),
        grouped.weighted_design_gram(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert history._row_weight == pytest.approx(1.0 / (3 * C))
    assert len(history._parent_chunks) == 3
    assert len(history._packed_child_chunks) == 3
    assert all(chunk.dtype == np.uint8 for chunk in history._parent_chunks)
    assert all(
        chunk.dtype == np.uint8 for chunk in history._packed_child_chunks
    )


@pytest.mark.parametrize("K", (4, 12, 21))
def test_batched_bernoulli_accumulator_matches_rowwise_objectives(K):
    rng = np.random.default_rng(1020 + K)
    N, R, C = 31, 11, 2
    accumulator = _make_bernoulli_layer_accumulator(K, R)
    for iteration in (1, 2, 3):
        upper = np.concatenate(
            [
                np.ones((N, 1, C), dtype=np.uint8),
                rng.integers(0, 2, size=(N, K, C), dtype=np.uint8),
            ],
            axis=1,
        )
        lower = np.concatenate(
            [
                np.ones((N, 1, C), dtype=np.uint8),
                rng.integers(0, 2, size=(N, R, C), dtype=np.uint8),
            ],
            axis=1,
        )
        accumulator.update(lower, upper, iteration)

    coefficients = rng.normal(scale=0.15, size=(R, K + 1))
    expected_values = []
    expected_gradients = []
    for child_index in range(R):
        value, gradient = accumulator.value_and_grad(
            child_index, coefficients[child_index]
        )
        expected_values.append(value)
        expected_gradients.append(gradient)

    actual_value, actual_gradient = accumulator.batched_value_and_grad(
        coefficients, row_chunk_size=7
    )
    assert actual_value == pytest.approx(
        np.sum(expected_values), rel=1e-12, abs=1e-10
    )
    np.testing.assert_allclose(
        actual_gradient,
        np.asarray(expected_gradients),
        rtol=1e-12,
        atol=1e-11,
    )


def test_binary_history_sufficient_statistics_preserve_clipped_eta_fallback():
    rng = np.random.default_rng(1013)
    N, R, U, C = 19, 5, 21, 1
    grouped = _BernoulliLayerAccumulator(U, R)
    history = _BinaryHistoryBernoulliLayerAccumulator(U, R)

    upper = np.concatenate(
        [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, U, C))],
        axis=1,
    ).astype(float)
    lower = np.concatenate(
        [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, R, C))],
        axis=1,
    ).astype(float)
    grouped.update(lower, upper, 1)
    history.update(lower, upper, 1)

    probe = np.full(U + 1, 40.0)
    for child_index in range(R):
        expected_value, expected_grad = grouped.value_and_grad(child_index, probe)
        actual_value, actual_grad = history.value_and_grad(child_index, probe)
        assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)

    coefficients = np.vstack(
        [probe + 0.25 * child_index for child_index in range(R)]
    )
    expected_values = []
    expected_gradients = []
    for child_index in range(R):
        value, gradient = history.value_and_grad(
            child_index, coefficients[child_index]
        )
        expected_values.append(value)
        expected_gradients.append(gradient)
    actual_value, actual_gradient = history.batched_value_and_grad(
        coefficients, row_chunk_size=5
    )
    assert actual_value == pytest.approx(
        np.sum(expected_values), rel=1e-12, abs=1e-10
    )
    np.testing.assert_allclose(
        actual_gradient,
        np.asarray(expected_gradients),
        rtol=1e-12,
        atol=1e-11,
    )


def test_direct_address_accumulator_matches_grouped_objective_and_gradient():
    rng = np.random.default_rng(1016)
    N, R, U, C = 29, 7, 4, 3
    grouped = _BernoulliLayerAccumulator(U, R)
    direct = _DirectAddressBernoulliLayerAccumulator(U, R)

    for t in (1, 2, 3):
        upper = np.concatenate(
            [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, U, C))],
            axis=1,
        ).astype(float)
        lower = np.concatenate(
            [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, R, C))],
            axis=1,
        ).astype(float)
        grouped.update(lower, upper, t)
        direct.update(lower, upper, t)

    grouped_positions = grouped.codes.astype(int)
    np.testing.assert_allclose(
        direct.counts[grouped_positions], grouped.counts, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        direct.sum_y[grouped_positions], grouped.sum_y, rtol=1e-12, atol=1e-12
    )
    for child_index in range(R):
        probe = rng.normal(scale=0.2, size=U + 1)
        expected_value, expected_grad = grouped.value_and_grad(child_index, probe)
        actual_value, actual_grad = direct.value_and_grad(child_index, probe)
        assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_wide_parent_layers_select_binary_history_backend():
    assert isinstance(
        _make_bernoulli_layer_accumulator(6, 18),
        _DirectAddressBernoulliLayerAccumulator,
    )
    assert isinstance(
        _make_bernoulli_layer_accumulator(18, 54),
        _BernoulliLayerAccumulator,
    )
    assert isinstance(
        _make_bernoulli_layer_accumulator(54, 162),
        _BinaryHistoryBernoulliLayerAccumulator,
    )


def test_fixed_seed_full_kernel_matches_all_grouped_accumulators(monkeypatch):
    rng = np.random.default_rng(1014)
    N, J = 13, 36
    K_list = [24, 21, 8, 3, 1]
    X = rng.normal(size=(N, J))
    B_list = [rng.normal(scale=0.04, size=(J, K_list[0] + 1))]
    B_list.extend(
        rng.normal(scale=0.04, size=(K_list[d - 1], K_list[d] + 1))
        for d in range(1, len(K_list))
    )
    A_list = [rng.integers(0, 2, size=(N, k)) for k in K_list]
    common = dict(
        X=X,
        prop_in=np.array([0.5]),
        B_list_in=B_list,
        gamma_in=np.ones(J),
        pen_list=[0.1] * len(K_list),
        tau=0.3,
        A_list_in=A_list,
        C=1,
        tol=-1.0,
        max_iter=3,
        verbose=False,
    )

    np.random.seed(1015)
    with np.errstate(over="ignore", invalid="ignore"):
        actual = get_SAEM_normal(**common)
    monkeypatch.setattr(
        semi_saem_module,
        "_make_bernoulli_layer_accumulator",
        lambda K_upper, n_children: _BernoulliLayerAccumulator(
            K_upper, n_children
        ),
    )
    np.random.seed(1015)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = get_SAEM_normal(**common)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    for actual_B, expected_B in zip(actual[1], expected[1]):
        np.testing.assert_allclose(actual_B, expected_B, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-10, atol=1e-10)
    for actual_A, expected_A in zip(actual[3], expected[3]):
        np.testing.assert_array_equal(actual_A, expected_A)
    assert actual[4] == expected[4]


def test_binary_history_backend_uses_less_storage_for_mostly_unique_patterns():
    rng = np.random.default_rng(1009)
    N, R, U, C = 64, 27, 21, 1
    grouped = _BernoulliLayerAccumulator(U, R)
    history = _BinaryHistoryBernoulliLayerAccumulator(U, R)
    for t in (1, 2, 3):
        upper = np.concatenate(
            [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, U, C))],
            axis=1,
        ).astype(float)
        lower = np.concatenate(
            [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, R, C))],
            axis=1,
        ).astype(float)
        grouped.update(lower, upper, t)
        history.update(lower, upper, t)

    _ = grouped.design
    grouped_nbytes = sum(
        value.nbytes
        for value in (
            grouped.codes,
            grouped.counts,
            grouped.sum_y,
            grouped.parent_states,
            grouped.design,
        )
    )
    assert history.storage_nbytes < grouped_nbytes
    packed_child_width = (R + 7) // 8
    expected_history_nbytes = (
        3 * N * C * (U + packed_child_width)
        + R * (U + 1) * np.dtype(float).itemsize
    )
    assert history.storage_nbytes == expected_history_nbytes

    # The previous float64 monolith doubled its capacity from 2*N to 4*N
    # on the third update.  The chunked binary representation must remain
    # substantially below that exact legacy allocation.
    legacy_capacity = 4 * N * C
    legacy_nbytes = (
        legacy_capacity * U * np.dtype(float).itemsize
        + legacy_capacity * packed_child_width
        + R * (U + 1) * np.dtype(float).itemsize
    )
    assert history.storage_nbytes <= 0.25 * legacy_nbytes


def _central_difference(fun, x, step=1e-6):
    grad = np.empty_like(x, dtype=float)
    for index in range(x.size):
        delta = np.zeros_like(x, dtype=float)
        delta[index] = step
        grad[index] = (fun(x + delta) - fun(x - delta)) / (2.0 * step)
    return grad


def test_tlp_analytic_gradient_matches_central_difference_away_from_knots():
    x = np.array([-0.25, -0.04, 0.08, 0.4])
    pen, tau = 1.7, 0.2
    value, grad = _tlp_penalty_value_grad(x, pen, tau)

    assert value == pytest.approx(_tlp_penalty(x, pen, tau))
    numerical = _central_difference(lambda z: _tlp_penalty(z, pen, tau), x)
    np.testing.assert_allclose(grad, numerical, rtol=1e-7, atol=1e-7)


def test_observed_objective_analytic_gradient_matches_central_difference():
    rng = np.random.default_rng(1003)
    N, J, P, C = 12, 4, 5, 2
    X = rng.normal(size=(N, J))
    samples = rng.integers(0, 2, size=(N, P, C)).astype(float)
    accumulator = _ObservedNormalAccumulator(J, P)
    accumulator.update(X, samples, 1)
    probe = rng.normal(scale=0.3, size=P)

    value, grad = accumulator.value_and_grad(2, probe)
    assert value == pytest.approx(accumulator.value(2, probe))
    numerical = _central_difference(lambda z: accumulator.value(2, z), probe)
    np.testing.assert_allclose(grad, numerical, rtol=1e-6, atol=1e-6)


def test_bernoulli_objective_analytic_gradient_matches_central_difference():
    rng = np.random.default_rng(1004)
    N, R, U, C = 13, 5, 4, 2
    upper = np.concatenate(
        [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, U, C))], axis=1
    ).astype(float)
    lower = np.concatenate(
        [np.ones((N, 1, C)), rng.integers(0, 2, size=(N, R, C))], axis=1
    ).astype(float)
    accumulator = _BernoulliLayerAccumulator(U, R)
    accumulator.update(lower, upper, 1)
    probe = rng.normal(scale=0.2, size=U + 1)

    value, grad = accumulator.value_and_grad(3, probe)
    assert value == pytest.approx(accumulator.value(3, probe))
    numerical = _central_difference(lambda z: accumulator.value(3, z), probe)
    np.testing.assert_allclose(grad, numerical, rtol=1e-6, atol=1e-6)


def test_deep_normal_observed_update_matches_matlab_closed_form():
    rng = np.random.default_rng(1005)
    N, J, K1, C = 30, 9, 4, 3
    X = rng.normal(size=(N, J))
    A_samples = np.concatenate(
        [
            np.ones((N, 1, C)),
            rng.integers(0, 2, size=(N, K1, C)),
        ],
        axis=1,
    ).astype(float)
    tau = 0.15

    Q, q_all = _normal_sample_statistics(X, A_samples)
    actual_B, actual_gamma = _update_deep_normal_observed(X, Q, q_all, tau)

    matlab_B = (np.linalg.inv(Q) @ q_all.T).T
    matlab_B[:, 1:][np.abs(matlab_B[:, 1:]) < tau] = 0.0
    eta = np.einsum("nkc,jk->njc", A_samples, matlab_B, optimize=True)
    matlab_gamma = np.mean((X[:, :, None] - eta) ** 2, axis=(0, 2))

    np.testing.assert_allclose(actual_B, matlab_B, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual_gamma, matlab_gamma, rtol=1e-11, atol=1e-11)

    cached_B, cached_gamma = _update_deep_normal_observed(
        X,
        Q,
        q_all,
        tau,
        x_sq_sum=np.sum(X ** 2, axis=0),
    )
    np.testing.assert_array_equal(cached_B, actual_B)
    np.testing.assert_array_equal(cached_gamma, actual_gamma)


def test_deep_normal_debias_matches_matlab_rule():
    values = np.array([-0.4, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.4])
    expected = np.array([-0.5, -0.25, -0.2, 0.0, 0.0, 0.0, 0.1, 0.25, 0.5])

    np.testing.assert_allclose(_matlab_debias(values, tau=0.1), expected)


@pytest.mark.parametrize(
    ("J", "K_list"),
    [
        (162, [54, 18, 6, 2]),
        (486, [162, 54, 18, 6, 2]),
    ],
)
def test_deep_normal_kernel_one_iteration_smoke(J, K_list):
    rng = np.random.default_rng(1006 + len(K_list))
    N = 8
    X = rng.normal(size=(N, J))
    B_list = [rng.normal(scale=0.05, size=(J, K_list[0] + 1))]
    B_list.extend(
        rng.normal(scale=0.05, size=(K_list[d - 1], K_list[d] + 1))
        for d in range(1, len(K_list))
    )
    A_list = [
        rng.integers(0, 2, size=(N, width)).astype(float)
        for width in K_list
    ]

    np.random.seed(1007)
    prop, B_fit, gamma, A_fit, iterations = get_SAEM_normal(
        X,
        np.full(K_list[-1], 0.5),
        B_list,
        np.ones(J),
        [0.1] * len(K_list),
        0.3,
        A_list,
        1,
        -1.0,
        max_iter=2,
        verbose=False,
    )

    assert iterations == 2
    assert prop.shape == (K_list[-1],)
    assert [B.shape for B in B_fit] == [
        (J, K_list[0] + 1),
        *[
            (K_list[d - 1], K_list[d] + 1)
            for d in range(1, len(K_list))
        ],
    ]
    assert [A.shape for A in A_fit] == [(N, width) for width in K_list]
    assert all(A.dtype == np.uint8 for A in A_fit)
    assert all(np.all(np.isfinite(B)) for B in B_fit)
    assert np.all(np.isfinite(gamma))
