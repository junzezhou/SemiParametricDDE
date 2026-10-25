import numpy as np

from dde.algorithms.dde import fit_dde_normal, fit_parametric_dde
from dde.algorithms.parametric_families import (
    NORMAL_FAMILY,
    resolve_parametric_family,
)
from dde.algorithms.semi_saem import get_SAEM_normal, get_SAEM_parametric


def _normal_state(seed=20260901):
    rng = np.random.default_rng(seed)
    N, J = 8, 9
    K = (3, 1)
    X = rng.normal(size=(N, J))
    prop = np.array([0.43])
    B_list = [
        rng.normal(scale=0.15, size=(J, K[0] + 1)),
        rng.normal(scale=0.15, size=(K[0], K[1] + 1)),
    ]
    gamma = rng.uniform(0.7, 1.3, size=J)
    A_list = [
        rng.integers(0, 2, size=(N, width), dtype=np.uint8)
        for width in K
    ]
    return X, prop, B_list, gamma, A_list


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


def test_normal_adapter_exposes_the_observed_family_contract():
    assert resolve_parametric_family("normal") is NORMAL_FAMILY
    assert resolve_parametric_family("Gaussian") is NORMAL_FAMILY
    assert resolve_parametric_family(NORMAL_FAMILY) is NORMAL_FAMILY
    assert NORMAL_FAMILY.has_dispersion is True
    assert NORMAL_FAMILY.dispersion_name == "conditional_variance"

    X = np.array([[1.0, -2.0], [3.0, 4.0]])
    working = NORMAL_FAMILY.initial_working_response(X, epsilon=1e-4)
    np.testing.assert_array_equal(working, X)
    assert working is not X

    eta = np.array([[0.2, -0.1], [0.7, 1.2]])
    gamma = np.array([0.5, 1.5])
    expected = -0.5 * (
        np.log(2.0 * np.pi * gamma[None, :])
        + (X - eta) ** 2 / gamma[None, :]
    )
    np.testing.assert_allclose(
        NORMAL_FAMILY.conditional_log_prob(X, eta, gamma), expected
    )

    actual_sample = NORMAL_FAMILY.sample_observed(
        eta, gamma, np.random.default_rng(81)
    )
    expected_sample = np.random.default_rng(81).normal(
        loc=eta, scale=np.sqrt(gamma)[None, :], size=eta.shape
    )
    np.testing.assert_array_equal(actual_sample, expected_sample)


def test_fixed_normal_compatibility_runner_is_bitwise_equivalent():
    X, prop, B_list, gamma, A_list = _normal_state()
    common = dict(
        X=X,
        prop_in=prop,
        B_list_in=B_list,
        pen_list=[0.03, 0.04],
        tau=0.04,
        A_list_in=A_list,
        C=2,
        tol=0.0,
        max_iter=3,
        verbose=False,
        estep_row_chunk_size=3,
    )

    np.random.seed(7319)
    expected = get_SAEM_normal(gamma_in=gamma, **common)
    np.random.seed(7319)
    actual = get_SAEM_parametric(
        dispersion_in=gamma,
        family=NORMAL_FAMILY,
        **common,
    )
    _assert_nested_equal(actual, expected)


def test_public_normal_wrapper_is_bitwise_equivalent_to_parametric_dispatch():
    X, prop, B_list, gamma, A_list = _normal_state()
    common = dict(
        D=2,
        K_list=[3, 1],
        dimension_method="fixed",
        pen_list=[0.03, 0.04],
        tau=0.04,
        C=2,
        tol=0.0,
        max_it=3,
        prop_in=prop,
        B_list_in=B_list,
        gamma_in=gamma,
        A_list_in=A_list,
        verbose=False,
    )

    np.random.seed(519)
    expected = fit_dde_normal(X, **common)
    np.random.seed(519)
    actual = fit_parametric_dde(X, family="normal", **common)
    _assert_nested_equal(actual.raw, expected.raw)
    _assert_nested_equal(actual.diagnostics, expected.diagnostics)
