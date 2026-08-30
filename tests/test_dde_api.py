import numpy as np
import pytest

from dde import fit
from dde.algorithms.normal_init_semi import Normal_init_semi
from dde.algorithms.semi_saem import get_SAEM_normal
from dde.simulation import generate_X_normal


def _normal_parameters(seed=20260827):
    rng = np.random.default_rng(seed)
    N, J = 16, 18
    K_list = [6, 2]
    prop = np.array([0.4, 0.6])
    B0 = rng.normal(0.0, 0.35, size=(J, 7))
    B1 = rng.normal(0.0, 0.35, size=(6, 3))
    gamma = np.full(J, 0.7)
    B_list = [B0, B1]
    G_list = [(B[:, 1:] != 0).astype(int) for B in B_list]
    np.random.seed(seed)
    X, _ = generate_X_normal(N, prop, B_list, gamma)
    return X, K_list, B_list, G_list


def test_fit_dde_normal_fixed_public_smoke():
    X, K_list, _, _ = _normal_parameters()
    np.random.seed(91)
    result = fit(
        X,
        method="DDE",
        D=2,
        family="Normal",
        dimension_method="fixed",
        K_list=K_list,
        pen_list=[0.0, 0.0],
        tau=0.3,
        C=1,
        tol=0.0,
        max_it=2,
        verbose=False,
    )

    assert result.model == "DDE"
    assert result.K_list == K_list
    assert result.prop.shape == (2,)
    assert [B.shape for B in result.B_list] == [(18, 7), (6, 3)]
    assert [A.shape for A in result.A_list] == [(16, 6), (16, 2)]
    assert result.gamma.shape == (18,)
    assert np.array_equal(result.transformed, X)
    assert result.raw[0] is result.transformed
    assert result.diagnostics["family"] == "Normal"
    assert result.diagnostics["dimension_method"] == "fixed"
    assert result.diagnostics["initialization"] == "data"


def test_fit_dde_normal_is_equivalent_to_direct_kernel_with_same_initial_state():
    X, K_list, _, _ = _normal_parameters()
    initial = Normal_init_semi(X, K_list[0], K_list, mode="data")[:4]
    prop0, B0, gamma0, A0 = initial

    np.random.seed(31415)
    expected = get_SAEM_normal(
        X,
        prop0,
        B0,
        gamma0,
        [0.0, 0.0],
        0.3,
        A0,
        1,
        0.0,
        max_iter=2,
        verbose=False,
    )
    np.random.seed(31415)
    actual = fit(
        X,
        method="DDE",
        D=2,
        family="Normal",
        dimension_method="fixed",
        K_list=K_list,
        prop_in=prop0,
        B_list_in=B0,
        gamma_in=gamma0,
        A_list_in=A0,
        pen_list=[0.0, 0.0],
        tau=0.3,
        C=1,
        tol=0.0,
        max_it=2,
        verbose=False,
    )

    assert np.allclose(actual.prop, expected[0])
    for fitted, direct in zip(actual.B_list, expected[1]):
        assert np.allclose(fitted, direct)
    assert np.allclose(actual.gamma, expected[2])
    for fitted, direct in zip(actual.A_list, expected[3]):
        assert np.array_equal(fitted, direct)
    assert actual.diagnostics["iterations"] == expected[4]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"D": None}, "D is required"),
        ({"family": None}, "family is required"),
        ({"family": "Poisson"}, "Normal.*only"),
        ({"K_list": [3, 2]}, "K_list"),
        ({"pen_list": [0.0]}, "length D=2"),
    ],
)
def test_fit_dde_normal_rejects_invalid_path_arguments(overrides, message):
    X, K_list, _, _ = _normal_parameters()
    args = dict(
        method="DDE",
        D=2,
        family="Normal",
        dimension_method="fixed",
        K_list=K_list,
        pen_list=[0.0, 0.0],
        tau=0.3,
        C=1,
        tol=0.0,
        max_it=2,
        verbose=False,
    )
    args.update(overrides)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        fit(X, **args)


def test_fit_rejects_conflicting_method_and_model():
    X, _, _, _ = _normal_parameters()
    with pytest.raises(ValueError, match="different DDE formulations"):
        fit(X, method="DDE", model="SemiDDE")


def test_fit_dde_normal_csp_public_smoke_and_is_reproducible():
    X, _, _, _ = _normal_parameters()
    args = dict(
        method="DDE",
        D=2,
        family="Normal",
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        tol=0.0,
        max_it=2,
        optimizer_maxiter=20,
        random_state=731,
        verbose=False,
    )

    first = fit(X, **args)
    second = fit(X, **args)

    assert first.model == "DDE"
    assert first.diagnostics["family"] == "Normal"
    assert first.diagnostics["dimension_method"] == "csp"
    assert first.diagnostics["backend"] == "approximate_mcem_csp"
    assert first.diagnostics["K_max_list"] == (6, 2)
    assert first.diagnostics["threshold_rule"] == "first_two_layers"
    fitted_K = tuple(first.diagnostics["K_list"])
    assert first.K_list == list(fitted_K)
    assert [B.shape for B in first.B_list] == [
        (18, fitted_K[0] + 1),
        (fitted_K[0], fitted_K[1] + 1),
    ]
    assert [A.shape for A in first.A_list] == [
        (X.shape[0], fitted_K[0]),
        (X.shape[0], fitted_K[1]),
    ]
    assert first.prop.shape == (fitted_K[-1],)
    assert "A_sample_long" not in first.raw
    assert first.diagnostics["K_list"] == second.diagnostics["K_list"]
    np.testing.assert_allclose(first.prop, second.prop)
    for B_first, B_second in zip(first.B_list, second.B_list):
        np.testing.assert_allclose(B_first, B_second)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"K_list": [6, 2]}, "K_list must be omitted"),
        ({"pen_list": [0.1, 0.1]}, "pen_list.*fixed"),
        ({"t_spike": None}, "requires t_spike and temp"),
        ({"temp": None}, "requires t_spike and temp"),
    ],
)
def test_fit_dde_normal_csp_rejects_path_incompatible_arguments(overrides, message):
    X, _, _, _ = _normal_parameters()
    args = dict(
        method="DDE",
        D=2,
        family="Normal",
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        tol=0.005,
        max_it=1,
        verbose=False,
    )
    args.update(overrides)
    with pytest.raises(ValueError, match=message):
        fit(X, **args)
