import numpy as np
import pytest

from dde import fit
from dde.algorithms.normal_init_semi import Normal_init_semi, rank_gaussianize_columns
from dde.simulation import generate_X_normal
from dde.utils import dimension_recovery_metrics


def _semidde_sample(seed=20260916):
    rng = np.random.default_rng(seed)
    N, J = 24, 18
    B_true = [
        rng.normal(scale=0.35, size=(J, 4)),
        rng.normal(scale=0.4, size=(3, 2)),
    ]
    prop_true = np.array([0.5])
    gamma_true = np.ones(J)
    np.random.seed(seed)
    Y, _ = generate_X_normal(N, prop_true, B_true, gamma_true)
    return Y


def _maximal_initial_values(Y):
    X = rank_gaussianize_columns(Y)
    prop, B_list, gamma, A_list, _ = Normal_init_semi(
        X, 6, [6, 2], mode="data"
    )
    return X, prop, B_list, gamma, A_list


def test_semidde_csp_public_path_uses_maximal_width_and_compresses_result():
    Y = _semidde_sample()
    result = fit(
        Y,
        model="SemiDDE",
        D=2,
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        tol=0.0,
        max_it=1,
        optimizer_maxiter=2,
        random_state=731,
        verbose=False,
    )

    fitted = tuple(result.K_list)
    assert result.model == "SemiDDE"
    assert result.diagnostics["dimension_method"] == "csp"
    assert result.diagnostics["K_max_list"] == (6, 2)
    assert result.diagnostics["original_K_list"] == (6, 2)
    assert result.diagnostics["K_list"] == fitted
    assert result.diagnostics["num_active"].tolist() == list(fitted)
    assert result.diagnostics["backend"] == "copula"
    assert result.diagnostics["pseudo_observation_update"] == "semiparametric_g_every_iteration"
    assert [B.shape for B in result.B_list] == [
        (Y.shape[1], fitted[0] + 1),
        (fitted[0], fitted[1] + 1),
    ]
    assert [A.shape for A in result.A_list] == [
        (Y.shape[0], fitted[0]),
        (Y.shape[0], fitted[1]),
    ]
    assert result.prop.shape == (fitted[-1],)
    assert result.gamma.shape == (Y.shape[1],)
    assert result.transformed.shape == Y.shape
    assert np.isfinite(result.transformed).all()
    assert "active_columns_original_index" in result.diagnostics


def test_semidde_csp_is_reproducible_from_supplied_initial_state():
    Y = _semidde_sample(seed=20260917)
    X, prop, B_list, gamma, A_list = _maximal_initial_values(Y)
    args = dict(
        model="semi",
        D=2,
        dimension_method="csp",
        X_in=X,
        prop_in=prop,
        B_list_in=B_list,
        gamma_in=gamma,
        A_list_in=A_list,
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        tol=0.0,
        max_inner_iter=2,
        optimizer_maxiter=3,
        random_state=732,
        verbose=False,
    )

    first = fit(Y, **args)
    second = fit(Y, **args)

    assert first.K_list == second.K_list
    np.testing.assert_allclose(first.prop, second.prop)
    np.testing.assert_allclose(first.gamma, second.gamma)
    np.testing.assert_allclose(first.transformed, second.transformed)
    for first_B, second_B in zip(first.B_list, second.B_list):
        np.testing.assert_allclose(first_B, second_B)


def test_semidde_csp_dimension_recovery_metrics_match_primary_endpoints():
    exact = dimension_recovery_metrics([9, 3], [9, 3])
    missed = dimension_recovery_metrics([8, 4], [9, 3])

    assert exact == {
        "exact_dimension_recovery": 1,
        "dimension_l1_error": 0,
    }
    assert missed == {
        "exact_dimension_recovery": 0,
        "dimension_l1_error": 2,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"D": None}, "D is required"),
        ({"K_list": [7, 2]}, "J >= 3"),
        ({"pen_list": [0.1, 0.1]}, "pen_list.*fixed"),
    ],
)
def test_semidde_csp_rejects_incompatible_path_arguments(overrides, message):
    Y = _semidde_sample()
    args = dict(
        model="semi",
        D=2,
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        max_it=1,
        verbose=False,
    )
    args.update(overrides)
    with pytest.raises(ValueError, match=message):
        fit(Y, **args)


def test_semidde_csp_copula_backend_updates_g_every_iteration():
    rng = np.random.default_rng(5)
    N, J = 240, 18
    A2 = rng.binomial(1, 0.5, (N, 2))
    A1 = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + A2 @ np.repeat(np.eye(2), 3, axis=0).T * 4.0))))
    B1 = np.zeros((J, 6))
    for k in range(6):
        B1[[k, 6 + k, 12 + k], k] = 2.0
    Z = A1 @ B1.T + rng.normal(scale=0.7, size=(N, J))
    Y = np.exp(Z)
    from dde import fit

    res = fit(
        Y, model="SemiDDE", D=2, dimension_method="csp", csp_backend="copula",
        t_spike=(0.02, 0.04), temp=0.7, tau=0.4, C=1, max_it=8, random_state=3, verbose=False,
    )
    assert res.diagnostics["backend"] == "copula"
    assert len(res.diagnostics["g_relative_history"]) >= 1
    assert res.transformed.shape == Y.shape
    assert np.all(np.isfinite(res.transformed))
    with pytest.raises(ValueError, match="csp_backend"):
        fit(Y, model="SemiDDE", D=2, dimension_method="csp", csp_backend="bogus",
            t_spike=(0.02, 0.04), temp=0.7, tau=0.4, C=1, max_it=2, verbose=False)


def test_semidde_csp_defaults_to_the_copula_engine():
    Y = _semidde_sample()
    args = dict(model="SemiDDE", D=2, dimension_method="csp", t_spike=[0.03, 0.05],
                temp=0.9, tau=0.3, C=1, max_it=2, random_state=3, verbose=False)
    result = fit(Y, **args)
    assert result.diagnostics["backend"] == "copula"
    with pytest.raises(ValueError, match="MCEM path was removed"):
        fit(Y, csp_backend="mcem", **args)
    with pytest.raises(TypeError, match="max_outer_iter belonged to the removed MCEM path"):
        fit(Y, max_outer_iter=1, **args)
