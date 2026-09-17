import numpy as np
import pytest

from dde import fit
import dde.algorithms.semi_csp as semi_csp_module
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
        max_outer_iter=1,
        optimizer_maxiter=2,
        random_state=731,
        verbose=False,
    )

    fitted = tuple(result.K_list)
    assert result.model == "semi"
    assert result.diagnostics["dimension_method"] == "csp"
    assert result.diagnostics["K_max_list"] == (6, 2)
    assert result.diagnostics["original_K_list"] == (6, 2)
    assert result.diagnostics["K_list"] == fitted
    assert result.diagnostics["num_active"].tolist() == list(fitted)
    assert result.diagnostics["pseudo_observation_update"] == "semiparametric_g"
    assert result.diagnostics["threshold_rule"] == "first_two_layers"
    assert (
        result.diagnostics["csp_state_continuation"]
        == "across_outer_iterations"
    )
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
        max_outer_iter=2,
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


def test_semidde_csp_compresses_a_known_inactive_coefficient_graph(monkeypatch):
    Y = _semidde_sample(seed=20260918)

    def fake_inner(X, prop, B_list, gamma, A_list, **kwargs):
        B0 = np.zeros_like(B_list[0])
        B0[:, 1:4] = 0.5
        B1 = np.zeros_like(B_list[1])
        B1[:3, 1] = 0.5
        return {
            "prop": np.asarray(prop).copy(),
            "B_list": [B0, B1],
            "gamma": np.asarray(gamma).copy(),
            "A_list": [np.asarray(A).copy() for A in A_list],
            "allocation": [np.full(6, 7), np.full(2, 3)],
            "omega": [np.ones(7) / 7, np.ones(3) / 3],
            "slab_scale": [np.ones(6), np.ones(2)],
            "penalty_scale": [np.ones((18, 6)), np.ones((6, 2))],
            "active_masks": [
                np.array([True, True, True, False, False, False]),
                np.array([True, False]),
            ],
            "num_active": np.array([3, 1]),
            "iterations": 1,
            "temperature": kwargs["temperature"],
            "latent_transition": "legacy_synchronous",
            "mc_state_update": "draw_majority",
            "relative_history": [(0.0,), (0.0,)],
            "trace": None,
        }

    monkeypatch.setattr(semi_csp_module, "fit_normal_csp_mcem", fake_inner)
    result = fit(
        Y,
        model="semi",
        D=2,
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        max_it=1,
        max_outer_iter=1,
        verbose=False,
    )

    assert result.K_list == [3, 1]
    assert result.diagnostics["num_active"].tolist() == [3, 1]
    assert [B.shape for B in result.B_list] == [(18, 4), (3, 2)]
    assert [A.shape for A in result.A_list] == [(24, 3), (24, 1)]


def test_semidde_csp_does_not_cap_overspecified_parent_width(monkeypatch):
    Y = _semidde_sample(seed=20260919)

    def fake_inner(X, prop, B_list, gamma, A_list, **kwargs):
        B0 = np.zeros_like(B_list[0])
        B0[:, 1:4] = 0.5
        B1 = np.zeros_like(B_list[1])
        B1[:3, 1:] = 0.5
        return {
            "prop": np.asarray(prop).copy(),
            "B_list": [B0, B1],
            "gamma": np.asarray(gamma).copy(),
            "A_list": [np.asarray(A).copy() for A in A_list],
            "allocation": [np.full(6, 7), np.full(2, 3)],
            "omega": [np.ones(7) / 7, np.ones(3) / 3],
            "slab_scale": [np.ones(6), np.ones(2)],
            "penalty_scale": [np.ones((18, 6)), np.ones((6, 2))],
            "active_masks": [
                np.array([True, True, True, False, False, False]),
                np.array([True, True]),
            ],
            "num_active": np.array([3, 2]),
            "iterations": 1,
            "temperature": kwargs["temperature"],
            "latent_transition": "legacy_synchronous",
            "mc_state_update": "draw_majority",
            "relative_history": [(0.0,), (0.0,)],
            "trace": None,
        }

    monkeypatch.setattr(semi_csp_module, "fit_normal_csp_mcem", fake_inner)
    result = fit(
        Y,
        model="semi",
        D=2,
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        max_it=1,
        max_outer_iter=1,
        verbose=False,
    )

    # Overspecified CSP fits are reported as selected.  In particular, the
    # second layer is not forcibly capped at floor(3 / 3) = 1.
    assert result.K_list == [3, 2]
    assert result.diagnostics["num_active"].tolist() == [3, 2]


def test_semidde_csp_carries_copula_state_across_outer_updates(monkeypatch):
    Y = _semidde_sample(seed=20260920)
    received_states = []
    returned_states = []

    def fake_inner(X, prop, B_list, gamma, A_list, **kwargs):
        received_states.append(kwargs["initial_csp_state"])
        K = [B.shape[1] - 1 for B in B_list]
        raw = {
            "prop": np.asarray(prop).copy(),
            "B_list": [np.asarray(B).copy() for B in B_list],
            "gamma": np.asarray(gamma).copy(),
            "A_list": [np.asarray(A).copy() for A in A_list],
            "allocation": [np.full(Kd, Kd + 1) for Kd in K],
            "omega": [np.ones(Kd + 1) / (Kd + 1) for Kd in K],
            "slab_scale": [np.ones(Kd) for Kd in K],
            "penalty_scale": [np.ones_like(B[:, 1:]) for B in B_list],
            "active_masks": [np.ones(Kd, dtype=bool) for Kd in K],
            "num_active": np.asarray(K),
            "iterations": 1,
            "temperature": 1.0,
            "latent_transition": "legacy_synchronous",
            "mc_state_update": "draw_majority",
            "relative_history": [(0.0,) for _ in K],
            "trace": None,
        }
        returned_states.append(raw)
        return raw

    def fake_g_step(Y, X, prop, B_list, A1, **kwargs):
        return X.copy(), X.copy(), 1.0

    monkeypatch.setattr(semi_csp_module, "fit_normal_csp_mcem", fake_inner)
    monkeypatch.setattr(semi_csp_module, "_semiparametric_g_step", fake_g_step)
    result = fit(
        Y,
        model="semi",
        D=2,
        dimension_method="csp",
        t_spike=[0.03, 0.05],
        temp=0.9,
        tau=0.3,
        C=1,
        max_it=1,
        max_outer_iter=2,
        verbose=False,
    )

    assert received_states[0] is None
    assert received_states[1] is returned_states[0]
    assert result.diagnostics["outer_iterations"] == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"D": None}, "D is required"),
        ({"K_list": [6, 2]}, "K_list must be omitted"),
        ({"pen_list": [0.1, 0.1]}, "pen_list.*fixed"),
        ({"t_spike": None}, "t_spike"),
        ({"temp": None}, "temp"),
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
        max_outer_iter=1,
        verbose=False,
    )
    args.update(overrides)
    with pytest.raises(ValueError, match=message):
        fit(Y, **args)
