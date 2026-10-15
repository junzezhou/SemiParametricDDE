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
    assert (
        result.diagnostics["latent_transition"]
        == "systematic_coordinate_gibbs"
    )
    assert result.diagnostics["mc_state_update"] == "last_draw"
    assert (
        result.diagnostics["convergence_metric"]
        == "postprocessed_parameter_rmse"
    )
    assert result.diagnostics["convergence_tol"] == 0.0


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
        ({"family": "Gamma"}, "Normal, Bernoulli, and Poisson.*only"),
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


def test_fit_common_max_iter_alias_and_conflict():
    X, K_list, _, _ = _normal_parameters()
    args = dict(
        method="DDE", D=2, family="Normal", dimension_method="fixed",
        K_list=K_list, pen_list=[0.0, 0.0], tau=0.3, C=1,
        max_iter=1, verbose=False,
    )
    np.random.seed(99)
    result = fit(X, **args)
    legacy_args = {key: value for key, value in args.items() if key != "max_iter"}
    np.random.seed(99)
    legacy = fit(X, **legacy_args, max_it=1)
    assert result.diagnostics["iterations"] == legacy.diagnostics["iterations"]
    for actual, expected in zip(result.B_list, legacy.B_list):
        np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="max_iter conflicts with max_it"):
        fit(X, **args, max_it=2)


def test_fit_copula_csp_derives_widths_from_D():
    X, _, _, _ = _normal_parameters()
    result = fit(
        X, method="CopDDE", D=2, dimension_method="csp",
        t_spike=[0.03, 0.05], temp=0.7, tau=0.3, C=1,
        max_iter=1, optimizer_maxiter=1, random_state=4, verbose=False,
    )
    assert result.diagnostics["original_K_list"] == (6, 2)
    assert result.diagnostics["dimension_method"] == "csp"


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
        csp_backend="mcem",
        random_state=731,
        verbose=False,
    )

    first = fit(X, **args)
    second = fit(X, **args)

    assert first.model == "DDE"
    assert first.diagnostics["family"] == "Normal"
    assert first.diagnostics["dimension_method"] == "csp"
    assert first.diagnostics["backend"] == "mcem"
    assert first.diagnostics["kernel"] == "approximate_mcem_csp"
    assert first.diagnostics["transition_mode"] == "none"
    assert first.diagnostics["latent_transition"] == "legacy_synchronous"
    assert first.diagnostics["mc_state_update"] == "draw_majority"
    assert first.diagnostics["canonicalized_initial_order"] is False
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
    ("backend", "transition_mode"),
    [
        ("mcem", "none"),
        ("mcem_reinitialize", "reinitialize"),
        ("mcem_cap_only", "cap_only"),
    ],
)
def test_fit_dde_normal_csp_routes_three_mcem_backends(backend, transition_mode):
    X, _, _, _ = _normal_parameters()
    result = fit(
        X,
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
        optimizer_maxiter=2,
        mcem_stage_min_iter=1,
        mcem_stage_width_window=1,
        csp_backend=backend,
        random_state=732,
        verbose=False,
    )

    assert result.diagnostics["backend"] == backend
    assert result.diagnostics["transition_mode"] == transition_mode
    assert result.diagnostics["stagewise"] == (backend != "mcem")


def test_fit_dde_normal_csp_rejects_removed_saem_backend():
    X, _, _, _ = _normal_parameters()
    with pytest.raises(ValueError, match="mcem_cap_only"):
        fit(
            X,
            method="DDE",
            D=2,
            family="Normal",
            dimension_method="csp",
            t_spike=[0.03, 0.05],
            temp=0.9,
            tau=0.3,
            C=1,
            tol=0.0,
            max_it=1,
            csp_backend="saem",
            random_state=733,
            verbose=False,
        )


@pytest.mark.parametrize(
    "removed_option",
    ("csp_min_iter", "mcem_stagewise", "mcem_stage_reinitialize_next"),
)
def test_fit_dde_normal_csp_rejects_removed_backend_switches(removed_option):
    X, _, _, _ = _normal_parameters()
    with pytest.raises(TypeError, match="Unexpected Normal DDE argument"):
        fit(
            X,
            method="DDE",
            D=2,
            family="Normal",
            dimension_method="csp",
            t_spike=[0.03, 0.05],
            temp=0.9,
            tau=0.3,
            C=1,
            tol=0.0,
            max_it=1,
            csp_backend="mcem",
            random_state=734,
            verbose=False,
            **{removed_option: True},
        )


def test_cap_only_backend_rejects_spectral_transition_option():
    X, _, _, _ = _normal_parameters()
    with pytest.raises(ValueError, match="mcem_reinitialize"):
        fit(
            X,
            method="DDE",
            D=2,
            family="Normal",
            dimension_method="csp",
            t_spike=[0.03, 0.05],
            temp=0.9,
            tau=0.3,
            C=1,
            tol=0.0,
            max_it=1,
            csp_backend="mcem_cap_only",
            mcem_stage_spectral_gap_threshold=1.5,
            random_state=735,
            verbose=False,
        )


def test_fit_dde_normal_csp_uses_copula_backend_by_default():
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
        max_it=1,
        optimizer_maxiter=1,
        random_state=733,
        verbose=False,
    )
    result = fit(X, **args)

    assert result.diagnostics["backend"] == "copula"
    assert result.diagnostics["kernel"] == "copdde_csp_mean_field_mcem"
    assert result.diagnostics["standardized"] is True
    assert result.diagnostics["initialization"] == "data"
    assert result.diagnostics["initializer"] == "family"
    assert result.diagnostics["mstep_solver"] == "fista_lower"
    assert result.diagnostics["fista_maxiter"] == 300

    legacy = fit(X, csp_backend="mcem", **args)
    assert legacy.diagnostics["backend"] == "mcem"
    assert legacy.diagnostics["kernel"] == "approximate_mcem_csp"
    assert legacy.diagnostics["standardized"] is False


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
