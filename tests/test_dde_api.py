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
        model="DDE",
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
    assert result.raw["X"] is result.transformed
    assert result.diagnostics["active_by_layer"] == [
        int((B[:, 1:] != 0).any(axis=0).sum()) for B in result.B_list
    ]
    assert result.diagnostics["n_active_edges"] == [
        int((B[:, 1:] != 0).sum()) for B in result.B_list
    ]
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
        == "relative_parameter_change"
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
        model="DDE",
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
        model="DDE",
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
    with pytest.warns(DeprecationWarning, match="'method' argument is deprecated"):
        with pytest.raises(ValueError, match="different DDE formulations"):
            fit(X, method="DDE", model="SemiDDE")


def test_fit_method_is_a_deprecated_spelling_of_model():
    X, K_list, _, _ = _normal_parameters()
    args = dict(
        D=2, family="Normal", dimension_method="fixed", K_list=K_list,
        pen_list=[0.0, 0.0], tau=0.3, max_it=1, verbose=False,
    )
    np.random.seed(5)
    current = fit(X, model="DDE", **args)
    np.random.seed(5)
    with pytest.warns(DeprecationWarning, match="use 'model'"):
        legacy = fit(X, method="DDE", **args)
    assert legacy.model == current.model == "DDE"
    for actual, expected in zip(legacy.B_list, current.B_list):
        np.testing.assert_allclose(actual, expected)


def test_fit_warns_when_dimension_method_is_omitted():
    X, K_list, _, _ = _normal_parameters()
    with pytest.warns(FutureWarning, match="dimension_method was not given"):
        result = fit(
            X, model="DDE", D=2, family="Normal", K_list=K_list,
            pen_list=[0.0, 0.0], tau=0.3, max_it=1, verbose=False,
        )
    assert result.diagnostics["dimension_method"] == "fixed"


def test_fit_common_max_iter_alias_and_conflict():
    X, K_list, _, _ = _normal_parameters()
    args = dict(
        model="DDE", D=2, family="Normal", dimension_method="fixed",
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
    with pytest.raises(ValueError, match="Conflicting iteration limits: max_it, max_iter"):
        fit(X, **args, max_it=2)


def test_fit_copula_csp_derives_widths_from_D():
    X, _, _, _ = _normal_parameters()
    result = fit(
        X, model="CopDDE", D=2, dimension_method="csp",
        t_spike=[0.03, 0.05], temp=0.7, tau=0.3, C=1,
        max_iter=1, optimizer_maxiter=1, random_state=4, verbose=False,
    )
    assert result.diagnostics["original_K_list"] == (6, 2)
    assert result.diagnostics["dimension_method"] == "csp"


def test_fit_dde_normal_csp_public_smoke_and_is_reproducible():
    X, _, _, _ = _normal_parameters()
    args = dict(
        model="DDE",
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
    assert first.diagnostics["backend"] == "copula"
    assert first.diagnostics["kernel"] == "copdde_csp"
    assert first.diagnostics["latent_transition"] == "synchronous_mean_field"
    assert first.diagnostics["mc_state_update"] == "draw_majority"
    assert first.diagnostics["K_max_list"] == (6, 2)
    assert first.diagnostics["threshold_rule"] == "first_two"
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


def _binary_matrix(seed=732, N=60, J=18):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(N, J)).astype(float)


@pytest.mark.parametrize("family", ["Normal", "Bernoulli", "Poisson"])
@pytest.mark.parametrize("backend", ["mcem", "mcem_reinitialize", "mcem_cap_only"])
def test_fit_dde_csp_rejects_removed_mcem_backends(backend, family):
    X = _normal_parameters()[0] if family == "Normal" else np.round(np.abs(_normal_parameters()[0]))
    if family == "Bernoulli":
        X = (X > 0).astype(float)
    with pytest.raises(ValueError, match="were removed"):
        fit(
            X,
            model="DDE",
            D=2,
            family=family,
            dimension_method="csp",
            t_spike=[0.03, 0.05],
            temp=0.9,
            tau=0.3,
            C=1,
            tol=0.0,
            max_it=1,
            csp_backend=backend,
            random_state=733,
            verbose=False,
        )


def test_fit_dde_normal_csp_rejects_removed_saem_backend():
    X, _, _, _ = _normal_parameters()
    with pytest.raises(ValueError, match="not available"):
        fit(
            X,
            model="DDE",
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
    ("csp_min_iter", "mcem_stagewise", "mcem_stage_reinitialize_next", "mcem_shrinkage_burn"),
)
def test_fit_dde_normal_csp_rejects_removed_backend_switches(removed_option):
    X, _, _, _ = _normal_parameters()
    with pytest.raises(TypeError, match="Unexpected Normal DDE argument"):
        fit(
            X,
            model="DDE",
            D=2,
            family="Normal",
            dimension_method="csp",
            t_spike=[0.03, 0.05],
            temp=0.9,
            tau=0.3,
            C=1,
            tol=0.0,
            max_it=1,
            random_state=734,
            verbose=False,
            **{removed_option: True},
        )


def test_fit_dde_normal_csp_uses_copula_backend_by_default():
    X, _, _, _ = _normal_parameters()
    args = dict(
        model="DDE",
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
    assert result.diagnostics["kernel"] == "copdde_csp"
    assert result.diagnostics["standardized"] is True
    assert result.diagnostics["initialization"] == "data"
    assert result.diagnostics["initializer"] == "family"
    assert result.diagnostics["mstep_solver"] == "fista_lower"
    assert result.diagnostics["fista_maxiter"] == 300

    with pytest.raises(ValueError, match="were removed"):
        fit(X, csp_backend="mcem", **args)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"K_list": [7, 2]}, "J >= 3"),
        ({"pen_list": [0.1, 0.1]}, "pen_list.*fixed"),
    ],
)
def test_fit_dde_normal_csp_rejects_path_incompatible_arguments(overrides, message):
    X, _, _, _ = _normal_parameters()
    args = dict(
        model="DDE",
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


def test_csp_stops_on_windowed_mean_relative_change():
    rng = np.random.default_rng(4)
    A = rng.binomial(1, 0.5, size=(300, 3))
    X = A @ rng.normal(2.0, 0.3, size=(3, 18)) + rng.normal(size=(300, 18))
    common = dict(model="DDE", family="Normal", D=2, dimension_method="csp", t_spike=[0.05, 0.1],
                  temp=0.8, tau=0.3, C=1, max_it=30, random_state=0, verbose=False)
    loose = fit(X, tol=10.0, **common)
    history = np.asarray(loose.diagnostics["relative_history"], dtype=float)
    window = loose.diagnostics["convergence_window"]
    # Every layer's mean change over the window is far below 10, so the fit
    # stops as soon as the window is full.
    assert loose.diagnostics["stopped_by_convergence"]
    assert loose.diagnostics["iterations"] == window == history.shape[1]
    strict = fit(X, tol=0.0, **common)
    assert not strict.diagnostics["stopped_by_convergence"]
    assert strict.diagnostics["iterations"] == 30


def test_fit_fills_the_default_hyperparameters():
    from dde.utils.hyperparameter_selection import copula_tau_default, default_fit_hyperparameters

    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 18))
    np.random.seed(0)
    fixed = fit(X, model="DDE", family="Normal", D=2, dimension_method="fixed", K_list=[6, 2],
                max_it=2, verbose=False)
    assert fixed.diagnostics["default_arguments"] == ["pen_list", "tau"]
    assert fixed.diagnostics["fit_arguments"] == {"pen_list": [4.0, 4.0], "tau": 0.3}
    np.random.seed(0)
    csp = fit(rng.normal(size=(256, 27)), model="DDE", family="Normal", D=3, dimension_method="csp",
              K_list=[9, 3, 1], tau=0.2, max_it=2, verbose=False)
    assert csp.diagnostics["default_arguments"] == ["t_spike", "temp"]
    assert csp.diagnostics["fit_arguments"]["t_spike"] == [0.01, 0.03, 0.005]
    assert csp.diagnostics["fit_arguments"]["temp"] == 0.9
    assert csp.diagnostics["fit_arguments"]["tau"] == 0.2
    copula = default_fit_hyperparameters("copula", "fixed", 1000, 2)
    assert copula["tau"] == copula_tau_default(1000) and copula["tol"] == 0.005
    assert default_fit_hyperparameters("semi", "csp", 1000, 2)["temp"] == 0.7
    # the low-level kernel entry point still needs every hyperparameter
    from dde.algorithms.dde import fit_parametric_dde

    with pytest.raises(ValueError, match="requires t_spike and temp"):
        fit_parametric_dde(X, family="normal", D=2, dimension_method="csp", tau=0.3, verbose=False)


def test_np_random_seed_reproduces_copdde_and_fixed_paths_reject_random_state():
    rng = np.random.default_rng(1)
    Y = rng.poisson(2.0, size=(150, 12))
    fits = []
    for _ in range(2):
        np.random.seed(5)
        fits.append(fit(Y, model="CopDDE", dimension_method="fixed", K_list=[4, 1], max_it=2, verbose=False))
    for B, B_again in zip(fits[0].B_list, fits[1].B_list):
        np.testing.assert_array_equal(B, B_again)
    for model in ("DDE", "SemiDDE"):
        with pytest.raises(ValueError, match="np.random.seed"):
            fit(np.exp(rng.normal(size=(150, 12))), model=model, family="Normal", D=2,
                dimension_method="fixed", K_list=[4, 1], random_state=3, verbose=False)
