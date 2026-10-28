import numpy as np
import pytest

from dde import (
    auto_grid,
    auto_grid_copula_csp,
    auto_grid_copula_fixed,
    copula_predictive_pmse_score,
    copula_tau_default,
    estimate_K_list_copula,
    estimate_K_list_semidde,
    normal_mixture_loglik_score,
    select_hyperparameters,
    simulate_copula_predictive,
)
from dde.results import DDEResult
from dde.model_selection import _as_candidate_list
from dde.utils.hyperparameter_selection import parametric_mixture_loglik_score
from dde.simulation import generate_X_normal
from dde.simulation import generate_X_Cop_marg_emp_D


def _parameters(J, K_list, seed=123):
    rng = np.random.default_rng(seed)
    B_list = []
    B0 = rng.normal(0.0, 0.3, size=(J, K_list[0] + 1))
    B0[:, 0] = 0.0
    B_list.append(B0)
    for d in range(1, len(K_list)):
        Bd = rng.normal(0.0, 0.35, size=(K_list[d - 1], K_list[d] + 1))
        Bd[:, 0] = 0.0
        B_list.append(Bd)
    prop = np.full(K_list[-1], 0.45)
    gamma = np.ones(J)
    return prop, B_list, gamma


def test_normal_mixture_loglik_score_is_finite():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        20,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=42,
    )
    result = DDEResult(model="copula", prop=prop, B_list=B_list, gamma=gamma)

    score = normal_mixture_loglik_score(sim["X"], result, model="copula")

    assert np.isfinite(score)


def test_normal_mixture_loglik_score_chunk_size_invariant():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        20,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=42,
    )
    result = DDEResult(model="copula", prop=prop, B_list=B_list, gamma=gamma)

    score_small_chunks = normal_mixture_loglik_score(
        sim["X"],
        result,
        model="copula",
        pattern_chunk_size=3,
    )
    score_large_chunk = normal_mixture_loglik_score(
        sim["X"],
        result,
        model="copula",
        pattern_chunk_size=1 << K_list[0],
    )

    assert np.isclose(score_small_chunks, score_large_chunk)


def test_select_hyperparameters_copula_smoke():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        18,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=43,
    )

    result = select_hyperparameters(
        sim["X"],
        D=len(K_list),
        model="copula",
        dimension_method="fixed",
        candidates=[
            {
                "name": "weak_penalty",
                "K_list": K_list,
                "C": 1,
                "tol": 0.0,
                "it": 1,
                "temp": 1.0,
                "pen_list": [0.0, 0.0],
                "tau": 0.0,
                "verbose": False,
                "optimizer_maxiter": 1,
            },
            {
                "name": "strong_penalty",
                "K_list": K_list,
                "C": 1,
                "tol": 0.0,
                "it": 1,
                "temp": 1.0,
                "pen_list": [0.1, 0.1],
                "tau": 0.1,
                "verbose": False,
                "optimizer_maxiter": 1,
            },
        ],
        n_splits=2,
        random_state=1,
    )

    cv_table = result.diagnostics["cv_table"]
    assert result.model == "copula"
    assert len(cv_table) == 2
    assert all(np.isfinite(row["mean_score"]) for row in cv_table)
    assert result.diagnostics["hyperparameter_selection"].best_index in {0, 1}


def test_copula_predictive_pmse_score_is_finite():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        18,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=44,
    )
    result = DDEResult(model="copula", prop=prop, B_list=B_list, gamma=gamma)

    synth = simulate_copula_predictive(sim["X"], result, n_samples=10, random_state=1)
    pmse, values = copula_predictive_pmse_score(
        sim["X"],
        result,
        n_repeats=2,
        random_state=2,
        n_estimators=5,
    )

    assert synth.shape == (10, J)
    assert np.isfinite(pmse)
    assert len(values) == 2
    parallel_pmse, parallel_values = copula_predictive_pmse_score(
        sim["X"], result, n_repeats=2, random_state=2,
        n_estimators=5, n_jobs=2,
    )
    assert parallel_values == values
    assert parallel_pmse == pmse


def test_select_hyperparameters_copula_pmse_fixed_smoke():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        18,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=45,
    )

    result = select_hyperparameters(
        sim["X"],
        D=2,
        model="copula",
        dimension_method="fixed",
        criterion="pmse",
        candidates=[
            {
                "name": "fast_fixed",
                "K_list": K_list,
                "C": 1,
                "tol": 0.0,
                "it": 1,
                "temp": 0.7,
                "pen_list": [0.0, 0.0],
                "tau": copula_tau_default(sim["X"].shape[0]),
                "verbose": False,
                "optimizer_maxiter": 1,
            }
        ],
        n_repeats=2,
        random_state=1,
        pmse_n_estimators=5,
    )

    cv_table = result.diagnostics["cv_table"]
    assert result.model == "copula"
    assert len(cv_table) == 1
    assert cv_table[0]["criterion"] == "pmse"
    assert np.isfinite(cv_table[0]["mean_pmse"])


def test_select_hyperparameters_copula_csp_uses_default_t_spike():
    J = 6
    K_list = [2]
    prop, B_list, gamma = _parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        12,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=46,
    )

    result = select_hyperparameters(
        sim["X"],
        D=1,
        model="copula",
        dimension_method="csp",
        criterion="pmse",
        candidates=[
            {
                "name": "default_csp",
                "K_list": K_list,
                "C": 1,
                "it": 1,
                "temp": 0.7,
                "tau": copula_tau_default(sim["X"].shape[0]),
                "verbose": False,
                "optimizer_maxiter": 1,
            }
        ],
        n_repeats=1,
        random_state=1,
        pmse_n_estimators=5,
    )

    assert result.diagnostics["cv_table"][0]["t_spike"] == [0.03]


def test_auto_grid_uses_dde_penalty_grid():
    grid = auto_grid(256, 3)

    assert len(grid) == 24
    assert all(len(candidate["pen_list"]) == 3 for candidate in grid)
    assert {tuple(candidate["lambda_exp"]) for candidate in grid} == {
        (a, b, c) for a in (2 / 8, 3 / 8)
        for b in (2 / 8, 3 / 8) for c in (2 / 8, 3 / 8)
    }
    assert sum(candidate["tau"] == 0.3 for candidate in grid) == 8


def test_auto_grid_copula_fixed_crosses_layer_penalties():
    grid = auto_grid_copula_fixed(256, 2)

    assert len(grid) == 18
    assert {candidate["temp"] for candidate in grid} == {0.7, 0.9}
    assert all(candidate["tau"] == copula_tau_default(256) for candidate in grid)
    assert all("dimension_method" not in candidate for candidate in grid)


def test_auto_grid_copula_csp_defaults():
    full_d2 = auto_grid_copula_csp(256, 2)
    full_d3 = auto_grid_copula_csp(256, 3)

    assert len(full_d2) == 9
    assert all("t_spike" in candidate for candidate in full_d2)
    assert len(full_d3) == 27
    assert all(candidate["temp"] == 0.9 for candidate in full_d3)
    assert len({tuple(candidate["t_spike"]) for candidate in full_d3}) == 27


def test_auto_grid_copula_csp_full_grid_is_available():
    full = auto_grid_copula_csp(256, 2, strategy="full")

    assert len(full) == 9


def test_candidate_axes_expand_each_layer_and_scalar_axis():
    rows = _as_candidate_list({
        "pen_list": {"per_layer": [[0.1, 0.2], [0.3, 0.4, 0.5]]},
        "tau": [0.2, 0.3], "K_list": [6, 2],
    })
    assert len(rows) == 12
    assert len({tuple(row["pen_list"]) for row in rows}) == 6
    assert all(row["K_list"] == [6, 2] for row in rows)
    assert len(auto_grid_copula_csp(
        256, 2, t_spike_values=[[0.01, 0.02], [0.03, 0.04, 0.05]],
        temp_values=[0.7, 0.9], tau_values=[0.2, 0.3],
    )) == 24
    with pytest.raises(ValueError, match="full Cartesian grid"):
        _as_candidate_list({"strategy": "coordinate", "t_spike_values": [[0.01, 0.02]]})


def test_selection_accepts_one_pass_candidate_generator():
    Y = np.random.default_rng(28).normal(size=(18, 9))
    candidates = (
        {"K_list": [3], "pen_list": [pen], "tau": 0.3}
        for pen in (0.1, 0.2)
    )
    result = select_hyperparameters(
        Y, D=1, method="DDE", family="Normal", criterion="ebic",
        candidates=candidates, base_fit_kwargs={"max_iter": 1, "verbose": False},
        refit=False,
    )
    assert len(result.diagnostics["cv_table"]) == 2


@pytest.mark.parametrize("family,Y,expected", [
    ("normal", np.array([[0.0], [1.0]]), -0.5 * (0.5 + np.log(2 * np.pi))),
    ("bernoulli", np.array([[0.0], [1.0]]), -np.log(2.0)),
    ("poisson", np.array([[0.0], [1.0]]), -1.0),
])
def test_parametric_mixture_scores_match_one_component_likelihood(family, Y, expected):
    result = DDEResult(
        model="DDE", prop=np.array([0.5]), B_list=[np.zeros((1, 2))],
        gamma=np.ones(1) if family == "normal" else None,
    )
    score = parametric_mixture_loglik_score(Y, result, family=family)
    assert np.isclose(score, expected)
    sampled = parametric_mixture_loglik_score(
        Y, result, family=family, max_enum_bits=0, mc_samples=16,
    )
    assert np.isclose(sampled, expected)


def test_normal_mixture_scoring_ignores_missing_observations():
    result = DDEResult(
        model="DDE", prop=np.array([0.5]), B_list=[np.zeros((1, 2))],
        gamma=np.ones(1),
    )
    score = parametric_mixture_loglik_score(
        np.array([[np.nan], [1.0]]), result, family="normal",
    )
    assert np.isclose(score, -0.25 * (1.0 + np.log(2.0 * np.pi)))


@pytest.mark.parametrize("method,path,family", [
    ("DDE", "fixed", "Normal"), ("DDE", "fixed", "Bernoulli"),
    ("DDE", "fixed", "Poisson"), ("DDE", "csp", "Normal"),
    ("DDE", "csp", "Bernoulli"), ("DDE", "csp", "Poisson"),
    ("SemiDDE", "fixed", None), ("SemiDDE", "csp", None),
    ("CopDDE", "fixed", None), ("CopDDE", "csp", None),
])
def test_selection_connects_every_supported_method_and_refits(method, path, family):
    rng = np.random.default_rng(27)
    Y = rng.normal(size=(22, 9))
    if family == "Bernoulli":
        Y = rng.binomial(1, 0.4, size=Y.shape)
    elif family == "Poisson":
        Y = rng.poisson(1.5, size=Y.shape)
    candidates = (
        [{"K_list": [3], "pen_list": [0.1], "tau": 0.3}]
        if path == "fixed" else
        [{"t_spike": [0.03], "temp": 0.7, "tau": 0.3}]
    )
    if method == "CopDDE":
        candidates[0]["K_list"] = [3]
        if path == "fixed":
            candidates[0]["temp"] = 0.7
    base = {"max_iter": 1, "verbose": False}
    if method == "SemiDDE" and path == "fixed":
        base["max_outer_iter"] = 1  # the default (copula) CSP path has no outer loop
    if method == "CopDDE":
        base["optimizer_maxiter"] = 1
    result = select_hyperparameters(
        Y, D=1, method=method, family=family, dimension_method=path,
        candidates=candidates, criterion="pmse" if method == "CopDDE" else "ebic",
        base_fit_kwargs=base, n_repeats=1, pmse_n_estimators=5,
        random_state=4,
    )
    assert len(result.diagnostics["cv_table"]) == 1
    assert np.isfinite(result.diagnostics["cv_table"][0]["mean_score"])
    assert result.diagnostics["dimension_method"] == path
    assert len(result.K_list) == 1


@pytest.mark.parametrize("method", ["DDE", "SemiDDE", "CopDDE"])
def test_three_layer_default_csp_search_evaluates_all_spike_combinations(method):
    Y = np.random.default_rng(13).normal(size=(36, 27))
    base = {"max_iter": 1, "verbose": False}
    if method == "CopDDE":
        base["optimizer_maxiter"] = 1
    result = select_hyperparameters(
        Y, D=3, method=method, family="Normal" if method == "DDE" else None,
        dimension_method="csp", criterion="pmse" if method == "CopDDE" else "ebic",
        base_fit_kwargs=base, n_repeats=1, pmse_n_estimators=5,
        random_state=11, refit=False,
    )
    rows = result.diagnostics["cv_table"]
    # DDE and SemiDDE also tune the annealing temperature over {0.7, 0.9}.
    expected = 27 if method == "CopDDE" else 54
    assert len(rows) == expected
    assert len({tuple(row["t_spike"]) for row in rows}) == 27
    assert {row["temp"] for row in rows} == ({0.9} if method == "CopDDE" else {0.7, 0.9})
    assert all(np.isfinite(row["mean_score"]) for row in rows)
    if method != "CopDDE":
        assert all(len(row["fitted_K"]) == 3 for row in rows)


def test_estimate_K_list_copula_uses_identifiable_maximal_widths():
    Y = np.zeros((10, 81))

    assert estimate_K_list_copula(Y, D=3) == [27, 9, 3]
    assert estimate_K_list_copula(Y, D=4) == [27, 9, 3, 1]


def test_estimate_K_list_copula_rejects_too_deep_model():
    Y = np.zeros((10, 8))

    try:
        estimate_K_list_copula(Y, D=2)
    except ValueError as exc:
        assert "Data support D at most 1" in str(exc)
    else:
        raise AssertionError("estimate_K_list_copula should reject unsupported D.")


def test_estimate_K_list_semidde_smoke():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(J, K_list)
    X, _ = generate_X_normal(30, prop, B_list, gamma)

    estimated = estimate_K_list_semidde(X, D=2)

    assert len(estimated) == 2
    assert all(k >= 1 for k in estimated)


def test_estimate_K_list_semidde_allows_single_node_deepest_layer():
    rng = np.random.default_rng(123)
    Y = rng.normal(size=(24, 9))

    estimated, details = estimate_K_list_semidde(
        Y,
        D=2,
        k1_grid=[3],
        spectral_ratio_tolerance=0.0,
        return_details=True,
    )

    assert estimated == [3, 1]
    assert details[1]["grid"] == [1]


def test_select_hyperparameters_semidde_defaults_smoke():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(J, K_list)
    X, _ = generate_X_normal(18, prop, B_list, gamma)

    result = select_hyperparameters(
        X,
        D=2,
        model="semi",
        candidates=[
            {
                "name": "default_penalty",
                "max_outer_iter": 1,
                "max_inner_iter": 1,
            },
            {
                "name": "damped",
                "g_update_rho": 0.5,
                "max_outer_iter": 1,
                "max_inner_iter": 1,
            },
        ],
        fixed_hp={"K_list": K_list},
        n_splits=2,
        random_state=1,
    )

    cv_table = result.diagnostics["cv_table"]
    assert result.model == "semi"
    assert len(cv_table) == 2
    assert all(row["K_list"] == K_list for row in cv_table)
    assert all("tau" in row for row in cv_table)
    assert {row["tol"] for row in cv_table} == {K_list[-1] / 2}
    assert len(result.diagnostics["timing_table"]) == 4
    assert all("fit_seconds" in row for row in result.diagnostics["timing_table"])
    assert all("score_seconds" in row for row in cv_table)


def test_select_hyperparameters_semidde_default_grid_has_full_penalty_product():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(J, K_list)
    X, _ = generate_X_normal(18, prop, B_list, gamma)

    result = select_hyperparameters(
        X,
        D=2,
        model="semi",
        fixed_hp={"K_list": K_list},
        base_fit_kwargs={"max_outer_iter": 1, "max_inner_iter": 1, "tol": 0.0},
        n_splits=2,
        random_state=1,
        refit=False,
    )

    cv_table = result.diagnostics["cv_table"]
    assert len(cv_table) == 12
    assert {tuple(row["lambda_exp"]) for row in cv_table} == {
        (a, b) for a in (2 / 8, 3 / 8) for b in (2 / 8, 3 / 8)
    }
    assert sum(row["tau"] == 0.3 for row in cv_table) == 4
    assert {row["tol"] for row in cv_table} == {0.0}
    assert {row["max_enum_bits"] for row in cv_table} == {12}


def test_select_hyperparameters_semidde_csp_smoke():
    Y = np.random.default_rng(34).normal(size=(20, 9))
    result = select_hyperparameters(
        Y, D=1, model="semi", dimension_method="csp", criterion="ebic",
        candidates={"t_spike": {"per_layer": [[0.02, 0.03]]}, "temp": [0.7], "tau": [0.3]},
        base_fit_kwargs={"max_iter": 1, "verbose": False},
        random_state=3,
    )
    assert len(result.diagnostics["cv_table"]) == 2
    assert result.diagnostics["dimension_method"] == "csp"


def test_select_hyperparameters_copula_csp_rejects_pen_list_candidate():
    Y = np.zeros((12, 9))

    try:
        select_hyperparameters(
            Y,
            D=2,
            model="copula",
            dimension_method="csp",
            candidates=[{"K_list": [6, 2], "pen_list": [1.0, 1.0]}],
        )
    except ValueError as exc:
        assert "pen_list is only valid for CopDDE dimension_method='fixed'" in str(exc)
    else:
        raise AssertionError("CopDDE CSP path should reject pen_list candidates.")


_TIMING_FIELDS = {"elapsed_seconds", "init_seconds", "fit_seconds", "score_seconds"}


def _selection_signature(result):
    rows = [
        {key: value for key, value in row.items() if key not in _TIMING_FIELDS}
        for row in result.diagnostics["cv_table"]
    ]
    arrays = [np.asarray(result.prop)] + [np.asarray(B) for B in result.B_list]
    arrays += [np.asarray(A) for A in result.A_list]
    if result.gamma is not None:
        arrays.append(np.asarray(result.gamma))
    return rows, result.diagnostics["fold_scores"], list(result.K_list), arrays


def _assert_same_selection(first, second):
    rows_a, folds_a, K_a, arrays_a = _selection_signature(first)
    rows_b, folds_b, K_b, arrays_b = _selection_signature(second)
    assert rows_a == rows_b
    assert folds_a == folds_b
    assert K_a == K_b
    assert len(arrays_a) == len(arrays_b)
    for a, b in zip(arrays_a, arrays_b):
        np.testing.assert_array_equal(a, b)
    assert len(first.diagnostics["timing_table"]) == len(second.diagnostics["timing_table"])


def _small_selection_data(kind):
    rng = np.random.default_rng(91)
    X = rng.normal(size=(60, 18))
    if kind == "binary":
        return (X > 0).astype(float)
    return X


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        ("normal", dict(method="DDE", family="Normal")),  # draws from the global generator
        ("binary", dict(method="DDE", family="Bernoulli")),
        ("normal", dict(model="SemiDDE")),
    ],
)
def test_parallel_candidates_match_serial_bitwise_ebic(kind, spec):
    Y = _small_selection_data(kind)
    args = dict(
        D=2, dimension_method="csp", criterion="ebic", random_state=17,
        candidates={"t_spike": {"per_layer": [[0.01, 0.05], [0.03, 0.1]]}, "temp": [0.9], "tau": [0.3]},
        base_fit_kwargs={"max_iter": 3, "verbose": False}, **spec,
    )
    np.random.seed(1)
    serial = select_hyperparameters(Y, n_jobs=1, **args)
    np.random.seed(2)  # a different caller state must not matter
    parallel = select_hyperparameters(Y, n_jobs=2, **args)
    assert len(serial.diagnostics["cv_table"]) == 4
    _assert_same_selection(serial, parallel)


def test_parallel_candidates_match_serial_bitwise_cross_validated_bic():
    K_list = [6, 2]
    prop, B_list, gamma = _parameters(18, K_list)
    X, _ = generate_X_normal(40, prop, B_list, gamma)
    args = dict(
        D=2, method="DDE", family="Normal", dimension_method="fixed", criterion="bic",
        n_splits=2, random_state=5, fixed_hp={"K_list": K_list},
        candidates=[{"pen_list": [0.5, 0.5], "tau": 0.3}, {"pen_list": [2.0, 2.0], "tau": 0.3}],
        base_fit_kwargs={"max_iter": 2, "verbose": False},
    )
    np.random.seed(3)
    serial = select_hyperparameters(X, n_jobs=1, **args)
    np.random.seed(4)
    parallel = select_hyperparameters(X, n_jobs=2, **args)
    assert len(serial.diagnostics["fold_scores"][0]) == 2
    _assert_same_selection(serial, parallel)


def test_selection_rejects_nonpositive_n_jobs():
    with pytest.raises(ValueError, match="n_jobs"):
        select_hyperparameters(_small_selection_data("normal"), D=2, method="DDE", family="Normal",
                               dimension_method="csp", n_jobs=0)
