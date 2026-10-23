"""Tests for the Bernoulli observed layer of the copula CSP kernel."""

import numpy as np
import pytest
from scipy.special import expit

from dde import fit
from dde.algorithms.normal_csp import fit_bernoulli_csp, fit_parametric_csp_copula


def _simulate(seed, N=400, J=27, K_true=(6, 2)):
    rng = np.random.default_rng(seed)
    B1 = np.zeros((J, K_true[0]))
    for k in range(K_true[0]):
        B1[k, k] = 4.0
        B1[K_true[0] + k, k] = 4.0
        B1[2 * K_true[0] + k, k] = 4.0
    B2 = np.zeros((K_true[0], K_true[1]))
    B2[:3, 0] = 4.0
    B2[3:, 1] = 4.0
    A2 = rng.binomial(1, 0.5, (N, K_true[1]))
    A1 = rng.binomial(1, expit(-2.0 + A2 @ B2.T))
    X = rng.binomial(1, expit(-2.0 + A1 @ B1.T))
    return X.astype(float), (A1, A2)


def _inputs(seed=11, N=50, J=18, widths=(6, 2)):
    rng = np.random.default_rng(seed)
    return {
        "X": rng.integers(0, 2, size=(N, J)).astype(float),
        "prop": np.full(widths[-1], 0.5),
        "B_list": [
            rng.normal(scale=0.3, size=(J, widths[0] + 1)),
            rng.normal(scale=0.3, size=(widths[0], widths[1] + 1)),
        ],
        "A_list": [rng.integers(0, 2, size=(N, w)).astype(float) for w in widths],
    }


def _copy(values):
    return [np.asarray(v).copy() for v in values]


@pytest.mark.parametrize("solver", ["lbfgsb", "fista_lower", "fista"])
def test_bernoulli_kernel_runs_with_each_solver(solver):
    inputs = _inputs()
    raw = fit_bernoulli_csp(
        inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        C=1, it=3, t_spike=[0.03, 0.05], temp=0.9, tau=0.3, verbose=False,
        mstep_solver=solver, random_state=2,
    )
    assert raw["family"] == "Bernoulli"
    assert raw["gamma"] is None
    assert raw["iterations"] == 3
    assert all(np.all(np.isfinite(B)) for B in raw["B_list"])
    assert raw["B_list"][0].shape == (18, 7)
    # Bernoulli bounds: intercept within [-5, 5], slopes within [-2.5, 5].
    assert np.all(np.abs(raw["B_list"][0][:, 0]) <= 5.0 + 1e-9)
    assert np.all(raw["B_list"][0][:, 1:] >= -2.5 - 1e-9)


def test_bernoulli_kernel_rejects_gamma_and_normal_only_options():
    inputs = _inputs()
    base = dict(C=1, it=1, t_spike=0.03, temp=0.9, tau=0.3, verbose=False)
    with pytest.raises(ValueError, match="no dispersion"):
        fit_parametric_csp_copula(
            inputs["X"], inputs["prop"], np.ones(18), _copy(inputs["B_list"]),
            _copy(inputs["A_list"]), family="bernoulli", **base,
        )
    with pytest.raises(ValueError, match="Normal family only"):
        fit_bernoulli_csp(
            inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
            observed_noise_scale=0.5, **base,
        )


def test_bernoulli_gibbs_transition_runs():
    inputs = _inputs()
    raw = fit_bernoulli_csp(
        inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        C=2, it=2, t_spike=0.03, temp=0.9, tau=0.3, verbose=False,
        latent_transition="systematic_coordinate_gibbs", random_state=3,
    )
    assert raw["latent_transition"] == "systematic_coordinate_gibbs"
    assert all(A.shape[0] == 50 for A in raw["A_list"])


def test_fit_routes_bernoulli_to_copula_backend_explicitly():
    X, _ = _simulate(5)
    result = fit(
        X, method="DDE", family="Bernoulli", D=2, dimension_method="csp",
        t_spike=[0.03, 0.05], temp=0.9, tau=0.3, C=1, max_iter=30,
        random_state=7, verbose=False, csp_backend="copula",
    )
    d = result.diagnostics
    assert d["backend"] == "copula"
    assert d["family"] == "Bernoulli"
    assert d["standardized"] is False
    assert d["initializer"] == "family"
    assert d["mstep_solver"] == "fista_lower"
    assert d["latent_transition"] == "systematic_coordinate_gibbs"
    assert d["penalty_scaling_power"] == (0.5, 0.5)
    assert d["shrinkage_burn"] == 0  # D=2: no automatic burn-in
    assert result.gamma is None
    assert tuple(d["K_max_list"]) == (9, 3)
    K1, K2 = result.K_list
    assert 1 <= K1 <= 9 and 1 <= K2 <= 3
    assert result.B_list[0].shape == (27, K1 + 1)
    default = fit(
        X, method="DDE", family="Bernoulli", D=2, dimension_method="csp",
        t_spike=[0.03, 0.05], temp=0.9, tau=0.3, C=1, max_iter=1,
        random_state=7, verbose=False,
    )
    assert default.diagnostics["backend"] == "mcem"


def test_bernoulli_copula_backend_rejects_standardize():
    X, _ = _simulate(6, N=120)
    with pytest.raises(ValueError, match="standardize"):
        fit(
            X, method="DDE", family="Bernoulli", D=2, dimension_method="csp",
            t_spike=0.03, temp=0.9, tau=0.3, C=1, max_iter=1, verbose=False,
            csp_backend="copula", standardize=True,
        )


def test_shrinkage_burn_holds_every_column_in_the_slab():
    inputs = _inputs()
    raw = fit_bernoulli_csp(
        inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        C=1, it=4, t_spike=[0.03, 0.05], temp=0.9, tau=0.3, verbose=False,
        latent_transition="systematic_coordinate_gibbs", shrinkage_burn=2,
        keep_trace=True, random_state=4,
    )
    assert raw["shrinkage_burn"] == 2
    assert raw["iterations"] == 4
    trace = raw["trace"]
    for entry in trace[:2]:
        assert entry["shrinkage_active"] is False
        assert entry["will_stop"] is False
        assert tuple(entry["num_active"]) == (6, 2)
        assert np.array_equal(entry["allocation"][0], np.full(6, 7))
        assert np.array_equal(entry["allocation"][1], np.full(2, 3))
        # slab scales are untouched (unit) while the CUSP update is off
        assert np.allclose(entry["slab_scale"][0], 1.0)
    assert trace[2]["shrinkage_active"] is True
    # the strength ordering ran once at the end of the burn-in
    assert raw["reorder_after_shrinkage_burn"] is True
    assert len(raw["burn_permutations"]) == 2
    for permutation, K in zip(raw["burn_permutations"], (6, 2)):
        assert sorted(permutation.tolist()) == list(range(K))
    assert all(np.all(np.isfinite(B)) for B in raw["B_list"])


def test_shrinkage_burn_without_reordering_and_validation():
    inputs = _inputs()
    base = dict(C=1, t_spike=0.03, temp=0.9, tau=0.3, verbose=False, random_state=4)
    raw = fit_bernoulli_csp(
        inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        it=3, shrinkage_burn=1, reorder_after_shrinkage_burn=False, **base,
    )
    assert raw["burn_permutations"] is None
    assert raw["reorder_after_shrinkage_burn"] is False
    unchanged = fit_bernoulli_csp(
        inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        it=3, **base,
    )
    assert unchanged["shrinkage_burn"] == 0
    assert unchanged["burn_permutations"] is None
    with pytest.raises(ValueError, match="nonnegative"):
        fit_bernoulli_csp(
            inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
            it=3, shrinkage_burn=-1, **base,
        )
    with pytest.raises(ValueError, match="smaller than it"):
        fit_bernoulli_csp(
            inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
            it=3, shrinkage_burn=3, **base,
        )


def test_shrinkage_burn_reordering_sorts_columns_by_strength():
    inputs = _inputs()
    raw = fit_bernoulli_csp(
        inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        C=1, it=2, t_spike=0.03, temp=0.9, tau=0.3, verbose=False, random_state=4,
        latent_transition="systematic_coordinate_gibbs", shrinkage_burn=1,
        keep_trace=True,
    )
    # The reordering runs at the start of iteration 2 on the coefficients
    # produced by the burn-in iteration, whose column L1 norms the trace
    # records.
    for d, K in enumerate((6, 2)):
        strength = raw["trace"][0]["updated_column_l1"][d]
        expected = np.argsort(-strength, kind="stable")
        assert np.array_equal(raw["burn_permutations"][d], expected)
    assert not np.array_equal(raw["burn_permutations"][0], np.arange(6))


def test_fit_passes_shrinkage_burn_to_copula_backend():
    X, _ = _simulate(5, N=200)
    result = fit(
        X, method="DDE", family="Bernoulli", D=2, dimension_method="csp",
        t_spike=[0.03, 0.05], temp=0.9, tau=0.3, C=1, max_iter=6,
        random_state=7, verbose=False, csp_backend="copula",
        csp_shrinkage_burn=3, keep_trace=True,
    )
    d = result.diagnostics
    assert d["shrinkage_burn"] == 3
    assert d["reorder_after_shrinkage_burn"] is True
    assert len(d["burn_permutations"]) == 2
    assert all(tuple(e["num_active"]) == (9, 3) for e in d["trace"][:3])
    assert d["iterations"] == 6


def test_bernoulli_copula_defaults_burn_in_for_three_layers():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, size=(120, 27)).astype(float)
    common = dict(
        method="DDE", family="Bernoulli", D=3, dimension_method="csp",
        t_spike=[0.01, 0.03, 0.005], temp=0.9, tau=0.3, C=1, random_state=1,
        verbose=False, csp_backend="copula", keep_trace=True,
    )
    auto = fit(X, max_iter=20, **common)
    assert auto.diagnostics["shrinkage_burn"] == 10
    assert auto.diagnostics["penalty_scaling_power"] == (0.5, 0.5, 0.5)
    assert all(
        tuple(e["num_active"]) == (9, 3, 1) for e in auto.diagnostics["trace"][:10]
    )
    short = fit(X, max_iter=12, **common)
    assert short.diagnostics["shrinkage_burn"] == 0
    explicit = fit(X, max_iter=12, csp_shrinkage_burn=4, **common)
    assert explicit.diagnostics["shrinkage_burn"] == 4


def test_sequential_transition_matches_synchronous_when_layer_one_is_unchanged():
    # Near-deterministic observed layer started at the true layer-1 state: the
    # fresh layer-1 draw reproduces the previous state, so the layer-2
    # probabilities (and with the same stream, every draw) coincide.
    rng = np.random.default_rng(5)
    N, widths = 60, (6, 2)
    A1 = rng.integers(0, 2, size=(N, widths[0])).astype(float)
    B1 = np.zeros((18, widths[0] + 1))
    B1[:, 0] = -15.0
    for k in range(widths[0]):
        B1[[k, k + 6, k + 12], k + 1] = 30.0
    X = (B1[:, 0] + A1 @ B1[:, 1:].T > 0).astype(float)
    B2 = rng.normal(scale=1.0, size=(widths[0], widths[1] + 1))
    A2 = rng.integers(0, 2, size=(N, widths[1])).astype(float)
    raws = {}
    for transition in ("synchronous_mean_field", "sequential_mean_field"):
        raws[transition] = fit_bernoulli_csp(
            X, np.full(widths[1], 0.5), [B1.copy(), B2.copy()], [A1.copy(), A2.copy()],
            C=1, it=1, t_spike=[0.03, 0.05], temp=0.9, tau=0.3, verbose=False,
            latent_transition=transition, random_state=3,
        )
    sync, seq = raws["synchronous_mean_field"], raws["sequential_mean_field"]
    np.testing.assert_array_equal(seq["A_list"][0], A1.astype(np.uint8))
    for d in range(2):
        np.testing.assert_array_equal(seq["A_list"][d], sync["A_list"][d])
        np.testing.assert_allclose(seq["B_list"][d], sync["B_list"][d])
    np.testing.assert_allclose(seq["prop"], sync["prop"])
    assert seq["latent_transition"] == "sequential_mean_field"


def test_sequential_transition_differs_from_synchronous_on_uncertain_layers():
    inputs = _inputs()
    raws = {}
    for transition in ("synchronous_mean_field", "sequential_mean_field"):
        raws[transition] = fit_bernoulli_csp(
            inputs["X"], inputs["prop"], _copy(inputs["B_list"]), _copy(inputs["A_list"]),
            C=1, it=2, t_spike=[0.03, 0.05], temp=0.9, tau=0.3, verbose=False,
            latent_transition=transition, random_state=4,
        )
    # Same random stream and same first-layer probabilities in the first sweep.
    assert raws["sequential_mean_field"]["iterations"] == 2
    assert not all(
        np.array_equal(a, b)
        for a, b in zip(raws["sequential_mean_field"]["A_list"], raws["synchronous_mean_field"]["A_list"])
    )


def test_threshold_rule_all_thresholds_the_deepest_layer():
    rng = np.random.default_rng(13)
    N, J, widths = 80, 18, (6, 3, 2)
    inputs = {
        "X": rng.integers(0, 2, size=(N, J)).astype(float),
        "B_list": [
            rng.normal(scale=0.3, size=(J, widths[0] + 1)),
            rng.normal(scale=0.3, size=(widths[0], widths[1] + 1)),
            rng.normal(scale=0.3, size=(widths[1], widths[2] + 1)),
        ],
        "A_list": [rng.integers(0, 2, size=(N, w)).astype(float) for w in widths],
    }
    tops = {}
    for rule in ("first_two", "all"):
        raw = fit_bernoulli_csp(
            inputs["X"], np.full(widths[-1], 0.5), _copy(inputs["B_list"]), _copy(inputs["A_list"]),
            C=1, it=1, t_spike=[0.03, 0.05, 0.001], temp=0.9, tau=0.3, verbose=False,
            threshold_rule=rule, random_state=6,
        )
        assert raw["threshold_rule"] == rule
        tops[rule] = raw["B_list"][2][:, 1:]
    small = (tops["first_two"] != 0) & (np.abs(tops["first_two"]) < 0.3)
    assert small.any()
    assert np.all((tops["all"] == 0) | (np.abs(tops["all"]) >= 0.3))
