"""Tests for the CopDDE-derived Normal CSP kernel."""

from importlib import import_module

import numpy as np
import pytest
from scipy.special import expit

from dde import fit
from dde.algorithms.normal_csp_copula import fit_normal_csp_copula


copula_csp = import_module("dde.algorithms.get_SAEM_RL_CSP_D")


def _inputs(seed=3319, N=40, J=18, widths=(6, 2)):
    rng = np.random.default_rng(seed)
    return {
        "X": rng.normal(size=(N, J)),
        "prop": np.full(widths[-1], 0.5),
        "gamma": np.ones(J),
        "B_list": [
            rng.normal(scale=0.3, size=(J, widths[0] + 1)),
            rng.normal(scale=0.3, size=(widths[0], widths[1] + 1)),
        ],
        "A_list": [
            rng.integers(0, 2, size=(N, width)).astype(float) for width in widths
        ],
    }


def _copy(values):
    return [np.asarray(value).copy() for value in values]


def test_kernel_equals_copdde_kernel_without_rank_resampling(monkeypatch):
    """With Z fixed at X, the Normal kernel is the CopDDE kernel exactly."""
    inputs = _inputs()
    monkeypatch.setattr(
        copula_csp, "_sample_rank_likelihood_Z", lambda *args, **kwargs: None
    )
    common = dict(C=2, t_spike=[0.02, 0.03], tau=0.3)
    # The package default is the hybrid M-step; the CopDDE kernel is L-BFGS-B.

    np.random.seed(1817)
    reference = copula_csp.get_SAEM_RL_CSP_D(
        X=inputs["X"],
        Z=inputs["X"].copy(),
        R=np.zeros_like(inputs["X"]),
        Rlevels=np.zeros(inputs["X"].shape[1], dtype=int),
        prop_in=inputs["prop"],
        gamma_in=inputs["gamma"],
        B_list_in=_copy(inputs["B_list"]),
        A_list_in=_copy(inputs["A_list"]),
        it=6,
        temp=0.9,
        optimizer_maxiter=15,
        verbose=False,
        **common,
    )

    np.random.seed(1817)
    candidate = fit_normal_csp_copula(
        inputs["X"],
        inputs["prop"],
        inputs["gamma"],
        _copy(inputs["B_list"]),
        _copy(inputs["A_list"]),
        it=6,
        temp=0.9,
        optimizer_maxiter=15,
        mstep_solver="lbfgsb",
        penalty_scaling_power=0.0,
        random_state=None,
        verbose=False,
        **common,
    )

    assert candidate["iterations"] == reference["it_num"]
    np.testing.assert_array_equal(candidate["num_active"], reference["num_act"])
    for key_c, key_r in (("allocation", "z_cusp"), ("omega", "omega"),
                         ("slab_scale", "theta_slab"), ("penalty_scale", "theta")):
        for value_c, value_r in zip(candidate[key_c], reference[key_r]):
            np.testing.assert_array_equal(value_c, value_r)
    for B_c, B_r in zip(candidate["B_list"], reference["B_list"]):
        np.testing.assert_allclose(B_c, B_r, rtol=0, atol=0)
    for A_c, A_r in zip(candidate["A_list"], reference["A_list"]):
        np.testing.assert_array_equal(A_c, A_r)
    np.testing.assert_allclose(candidate["gamma"], reference["gamma"])
    np.testing.assert_allclose(candidate["prop"], reference["prop"])


def test_generator_seed_is_reproducible_and_differs_from_legacy_stream():
    inputs = _inputs()
    kwargs = dict(C=1, it=3, t_spike=0.02, temp=0.9, tau=0.3, verbose=False)
    first = fit_normal_csp_copula(
        inputs["X"], inputs["prop"], inputs["gamma"],
        _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        random_state=7, **kwargs,
    )
    second = fit_normal_csp_copula(
        inputs["X"], inputs["prop"], inputs["gamma"],
        _copy(inputs["B_list"]), _copy(inputs["A_list"]),
        random_state=np.random.default_rng(7), **kwargs,
    )
    for B_1, B_2 in zip(first["B_list"], second["B_list"]):
        np.testing.assert_array_equal(B_1, B_2)


def test_fista_solver_and_threshold_rules_run():
    inputs = _inputs()
    for solver, rule in (("fista", "all_but_last"), ("lbfgsb", "none")):
        raw = fit_normal_csp_copula(
            inputs["X"], inputs["prop"], inputs["gamma"],
            _copy(inputs["B_list"]), _copy(inputs["A_list"]),
            C=1, it=2, t_spike=0.02, temp=0.9, tau=0.3, verbose=False,
            mstep_solver=solver, threshold_rule=rule, random_state=1,
            keep_trace=True,
        )
        assert raw["mstep_solver"] == solver
        assert raw["threshold_rule"] == rule
        assert len(raw["trace"]) == 2
        assert all(np.all(np.isfinite(B)) for B in raw["B_list"])


def _run_kernel(inputs, **overrides):
    kwargs = dict(C=1, it=1, t_spike=[0.02, 0.03], temp=0.9, tau=0.3,
                  verbose=False, random_state=4, penalty_scaling_power=0.0)
    kwargs.update(overrides)
    return fit_normal_csp_copula(
        inputs["X"], inputs["prop"], inputs["gamma"],
        _copy(inputs["B_list"]), _copy(inputs["A_list"]), **kwargs,
    )


def test_hybrid_solver_uses_fista_observed_layer_and_lbfgsb_logistic_layers():
    """One iteration: hybrid B_1 equals the FISTA B_1, hybrid B_2 the L-BFGS-B B_2."""
    inputs = _inputs()
    lbfgsb = _run_kernel(inputs, mstep_solver="lbfgsb", optimizer_maxiter=15)
    fista = _run_kernel(inputs, mstep_solver="fista", optimizer_maxiter=15,
                        fista_maxiter=60)
    hybrid = _run_kernel(inputs, mstep_solver="hybrid", optimizer_maxiter=15,
                         fista_maxiter=60)
    assert hybrid["mstep_solver"] == "hybrid"
    assert hybrid["fista_maxiter"] == 60
    assert lbfgsb["fista_maxiter"] == 300
    assert _run_kernel(inputs)["mstep_solver"] == "fista_lower"
    assert _run_kernel(inputs)["layer_solvers"] == ("fista", "lbfgsb")
    np.testing.assert_array_equal(hybrid["B_list"][0], fista["B_list"][0])
    np.testing.assert_array_equal(hybrid["B_list"][1], lbfgsb["B_list"][1])
    assert not np.array_equal(hybrid["B_list"][0], lbfgsb["B_list"][0])


def test_mean_field_mstep_and_observed_noise_options():
    inputs = _inputs()
    base = _run_kernel(inputs)
    assert base["mstep_latent"] == "draw"
    assert base["observed_noise_scale"] == 0.0
    mean_field = _run_kernel(inputs, mstep_latent="mean_field")
    assert mean_field["mstep_latent"] == "mean_field"
    assert all(np.all(np.isfinite(B)) for B in mean_field["B_list"])
    assert not np.array_equal(mean_field["B_list"][0], base["B_list"][0])
    # The committed latent state is still the majority vote of the draws.
    for A_mf, A_base in zip(mean_field["A_list"], base["A_list"]):
        np.testing.assert_array_equal(A_mf, A_base)
    noisy = _run_kernel(inputs, observed_noise_scale=0.5)
    assert noisy["observed_noise_scale"] == 0.5
    assert all(np.all(np.isfinite(B)) for B in noisy["B_list"])
    assert np.all(noisy["gamma"] > 0)
    assert not np.array_equal(noisy["B_list"][0], base["B_list"][0])
    np.testing.assert_array_equal(
        _run_kernel(inputs, observed_noise_scale=0.0)["B_list"][0], base["B_list"][0]
    )


def test_fit_forwards_hybrid_solver_controls():
    X = _simulate_two_layer(13, N=120)
    result = fit(
        X, method="DDE", family="Normal", D=2, dimension_method="csp",
        t_spike=[0.02, 0.04], temp=0.7, tau=0.3, C=1, max_iter=2,
        random_state=5, verbose=False, csp_mstep_solver="hybrid",
        csp_fista_maxiter=50, csp_initializer="family",
    )
    assert result.diagnostics["mstep_solver"] == "hybrid"
    assert result.diagnostics["fista_maxiter"] == 50
    assert result.diagnostics["optimizer_maxiter"] == 15
    assert result.diagnostics["mstep_latent"] == "draw"
    assert result.diagnostics["observed_noise_scale"] == 0.0


def test_per_layer_mstep_solver_sequence():
    inputs = _inputs()
    args = (inputs["X"], inputs["prop"], inputs["gamma"],
            _copy(inputs["B_list"]), _copy(inputs["A_list"]))
    base = dict(C=1, it=2, t_spike=0.02, temp=0.9, tau=0.3, verbose=False,
                random_state=3)
    hybrid = fit_normal_csp_copula(*args, mstep_solver="hybrid", **base)
    seq = fit_normal_csp_copula(*args, mstep_solver=["fista", "lbfgsb"], **base)
    assert seq["mstep_solver"] == "hybrid"
    assert seq["layer_solvers"] == ("fista", "lbfgsb")
    for B_1, B_2 in zip(hybrid["B_list"], seq["B_list"]):
        np.testing.assert_array_equal(B_1, B_2)
    other = fit_normal_csp_copula(*args, mstep_solver=["lbfgsb", "fista"], **base)
    assert other["mstep_solver"] == "lbfgsb+fista"
    lower = fit_normal_csp_copula(*args, mstep_solver="fista_lower", **base)
    assert lower["layer_solvers"] == ("fista", "lbfgsb")
    assert lower["mstep_solver"] == "fista_lower"
    for B_1, B_2 in zip(hybrid["B_list"], lower["B_list"]):
        np.testing.assert_array_equal(B_1, B_2)  # identical for D=2
    scaled_lower = fit_normal_csp_copula(
        *args, penalty_scaling_power=1.0, penalty_scaling_reference=10.0,
        penalty_scaling_layers="lower", **base
    )
    assert scaled_lower["penalty_factor"] == (inputs["X"].shape[0] / 10.0, 1.0)
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, mstep_solver=["fista"], **base)
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, mstep_solver=["fista", "bogus"], **base)


def test_invalid_controls_raise():
    inputs = _inputs()
    base = dict(C=1, it=1, t_spike=0.02, temp=0.9, tau=0.3, verbose=False)
    args = (inputs["X"], inputs["prop"], inputs["gamma"],
            _copy(inputs["B_list"]), _copy(inputs["A_list"]))
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, **{**base, "threshold_rule": "bogus"})
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, **{**base, "mstep_solver": "bogus"})
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, **{**base, "t_spike": [0.02, 0.03, 0.04]})
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, **{**base, "temp": 1.5})
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, **{**base, "mstep_latent": "bogus"})
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, **{**base, "observed_noise_scale": -1.0})
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, **{**base, "fista_maxiter": 0})


def _simulate_two_layer(seed, N=600, J=36, K_true=(6, 2)):
    rng = np.random.default_rng(seed)
    B1 = np.zeros((J, K_true[0]))
    block = J // K_true[0]
    for k in range(K_true[0]):
        B1[k * block:(k + 1) * block, k] = 8.0
        B1[k * block:(k + 1) * block, (k + 1) % K_true[0]] = -4.0
    B2 = np.zeros((K_true[0], K_true[1]))
    B2[: K_true[0] // 2, 0] = 4.0
    B2[K_true[0] // 2 :, 1] = 4.0
    A2 = rng.binomial(1, 0.5, (N, K_true[1]))
    A1 = rng.binomial(1, expit(-2.0 + A2 @ B2.T))
    X = -2.0 + A1 @ B1.T + rng.normal(size=(N, J))
    return X


def test_copula_backend_is_default_for_normal_and_shrinks_overspecified_widths():
    X = _simulate_two_layer(11)
    result = fit(
        X, method="DDE", family="Normal", D=2, dimension_method="csp",
        t_spike=[0.02, 0.04], temp=0.7, tau=0.3, C=1, max_iter=40,
        random_state=5, verbose=False,
    )
    assert result.diagnostics["backend"] == "copula"
    assert result.diagnostics["initialization"] == "data"
    assert result.diagnostics["initializer"] == "family"
    assert result.diagnostics["optimizer_maxiter"] == 15
    assert result.diagnostics["mstep_solver"] == "fista_lower"
    assert result.diagnostics["fista_maxiter"] == 300
    assert tuple(result.diagnostics["K_max_list"]) == (12, 4)
    K1, K2 = result.K_list
    assert 6 <= K1 < 12
    assert 1 <= K2 < 4
    assert result.B_list[0].shape == (36, K1 + 1)
    assert result.B_list[1].shape == (K1, K2 + 1)


def test_explicit_mcem_backend_and_copula_initializer_still_available():
    X = _simulate_two_layer(12, N=120)
    result = fit(
        X, method="DDE", family="Normal", D=2, dimension_method="csp",
        t_spike=[0.02, 0.04], temp=0.7, tau=0.3, C=1, max_iter=2,
        random_state=5, verbose=False, csp_backend="mcem",
    )
    assert result.diagnostics["backend"] == "mcem"
    assert result.diagnostics["initialization"] == "data"
    result = fit(
        X, method="DDE", family="Normal", D=2, dimension_method="csp",
        t_spike=[0.02, 0.04], temp=0.7, tau=0.3, C=1, max_iter=2,
        random_state=5, verbose=False, csp_initializer="copula",
    )
    assert result.diagnostics["backend"] == "copula"
    assert result.diagnostics["initializer"] == "copula"
    assert result.diagnostics["initialization"] == "data_copula"


def test_copula_backend_rejects_discrete_families():
    rng = np.random.default_rng(0)
    Y = rng.integers(0, 2, size=(60, 18)).astype(float)
    with pytest.raises(NotImplementedError):
        fit(
            Y, method="DDE", family="Bernoulli", D=2, dimension_method="csp",
            t_spike=0.02, temp=0.9, tau=0.3, C=1, max_iter=1,
            csp_backend="copula", verbose=False,
        )


def test_standardized_copula_backend_is_affine_equivariant():
    """Raw-scale estimates must transform exactly with an affine change of X."""
    X = _simulate_two_layer(21, N=300)
    scale = np.linspace(0.5, 4.0, X.shape[1])
    shift = np.linspace(-3.0, 3.0, X.shape[1])
    kwargs = dict(
        method="DDE", family="Normal", D=2, dimension_method="csp",
        t_spike=[0.02, 0.04], temp=0.7, tau=0.3, C=1, max_iter=6,
        random_state=9, verbose=False,
    )
    first = fit(X, **kwargs)
    second = fit(X * scale[None, :] + shift[None, :], **kwargs)
    assert first.diagnostics["standardized"] is True
    assert first.K_list == second.K_list
    B_first = first.B_list[0]
    B_second = second.B_list[0]
    np.testing.assert_allclose(B_second[:, 1:], B_first[:, 1:] * scale[:, None], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(B_second[:, 0], B_first[:, 0] * scale + shift, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(second.gamma, first.gamma * scale**2, rtol=1e-6)
    raw = fit(X, standardize=False, **kwargs)
    assert raw.diagnostics["standardized"] is False


def test_standardize_applies_to_supplied_initialization():
    from dde.algorithms.normal_init_cop import Normal_init_cop

    X = _simulate_two_layer(22, N=200)
    prop0, B0, gamma0, A0 = Normal_init_cop(X, (12, 4))
    kwargs = dict(
        method="DDE", family="Normal", D=2, dimension_method="csp",
        t_spike=[0.02, 0.04], temp=0.7, tau=0.3, C=1, max_iter=3,
        random_state=9, verbose=False,
    )
    result = fit(X, prop_in=prop0, B_list_in=B0, gamma_in=gamma0, A_list_in=A0, **kwargs)
    assert result.diagnostics["initialization"] == "provided"
    assert result.diagnostics["standardized"] is True
    assert np.all(np.isfinite(result.B_list[0]))
    assert np.all(result.gamma > 0)


def test_slab_hyperprior_defaults_reproduce_copdde_and_change_allocation():
    from dde.algorithms.csp import update_csp_layer

    rng = np.random.default_rng(4)
    B = rng.normal(scale=0.3, size=(20, 7))
    B[:, 1:3] *= 6.0  # two strong columns
    B[:, 5:] *= 0.05  # two near-zero columns
    omega = np.ones(7) / 7
    slab = np.ones(6)
    base = update_csp_layer(B, omega, 0.03, slab.copy(), 20, rng=np.random.default_rng(1))
    same = update_csp_layer(
        B, omega, 0.03, slab.copy(), 20, rng=np.random.default_rng(1),
        slab_shape=1.0, slab_rate=5.0,
    )
    for a, b in zip(base, same):
        np.testing.assert_array_equal(a, b)
    strong = update_csp_layer(
        B, omega, 0.03, slab.copy(), 20, rng=np.random.default_rng(1),
        slab_shape=20.0, slab_rate=1.0,
    )
    # A stronger slab hyperprior yields smaller slab scales (larger weights)
    # on the active columns.
    active = base[0] > np.arange(1, 7)
    assert np.all(strong[2][active] < base[2][active])
    with pytest.raises(ValueError):
        update_csp_layer(B, omega, 0.03, slab.copy(), 20, slab_shape=0.0)


def test_penalty_scaling_multiplies_mstep_weights_and_is_recorded():
    inputs = _inputs()
    args = (inputs["X"], inputs["prop"], inputs["gamma"],
            _copy(inputs["B_list"]), _copy(inputs["A_list"]))
    base = dict(C=1, it=2, t_spike=0.02, temp=0.9, tau=0.3, verbose=False,
                random_state=3)
    unscaled = fit_normal_csp_copula(*args, penalty_scaling_power=0.0, **base)
    default = fit_normal_csp_copula(*args, **base)
    assert default["penalty_scaling_layers"] == "lower"
    assert default["penalty_factor"] == (inputs["X"].shape[0] / 500.0, 1.0)
    scaled = fit_normal_csp_copula(
        *args, penalty_scaling_power=1.0, penalty_scaling_reference=10.0,
        penalty_scaling_layers="all", **base
    )
    N = inputs["X"].shape[0]
    assert scaled["penalty_factor"] == (N / 10.0, N / 10.0)
    assert unscaled["penalty_factor"] == (1.0, 1.0)
    deep = fit_normal_csp_copula(
        *args, penalty_scaling_power=0.5, penalty_scaling_reference=10.0,
        penalty_scaling_layers="deep", **base
    )
    assert deep["penalty_factor"] == (1.0, (N / 10.0) ** 0.5)
    per_layer = fit_normal_csp_copula(
        *args, penalty_scaling_power=[1.0, 0.5], penalty_scaling_reference=10.0,
        penalty_scaling_layers="all", **base
    )
    assert per_layer["penalty_factor"] == (N / 10.0, (N / 10.0) ** 0.5)
    assert per_layer["penalty_scaling_power"] == (1.0, 0.5)
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, penalty_scaling_power=[1.0, 0.5, 0.2], **base)
    # A much larger penalty shrinks the coefficient mass.
    heavy = fit_normal_csp_copula(
        *args, penalty_scaling_power=1.0, penalty_scaling_reference=1e-3, **base
    )
    assert np.sum(np.abs(heavy["B_list"][0][:, 1:])) < np.sum(
        np.abs(unscaled["B_list"][0][:, 1:])
    )
    consistent = fit_normal_csp_copula(
        *args, penalty_scaling_power=1.0, penalty_scaling_reference=10.0,
        penalty_scaling_allocation=True, **base
    )
    assert consistent["penalty_scaling_allocation"] is True
    assert all(np.all(np.isfinite(B)) for B in consistent["B_list"])
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, penalty_scaling_layers="bogus", **base)
    with pytest.raises(ValueError):
        fit_normal_csp_copula(*args, penalty_scaling_power=-1.0, **base)


def test_fit_passes_slab_and_penalty_scaling_controls():
    X = _simulate_two_layer(31, N=150)
    result = fit(
        X, method="DDE", family="Normal", D=2, dimension_method="csp",
        t_spike=[0.02, 0.04], temp=0.7, tau=0.3, C=1, max_iter=2,
        random_state=5, verbose=False,
        csp_slab_shape=5.0, csp_slab_rate=1.0,
        csp_penalty_scaling_power=0.5, csp_penalty_scaling_reference=500.0,
        csp_penalty_scaling_layers="deep", csp_penalty_scaling_allocation=True,
    )
    d = result.diagnostics
    assert d["slab_shape"] == 5.0 and d["slab_rate"] == 1.0
    assert d["penalty_scaling_power"] == (0.5, 0.5)
    assert d["penalty_scaling_layers"] == "deep"
    assert d["penalty_scaling_allocation"] is True
    assert d["penalty_factor"] == (1.0, (150 / 500.0) ** 0.5)
