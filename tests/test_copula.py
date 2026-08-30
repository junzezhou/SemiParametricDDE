import numpy as np
import pytest

from dde.algorithms.copula import (
    fit_copula_dde,
    normal_initialize_copula_dde,
    prepare_rank_likelihood_inputs,
)
from dde.copula import format_estimates_D, implied_Z_moments_D, rescale_B1_D
from dde.simulation import generate_X_Cop_marg_emp_D


def _make_parameters(J, K_list, seed=123):
    rng = np.random.default_rng(seed)
    B_list = []
    B0 = rng.normal(0.0, 0.35, size=(J, K_list[0] + 1))
    B0[:, 0] = rng.normal(0.0, 0.1, size=J)
    B_list.append(B0)
    for d in range(1, len(K_list)):
        Bd = rng.normal(0.0, 0.45, size=(K_list[d - 1], K_list[d] + 1))
        Bd[:, 0] = rng.normal(0.0, 0.1, size=K_list[d - 1])
        B_list.append(Bd)
    prop = np.linspace(0.35, 0.65, K_list[-1])
    gamma = np.ones(J) * 0.8
    return prop, B_list, gamma


@pytest.mark.parametrize("K_list", [[2], [6, 2], [18, 6, 2]])
def test_generate_copula_shapes(K_list):
    J = max(6, 3 * K_list[0])
    N = 20
    prop, B_list, gamma = _make_parameters(J, K_list)

    sim = generate_X_Cop_marg_emp_D(
        N,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=42,
    )

    assert sim["X"].shape == (N, J)
    assert sim["Z"].shape == (N, J)
    assert [A.shape for A in sim["A_list"]] == [(N, k) for k in K_list]
    assert [P.shape for P in sim["prob_list"]] == [(N, k) for k in K_list]


@pytest.mark.parametrize("K_list", [[2], [6, 2], [18, 6, 2]])
def test_fit_copula_dde_fixed_smoke(K_list):
    J = max(6, 3 * K_list[0])
    N = 18
    prop, B_list, gamma = _make_parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        N,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=43,
    )
    Z, R, Rlevels = prepare_rank_likelihood_inputs(sim["X"])
    init = normal_initialize_copula_dde(sim["X"], K_list, Z=Z)

    result = fit_copula_dde(
        sim["X"],
        K_list=K_list,
        dimension_method="fixed",
        initialize="data",
        Z=Z,
        R=R,
        Rlevels=Rlevels,
        prop_in=init["prop"],
        gamma_in=init["gamma"],
        B_list_in=init["B_list"],
        A_list_in=init["A_list"],
        C=2,
        tol=0.0,
        it=1,
        temp=1.0,
        pen_list=np.zeros(len(K_list)),
        tau=0.0,
        verbose=False,
        optimizer_maxiter=2,
    )

    assert result.prop.shape == (K_list[-1],)
    assert [B.shape for B in result.B_list] == [B.shape for B in B_list]
    assert [A.shape for A in result.A_list] == [(N, k) for k in K_list]
    assert result.gamma.shape == (J,)


def test_normal_initialize_copula_dde_and_default_fit(capsys):
    J = 18
    N = 28
    K_list = [6, 2]
    prop, B_list, gamma = _make_parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        N,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=45,
    )
    Z, R, Rlevels = prepare_rank_likelihood_inputs(sim["X"])

    init = normal_initialize_copula_dde(sim["X"], K_list, Z=Z)
    assert [B.shape for B in init["B_list"]] == [B.shape for B in B_list]
    assert [A.shape for A in init["A_list"]] == [(N, k) for k in K_list]
    assert init["prop"].shape == (K_list[-1],)
    assert init["gamma"].shape == (J,)
    assert init["used_oracle"] is False

    result = fit_copula_dde(
        sim["X"],
        K_list=K_list,
        dimension_method="fixed",
        Z=Z,
        R=R,
        Rlevels=Rlevels,
        C=2,
        tol=0.0,
        it=1,
        temp=1.0,
        pen_list=np.zeros(len(K_list)),
        tau=0.0,
        verbose=False,
        optimizer_maxiter=2,
    )
    assert [B.shape for B in result.B_list] == [B.shape for B in B_list]
    assert [A.shape for A in result.A_list] == [(N, k) for k in K_list]
    assert "CopDDE initialization:" not in capsys.readouterr().out


def test_fit_copula_dde_csp_uses_canonical_kernel():
    J = 54
    N = 20
    K_list = [18, 6, 2]
    prop, B_list, gamma = _make_parameters(J, K_list)
    sim = generate_X_Cop_marg_emp_D(
        N,
        prop,
        lambdas=np.ones(J) * 2.0,
        gamma=gamma,
        B_list=B_list,
        random_state=46,
    )
    Z, R, Rlevels = prepare_rank_likelihood_inputs(sim["X"])
    init = normal_initialize_copula_dde(sim["X"], K_list, Z=Z)

    result = fit_copula_dde(
        sim["X"],
        K_list=K_list,
        dimension_method="csp",
        Z=Z,
        R=R,
        Rlevels=Rlevels,
        prop_in=init["prop"],
        gamma_in=init["gamma"],
        B_list_in=init["B_list"],
        A_list_in=init["A_list"],
        C=1,
        it=1,
        t_spike=[0.01, 0.03, 0.005],
        temp=0.9,
        tau=0.3,
        verbose=False,
        optimizer_maxiter=2,
    )
    assert result.diagnostics["backend"] == "csp"
    assert result.diagnostics["threshold_rule"] == "first_two_layers"
    assert result.diagnostics["num_active"].shape == (len(K_list),)
    assert result.diagnostics["original_K_list"] == tuple(K_list)
    assert result.diagnostics["K_list"] == tuple(B.shape[1] - 1 for B in result.B_list)
    assert "active_columns_original_index" in result.diagnostics
    assert "trace" not in result.raw


def test_rescale_and_format_estimates_D_shapes():
    J = 18
    K_list = [6, 2]
    prop, B_list, gamma = _make_parameters(J, K_list)
    mu, sd = implied_Z_moments_D(prop, B_list, gamma)
    assert mu.shape == (J,)
    assert sd.shape == (J,)

    B_scaled, gamma_scaled, mu_used, sd_used = rescale_B1_D(prop, B_list, gamma)
    assert [B.shape for B in B_scaled] == [B.shape for B in B_list]
    assert gamma_scaled.shape == gamma.shape
    assert mu_used.shape == (J,)
    assert sd_used.shape == (J,)

    formatted = format_estimates_D(B_list, B_list, gamma, prop)
    assert [B.shape for B in formatted["B_list"]] == [B.shape for B in B_list]
    assert formatted["prop"].shape == prop.shape
    assert len(formatted["assignments"]) == len(K_list)
