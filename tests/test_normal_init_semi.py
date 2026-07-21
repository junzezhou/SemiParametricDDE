import numpy as np

from semidde.algorithms.normal_init_semi import Normal_init_semi, rank_gaussianize_columns
from semidde.algorithms.semi import fit_semidde
from semidde.simulation import generate_X_normal


def _make_normal_params(J, K_list, seed=321):
    rng = np.random.default_rng(seed)
    B_list = []
    B0 = rng.normal(0.0, 0.6, size=(J, K_list[0] + 1))
    B0[:, 0] = rng.normal(0.0, 0.2, size=J)
    B_list.append(B0)
    for d in range(1, len(K_list)):
        Bd = rng.normal(0.0, 0.7, size=(K_list[d - 1], K_list[d] + 1))
        Bd[:, 0] = rng.normal(0.0, 0.2, size=K_list[d - 1])
        B_list.append(Bd)
    prop = np.linspace(0.35, 0.65, K_list[-1])
    gamma = np.ones(J) * 0.5
    G_list = [(B[:, 1:] != 0).astype(int) for B in B_list]
    return prop, B_list, gamma, G_list


def _simulate_X(N, J, K_list):
    prop, B_list, gamma, G_list = _make_normal_params(J, K_list)
    X, _ = generate_X_normal(N, prop, B_list, gamma)
    return X, B_list, G_list


def test_rank_gaussianize_columns_shape_and_finiteness():
    Y = np.array([[1.0, 3.0], [2.0, np.nan], [4.0, 1.0], [3.0, 2.0]])
    X0 = rank_gaussianize_columns(Y)
    assert X0.shape == Y.shape
    assert np.all(np.isfinite(X0))
    assert X0[1, 1] == 0.0


def test_normal_init_semi_shapes_D1_to_D3():
    for K_list in ([2], [3, 1], [6, 2], [18, 6, 2]):
        N, J = 40, max(12, 3 * K_list[0])
        X, B_list_true, G_list_true = _simulate_X(N, J, list(K_list))
        prop, B_list, gamma, A_list, err_G_list = Normal_init_semi(
            X,
            K_list[0],
            K_list,
            B_list_true,
            G_list_true,
            1e-4,
        )
        assert prop.shape == (K_list[-1],)
        assert gamma.shape == (J,)
        assert [A.shape for A in A_list] == [(N, k) for k in K_list]
        expected_B = [(J, K_list[0] + 1)] + [
            (K_list[d - 1], K_list[d] + 1) for d in range(1, len(K_list))
        ]
        assert [B.shape for B in B_list] == expected_B
        assert len(err_G_list) == len(K_list)


def test_normal_init_semi_data_mode_does_not_need_oracle():
    N, J = 40, 18
    K_list = [6, 2]
    X, _, _ = _simulate_X(N, J, K_list)

    prop, B_list, gamma, A_list, err_G_list = Normal_init_semi(
        X,
        K_list[0],
        K_list,
        mode="data",
        epsilon=1e-4,
    )

    assert prop.shape == (K_list[-1],)
    assert gamma.shape == (J,)
    assert [A.shape for A in A_list] == [(N, k) for k in K_list]
    assert [B.shape for B in B_list] == [(J, 7), (6, 3)]
    assert err_G_list == [None, None]


def test_fit_semidde_initialize_data_mode_smoke(capsys):
    N, J = 18, 18
    K_list = [6, 2]
    X, _, _ = _simulate_X(N, J, K_list)

    result = fit_semidde(
        X,
        K_list=K_list,
        initialize="data",
        pen_list=[0.0, 0.0],
        tau=1e-6,
        tol=0.0,
        C=1,
        max_outer_iter=1,
        max_inner_iter=1,
        verbose=False,
    )

    assert result.prop.shape == (2,)
    assert [A.shape for A in result.A_list] == [(N, 6), (N, 2)]
    assert "SemiDDE initialization:" not in capsys.readouterr().out


def test_fit_semidde_complete_initial_values_bypass_initialize_method(capsys):
    N, J = 18, 18
    K_list = [6, 2]
    X, _, _ = _simulate_X(N, J, K_list)
    X_init = rank_gaussianize_columns(X)
    prop, B_list, gamma, A_list, _ = Normal_init_semi(
        X_init,
        K_list[0],
        K_list,
        mode="data",
        epsilon=1e-4,
    )

    result = fit_semidde(
        X,
        K_list=K_list,
        initialize="oracle",
        X_in=X_init,
        prop_in=prop,
        B_list_in=B_list,
        gamma_in=gamma,
        A_list_in=A_list,
        pen_list=[0.0, 0.0],
        tau=1e-6,
        tol=0.0,
        C=1,
        max_outer_iter=1,
        max_inner_iter=1,
        verbose=False,
    )

    assert result.prop.shape == (2,)
    assert "SemiDDE initialization:" not in capsys.readouterr().out
