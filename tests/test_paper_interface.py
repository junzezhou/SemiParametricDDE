"""The calls of Section 5 of the package manuscript, on small data.

Each test follows one manuscript example with the same arguments (smaller
data, iteration budgets and pMSE repeats) and fails on any deprecated or
future-changing argument the examples might rely on.
"""

import numpy as np
import pytest

from dde import fit, select_hyperparameters

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning:dde", "error::FutureWarning:dde",
    "error::DeprecationWarning:test_paper_interface", "error::FutureWarning:test_paper_interface",
)

N, J = 200, 30


def _simulate(K_list, seed):
    """Normal DDE with pure-child blocks: J observed variables, widths K_list."""
    rng = np.random.default_rng(seed)
    widths = [J, *K_list]
    B_list = []
    for d in range(1, len(widths)):
        B = np.zeros((widths[d - 1], widths[d] + 1))
        B[:, 0] = 0.0 if d == 1 else -1.0
        B[np.arange(widths[d - 1]), 1 + np.arange(widths[d - 1]) % widths[d]] = 2.0
        B_list.append(B)
    A = (rng.random((N, widths[-1])) < 0.5).astype(float)
    for B in reversed(B_list[1:]):
        eta = B[:, 0] + A @ B[:, 1:].T
        A = (rng.random(eta.shape) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return B_list[0][:, 0] + A @ B_list[0][:, 1:].T + rng.normal(size=(N, J))


def _ordinal(Y):
    return np.digitize(Y, np.quantile(Y, [0.2, 0.4, 0.6, 0.8]))


def _full_graph(K_list):
    widths = [J, *K_list]
    return [np.ones((widths[d], widths[d + 1]), dtype=int) for d in range(len(K_list))]


def test_section_5_1_parametric_dde_fixed():
    Y = _simulate([10, 3], seed=1)
    np.random.seed(0)
    dde_res = fit(
        Y, D=2, model="DDE", family="Normal", K_list=[10, 3],
        dimension_method="fixed", pen_list=[0.1, 0.1], tau=0.3, C=1, tol=0.01,
        max_it=3, verbose=False,
    )
    assert dde_res.model == "DDE"
    assert dde_res.K_list == [10, 3]
    assert [B.shape for B in dde_res.B_list] == [(J, 11), (10, 4)]
    assert [A.shape for A in dde_res.A_list] == [(N, 10), (N, 3)]
    assert dde_res.transformed.shape == (N, J)
    assert dde_res.diagnostics["iterations"] <= 3
    assert len(dde_res.diagnostics["active_by_layer"]) == 2
    assert isinstance(dde_res.raw, dict)


def test_section_5_1_semidde_csp():
    Y = np.exp(_simulate([10, 3], seed=2) / 2.0)
    semi_res = fit(
        Y, D=2, model="SemiDDE", dimension_method="csp", t_spike=[0.04, 0.08],
        temp=0.8, tau=0.3, C=1, tol=0.01, max_it=3, random_state=0, verbose=False,
    )
    assert semi_res.model == "SemiDDE"
    assert semi_res.diagnostics["K_max_list"] == (10, 3)
    assert [B.shape[1] - 1 for B in semi_res.B_list] == semi_res.K_list


def test_section_5_1_semidde_fixed_takes_max_it():
    Y = np.exp(_simulate([10, 3], seed=3) / 2.0)
    np.random.seed(1)
    semi_res = fit(
        Y, D=2, model="SemiDDE", K_list=[10, 3], dimension_method="fixed",
        pen_list=[N ** 0.25] * 2, tau=0.3, C=1, tol=0.01, max_it=2,
        max_outer_iter=1, verbose=False,
    )
    assert semi_res.model == "SemiDDE"
    assert semi_res.diagnostics["final_iterations"] <= 2


def test_section_5_1_copdde_fixed_with_graph():
    Y = _ordinal(_simulate([9, 3, 1], seed=4))
    G_list = _full_graph([9, 3, 1])
    cop_res = fit(
        Y, D=3, model="CopDDE", K_list=[9, 3, 1], G_list=G_list,
        dimension_method="fixed", pen_list=[0.1] * 3, tau=0.3, C=1, tol=0.01,
        max_it=3, n_init=2, random_state=0, verbose=False,
    )
    assert cop_res.model == "CopDDE"
    assert cop_res.K_list == [9, 3, 1]
    assert cop_res.diagnostics["iterations"] <= 3


def test_copdde_fixed_and_csp_take_the_shared_arguments():
    Y = _ordinal(_simulate([10, 3], seed=5))
    fixed = fit(
        Y, D=2, model="CopDDE", K_list=[10, 3], dimension_method="fixed",
        pen_list=[0.1, 0.1], tau=0.3, tol=0.01, max_it=2, random_state=0, verbose=False,
    )
    assert fixed.model == "CopDDE" and fixed.diagnostics["iterations"] <= 2
    csp = fit(
        Y, D=2, model="CopDDE", dimension_method="csp", t_spike=[0.04, 0.08],
        temp=0.8, tau=0.3, C=1, tol=0.01, max_it=2, optimizer_maxiter=2,
        random_state=0, verbose=False,
    )
    assert csp.model == "CopDDE" and csp.diagnostics["iterations"] <= 2


def test_section_5_2_parametric_dde_grid_of_scalar_penalties():
    Y = _simulate([10, 3], seed=6)
    sel_DDE = select_hyperparameters(
        Y, D=2, model="DDE", family="Normal", dimension_method="fixed",
        K_list=[10, 3],
        candidates={"pen_list": [.1, .05], "tau": [.2, .3]},
        C=1, tol=.01, max_it=2, random_state=0, verbose=False,
    )
    rows = sel_DDE.diagnostics["cv_table"]
    assert len(rows) == 4
    assert [row["pen_list"] for row in rows] == [[.1, .1], [.1, .1], [.05, .05], [.05, .05]]
    assert {row["criterion"] for row in rows} == {"ebic"}
    assert {row["max_it"] for row in rows} == {2}
    for row in rows:
        assert row["ebic"] == pytest.approx(-2.0 * N * row["mean_score"])
    best = min(range(4), key=lambda i: rows[i]["ebic"])
    assert sel_DDE.diagnostics["hyperparameter_selection"].best_index == best
    assert sel_DDE.model == "DDE"
    assert sel_DDE.K_list == [10, 3]
    assert sel_DDE.raw["selection_method"] == "fixed"
    assert sel_DDE.raw["selected_hyperparameters"] == {
        "pen_list": rows[best]["pen_list"], "tau": rows[best]["tau"],
    }


def test_section_5_2_semidde_csp_grid():
    Y = np.exp(_simulate([10, 3], seed=7) / 2.0)
    sel_semiDDE = select_hyperparameters(
        Y, D=2, model="SemiDDE", dimension_method="csp",
        candidates={"t_spike": [.1, .05], "tau": [.3], "temp": [.8]},
        C=1, tol=.01, max_it=2, random_state=0, verbose=False,
    )
    rows = sel_semiDDE.diagnostics["cv_table"]
    assert [row["t_spike"] for row in rows] == [[.1, .1], [.05, .05]]
    assert sel_semiDDE.model == "SemiDDE"
    assert sel_semiDDE.raw["selection_method"] == "csp"
    assert [B.shape[1] - 1 for B in sel_semiDDE.B_list] == sel_semiDDE.K_list


def test_section_5_2_copdde_fixed_graph_uses_pmse():
    Y = _ordinal(_simulate([9, 3, 1], seed=8))
    sel_copDDE = select_hyperparameters(
        Y, D=3, model="CopDDE", dimension_method="fixed", K_list=[9, 3, 1],
        G_list=_full_graph([9, 3, 1]),
        candidates={"pen_list": [.1, .05], "tau": [.3]},
        C=1, tol=.01, max_it=2, n_init=2,
        n_repeats=2, pmse_n_estimators=5, random_state=0, verbose=False,
    )
    rows = sel_copDDE.diagnostics["cv_table"]
    assert len(rows) == 2
    assert {row["criterion"] for row in rows} == {"pmse"}
    assert all(np.isfinite(row["mean_pmse"]) for row in rows)
    assert sel_copDDE.model == "CopDDE"
    assert sel_copDDE.raw["selection_method"] == "fixed"
    assert sel_copDDE.raw["selection_criterion"] == "pmse"


def test_select_hyperparameters_rejects_an_argument_given_twice():
    Y = _simulate([10, 3], seed=9)
    with pytest.raises(ValueError, match="'K_list' is given both"):
        select_hyperparameters(
            Y, D=2, model="DDE", family="Normal", dimension_method="fixed",
            K_list=[10, 3], fixed_hp={"K_list": [10, 3]},
            candidates={"pen_list": [.1], "tau": [.3]},
        )
