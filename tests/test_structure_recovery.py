import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dde.utils import layer_structure_recovery_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _truth(seed=0, N=600):
    """D=3 hierarchy, widths (4, 3, 2), J=8, with distinct latent columns."""
    rng = np.random.default_rng(seed)
    B1 = np.zeros((8, 5))
    B1[:, 0] = -1.0
    for j in range(8):
        B1[j, 1 + j % 4] = 3.0
    B1[0, 2] = -1.5
    B1[5, 4] = 1.0
    B2 = np.array(
        [[-1.0, 2.0, 0.0, 0.0],
         [-1.0, 0.0, 2.0, 0.0],
         [-1.0, 0.0, 0.0, 2.0],
         [-1.0, 2.0, 0.0, -1.0]]
    )
    B3 = np.array([[-1.0, 2.0, 0.0], [-1.0, 0.0, 2.0], [-1.0, 2.0, 2.0]])
    A = [rng.integers(0, 2, size=(N, width)).astype(np.uint8) for width in (4, 3, 2)]
    return [B1, B2, B3], A


def _permuted(B_list, A_list, perms, flips=()):
    """Reorder every layer's columns by perms[d] (and rows of B_{d+1} with it)."""
    B_out = [B.copy() for B in B_list]
    A_out = [A.copy() for A in A_list]
    for d, perm in enumerate(perms):
        B_out[d] = np.column_stack([B_out[d][:, 0], B_out[d][:, 1:][:, perm]])
        A_out[d] = A_out[d][:, perm]
        if d + 1 < len(B_out):
            B_out[d + 1] = B_out[d + 1][perm]
    for d, column in flips:
        A_out[d][:, column] = 1 - A_out[d][:, column]
        B_out[d][:, 0] += B_out[d][:, 1 + column]
        B_out[d][:, 1 + column] *= -1.0
    return B_out, A_out


def _check_perfect(metrics, widths):
    assert metrics["layer_graph_accuracy"] == [1.0] * len(widths)
    assert metrics["layer_sensitivity"] == [1.0] * len(widths)
    assert metrics["layer_specificity"] == [1.0] * len(widths)
    assert metrics["layer_matched_columns"] == list(widths)
    for counts in metrics["layer_graph_counts"]:
        assert counts[1] == 0 and counts[2] == 0


def test_identical_fit_is_perfect():
    B_true, A_true = _truth()
    metrics = layer_structure_recovery_metrics(B_true, B_true, A_true, A_true)
    assert metrics["alignment"] == "latent_state"
    _check_perfect(metrics, (4, 3, 2))
    assert metrics["layer_column_pairs"][0] == [[0, 0], [1, 1], [2, 2], [3, 3]]
    assert metrics["layer_graph_counts"][0] == [10, 0, 0, 22]


@pytest.mark.parametrize("alignment", ["latent_state", "coefficient"])
def test_column_permutations_and_flips_are_undone(alignment):
    B_true, A_true = _truth()
    perms = ([2, 0, 3, 1], [1, 2, 0], [1, 0])
    B_fit, A_fit = _permuted(B_true, A_true, perms, flips=[(0, 1), (1, 2)])
    kwargs = {} if alignment == "coefficient" else {"A_list": A_fit, "A_true_list": A_true}
    metrics = layer_structure_recovery_metrics(B_fit, B_true, **kwargs)
    assert metrics["alignment"] == alignment
    _check_perfect(metrics, (4, 3, 2))
    for d, perm in enumerate(perms):
        assert sorted(metrics["layer_column_pairs"][d]) == sorted(
            [[true, fit] for fit, true in enumerate(perm)]
        )


def test_permuting_layer_columns_without_their_child_rows_is_detected():
    # Only the columns of B_1 / A_1 are permuted; B_2 keeps the old row order,
    # so its rows no longer belong to the right layer-1 columns.
    B_true, A_true = _truth()
    perm = [1, 0, 2, 3]
    B_fit = [B.copy() for B in B_true]
    A_fit = [A.copy() for A in A_true]
    B_fit[0] = np.column_stack([B_true[0][:, 0], B_true[0][:, 1:][:, perm]])
    A_fit[0] = A_true[0][:, perm]
    metrics = layer_structure_recovery_metrics(B_fit, B_true, A_fit, A_true)
    assert metrics["layer_graph_accuracy"][0] == 1.0
    assert metrics["layer_graph_accuracy"][1] < 1.0
    assert metrics["layer_graph_accuracy"][2] == 1.0


def test_over_selected_layer_pads_truth_and_shifts_child_rows():
    B_true, A_true = _truth()
    rng = np.random.default_rng(5)
    extra_state = rng.integers(0, 2, size=(A_true[0].shape[0], 1)).astype(np.uint8)
    extra_column = np.zeros((8, 1))
    extra_column[[2, 6], 0] = 2.0
    # The spurious column sits at fitted position 1 of layer 1.
    B1 = np.column_stack([B_true[0][:, :2], extra_column, B_true[0][:, 2:]])
    A1 = np.column_stack([A_true[0][:, :1], extra_state, A_true[0][:, 1:]])
    extra_row = np.array([[-1.0, 0.0, 0.0, 2.5]])
    B2 = np.row_stack([B_true[1][:1], extra_row, B_true[1][1:]])
    metrics = layer_structure_recovery_metrics(
        [B1, B2, B_true[2]], B_true, [A1, A_true[1], A_true[2]], A_true
    )
    assert metrics["layer_matched_columns"] == [4, 3, 2]
    assert [0, 0] in metrics["layer_column_pairs"][0]
    assert [1, 2] in metrics["layer_column_pairs"][0]
    # Layer 1: 8 x 5 union, two spurious edges.
    assert metrics["layer_graph_counts"][0] == [10, 2, 0, 28]
    # Layer 2: 5 x 3 union, the spurious row adds one false edge.
    assert metrics["layer_graph_counts"][1] == [5, 1, 0, 9]
    assert metrics["layer_sensitivity"][:2] == [1.0, 1.0]
    assert metrics["layer_specificity"][0] == pytest.approx(28 / 30)
    assert metrics["layer_graph_accuracy"][1] == pytest.approx(14 / 15)
    assert metrics["layer_graph_counts"][2] == [4, 0, 0, 2]


def test_under_selected_layer_pads_fit_and_drops_child_row():
    B_true, A_true = _truth()
    keep = [0, 2, 3]
    B1 = np.column_stack([B_true[0][:, 0], B_true[0][:, 1:][:, keep]])
    B2 = B_true[1][keep]
    A1 = A_true[0][:, keep]
    metrics = layer_structure_recovery_metrics(
        [B1, B2, B_true[2]], B_true, [A1, A_true[1], A_true[2]], A_true
    )
    assert metrics["layer_matched_columns"][0] == 3
    # True column 1 loads on observed rows 1 and 5, and negatively on row 0;
    # B_2 row 1 (the dropped column's parents) has one missed edge.
    true_column_edges = int(np.sum(B_true[0][:, 2] != 0))
    assert metrics["layer_graph_counts"][0] == [10 - true_column_edges, 0, true_column_edges, 22]
    assert metrics["layer_graph_counts"][1] == [4, 0, 1, 7]
    assert metrics["layer_sensitivity"][1] == pytest.approx(0.8)
    assert metrics["layer_specificity"][1] == 1.0


def test_width_mismatch_with_permutation_in_both_layers():
    # Layer 1 over-selects and is permuted, layer 2 under-selects; the layer-2
    # rows must follow the layer-1 alignment for the score to stay exact.
    B_true, A_true = _truth()
    rng = np.random.default_rng(9)
    extra_state = rng.integers(0, 2, size=(A_true[0].shape[0], 1)).astype(np.uint8)
    B1 = np.column_stack([B_true[0], np.zeros((8, 1))])
    A1 = np.column_stack([A_true[0], extra_state])
    B2 = np.row_stack([B_true[1], np.zeros((1, 4))])
    perm1 = [4, 3, 0, 2, 1]
    B1 = np.column_stack([B1[:, 0], B1[:, 1:][:, perm1]])
    A1 = A1[:, perm1]
    B2 = B2[perm1]
    keep2 = [2, 0]  # drop true layer-2 column 1, reorder the rest
    B2 = np.column_stack([B2[:, 0], B2[:, 1:][:, keep2]])
    A2 = A_true[1][:, keep2]
    B3 = B_true[2][keep2]
    metrics = layer_structure_recovery_metrics([B1, B2, B3], B_true, [A1, A2, A_true[2]], A_true)
    assert metrics["layer_matched_columns"] == [4, 2, 2]
    assert metrics["layer_graph_counts"][0] == [10, 0, 0, 30]
    # Layer 2: union rows 5, columns 3; true column 1 has one edge (row 1).
    assert metrics["layer_graph_counts"][1] == [4, 0, 1, 10]
    # Layer 3: the true row of the dropped column has one edge.
    assert metrics["layer_graph_counts"][2] == [3, 0, 1, 2]


def test_threshold_controls_support():
    B_true, A_true = _truth()
    B_fit = [B.copy() for B in B_true]
    B_fit[0][1, 1] = 0.1  # spurious small entry
    B_fit[0][2, 3] = 0.2  # weak true entry
    loose = layer_structure_recovery_metrics(B_fit, B_true, A_true, A_true, threshold=0.0)
    strict = layer_structure_recovery_metrics(B_fit, B_true, A_true, A_true, threshold=0.3)
    assert loose["layer_graph_counts"][0] == [10, 1, 0, 21]
    assert strict["layer_graph_counts"][0] == [9, 0, 1, 22]


def test_empty_fitted_layer_and_unmatched_column():
    B_true, A_true = _truth()
    rng = np.random.default_rng(3)
    A1 = A_true[0].copy()
    A1[:, 3] = rng.integers(0, 2, size=A1.shape[0])  # unrelated state
    metrics = layer_structure_recovery_metrics(
        [B_true[0], B_true[1][:, :1], np.zeros((0, 3))],
        [B_true[0], B_true[1], B_true[2]],
        [A1, np.zeros((A1.shape[0], 0)), A_true[2]],
        A_true,
    )
    # Layer 3 keeps its latent states, so both columns still match.
    assert metrics["layer_matched_columns"] == [3, 0, 2]
    assert metrics["layer_graph_counts"][1] == [0, 0, 5, 7]
    assert metrics["layer_sensitivity"][1] == 0.0
    assert metrics["layer_column_pairs"][1] == []
    # Layer 3: no fitted rows, the true rows are padded in.
    assert metrics["layer_graph_counts"][2] == [0, 0, 4, 2]


def test_input_validation():
    B_true, A_true = _truth()
    with pytest.raises(ValueError):
        layer_structure_recovery_metrics(B_true[:2], B_true)
    with pytest.raises(ValueError):
        layer_structure_recovery_metrics(B_true, B_true, alignment="latent_state")
    with pytest.raises(ValueError):
        layer_structure_recovery_metrics([B_true[0], B_true[1][:3], B_true[2]], B_true)


def test_pooled_counts_equal_bernoulli_csp_driver():
    """Summed per-layer counts reproduce the driver's pooled graph metric."""
    import evaluate_bernoulli_csp_recovery as driver

    scenario = driver.build_scenario("d3")
    X, A_true = driver.simulate(scenario, 400, 11)
    rng = np.random.default_rng(11)
    # A perturbed, over/under-selected, permuted "fit".
    B_fit = [B.copy() + rng.normal(scale=0.2, size=B.shape) for B in scenario["B_true"]]
    A_fit = [A.copy() for A in A_true]
    extra = rng.integers(0, 2, size=(400, 2)).astype(np.uint8)
    A_fit[0] = np.column_stack([A_fit[0], extra])
    B_fit[0] = np.column_stack([B_fit[0], rng.normal(size=(64, 2))])
    B_fit[1] = np.row_stack([B_fit[1], rng.normal(size=(2, 7))])
    B_fit[1] = B_fit[1][:, :-1]  # drop the last layer-2 column
    A_fit[1] = A_fit[1][:, :-1]
    B_fit[2] = B_fit[2][:-1]
    perm = rng.permutation(20)
    B_fit[0] = np.column_stack([B_fit[0][:, 0], B_fit[0][:, 1:][:, perm]])
    A_fit[0] = A_fit[0][:, perm]
    B_fit[1] = B_fit[1][perm]
    result = SimpleNamespace(B_list=B_fit, A_list=A_fit, K_list=(20, 5, 2))
    pooled = driver.calculate_metrics(result, X, A_true, scenario, 0.3)
    metrics = layer_structure_recovery_metrics(
        B_fit, scenario["B_true"], A_fit, A_true, threshold=0.3
    )
    tp, fp, fn, tn = np.sum(metrics["layer_graph_counts"], axis=0)
    assert (tp, fp, fn, tn) == (
        pooled["true_positive"], pooled["false_positive"],
        pooled["false_negative"], pooled["true_negative"],
    )
