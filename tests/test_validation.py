import numpy as np
import pytest

from semidde.utils import DDEListShape, validate_K_list_identifiability, validate_dde_lists


def test_validate_dde_lists_accepts_three_layers():
    X = np.zeros((8, 54))
    B_list = [
        np.zeros((54, 19)),
        np.zeros((18, 7)),
        np.zeros((6, 3)),
    ]
    A_list = [
        np.zeros((8, 18)),
        np.zeros((8, 6)),
        np.zeros((8, 2)),
    ]

    shape = validate_dde_lists(
        X,
        B_list,
        A_list=A_list,
        prop=np.array([0.4, 0.6]),
        gamma=np.ones(54),
        K_list=[18, 6, 2],
        require_binary_A=True,
    )

    assert shape == DDEListShape(D=3, N=8, J=54, K_list=(18, 6, 2))


def test_validate_K_list_identifiability_rejects_narrow_parent_layer():
    with pytest.raises(ValueError, match=r"K_list\[d\] >= 3"):
        validate_K_list_identifiability([3, 2])


def test_validate_K_list_identifiability_rejects_too_few_observed_variables():
    with pytest.raises(ValueError, match=r"J >= 3 \* K_list\[0\]"):
        validate_K_list_identifiability([6, 2], J=17)


def test_validate_K_list_identifiability_accepts_single_node_deepest_layer():
    assert validate_K_list_identifiability([3, 1], J=9) == (3, 1)


def test_validate_dde_lists_accepts_one_layer():
    X = np.zeros((6, 6))
    B_list = [np.zeros((6, 3))]
    A_list = [np.ones((6, 2))]

    shape = validate_dde_lists(
        X,
        B_list,
        A_list=A_list,
        prop=np.array([0.2, 0.8]),
        gamma=np.ones(6),
        K_list=[2],
        require_binary_A=True,
    )

    assert shape.K_list == (2,)


def test_validate_dde_lists_rejects_mismatched_logistic_layer_rows():
    X = np.zeros((8, 9))
    B_list = [
        np.zeros((9, 4)),
        np.zeros((4, 3)),
    ]

    with pytest.raises(ValueError, match=r"B_list\[1\].*must have 3 rows"):
        validate_dde_lists(X, B_list)


def test_validate_dde_lists_rejects_mismatched_a_list_shape():
    X = np.zeros((8, 9))
    B_list = [np.zeros((9, 4))]
    A_list = [np.zeros((8, 2))]

    with pytest.raises(ValueError, match=r"A_list\[0\] shape"):
        validate_dde_lists(X, B_list, A_list=A_list)


def test_validate_dde_lists_rejects_invalid_prop_and_gamma():
    X = np.zeros((8, 9))
    B_list = [np.zeros((9, 4))]

    with pytest.raises(ValueError, match="prop entries"):
        validate_dde_lists(X, B_list, prop=np.array([0.2, 1.2, 0.5]))

    with pytest.raises(ValueError, match="gamma entries"):
        validate_dde_lists(X, B_list, gamma=np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
