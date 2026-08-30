import numpy as np

from dde.utils.result_compression import compress_csp_result_components


def test_compress_csp_result_components_keeps_active_columns_only():
    B1 = np.array(
        [
            [0.1, 1.0, 0.0, 0.0, 2.0],
            [0.2, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.5, 0.0, 0.0, 0.0],
        ]
    )
    B2 = np.array(
        [
            [0.0, 0.7, 0.0],
            [0.0, 8.0, 9.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ]
    )
    A1 = np.arange(20, dtype=float).reshape(5, 4)
    A2 = np.arange(10, dtype=float).reshape(5, 2)
    prop = np.array([0.4, 0.6])
    samples = [
        np.zeros((5, 5, 2), dtype=float),
        np.zeros((5, 3, 2), dtype=float),
    ]
    raw = {
        "prop": prop,
        "B_list": [B1, B2],
        "A_list": [A1, A2],
        "gamma": np.ones(3),
        "A_sample_long": samples,
    }

    out = compress_csp_result_components(
        prop=prop,
        B_list=[B1, B2],
        A_list=[A1, A2],
        diagnostics={"K_list": (4, 2), "num_active": np.array([2, 1])},
        raw=raw,
    )

    assert out["diagnostics"]["original_K_list"] == (4, 2)
    assert out["diagnostics"]["K_list"] == (2, 1)
    assert out["diagnostics"]["active_columns_original_index"] == {
        "layer_1": [1, 4],
        "layer_2": [1],
    }
    assert out["B_list"][0].shape == (3, 3)
    assert out["B_list"][1].shape == (2, 2)
    np.testing.assert_array_equal(out["A_list"][0], A1[:, [0, 3]])
    np.testing.assert_array_equal(out["A_list"][1], A2[:, [0]])
    np.testing.assert_array_equal(out["prop"], prop[[0]])
    assert out["raw"]["B_list"][1].shape == (2, 2)
    assert out["raw"]["A_sample_long"][0].shape == (5, 3, 2)
    assert out["raw"]["A_sample_long"][1].shape == (5, 2, 2)


def test_compression_can_use_csp_allocation_masks_instead_of_nonzero_loadings():
    B1 = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 4.0, 5.0, 6.0]])
    B2 = np.array(
        [[0.0, 1.0, 2.0], [0.0, 3.0, 4.0], [0.0, 5.0, 6.0]]
    )
    A1 = np.arange(12).reshape(4, 3)
    A2 = np.arange(8).reshape(4, 2)

    out = compress_csp_result_components(
        prop=np.array([0.3, 0.7]),
        B_list=[B1, B2],
        A_list=[A1, A2],
        active_masks=[np.array([False, True, False]), np.array([True, False])],
        min_active=False,
    )

    assert out["diagnostics"]["K_list"] == (1, 1)
    np.testing.assert_array_equal(out["B_list"][0], B1[:, [0, 2]])
    np.testing.assert_array_equal(out["B_list"][1], B2[[1], :][:, [0, 1]])
    np.testing.assert_array_equal(out["A_list"][0], A1[:, [1]])
    np.testing.assert_array_equal(out["prop"], np.array([0.3]))
