import numpy as np

from semidde.utils.result_compression import compress_csp_result_components


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
