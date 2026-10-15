"""Check the scientific metric used by the validation runner."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_hp_selection_validation import _aligned_mse, _data  # noqa: E402
from evaluate_bernoulli_fixed_recovery import _paper_sparse_coefficients  # noqa: E402


def test_fixed_parameter_mse_is_invariant_to_latent_label_permutations():
    B1 = np.array([[0.2, 2.0, -1.0, 0.0],
                   [-0.4, 0.0, 1.7, -2.1],
                   [0.1, -1.4, 0.0, 0.9],
                   [0.3, 1.2, 0.0, -1.8]])
    B2 = np.array([[0.1, 1.3], [-0.2, -1.2], [0.5, 0.8]])
    true = {"B_true": [B1, B2], "prop": np.array([0.4]),
            "gamma": np.ones(4)}
    order = [2, 0, 1]
    fitted_B1 = B1.copy()
    fitted_B1[:, 1:] = fitted_B1[:, 1:][:, order]
    fitted_B2 = B2[order, :]
    result = SimpleNamespace(B_list=[fitted_B1, fitted_B2],
                             prop=true["prop"], gamma=true["gamma"])
    raw, thresholded = _aligned_mse(result, true, tau=0.01, copula=False)
    assert raw < 1e-28
    assert thresholded < 1e-28


def test_normal_and_poisson_fixed_truth_use_paper_sparse_coefficients():
    for kernel in ("dde_normal_fixed", "dde_poisson_fixed"):
        for depth in (2, 3):
            observed, scenario = _data(kernel, depth, 12, 42)
            assert observed.shape == (12, 3 * scenario["K_true"][0])
            for B, width in zip(scenario["B_true"], scenario["K_true"]):
                np.testing.assert_array_equal(B, _paper_sparse_coefficients(width))
            if kernel == "dde_normal_fixed":
                np.testing.assert_array_equal(scenario["gamma"], np.ones(observed.shape[1]))
