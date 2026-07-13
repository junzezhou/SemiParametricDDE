"""Shared D=2/D=3 CopDDE simulation helpers for script experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semidde.algorithms.copula_D import normal_initialize_copula_dde_D
from semidde.copula import rescale_B1_D, simulate_gaussian_mixture
from semidde.simulation import generate_X_Cop_marg_emp_D


MATLAB_APPENDIX_C_NEG_COLS = {
    (108, 18, 6, 1): np.array([4, 14, 10, 13, 2, 16, 14, 9, 4, 17, 1, 17, 4, 1, 8, 4, 4, 5]) - 1,
    (18, 6, 3, 1): np.array([2, 5, 4, 5, 1, 5]) - 1,
}


def compute_rank_matrix(X):
    X = np.asarray(X)
    N, J = X.shape
    R = np.full((N, J), np.nan, dtype=float)
    Rlevels = np.zeros(J, dtype=float)
    for j in range(J):
        obs = ~np.isnan(X[:, j])
        if not np.any(obs):
            continue
        _, inverse = np.unique(X[obs, j], return_inverse=True)
        R[obs, j] = inverse + 1
        Rlevels[j] = np.nanmax(R[:, j])
    return R, Rlevels


def sim_block_loadings(J, K1, block_size=10, pos_mag=3.0, neg_mag=None, noise_sd=0.0, seed=None):
    if neg_mag is None:
        neg_mag = pos_mag
    rng = np.random.default_rng(seed)
    J = int(J)
    K1 = int(K1)
    B = float(noise_sd) * rng.standard_normal((J, K1))
    n_blocks = int(np.ceil(J / int(block_size)))
    all_cols = np.arange(K1)
    for block in range(n_blocks):
        rows = slice(block * int(block_size), min((block + 1) * int(block_size), J))
        pos_col = min(block, K1 - 1)
        neg_choices = all_cols[all_cols != pos_col]
        neg_col = neg_choices[rng.integers(len(neg_choices))]
        B[rows, pos_col] += float(pos_mag)
        B[rows, neg_col] -= float(neg_mag)
    return B


def sim_block_loadings_matlab_appendix_c(J, K1, block_size=10, pos_mag=3.0, neg_mag=None, noise_sd=0.0, seed=None):
    if neg_mag is None:
        neg_mag = pos_mag
    key = (int(J), int(K1), int(block_size), int(seed) if seed is not None else None)
    if float(noise_sd) != 0.0 or key not in MATLAB_APPENDIX_C_NEG_COLS:
        return sim_block_loadings(J, K1, block_size, pos_mag, neg_mag, noise_sd, seed)

    neg_cols = MATLAB_APPENDIX_C_NEG_COLS[key]
    B = np.zeros((int(J), int(K1)), dtype=float)
    for block, neg_col in enumerate(neg_cols):
        rows = slice(block * int(block_size), min((block + 1) * int(block_size), int(J)))
        B[rows, block] += float(pos_mag)
        B[rows, neg_col] -= float(neg_mag)
    return B


def make_middle_loadings(K_child, K_parent, pos=4.0, neg=None):
    if neg is None:
        neg = pos / 3.0
    B = np.zeros((int(K_child), int(K_parent)), dtype=float)
    idx = np.floor(np.linspace(1, int(K_child), int(K_parent) + 1)).astype(int)
    for one_based in range(1, len(idx)):
        rows = slice(idx[one_based - 1] - 1, idx[one_based])
        col = one_based - 1
        if one_based % 2 == 0:
            B[rows, col] = pos
            B[rows, col - 1] = -neg
        else:
            B[rows, col] = pos
            if one_based < len(idx) - 2:
                B[rows, col + 1] = -neg
    return B


def make_true_model(scenario):
    D = int(scenario["D"])
    J = int(scenario["J"])
    K_true = list(map(int, scenario["K_true"]))
    K_fit = list(map(int, scenario["K_fit"]))
    intercept = -2.0

    if D == 2:
        B1_sub = sim_block_loadings(
            J,
            K_true[0],
            block_size=J // K_true[0],
            pos_mag=10.0,
            neg_mag=5.0,
            seed=1,
        )
        B2_sub = make_middle_loadings(K_true[0], K_true[1], pos=4.0, neg=4.0 / 3.0)
        B_true = [
            np.column_stack((intercept * np.ones(J), B1_sub)),
            np.column_stack((intercept * np.ones(K_true[0]), B2_sub)),
        ]
    elif D == 3:
        block_loader = sim_block_loadings_matlab_appendix_c if J == 108 else sim_block_loadings
        B1_sub = block_loader(
            J,
            K_true[0],
            block_size=J // K_true[0],
            pos_mag=10.0,
            neg_mag=5.0,
            seed=1,
        )
        B2_sub = block_loader(
            K_true[0],
            K_true[1],
            block_size=K_true[0] // K_true[1],
            pos_mag=4.0,
            neg_mag=4.0 / 3.0,
            seed=1,
        )
        B3_sub = make_middle_loadings(K_true[1], K_true[2], pos=4.0, neg=4.0 / 3.0)
        B_true = [
            np.column_stack((intercept * np.ones(J), B1_sub)),
            np.column_stack((intercept * np.ones(K_true[0]), B2_sub)),
            np.column_stack((intercept * np.ones(K_true[1]), B3_sub)),
        ]
    else:
        raise ValueError("Only D=2 and D=3 are supported.")

    B_aug = []
    row_dims = [J] + K_fit[:-1]
    for d in range(D):
        B = np.zeros((row_dims[d], K_fit[d] + 1), dtype=float)
        B[: B_true[d].shape[0], : B_true[d].shape[1]] = B_true[d]
        B_aug.append(B)

    rng_lambda = np.random.default_rng(50)
    return {
        "B_true": B_true,
        "B_aug_unscaled": B_aug,
        "prop_true": 0.5 * np.ones(K_true[-1]),
        "gamma_true": np.ones(J),
        "lambdas": rng_lambda.integers(1, 11, size=J),
    }


def initialize_state(scenario, N, rep_index, seed_base):
    seed = int(seed_base + rep_index + 1)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    true = make_true_model(scenario)
    generated = generate_X_Cop_marg_emp_D(
        int(N),
        true["prop_true"],
        true["lambdas"],
        true["gamma_true"],
        true["B_true"],
        random_state=rng,
    )
    X = generated["X"]
    R, Rlevels = compute_rank_matrix(X)
    Z_init = simulate_gaussian_mixture(X, int(scenario["J"]), random_state=seed)
    init = normal_initialize_copula_dde_D(
        X,
        scenario["K_fit"],
        Z=Z_init,
        epsilon=1e-4,
    )
    B_scaled, gamma_scaled, _, _ = rescale_B1_D(
        true["prop_true"], true["B_true"], true["gamma_true"], emp=True, Z=generated["Z"]
    )
    B1_scaled_aug = np.zeros_like(true["B_aug_unscaled"][0])
    B1_scaled_aug[:, : B_scaled[0].shape[1]] = B_scaled[0]
    true_scaled = [B1_scaled_aug] + true["B_aug_unscaled"][1:]
    return {
        "seed": seed,
        "X": X,
        "R": R,
        "Rlevels": Rlevels,
        "Z_init": Z_init,
        "init": init,
        "true_scaled": true_scaled,
        "gamma_scaled": gamma_scaled,
        "true": true,
    }
