"""D-layer estimate alignment helpers for DDE copulas."""

from __future__ import annotations

import numpy as np

from dde.copula.rescale_b1_cop_D import rescale_B1_D
from dde.utils import validate_dde_lists
from dde.utils.munkres import munkres


def _assignment_from_cost(costmat: np.ndarray) -> np.ndarray:
    assignment, _ = munkres(costmat)
    assignment = np.asarray(assignment, dtype=int).reshape(-1)
    if np.any(assignment < 0):
        raise ValueError(f"Could not align all latent columns: {assignment}.")
    return assignment


def _layer_assignment(B_true_d: np.ndarray, B_est_d: np.ndarray) -> np.ndarray:
    K = B_true_d.shape[1] - 1
    costmat = np.zeros((K, K), dtype=float)
    est_slopes = B_est_d[:, 1:]
    for k in range(K):
        rows = B_true_d[:, k + 1] > 0
        if np.any(rows):
            costmat[k, :] = -np.sum(est_slopes[rows, :], axis=0)
        else:
            true_col = B_true_d[:, k + 1]
            costmat[k, :] = np.sum((est_slopes - true_col[:, None]) ** 2, axis=0)
    return _assignment_from_cost(costmat)


def format_estimates_D(
    B_list_true,
    B_list_final,
    gamma_final,
    prop_final,
    *,
    scale=False,
    Z=None,
    emp=False,
):
    """Align D-layer estimates to a true latent-label ordering.

    The alignment generalizes the original two-layer helper by moving from
    shallow to deep layers.  The permutation selected for layer ``d`` reorders
    slope columns in ``B_list[d]`` and rows in ``B_list[d + 1]``.
    """
    B_true = [np.asarray(B, dtype=float) for B in B_list_true]
    B_est = [np.asarray(B, dtype=float).copy() for B in B_list_final]
    gamma_est = np.asarray(gamma_final, dtype=float).reshape(-1)
    prop_est = np.asarray(prop_final, dtype=float).reshape(-1)

    J = B_true[0].shape[0]
    validate_dde_lists(np.zeros((1, J)), B_true)
    shape = validate_dde_lists(np.zeros((1, J)), B_est, prop=prop_est, gamma=gamma_est)
    D = shape.D

    assignments = []
    for d in range(D):
        assignment = _layer_assignment(B_true[d], B_est[d])
        assignments.append(assignment)
        B_est[d] = np.column_stack((B_est[d][:, 0], B_est[d][:, 1:][:, assignment]))
        if d + 1 < D:
            B_est[d + 1] = B_est[d + 1][assignment, :]

    prop_aligned = prop_est[assignments[-1]]
    gamma_aligned = gamma_est

    if scale:
        B_est, gamma_aligned, _, _ = rescale_B1_D(
            prop_aligned,
            B_est,
            gamma_aligned,
            emp=emp,
            Z=Z,
        )

    return {
        "B_list": B_est,
        "gamma": gamma_aligned,
        "prop": prop_aligned,
        "assignments": assignments,
    }


__all__ = ["format_estimates_D"]
