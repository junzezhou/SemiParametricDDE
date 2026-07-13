"""format_estimates.py fixed for a user-provided munkres(costmat) function.

This version assumes `munkres(costmat)` returns `(assignment, cost)`, where
`assignment` is a 0-based NumPy index vector, as in the uploaded `munkres.py`.
"""

from __future__ import annotations

import numpy as np

try:
    from semidde.utils.munkres import munkres
except ImportError:
    munkres = None

try:
    from semidde.copula.rescale_b1_cop import rescale_B1
except ImportError:
    rescale_B1 = None


def _as_index_vector(assignment, n_cols: int, name: str) -> np.ndarray:
    """Convert munkres assignment output to a valid 0-based index vector."""
    assignment = np.asarray(assignment, dtype=int).ravel()
    if assignment.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(assignment < 0):
        raise ValueError(f"{name} contains unassigned entries: {assignment}")
    if np.any(assignment >= n_cols):
        raise ValueError(
            f"{name} contains indices outside [0, {n_cols - 1}]: {assignment}. "
            "If your munkres implementation returns MATLAB-style 1-based indices, "
            "convert by subtracting 1 before using it for NumPy indexing."
        )
    return assignment


def _munkres_assignment(costmat: np.ndarray) -> np.ndarray:
    """Call user-provided munkres and return only the 0-based assignment vector."""
    if munkres is None:
        raise ImportError("format_estimates requires munkres.py in the Python path.")
    assignment, _ = munkres(costmat)   # critical: unpack `(assignment, cost)`
    return _as_index_vector(assignment, costmat.shape[1], "munkres assignment")


def format_estimates(
    B1_true_unscale,
    B2_true,
    B1_final,
    B2_final,
    gamma_final,
    prop_final,
    scale,
    Z,
    emp,
):
    """Align estimated parameters to the true latent-label ordering.

    This follows the MATLAB logic:
        [assignment1, ~] = munkres(costmat)
        B1_est = [B1_final(:,1), B1_final(:,2:end)(:, assignment1)]

    In Python, `assignment1` and `assignment2` must be 0-based index vectors.
    """
    B1_true_unscale = np.asarray(B1_true_unscale)
    B2_true = np.asarray(B2_true)
    B1_final = np.asarray(B1_final)
    B2_final = np.asarray(B2_final)
    gamma_final = np.asarray(gamma_final)
    prop_final = np.asarray(prop_final)

    J, K1 = B1_true_unscale[:, 1:].shape

    # -------- First-layer latent label alignment --------
    costmat = np.zeros((K1, K1), dtype=float)
    B1_no_intercept = B1_final[:, 1:]
    for k in range(K1):
        rows = B1_true_unscale[:, k + 1] > 0
        costmat[k, :] = -np.sum(B1_no_intercept[rows, :], axis=0)

    # print("1")

    assignment1 = _munkres_assignment(costmat)
    B1_est = np.concatenate(
        [B1_final[:, [0]], B1_no_intercept[:, assignment1]],
        axis=1,
    )

    # -------- Second-layer latent label alignment --------
    K2 = B2_true[:, 1:].shape[1]
    costmat = np.zeros((K2, K2), dtype=float)
    B2_no_intercept_reordered = B2_final[assignment1, 1:]
    for k in range(K2):
        rows = B2_true[:, k + 1] > 0
        costmat[k, :] = -np.sum(B2_no_intercept_reordered[rows, :], axis=0)

    # print("2")
    assignment2 = _munkres_assignment(costmat)
    B2_est = np.concatenate(
        [B2_final[assignment1, 0][:, None], B2_no_intercept_reordered[:, assignment2]],
        axis=1,
    )

    if assignment2.size == prop_final.size:
        prop_est = prop_final[assignment2]
    else:
        prop_est = np.nan

    gamma_est = np.ones((J, 1), dtype=float)
    # print("3")
    if bool(scale):
        if rescale_B1 is None:
            raise ImportError("format_estimates requires rescale_B1.py when scale=True.")
        B1_est, gamma_est, *_ = rescale_B1(prop_final, B2_est, B1_est, gamma_final, emp, Z)

    return B1_est, B2_est, gamma_est, prop_est, assignment1, assignment2
