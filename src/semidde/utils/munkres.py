"""Fast Munkres/Hungarian assignment wrapper.

The project uses this helper from ``format_estimates`` to permute latent
columns.  Its public interface is intentionally kept as:

    assignment, cost = munkres(costMat)

where ``assignment`` is a 0-based NumPy index vector with ``-1`` for rows that
cannot be assigned to any finite column.  This matches the existing Python
callers in this repository.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _large_penalty(finite_values: np.ndarray, size: int) -> float:
    """Return a finite dummy cost larger than any useful assignment."""
    finite_values = np.asarray(finite_values, dtype=float)
    if finite_values.size == 0:
        return 1.0

    cmin = float(np.min(finite_values))
    cmax = float(np.max(finite_values))
    span = max(cmax - cmin, 1.0)
    return cmax + (size + 1.0) * span + 1.0


def munkres(costMat):
    """Solve a rectangular linear assignment problem.

    Parameters
    ----------
    costMat : array-like, shape (m, n)
        Cost matrix. ``NaN`` and ``+/-Inf`` are treated as invalid assignments.

    Returns
    -------
    assignment : ndarray, shape (m,)
        0-based assigned column for each row. Rows with no valid assignment are
        returned as ``-1``.
    cost : float
        Sum of the original finite costs for assigned row/column pairs.

    Notes
    -----
    This is equivalent to the repository's previous Munkres helper at the
    interface level, but delegates the Hungarian algorithm to SciPy's optimized
    ``linear_sum_assignment`` implementation.  The previous pure-Python state
    machine could become very slow on degenerate or negative cost matrices.
    """
    C0 = np.asarray(costMat, dtype=float)
    if C0.ndim != 2:
        raise ValueError(f"costMat must be 2-D; got shape {C0.shape}.")

    m, n = C0.shape
    assignment = -np.ones(m, dtype=int)
    if m == 0 or n == 0:
        return assignment, 0.0

    valid = np.isfinite(C0)
    if not np.any(valid):
        return assignment, 0.0

    size = max(m, n)
    dummy_cost = _large_penalty(C0[valid], size)

    # Pad to square so unmatched rows/columns are represented by dummy matches.
    # Invalid real assignments also receive the dummy cost and are filtered out
    # after solving.
    C = np.full((size, size), dummy_cost, dtype=float)
    C[:m, :n] = np.where(valid, C0, dummy_cost)

    row_ind, col_ind = linear_sum_assignment(C)

    total_cost = 0.0
    for row, col in zip(row_ind, col_ind):
        if row < m and col < n and valid[row, col]:
            assignment[row] = int(col)
            total_cost += float(C0[row, col])

    return assignment, float(total_cost)
