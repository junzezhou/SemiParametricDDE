"""Dimension-recovery metrics shared by CSP simulation studies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def dimension_recovery_metrics(
    fitted_widths: Sequence[int], true_widths: Sequence[int]
) -> dict[str, int]:
    """Return exact width recovery and the width-vector L1 error."""
    fitted_raw = np.asarray(fitted_widths)
    truth_raw = np.asarray(true_widths)
    try:
        valid = (
            np.all(np.isfinite(fitted_raw.astype(float)))
            and np.all(np.isfinite(truth_raw.astype(float)))
            and np.all(fitted_raw.astype(float) == fitted_raw.astype(int))
            and np.all(truth_raw.astype(float) == truth_raw.astype(int))
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("dimension widths must be nonnegative integer sequences.")
    fitted = fitted_raw.astype(int).reshape(-1)
    truth = truth_raw.astype(int).reshape(-1)
    if fitted.shape != truth.shape:
        raise ValueError(
            "fitted_widths and true_widths must have the same number of layers."
        )
    if fitted.size < 1 or np.any(fitted < 0) or np.any(truth < 0):
        raise ValueError("dimension widths must be nonnegative integer sequences.")
    return {
        "exact_dimension_recovery": int(np.array_equal(fitted, truth)),
        "dimension_l1_error": int(np.sum(np.abs(fitted - truth))),
    }


__all__ = ["dimension_recovery_metrics"]
