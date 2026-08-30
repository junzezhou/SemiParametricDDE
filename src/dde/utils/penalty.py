"""Penalty and thresholding helpers."""

from __future__ import annotations

import numpy as np


def TLP(x, tau):
    """Truncated L1 penalty used by the original DDE implementation."""
    if tau <= 0:
        raise ValueError("tau must be positive.")
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.minimum(np.abs(x), tau)))


def thres(x, tau):
    """Hard-threshold entries with absolute value below ``tau``."""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    out[np.abs(out) < tau] = 0.0
    return out

