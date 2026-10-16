"""Visualization of fitted DDE models.

Two primary interfaces (manuscript Section 5.4): :func:`plot_weights` draws
the estimated weight matrices and :func:`get_anchors` ranks the observed
variables that anchor each shallow-layer latent feature.  The earlier
multi-panel report bundle lives in :mod:`dde.visualization.legacy`.
"""

from __future__ import annotations

import warnings

from . import legacy
from .weights import anchor_scores, get_anchors, plot_weights

__all__ = ["anchor_scores", "get_anchors", "legacy", "plot_weights"]


def __getattr__(name):
    if name in legacy.__all__:
        warnings.warn(
            f"dde.visualization.{name} is part of the legacy report interface; "
            "import it from dde.visualization.legacy.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(legacy, name)
    raise AttributeError(f"module 'dde.visualization' has no attribute {name!r}")
