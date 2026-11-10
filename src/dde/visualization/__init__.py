"""Visualization of fitted DDE models.

Primary interfaces (manuscript Section 5.4 and Appendix C):
:func:`plot_weights` draws the estimated weight matrices, :func:`get_anchors`
ranks the observed variables that anchor each shallow-layer latent feature,
and :func:`visualization_report` writes the standard bundle of diagnostic
plots of a SemiDDE or CopDDE fit.  The remaining report helpers live in
:mod:`dde.visualization.legacy`.
"""

from __future__ import annotations

import warnings

from . import legacy
from .report import visualization_report
from .weights import anchor_scores, get_anchors, plot_weights

__all__ = ["anchor_scores", "get_anchors", "legacy", "plot_weights", "visualization_report"]


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
