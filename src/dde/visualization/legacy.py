"""Legacy multi-panel reporting interface (kept for existing scripts).

The primary interfaces are :func:`dde.visualization.plot_weights` and
:func:`dde.visualization.get_anchors`.  The functions below produce the
earlier model-summary, latent-profile, theme-table and report bundles.
"""

from .copula import (
    make_copula_visualization_report,
    plot_copula_latent_profiles,
    plot_copula_model_summary,
    plot_copula_theme_table,
)
from .report import (
    plot_latent_profiles,
    plot_model_summary,
    plot_theme_table,
    visualization_report,
)
from .semi import (
    make_semi_visualization_report,
    plot_semi_latent_profiles,
    plot_semi_model_summary,
)

__all__ = [
    "make_copula_visualization_report",
    "make_semi_visualization_report",
    "plot_copula_latent_profiles",
    "plot_copula_model_summary",
    "plot_copula_theme_table",
    "plot_latent_profiles",
    "plot_model_summary",
    "plot_semi_latent_profiles",
    "plot_semi_model_summary",
    "plot_theme_table",
    "visualization_report",
]
