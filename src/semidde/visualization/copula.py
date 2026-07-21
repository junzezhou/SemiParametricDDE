"""Compatibility wrappers for CopDDE visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from semidde.results import DDEResult

from .report import (
    plot_latent_profiles,
    plot_model_summary,
    plot_theme_table,
    visualization_report,
)


def plot_copula_model_summary(
    result: DDEResult,
    *,
    output_dir: str | Path,
    prefix: str = "copula",
    cv_table=None,
    feature_names: Optional[Sequence[str]] = None,
    max_heatmap_rows: int = 80,
    top_loadings: int = 8,
) -> Dict[str, str]:
    """Create model-summary visualizations for a fitted CopDDE result."""
    return plot_model_summary(
        result,
        output_dir=output_dir,
        model="copula",
        prefix=prefix,
        cv_table=cv_table,
        feature_names=feature_names,
        max_heatmap_rows=max_heatmap_rows,
        top_loadings=top_loadings,
    )


def plot_copula_latent_profiles(
    result: DDEResult,
    *,
    output_dir: str | Path,
    prefix: str = "copula",
    latent_frame=None,
    label_col: Optional[str] = None,
    max_samples: int = 300,
    average_by_label: bool = False,
    continuous_label_bins: int = 10,
) -> Dict[str, str]:
    """Visualize sample latent profiles and optional label-averaged profiles."""
    return plot_latent_profiles(
        result,
        output_dir=output_dir,
        model="copula",
        prefix=prefix,
        latent_frame=latent_frame,
        label_col=label_col,
        max_samples=max_samples,
        average_by_label=average_by_label,
        continuous_label_bins=continuous_label_bins,
    )


def plot_copula_theme_table(
    comparison_table,
    *,
    output_dir: str | Path,
    prefix: str = "copula",
    title: str = "Latent theme comparison",
) -> Dict[str, str]:
    """Render a compact theme-comparison table or prediction confusion matrix."""
    return plot_theme_table(
        comparison_table,
        output_dir=output_dir,
        prefix=prefix,
        title=title,
    )


def make_copula_visualization_report(
    result: DDEResult,
    *,
    output_dir: str | Path,
    prefix: str = "copula",
    cv_table=None,
    feature_names: Optional[Sequence[str]] = None,
    latent_frame=None,
    label_col: Optional[str] = None,
    theme_table=None,
    average_by_label: bool = False,
    continuous_label_bins: int = 10,
) -> Dict[str, str]:
    """Generate the standard CopDDE visualization bundle."""
    return visualization_report(
        result,
        output_dir=output_dir,
        model="copula",
        prefix=prefix,
        cv_table=cv_table,
        feature_names=feature_names,
        latent_frame=latent_frame,
        label_col=label_col,
        theme_table=theme_table,
        average_by_label=average_by_label,
        continuous_label_bins=continuous_label_bins,
    )


__all__ = [
    "make_copula_visualization_report",
    "plot_copula_latent_profiles",
    "plot_copula_model_summary",
    "plot_copula_theme_table",
]
