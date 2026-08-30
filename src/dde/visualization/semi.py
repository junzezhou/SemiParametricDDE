"""Compatibility wrappers for SemiDDE visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from dde.results import DDEResult

from .report import plot_latent_profiles, plot_model_summary, visualization_report


def plot_semi_model_summary(
    result: DDEResult,
    *,
    output_dir: str | Path,
    prefix: str = "semi",
    cv_table=None,
    feature_names: Optional[Sequence[str]] = None,
    max_heatmap_rows: int = 80,
    top_loadings: int = 8,
    include_item_cluster: bool = False,
) -> Dict[str, str]:
    """Create model-summary visualizations for a fitted SemiDDE result."""
    return plot_model_summary(
        result,
        output_dir=output_dir,
        model="semi",
        prefix=prefix,
        cv_table=cv_table,
        feature_names=feature_names,
        max_heatmap_rows=max_heatmap_rows,
        top_loadings=top_loadings,
        include_item_cluster=include_item_cluster,
    )


def plot_semi_latent_profiles(
    result: DDEResult,
    *,
    output_dir: str | Path,
    prefix: str = "semi",
    latent_frame=None,
    label_col: Optional[str] = None,
    alignment_weights=None,
    max_samples: int = 300,
    average_by_label: bool = False,
    continuous_label_bins: int = 10,
) -> Dict[str, str]:
    """Visualize SemiDDE latent profiles and label/posterior quality summaries."""
    return plot_latent_profiles(
        result,
        output_dir=output_dir,
        model="semi",
        prefix=prefix,
        latent_frame=latent_frame,
        label_col=label_col,
        alignment_weights=alignment_weights,
        max_samples=max_samples,
        average_by_label=average_by_label,
        continuous_label_bins=continuous_label_bins,
    )


def make_semi_visualization_report(
    result: DDEResult,
    *,
    output_dir: str | Path,
    prefix: str = "semi",
    cv_table=None,
    feature_names: Optional[Sequence[str]] = None,
    latent_frame=None,
    label_col: Optional[str] = None,
    alignment_weights=None,
    average_by_label: bool = False,
    continuous_label_bins: int = 10,
) -> Dict[str, str]:
    """Generate the standard SemiDDE visualization bundle."""
    return visualization_report(
        result,
        output_dir=output_dir,
        model="semi",
        prefix=prefix,
        cv_table=cv_table,
        feature_names=feature_names,
        latent_frame=latent_frame,
        label_col=label_col,
        alignment_weights=alignment_weights,
        average_by_label=average_by_label,
        continuous_label_bins=continuous_label_bins,
    )


__all__ = [
    "make_semi_visualization_report",
    "plot_semi_latent_profiles",
    "plot_semi_model_summary",
]
