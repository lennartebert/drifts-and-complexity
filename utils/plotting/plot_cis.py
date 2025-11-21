from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import METRIC_BASIS_MAP, METRIC_DIMENSION_MAP


def _get_measure_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("sample_size", "sample_id")]


def _group_measures_by_breakdown(
    measure_cols: List[str], plot_breakdown: str | None
) -> dict[str | None, list[str]]:
    """Group measures by breakdown type (basis or dimension).

    Args:
        measure_cols: List of measure column names.
        plot_breakdown: None, "basis", or "dimension".

    Returns:
        Dictionary mapping breakdown key to list of measure columns.
    """
    measure_groups: dict[str | None, list[str]] = {}
    if plot_breakdown is None:
        # Single plot with all measures
        measure_groups[None] = measure_cols
    elif plot_breakdown == "basis":
        # Group by basis from METRIC_BASIS_MAP
        for col in measure_cols:
            basis = METRIC_BASIS_MAP.get(col, "Unknown")
            if basis not in measure_groups:
                measure_groups[basis] = []
            measure_groups[basis].append(col)
    elif plot_breakdown == "dimension":
        # Group by dimension from METRIC_DIMENSION_MAP
        for col in measure_cols:
            dimension = METRIC_DIMENSION_MAP.get(col, "Unknown")
            if dimension not in measure_groups:
                measure_groups[dimension] = []
            measure_groups[dimension].append(col)
    return measure_groups


def _prepare_breakdown_path_and_title(
    out_path: str, title: str | None, group_key: str | None
) -> tuple[str, str | None]:
    """Prepare output path and title for a breakdown group.

    Args:
        out_path: Original output path.
        title: Original title (may be None).
        group_key: Breakdown group key (None if no breakdown).

    Returns:
        Tuple of (modified_out_path, modified_title).
    """
    if group_key is None:
        # No breakdown - use original path and title
        return out_path, title
    else:
        # Add breakdown value to filename
        path_obj = Path(out_path)
        stem = path_obj.stem
        suffix = path_obj.suffix
        group_out_path = str(path_obj.parent / f"{stem}_{group_key}{suffix}")
        # Add breakdown to title if title is not None
        if title is not None:
            group_title = f"{title} — {group_key}"
        else:
            group_title = None
        return group_out_path, group_title


def plot_aggregated_measures_bootstrap_cis(
    long_df: pd.DataFrame,
    out_path: str,
    agg: str = "mean",
    title: str | None = None,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 3.2),
    plot_breakdown: str | None = None,
    metric_order: List[str] | None = None,
) -> str:
    """
    Create a multi-panel plot (one subplot per measure) that shows, over sample_size:
      - aggregated center line (mean or median) of the measure values
      - aggregated CI low
      - aggregated CI high

    Parameters
    ----------
    long_df : DataFrame in long format with columns:
              ['Sample Size', 'Sample ID', 'Metric', 'Value', 'Bootstrap CI Low', 'Bootstrap CI High']
              May have a MultiIndex (will be reset if present).
    out_path    : File path to save the figure (PNG, PDF, etc.). Parent dir will be created.
    agg         : 'mean' or 'median' — which center to plot
    title       : Optional figure title
    ncols       : Number of subplot columns
    figsize_per_panel : Size per subplot; overall size is scaled by number of panels.
    plot_breakdown : None, "basis", or "dimension" — if specified, creates separate plots
                     grouped by metric basis or dimension. Each plot is saved with the
                     breakdown value appended to the filename.
    metric_order : List[str] | None
                   Optional list of metric names in desired order. If provided, metrics will be
                   sorted according to this order instead of alphabetically.

    Returns
    -------
    str : The path where the figure was saved (or last path if multiple plots created).
    """
    agg = agg.lower()
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'.")

    if plot_breakdown not in (None, "basis", "dimension"):
        raise ValueError("plot_breakdown must be None, 'basis', or 'dimension'.")

    # Reset index if present
    if isinstance(long_df.index, pd.MultiIndex) or any(
        name is not None for name in long_df.index.names
    ):
        long_df = long_df.reset_index()

    # Check required columns
    required = {
        "Sample Size",
        "Sample ID",
        "Metric",
        "Value",
        "Bootstrap CI Low",
        "Bootstrap CI High",
    }
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"long_df is missing required columns: {missing}")

    # Get unique metrics, preserving order if metric_order is provided
    unique_metrics = long_df["Metric"].unique().tolist()
    if metric_order is not None:
        # Sort according to metric_order, keeping only metrics that exist in the data
        metric_order_dict = {metric: idx for idx, metric in enumerate(metric_order)}
        measure_cols = sorted(
            unique_metrics,
            key=lambda x: metric_order_dict.get(x, 9999),  # Unknown metrics go to end
        )
    else:
        measure_cols = sorted(unique_metrics)

    # Group measures by breakdown type if specified
    measure_groups = _group_measures_by_breakdown(measure_cols, plot_breakdown)

    # Generate plots for each group
    saved_paths = []
    for group_key, group_measures in measure_groups.items():
        if not group_measures:
            continue  # Skip empty groups

        # Preserve metric order within each group
        if metric_order is not None:
            metric_order_dict = {metric: idx for idx, metric in enumerate(metric_order)}
            group_measures = sorted(
                group_measures,
                key=lambda x: metric_order_dict.get(
                    x, 9999
                ),  # Unknown metrics go to end
            )

        # Prepare file path and title with breakdown suffix
        group_out_path, group_title = _prepare_breakdown_path_and_title(
            out_path, title, group_key
        )

        # Filter to this group's metrics
        group_long_df = long_df[long_df["Metric"].isin(group_measures)]

        # Create plot for this group
        saved_path = _create_bootstrap_ci_plot_long(
            group_long_df,
            group_measures,
            group_out_path,
            agg,
            group_title,
            ncols,
            figsize_per_panel,
        )
        saved_paths.append(saved_path)

    # Return the last saved path (or first if only one)
    return saved_paths[-1] if saved_paths else out_path


def _create_bootstrap_ci_plot_long(
    long_df: pd.DataFrame,
    measure_cols: List[str],
    out_path: str,
    agg: str,
    title: str | None,
    ncols: int,
    figsize_per_panel: Tuple[float, float],
) -> str:
    """Helper function to create a single bootstrap CI plot for a set of measures from long format."""

    # Group by Sample Size and Metric, then aggregate across Sample ID
    if agg == "mean":
        center = (
            long_df.groupby(["Sample Size", "Metric"], as_index=True)["Value"]
            .mean()
            .unstack("Metric")
        )
        low = (
            long_df.groupby(["Sample Size", "Metric"], as_index=True)[
                "Bootstrap CI Low"
            ]
            .mean()
            .unstack("Metric")
        )
        high = (
            long_df.groupby(["Sample Size", "Metric"], as_index=True)[
                "Bootstrap CI High"
            ]
            .mean()
            .unstack("Metric")
        )
    else:  # median
        center = (
            long_df.groupby(["Sample Size", "Metric"], as_index=True)["Value"]
            .median()
            .unstack("Metric")
        )
        low = (
            long_df.groupby(["Sample Size", "Metric"], as_index=True)[
                "Bootstrap CI Low"
            ]
            .median()
            .unstack("Metric")
        )
        high = (
            long_df.groupby(["Sample Size", "Metric"], as_index=True)[
                "Bootstrap CI High"
            ]
            .median()
            .unstack("Metric")
        )

    # Sort by sample_size to ensure monotone x
    center = center.sort_index()
    low = low.reindex(center.index)
    high = high.reindex(center.index)

    # Figure layout
    n_measures = len(measure_cols)
    nrows = math.ceil(n_measures / ncols)
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    x = center.index.values

    for i, m in enumerate(measure_cols):
        ax = axes_flat[i]

        # Check if measure exists in center DataFrame
        if m not in center.columns:
            # Measure doesn't exist in center - create empty subplot
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                alpha=0.7,
            )
        else:
            # Check if measure has any non-None values
            center_values = center[m].values
            has_valid_data = not all(pd.isna(center_values))

            if has_valid_data:
                # center line
                ax.plot(x, center_values, label=f"{agg.capitalize()}", linewidth=2)
                # CI low/high as lines
                ax.plot(x, low[m].values, label="CI low", linestyle="--")
                ax.plot(x, high[m].values, label="CI high", linestyle="--")
            else:
                # No valid data - create empty subplot with just title and grid
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    alpha=0.7,
                )

        ax.set_title(m, fontsize=10)
        ax.set_xlabel("Sample size")
        ax.set_xlim(x.min(), x.max())
        ax.grid(True, alpha=0.25)

        # Y label only on left-most column
        if i % ncols == 0:
            ax.set_ylabel("Value")

        # Keep legends tight
        ax.legend(fontsize=8, loc="best")

    # Hide any unused axes
    for j in range(n_measures, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_aggregated_measures_sample_cis(
    long_df: pd.DataFrame,
    out_path: str,
    agg: str = "mean",
    title: str | None = None,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 3.2),
    plot_breakdown: str | None = None,
    metric_order: List[str] | None = None,
) -> str:
    """
    Create a multi-panel plot (one subplot per measure) that shows, over sample_size:
      - aggregated center line (mean or median) of the measure values across samples
      - aggregated CI low across samples
      - aggregated CI high across samples

    Parameters
    ----------
    long_df : pd.DataFrame
        Long format DataFrame with columns:
        ['Sample Size', 'Sample ID', 'Metric', 'Value', 'Sample CI Low', 'Sample CI High']
        May have a MultiIndex (will be reset if present).
    out_path : str
        File path to save the figure (PNG, PDF, etc.). Parent directories are created.
    agg : {"mean","median"}, default "mean"
        How to aggregate the center line across samples at each sample_size.
    title : str | None
        Optional super-title for the whole figure.
    ncols : int
        Number of subplot columns.
    figsize_per_panel : (float, float)
        Size per subplot; overall size scales with number of panels.
    plot_breakdown : None, "basis", or "dimension" — if specified, creates separate plots
                     grouped by metric basis or dimension. Each plot is saved with the
                     breakdown value appended to the filename.
    metric_order : List[str] | None
                   Optional list of metric names in desired order. If provided, metrics will be
                   sorted according to this order instead of alphabetically.

    Returns
    -------
    str
        The saved figure path (or last path if multiple plots created).
    """
    agg = agg.lower()
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'.")

    if plot_breakdown not in (None, "basis", "dimension"):
        raise ValueError("plot_breakdown must be None, 'basis', or 'dimension'.")

    # Reset index if present
    if isinstance(long_df.index, pd.MultiIndex) or any(
        name is not None for name in long_df.index.names
    ):
        long_df = long_df.reset_index()

    # Check required columns
    required = {
        "Sample Size",
        "Sample ID",
        "Metric",
        "Value",
        "Sample CI Low",
        "Sample CI High",
    }
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"long_df is missing required columns: {missing}")

    # Get unique metrics, preserving order if metric_order is provided
    unique_metrics = long_df["Metric"].unique().tolist()
    if metric_order is not None:
        # Sort according to metric_order, keeping only metrics that exist in the data
        metric_order_dict = {metric: idx for idx, metric in enumerate(metric_order)}
        measure_cols = sorted(
            unique_metrics,
            key=lambda x: metric_order_dict.get(x, 9999),  # Unknown metrics go to end
        )
    else:
        measure_cols = sorted(unique_metrics)

    # Group measures by breakdown type if specified
    measure_groups = _group_measures_by_breakdown(measure_cols, plot_breakdown)

    # Generate plots for each group
    saved_paths = []
    for group_key, group_measures in measure_groups.items():
        if not group_measures:
            continue  # Skip empty groups

        # Preserve metric order within each group
        if metric_order is not None:
            metric_order_dict = {metric: idx for idx, metric in enumerate(metric_order)}
            group_measures = sorted(
                group_measures,
                key=lambda x: metric_order_dict.get(
                    x, 9999
                ),  # Unknown metrics go to end
            )

        # Prepare file path and title with breakdown suffix
        group_out_path, group_title = _prepare_breakdown_path_and_title(
            out_path, title, group_key
        )

        # Filter to this group's metrics
        group_long_df = long_df[long_df["Metric"].isin(group_measures)]

        # Create plot for this group
        saved_path = _create_sample_ci_plot_long(
            group_long_df,
            group_measures,
            group_out_path,
            agg,
            group_title,
            ncols,
            figsize_per_panel,
        )
        saved_paths.append(saved_path)

    # Return the last saved path (or first if only one)
    return saved_paths[-1] if saved_paths else out_path


def _create_sample_ci_plot_long(
    long_df: pd.DataFrame,
    measure_cols: List[str],
    out_path: str,
    agg: str,
    title: str | None,
    ncols: int,
    figsize_per_panel: Tuple[float, float],
) -> str:
    """Helper function to create a single sample CI plot for a set of measures from long format."""

    # Aggregate center (mean/median) across samples for each size and metric
    if agg == "mean":
        center = (
            long_df.groupby(["Sample Size", "Metric"], as_index=True)["Value"]
            .mean()
            .unstack("Metric")
        )
    else:
        center = (
            long_df.groupby(["Sample Size", "Metric"], as_index=True)["Value"]
            .median()
            .unstack("Metric")
        )

    # Sample CIs are already aggregated per Sample Size (no Sample ID), so just take first value
    # and unstack to wide format
    low = (
        long_df.groupby(["Sample Size", "Metric"], as_index=True)["Sample CI Low"]
        .first()
        .unstack("Metric")
    )
    high = (
        long_df.groupby(["Sample Size", "Metric"], as_index=True)["Sample CI High"]
        .first()
        .unstack("Metric")
    )

    # Reindex low/high to center's index to keep the same x-grid
    low = low.reindex(center.index)
    high = high.reindex(center.index)

    # ---- Figure layout ----
    n_measures = len(measure_cols)
    nrows = math.ceil(n_measures / ncols)
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    x = center.index.values

    for i, m in enumerate(measure_cols):
        ax = axes_flat[i]

        y = center[m].values
        y_low = low[m].values
        y_high = high[m].values

        # Plot center line and CI bounds
        ax.plot(x, y, label=f"{agg.capitalize()}", linewidth=2)
        ax.plot(x, y_low, label="CI low", linestyle="--")
        ax.plot(x, y_high, label="CI high", linestyle="--")

        # Axis cosmetics
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("Sample size")
        ax.set_xlim(x.min(), x.max())
        ax.grid(True, alpha=0.25)
        if i % ncols == 0:
            ax.set_ylabel("Value")

        # Keep legends compact
        ax.legend(fontsize=8, loc="best")

    # Hide any unused subplots
    for j in range(n_measures, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ci_results(
    metrics_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    out_dir: Path,
    plot_breakdown: str | None = None,
    ncols: int = 3,
    metric_order: List[str] | None = None,
) -> None:
    """Wrapper function that prepares data and triggers CI plotting.

    This function merges sample CIs from analysis_df into metrics_df for plotting,
    then creates both bootstrap CI and sample CI plots if the required columns are present.

    Args:
        metrics_df: DataFrame with raw metrics in long format (index: Metric, Sample Size).
        analysis_df: DataFrame with analysis results including Sample CI columns.
        out_dir: Directory where plots will be saved.
        plot_breakdown: Optional breakdown type ("basis" or "dimension") for grouping plots.
        ncols: Number of columns in the plot grid.
        metric_order: Optional list of metric names in desired order. If provided, metrics will be
                     sorted according to this order instead of alphabetically.
    """
    # Merge sample CIs back into metrics_df for plotting
    metrics_df_reset = metrics_df.reset_index()
    analysis_reset = analysis_df.reset_index()

    if (
        "Sample CI Low" in analysis_reset.columns
        and "Sample CI High" in analysis_reset.columns
    ):
        metrics_df_for_plotting = metrics_df_reset.merge(
            analysis_reset[
                ["Sample Size", "Metric", "Sample CI Low", "Sample CI High"]
            ],
            on=["Sample Size", "Metric"],
            how="left",
        )
    else:
        metrics_df_for_plotting = metrics_df_reset.copy()
        metrics_df_for_plotting["Sample CI Low"] = None
        metrics_df_for_plotting["Sample CI High"] = None

    # Create bootstrap CI plots if columns are present
    if (
        "Bootstrap CI Low" in metrics_df_for_plotting.columns
        and "Bootstrap CI High" in metrics_df_for_plotting.columns
    ):
        plot_aggregated_measures_bootstrap_cis(
            metrics_df_for_plotting,
            out_path=str(out_dir / "measures_bootstrap_cis_mean.png"),
            plot_breakdown=plot_breakdown,
            agg="mean",
            ncols=ncols,
            metric_order=metric_order,
        )

    # Create sample CI plots if columns are present
    if (
        "Sample CI Low" in metrics_df_for_plotting.columns
        and "Sample CI High" in metrics_df_for_plotting.columns
    ):
        plot_aggregated_measures_sample_cis(
            metrics_df_for_plotting,
            out_path=str(out_dir / "measures_sample_cis_mean.png"),
            agg="mean",
            plot_breakdown=plot_breakdown,
            ncols=ncols,
            metric_order=metric_order,
        )
