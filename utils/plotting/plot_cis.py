from __future__ import annotations

import math
import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_measure_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("sample_size", "sample_id")]


def plot_aggregated_measures_bootstrap_cis(
    measures_df: pd.DataFrame,
    bootstrap_ci_low_df: pd.DataFrame,
    bootstrap_ci_high_df: pd.DataFrame,
    out_path: str,
    agg: str = "mean",
    title: str | None = None,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 3.2),
) -> str:
    """
    Create a multi-panel plot (one subplot per measure) that shows, over sample_size:
      - aggregated center line (mean or median) of the measure values
      - aggregated CI low
      - aggregated CI high

    Parameters
    ----------
    measures_df : DataFrame with columns ['sample_size', 'sample_id', <measure columns...>]
    ci_low_df   : DataFrame with same shape/columns as measures_df for CI lows
    ci_high_df  : DataFrame with same shape/columns as measures_df for CI highs
    out_path    : File path to save the figure (PNG, PDF, etc.). Parent dir will be created.
    agg         : 'mean' or 'median' — which center to plot
    title       : Optional figure title
    ncols       : Number of subplot columns
    figsize_per_panel : Size per subplot; overall size is scaled by number of panels.

    Returns
    -------
    str : The path where the figure was saved.
    """
    agg = agg.lower()
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'.")

    # Ensure columns align and pick measure columns
    measure_cols = _get_measure_columns(measures_df)
    # guard: intersect with CI frames in case of drift
    measure_cols = [
        c
        for c in measure_cols
        if c in bootstrap_ci_low_df.columns and c in bootstrap_ci_high_df.columns
    ]

    # Group by sample_size and aggregate
    group = measures_df.groupby("sample_size", as_index=True)
    group_low = bootstrap_ci_low_df.groupby("sample_size", as_index=True)
    group_high = bootstrap_ci_high_df.groupby("sample_size", as_index=True)

    if agg == "mean":
        center = group[measure_cols].mean()
        low = group_low[measure_cols].mean()
        high = group_high[measure_cols].mean()
    else:  # median
        center = group[measure_cols].median()
        low = group_low[measure_cols].median()
        high = group_high[measure_cols].median()

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


def _get_measure_columns_sample(df: pd.DataFrame) -> List[str]:
    """Numeric metric columns excluding identifier fields."""
    id_cols = {"sample_size", "sample_id"}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in id_cols]


def plot_aggregated_measures_sample_cis(
    measures_df: pd.DataFrame,
    sample_ci_low_df: pd.DataFrame,
    sample_ci_high_df: pd.DataFrame,
    out_path: str,
    agg: str = "mean",
    title: str | None = None,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 3.2),
) -> str:
    """
    Create a multi-panel plot (one subplot per measure) that shows, over sample_size:
      - aggregated center line (mean or median) of the measure values across samples
      - aggregated CI low across samples
      - aggregated CI high across samples

    Parameters
    ----------
    measures_df : pd.DataFrame
        Columns: ['sample_size', 'sample_id', <metric columns...>]
        Contains raw per-sample metric values.
    sample_ci_low_df : pd.DataFrame
        CI-lower per sample_size for each metric (e.g., produced by your extractor).
        Must contain 'sample_size' and the same metric columns.
    sample_ci_high_df : pd.DataFrame
        CI-upper per sample_size for each metric (e.g., produced by your extractor).
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

    Returns
    -------
    str
        The saved figure path.
    """
    agg = agg.lower()
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'.")

    # Determine measure columns and ensure they exist in all inputs
    measure_cols = _get_measure_columns_sample(measures_df)
    measure_cols = [
        c
        for c in measure_cols
        if c in sample_ci_low_df.columns and c in sample_ci_high_df.columns
    ]
    if not measure_cols:
        raise ValueError("No overlapping metric columns found in the three DataFrames.")

    # ---- Aggregate center (mean/median) across samples for each size ----
    g = measures_df.groupby("sample_size", as_index=True, sort=True)
    if agg == "mean":
        center = g[measure_cols].mean()
    else:
        center = g[measure_cols].median()

    # Align CI frames to the same index (sample_size) and metrics
    # If the CI frames already have one row per sample_size, this is a no-op;
    # if they contain duplicate rows per size, we aggregate again for safety.
    low = sample_ci_low_df.groupby("sample_size", as_index=True, sort=True)[
        measure_cols
    ].mean()
    high = sample_ci_high_df.groupby("sample_size", as_index=True, sort=True)[
        measure_cols
    ].mean()

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
