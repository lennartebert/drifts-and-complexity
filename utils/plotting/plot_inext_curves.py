"""
iNEXT rarefaction and extrapolation curve plotting utilities.

This module provides specialized plotting functions for iNEXT-style
rarefaction and extrapolation curves with bootstrap confidence intervals.
"""

import math
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import (
    bootstrap_inext_curve,
)
from utils.population.chao1 import (
    chao1_unseen_estimation_inext,
    create_inext_bootstrap_community,
    inext_richness_at_size_m,
)
from utils.population.population_distribution import PopulationDistribution


def plot_inext_curves(
    counts: Counter,
    out_path: str,
    m_grid: List[int] | None = None,
    B: int = 200,
    title: str | None = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
    seed: int = 42,
) -> str:
    """
    Plot iNEXT rarefaction and extrapolation curves with bootstrap confidence intervals.

    This function:
    1. Computes the iNEXT estimator curve S_hat(m) for the original sample
    2. Generates bootstrap confidence intervals using the iNEXT methodology
    3. Plots the center line and confidence bands

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for species.
    out_path : str
        File path to save the figure.
    m_grid : List[int], optional
        Sample sizes to evaluate. If None, generates default grid.
    B : int, default=200
        Number of bootstrap replicates.
    title : str, optional
        Figure title.
    figsize : Tuple[float, float], default=(8.0, 6.0)
        Figure size.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    str
        Path where the figure was saved.
    """
    n = sum(counts.values())
    S_obs = len(counts)

    # Generate m_grid if not provided
    if m_grid is None:
        m_min = max(1, int(0.5 * n))
        m_max = min(2 * n, int(2 * n))
        m_grid = list(range(m_min, m_max + 1))

    # Compute iNEXT estimator curve for original sample
    xs = list(counts.values())
    curve_mean = [inext_richness_at_size_m(xs, m) for m in m_grid]

    # Create a temporary PopulationDistribution for bootstrap community
    from utils.population.population_distribution import (
        create_chao1_population_distribution,
        get_labels_and_probabilities,
    )

    temp_pop_dist = create_chao1_population_distribution(counts, n)
    bootstrap_comm = create_inext_bootstrap_community(temp_pop_dist)
    labels, probs = get_labels_and_probabilities(bootstrap_comm)
    n_ref = n

    # Generate bootstrap confidence intervals
    rng = np.random.RandomState(seed)
    boot_results = bootstrap_inext_curve(labels, probs, n_ref, m_grid, B, rng)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot center line (iNEXT estimator)
    ax.plot(m_grid, curve_mean, "b-", linewidth=2, label="Mean (iNEXT estimator)")

    # Plot confidence intervals
    ax.plot(
        m_grid,
        boot_results["lo"],
        "orange",
        linestyle="--",
        alpha=0.8,
        label="CI low (bootstrap)",
    )
    ax.plot(
        m_grid,
        boot_results["hi"],
        "green",
        linestyle="--",
        alpha=0.8,
        label="CI high (bootstrap)",
    )

    # Fill between confidence intervals
    ax.fill_between(
        m_grid, boot_results["lo"], boot_results["hi"], alpha=0.2, color="gray"
    )

    # Add reference line at m=n
    ax.axvline(
        x=n, color="red", linestyle=":", alpha=0.7, label=f"Reference size (n={n})"
    )
    ax.axhline(
        y=S_obs,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"Observed richness (S_obs={S_obs})",
    )

    # Formatting
    ax.set_xlabel("Sample size (m)")
    ax.set_ylabel("Species richness")
    ax.set_title(
        title or f"iNEXT Rarefaction and Extrapolation Curve (n={n}, S_obs={S_obs})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Set reasonable axis limits
    ax.set_xlim(min(m_grid), max(m_grid))
    ax.set_ylim(0, max(curve_mean) * 1.1)

    # Save figure
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_inext_curves_from_dataframe(
    measures_df: pd.DataFrame,
    ci_low_df: pd.DataFrame,
    ci_high_df: pd.DataFrame,
    out_path: str,
    metric_name: str = "Number of Distinct Traces (iNEXT)",
    title: str | None = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
) -> str:
    """
    Plot iNEXT curves from existing DataFrame data.

    This is a fallback function for when you already have computed
    iNEXT curves and just want to plot them.

    Parameters
    ----------
    measures_df : pd.DataFrame
        DataFrame with iNEXT curve data.
    ci_low_df : pd.DataFrame
        DataFrame with CI low data.
    ci_high_df : pd.DataFrame
        DataFrame with CI high data.
    out_path : str
        File path to save the figure.
    metric_name : str, default="Number of Distinct Traces (iNEXT)"
        Name of the metric to plot.
    title : str, optional
        Figure title.
    figsize : Tuple[float, float], default=(8.0, 6.0)
        Figure size.

    Returns
    -------
    str
        Path where the figure was saved.
    """
    if metric_name not in measures_df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in measures DataFrame")

    # Get data
    x = measures_df.index.values  # sample_size
    y_mean = measures_df[metric_name].values
    y_low = ci_low_df[metric_name].values
    y_hi = ci_high_df[metric_name].values

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot center line
    ax.plot(x, y_mean, "b-", linewidth=2, label="Mean (iNEXT estimator)")

    # Plot confidence intervals
    ax.plot(x, y_low, "orange", linestyle="--", alpha=0.8, label="CI low (bootstrap)")
    ax.plot(x, y_hi, "green", linestyle="--", alpha=0.8, label="CI high (bootstrap)")

    # Fill between confidence intervals
    ax.fill_between(x, y_low, y_hi, alpha=0.2, color="gray")

    # Formatting
    ax.set_xlabel("Sample size (m)")
    ax.set_ylabel("Species richness")
    ax.set_title(title or f"iNEXT Curve: {metric_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Save figure
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return out_path
