"""Plotting utilities for correlation analysis results."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_correlation_results(
    correlations_df: pd.DataFrame,
    out_path: Path,
    plot_type: str = "box",  # "box" or "dot"
) -> None:
    """Plot correlation results (Pearson r) across multiple logs.

    Creates a visualization showing the distribution of correlation coefficients
    between sample size and various complexity measures across different datasets.

    Args:
        correlations_df: DataFrame with measures as index and logs as columns.
        out_path: Path where the plot will be saved.
        plot_type: Either "box" for boxplot or "dot" for dotplot.
        
    Raises:
        ValueError: If plot_type is not "box" or "dot".
        FileNotFoundError: If output directory cannot be created.
        
    Examples:
        >>> df = pd.DataFrame({'measure1': [0.5, 0.3], 'measure2': [0.8, 0.2]})
        >>> plot_correlation_results(df, Path('output.png'), 'box')
    """
    measures = correlations_df.index.tolist()
    data_by_measure = [correlations_df.loc[m].dropna().values for m in measures]

    plt.figure(figsize=(10, 6))

    if plot_type == "box":
        plt.boxplot(data_by_measure, labels=measures, showfliers=False)
    elif plot_type == "dot":
        for i, vals in enumerate(data_by_measure, start=1):
            plt.scatter([i] * len(vals), vals, alpha=0.6)
        plt.xticks(range(1, len(measures) + 1), measures)
    else:
        raise ValueError("plot_type must be 'box' or 'dot'")

    # Common formatting
    plt.axhline(0, linewidth=0.8, color="grey")
    plt.ylabel("Pearson r")
    plt.xticks(rotation=90)
    plt.xlabel("Measure")
    plt.title("Distribution of Pearson r Across Datasets by Measure")
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)
    plt.close()
