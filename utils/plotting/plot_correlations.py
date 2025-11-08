"""Plotting utilities for correlation analysis results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_correlation_results(
    correlations_df: pd.DataFrame,
    out_path: Path,
    correlation_type: str = "Pearson",  # "Pearson" or "Spearman"
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
    # Choose the correct column
    if correlation_type == "Pearson":
        col = "Pearson_Rho"
        ylabel = "Pearson r"
        title = "Distribution of Pearson r Across Datasets by Measure"
    elif correlation_type == "Spearman":
        col = "Spearman_Rho"
        ylabel = "Spearman r"
        title = "Distribution of Spearman r Across Datasets by Measure"
    else:
        raise ValueError("correlation_type must be 'Pearson' or 'Spearman'")

    measures = correlations_df["Metric"].tolist()
    values = correlations_df[col].values

    plt.figure(figsize=(10, 6))

    if plot_type == "box":
        plt.boxplot([values], labels=[col], showfliers=False)
        plt.xticks([1], [col])
    elif plot_type == "dot":
        plt.scatter([1] * len(values), values, alpha=0.6)
        plt.xticks([1], [col])
    else:
        raise ValueError("plot_type must be 'box' or 'dot'")

    plt.axhline(0, linewidth=0.8, color="grey")
    plt.ylabel(ylabel)
    plt.ylim(-1, 1)
    plt.xticks(rotation=90)
    plt.xlabel("Measure")
    plt.title(title)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)
    plt.close()
