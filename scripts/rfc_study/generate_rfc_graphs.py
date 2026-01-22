"""Generate rank frequency curve (RFC) graphs for event logs.

This script analyzes trace variant frequencies in event logs and generates:
1. CSV with rank frequencies for all logs
2. CSV with correlation metrics (linear, exponential, power law)
3. All plots (4 scales: linear, logy, logx, loglog)
4. Per-dataset plots (4 scales) with correlation lines
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.util import xes

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import constants, helpers


def load_event_log(log_path: Path) -> EventLog:
    """Load an event log from a file path.

    Args:
        log_path: Path to the XES file (can be .xes or .xes.gz).

    Returns:
        PM4Py EventLog object.
    """
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    event_log = xes_importer.apply(
        str(log_path), variant=variant, parameters=parameters
    )
    return event_log


def extract_rank_frequencies(event_log: EventLog) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rank frequencies from an event log.

    Args:
        event_log: PM4Py EventLog object.

    Returns:
        Tuple of (ranks, relative_frequencies) as numpy arrays.
        Ranks are 1-indexed, frequencies are normalized to sum to 1.
    """
    # Get trace variants
    varmap = variants_filter.get_variants(event_log)

    # Count occurrences per variant
    variant_counts = []
    for traces in varmap.values():
        variant_counts.append(len(traces))

    # Sort by frequency (descending)
    variant_counts = sorted(variant_counts, reverse=True)

    # Convert to numpy arrays
    counts = np.array(variant_counts, dtype=float)
    total_traces = counts.sum()

    # Calculate relative frequencies
    relative_frequencies = counts / total_traces

    # Create ranks (1-indexed)
    ranks = np.arange(1, len(variant_counts) + 1)

    return ranks, relative_frequencies


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit a linear model y = a*x + b.

    Args:
        x: Independent variable (ranks).
        y: Dependent variable (frequencies).

    Returns:
        Tuple of (a, b, r_squared).
    """
    # Fit linear regression
    coeffs = np.polyfit(x, y, 1)
    a, b = coeffs

    # Calculate R-squared
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return a, b, r_squared


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit an exponential model y = a * exp(b*x).

    Args:
        x: Independent variable (ranks).
        y: Dependent variable (frequencies).

    Returns:
        Tuple of (a, b, r_squared).
    """
    # Filter out zeros and negative values for log transform
    mask = (y > 0) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, 0.0

    x_fit = x[mask]
    y_fit = y[mask]

    try:
        # Transform to linear: log(y) = log(a) + b*x
        log_y = np.log(y_fit)
        coeffs = np.polyfit(x_fit, log_y, 1)
        b, log_a = coeffs
        a = np.exp(log_a)

        # Calculate R-squared on original scale
        y_pred = a * np.exp(b * x_fit)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return a, b, r_squared
    except (ValueError, OverflowError):
        return np.nan, np.nan, 0.0


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit a power law model y = a * x^b.

    Args:
        x: Independent variable (ranks).
        y: Dependent variable (frequencies).

    Returns:
        Tuple of (a, b, r_squared).
    """
    # Filter out zeros and negative values for log transform
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, 0.0

    x_fit = x[mask]
    y_fit = y[mask]

    try:
        # Transform to linear: log(y) = log(a) + b*log(x)
        log_x = np.log(x_fit)
        log_y = np.log(y_fit)
        coeffs = np.polyfit(log_x, log_y, 1)
        b, log_a = coeffs
        a = np.exp(log_a)

        # Calculate R-squared on original scale
        y_pred = a * (x_fit**b)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return a, b, r_squared
    except (ValueError, OverflowError):
        return np.nan, np.nan, 0.0


def create_all_plots(
    rank_freq_df: pd.DataFrame,
    output_dir: Path,
    dataset_names: List[str],
) -> None:
    """Create 4 plots for all datasets with different scales.

    Args:
        rank_freq_df: DataFrame with rank frequencies (columns: rank, log1, log2, ...).
        output_dir: Directory to save plots.
        dataset_names: List of dataset names for column identification.
    """
    # Get rank column and log columns
    rank_col = rank_freq_df["rank"].values
    log_columns = [col for col in rank_freq_df.columns if col != "rank"]

    # Create 4 plots with different scales
    scales = [
        ("linear", False, False, "Linear Scale"),
        ("logy", False, True, "Log Y Scale"),
        ("logx", True, False, "Log X Scale"),
        ("loglog", True, True, "Log-Log Scale"),
    ]

    for scale_name, log_x, log_y, title_suffix in scales:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each log
        for col in log_columns:
            frequencies = rank_freq_df[col].values
            # Filter out NaN values
            mask = ~np.isnan(frequencies)
            if mask.sum() > 0:
                ax.plot(
                    rank_col[mask],
                    frequencies[mask],
                    "-",
                    alpha=0.6,
                    linewidth=1.0,
                    label=col,
                )

        ax.set_xlabel("Rank", fontsize=12)
        ax.set_ylabel("Relative Frequency", fontsize=12)
        ax.set_title(
            f"Rank Frequency Curves - {title_suffix}", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        plt.tight_layout()
        output_path = output_dir / f"rfc_all_{scale_name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path.name}")


def create_dataset_plots(
    rank_freq_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    log_column: str,
) -> None:
    """Create 4 plots for a single dataset with correlation lines.

    Args:
        rank_freq_df: DataFrame with rank frequencies.
        correlations_df: DataFrame with correlation parameters.
        output_dir: Base directory to save plots (will create subfolder).
        dataset_name: Name of the dataset.
        log_column: Column name in rank_freq_df for this dataset.
    """
    # Create subfolder for this dataset
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Get data
    rank_col = rank_freq_df["rank"].values
    frequencies = rank_freq_df[log_column].values

    # Filter out NaN values
    mask = ~np.isnan(frequencies)
    ranks = rank_col[mask]
    freqs = frequencies[mask]

    if len(ranks) == 0:
        print(f"  Warning: No data for {dataset_name}, skipping plots")
        return

    # Get correlation parameters for this dataset
    dataset_corr = correlations_df[correlations_df["dataset"] == dataset_name]
    if len(dataset_corr) == 0:
        print(f"  Warning: No correlation data for {dataset_name}, skipping plots")
        return

    # Extract parameters
    linear_a = dataset_corr["linear_a"].iloc[0]
    linear_b = dataset_corr["linear_b"].iloc[0]
    linear_r2 = dataset_corr["linear_r2"].iloc[0]

    exp_a = dataset_corr["exponential_a"].iloc[0]
    exp_b = dataset_corr["exponential_b"].iloc[0]
    exp_r2 = dataset_corr["exponential_r2"].iloc[0]

    power_a = dataset_corr["power_a"].iloc[0]
    power_b = dataset_corr["power_b"].iloc[0]
    power_r2 = dataset_corr["power_r2"].iloc[0]

    # Create 4 plots with different scales
    scales = [
        ("linear", False, False, "Linear Scale"),
        ("logy", False, True, "Log Y Scale"),
        ("logx", True, False, "Log X Scale"),
        ("loglog", True, True, "Log-Log Scale"),
    ]

    for scale_name, log_x, log_y, title_suffix in scales:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot actual data with line and markers
        ax.plot(
            ranks,
            freqs,
            "-o",
            markersize=3,
            linewidth=1.0,
            alpha=0.6,
            label="Data",
            color="black",
        )

        # Plot fitted curves
        x_plot = np.linspace(ranks.min(), ranks.max(), 1000)

        # Linear fit
        if not np.isnan(linear_a) and not np.isnan(linear_b):
            y_linear = linear_a * x_plot + linear_b
            # Only plot positive values
            y_linear = np.maximum(y_linear, 1e-10) if log_y else y_linear
            ax.plot(
                x_plot,
                y_linear,
                "--",
                label=f"Linear (R²={linear_r2:.3f})",
                linewidth=2,
                alpha=0.7,
            )

        # Exponential fit
        if not np.isnan(exp_a) and not np.isnan(exp_b):
            y_exp = exp_a * np.exp(exp_b * x_plot)
            # Only plot positive values
            y_exp = np.maximum(y_exp, 1e-10) if log_y else y_exp
            ax.plot(
                x_plot,
                y_exp,
                "--",
                label=f"Exponential (R²={exp_r2:.3f})",
                linewidth=2,
                alpha=0.7,
            )

        # Power law fit
        if not np.isnan(power_a) and not np.isnan(power_b):
            y_power = power_a * (x_plot**power_b)
            # Only plot positive values
            y_power = np.maximum(y_power, 1e-10) if log_y else y_power
            ax.plot(
                x_plot,
                y_power,
                "--",
                label=f"Power Law (R²={power_r2:.3f})",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_xlabel("Rank", fontsize=12)
        ax.set_ylabel("Relative Frequency", fontsize=12)
        ax.set_title(
            f"{dataset_name} - Rank Frequency Curve ({title_suffix})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        plt.tight_layout()
        output_path = dataset_dir / f"rfc_{scale_name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {dataset_name}/{output_path.name}")


def main() -> None:
    """Main function to generate RFC graphs."""
    parser = argparse.ArgumentParser(
        description="Generate rank frequency curve (RFC) graphs for event logs"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names to process (as defined in data dictionary)",
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        required=True,
        help='Name of the analysis (used in combined RFC plot filenames, e.g., "synthetic" or "real")',
    )

    args = parser.parse_args()

    # Load data dictionary
    data_dict_path = constants.get_data_dictionary_path()
    data_dictionary = helpers.load_data_dictionary(
        data_dict_path,
        get_real=True,
        get_synthetic=True,
    )

    # Validate datasets
    invalid_datasets = [ds for ds in args.datasets if ds not in data_dictionary]
    if invalid_datasets:
        print(f"Error: Invalid dataset names: {invalid_datasets}")
        print(f"Available datasets: {sorted(data_dictionary.keys())}")
        sys.exit(1)

    # Create output directory
    base_output_dir = PROJECT_ROOT / "results" / "rfc_study"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create analysis-specific subfolder
    analysis_dir = base_output_dir / f"all_{args.analysis_name}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(args.datasets)} dataset(s)...")

    # Process each dataset
    all_rank_freqs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    correlations_data = []

    for dataset_name in args.datasets:
        print(f"\nProcessing {dataset_name}...")
        dataset_info = data_dictionary[dataset_name]
        log_path = PROJECT_ROOT / dataset_info["path"]

        if not log_path.exists():
            print(f"  Warning: Log file not found: {log_path}")
            continue

        try:
            # Load event log
            print(f"  Loading log from {log_path}...")
            event_log = load_event_log(log_path)
            print(f"  Loaded {len(event_log)} traces")

            # Extract rank frequencies
            ranks, frequencies = extract_rank_frequencies(event_log)
            all_rank_freqs[dataset_name] = (ranks, frequencies)
            print(f"  Found {len(ranks)} distinct trace variants")

            # Fit correlations
            print(f"  Fitting correlations...")
            linear_a, linear_b, linear_r2 = fit_linear(ranks, frequencies)
            exp_a, exp_b, exp_r2 = fit_exponential(ranks, frequencies)
            power_a, power_b, power_r2 = fit_power_law(ranks, frequencies)

            # Determine best fit (highest R², only considering positive values)
            r2_values = {
                "Linear": (
                    linear_r2 if not np.isnan(linear_r2) and linear_r2 > 0 else -np.inf
                ),
                "Exponential": (
                    exp_r2 if not np.isnan(exp_r2) and exp_r2 > 0 else -np.inf
                ),
                "Power Law": (
                    power_r2 if not np.isnan(power_r2) and power_r2 > 0 else -np.inf
                ),
            }
            best_fit_model = (
                max(r2_values, key=r2_values.get)
                if max(r2_values.values()) > -np.inf
                else "None"
            )
            best_r2 = (
                max(r2_values.values()) if max(r2_values.values()) > -np.inf else np.nan
            )

            correlations_data.append(
                {
                    "dataset": dataset_name,
                    "linear_a": linear_a,
                    "linear_b": linear_b,
                    "linear_r2": linear_r2,
                    "exponential_a": exp_a,
                    "exponential_b": exp_b,
                    "exponential_r2": exp_r2,
                    "power_a": power_a,
                    "power_b": power_b,
                    "power_r2": power_r2,
                    "best_fit": best_fit_model,
                    "best_r2": best_r2,
                }
            )
            print(f"    Linear R²: {linear_r2:.4f}")
            print(f"    Exponential R²: {exp_r2:.4f}")
            print(f"    Power Law R²: {power_r2:.4f}")
            print(f"    Best Fit: {best_fit_model} (R²={best_r2:.4f})")

        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_rank_freqs:
        print("\nError: No datasets were successfully processed.")
        sys.exit(1)

    # Create rank frequency DataFrame
    print("\nCreating rank frequency CSV...")
    max_rank = max(len(ranks) for ranks, _ in all_rank_freqs.values())

    # Build DataFrame
    rank_freq_data = {"rank": np.arange(1, max_rank + 1)}
    for dataset_name, (ranks, frequencies) in all_rank_freqs.items():
        # Pad with NaN if needed
        padded_freqs = np.full(max_rank, np.nan)
        padded_freqs[: len(frequencies)] = frequencies
        rank_freq_data[dataset_name] = padded_freqs

    rank_freq_df = pd.DataFrame(rank_freq_data)
    rank_freq_csv_path = analysis_dir / "rank_frequencies.csv"
    rank_freq_df.to_csv(rank_freq_csv_path, index=False)
    print(f"  Saved: {rank_freq_csv_path.name}")

    # Create cumulative rank frequency DataFrame
    print("\nCreating cumulative rank frequency CSV...")
    cum_rank_freq_data = {"rank": rank_freq_df["rank"].values}
    for dataset_name in rank_freq_df.columns:
        if dataset_name != "rank":
            frequencies = rank_freq_df[dataset_name].values
            # Calculate cumulative sum, treating NaN as 0
            cum_frequencies = np.nancumsum(np.nan_to_num(frequencies, nan=0.0))
            # Set back to NaN where original was NaN (for ranks beyond the dataset's max)
            cum_frequencies[np.isnan(frequencies)] = np.nan
            cum_rank_freq_data[dataset_name] = cum_frequencies

    cum_rank_freq_df = pd.DataFrame(cum_rank_freq_data)
    cum_rank_freq_csv_path = analysis_dir / "cumulative_rank_frequencies.csv"
    cum_rank_freq_df.to_csv(cum_rank_freq_csv_path, index=False)
    print(f"  Saved: {cum_rank_freq_csv_path.name}")

    # Create correlations DataFrame
    print("\nCreating correlations CSV...")
    correlations_df = pd.DataFrame(correlations_data)
    correlations_csv_path = analysis_dir / "correlations.csv"
    correlations_df.to_csv(correlations_csv_path, index=False)
    print(f"  Saved: {correlations_csv_path.name}")

    # Create all plots
    print("\nCreating all plots...")
    create_all_plots(rank_freq_df, analysis_dir, args.datasets)

    # Create per-dataset plots
    print("\nCreating per-dataset plots...")
    for dataset_name in args.datasets:
        if dataset_name in all_rank_freqs:
            print(f"  Creating plots for {dataset_name}...")
            create_dataset_plots(
                rank_freq_df,
                correlations_df,
                base_output_dir,
                dataset_name,
                dataset_name,
            )

    print(f"\nAll outputs saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
