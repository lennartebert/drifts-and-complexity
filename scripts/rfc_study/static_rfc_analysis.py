"""Static RFC analysis from pre-extracted attachments.

This script performs a static rank-frequency-curve (RFC) study by reading
attachments CSV files (one per dataset/concept), then fitting and exporting
curve summaries and plots.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless backend for CLI/batch runs.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rfc_shared import load_attachments, parse_dataset_inputs


def extract_frequency_counts(attachments_df: pd.DataFrame) -> np.ndarray:
    """Return sorted absolute counts from node_id frequencies."""
    counts = attachments_df["node_id"].value_counts(dropna=False).to_numpy()
    return np.array(sorted(counts, reverse=True))


def extract_rank_frequencies(attachments_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return rank array and relative frequencies from attachments."""
    counts = extract_frequency_counts(attachments_df)
    if counts.size == 0:
        return np.array([]), np.array([])

    relative_frequencies = counts.astype(float) / counts.sum()
    ranks = np.arange(1, len(counts) + 1)
    return ranks, relative_frequencies


def _negative_log_likelihood(observed_freq: np.ndarray, predicted_values: np.ndarray) -> float:
    """Compute negative log-likelihood for observed frequencies.

    The observed frequencies are treated as empirical probabilities over rank.
    Predicted values are converted into a valid probability vector by clipping
    to a small positive epsilon and re-normalizing.
    """
    eps = 1e-12
    observed = np.asarray(observed_freq, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)

    # Ensure both vectors are finite and comparable.
    if observed.size == 0 or predicted.size != observed.size:
        return np.inf
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        return np.inf

    # Convert predictions to a probability distribution.
    predicted = np.maximum(predicted, eps)
    predicted_sum = predicted.sum()
    if predicted_sum <= 0:
        return np.inf
    predicted_prob = predicted / predicted_sum

    # Use empirical frequencies as weights in the log-likelihood sum.
    return float(-np.sum(observed * np.log(predicted_prob + eps)))


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = a*x + b and return a, b, negative log-likelihood."""
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    nll = _negative_log_likelihood(y, y_pred)
    return float(a), float(b), float(nll)


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = a*exp(b*x) and return a, b, negative log-likelihood."""
    mask = (y > 0) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.inf

    x_fit = x[mask]
    y_fit = y[mask]
    log_y = np.log(y_fit)
    b, log_a = np.polyfit(x_fit, log_y, 1)
    a = np.exp(log_a)
    # Evaluate likelihood on the full observed rank range.
    y_pred_full = a * np.exp(b * x)
    nll = _negative_log_likelihood(y, y_pred_full)
    return float(a), float(b), float(nll)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = C*x^(-alpha) and return C, alpha, negative log-likelihood."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.inf

    x_fit = x[mask]
    y_fit = y[mask]
    log_x = np.log(x_fit)
    log_y = np.log(y_fit)
    slope, log_c = np.polyfit(log_x, log_y, 1)
    alpha = -float(slope)
    c = float(np.exp(log_c))
    y_pred_full = c * (x ** (-alpha))
    nll = _negative_log_likelihood(y, y_pred_full)
    return c, alpha, float(nll)


def estimate_power_law_exponent(counts: np.ndarray, xmin: float = 1.0) -> float:
    """Estimate power-law exponent using the robust MLE form from Newman."""
    values = np.asarray(counts, dtype=float)
    tail = values[values >= xmin]
    if tail.size < 2:
        return float("nan")

    log_terms = np.log(tail / xmin)
    denom = np.sum(log_terms)
    if denom <= 0:
        return float("nan")
    return float(1.0 + tail.size / denom)


def create_all_plots(rank_freq_df: pd.DataFrame, output_dir: Path) -> None:
    """Create aggregate RFC plots across all datasets and scales."""
    rank_col = rank_freq_df["rank"].values
    dataset_cols = [col for col in rank_freq_df.columns if col != "rank"]
    scales = [
        ("linear", False, False, "Linear Scale"),
        ("logy", False, True, "Log Y Scale"),
        ("logx", True, False, "Log X Scale"),
        ("loglog", True, True, "Log-Log Scale"),
    ]

    for scale_name, log_x, log_y, _ in scales:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in dataset_cols:
            frequencies = rank_freq_df[col].values
            mask = ~np.isnan(frequencies)
            if mask.any():
                ax.plot(rank_col[mask], frequencies[mask], "-", linewidth=1.0, alpha=0.6, label=col)

        ax.set_xlabel("Rank")
        ax.set_ylabel("Relative Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        plt.tight_layout()
        out_path = output_dir / f"rfc_all_{scale_name}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")


def create_dataset_plots(
    rank_freq_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    base_output_dir: Path,
    dataset_name: str,
) -> None:
    """Create four per-dataset RFC plots (rfc_fitted_*.png) with fitted curves."""
    dataset_dir = base_output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rank_col = rank_freq_df["rank"].values
    freqs = rank_freq_df[dataset_name].values
    mask = ~np.isnan(freqs)
    ranks = rank_col[mask]
    y = freqs[mask]
    if len(ranks) == 0:
        return

    row = correlations_df[correlations_df["dataset"] == dataset_name].iloc[0]
    scales = [
        ("linear", False, False, "Linear Scale"),
        ("logy", False, True, "Log Y Scale"),
        ("logx", True, False, "Log X Scale"),
        ("loglog", True, True, "Log-Log Scale"),
    ]
    x_plot = np.linspace(ranks.min(), ranks.max(), 1000)

    for scale_name, log_x, log_y, _ in scales:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Raw RFC (same line style as aggregate "all" plots) plus fitted models.
        ax.plot(ranks, y, "-", linewidth=1.0, color="black", alpha=0.6, label="Data")

        # Plot each fit using already computed parameters.
        y_linear = row["linear_a"] * x_plot + row["linear_b"]
        y_exp = row["exponential_a"] * np.exp(row["exponential_b"] * x_plot)
        y_power = row["power_a"] * (x_plot ** (-row["power_b"]))
        if log_y:
            y_linear = np.maximum(y_linear, 1e-10)
            y_exp = np.maximum(y_exp, 1e-10)
            y_power = np.maximum(y_power, 1e-10)
        ax.plot(
            x_plot,
            y_linear,
            "--",
            linewidth=2,
            alpha=0.7,
            label=f"Linear (NLL={row['linear_nll']:.3f})",
        )
        ax.plot(
            x_plot,
            y_exp,
            "--",
            linewidth=2,
            alpha=0.7,
            label=f"Exponential (NLL={row['exponential_nll']:.3f})",
        )
        ax.plot(
            x_plot,
            y_power,
            "--",
            linewidth=2,
            alpha=0.7,
            label=f"Power Law (NLL={row['power_nll']:.3f})",
        )

        ax.set_xlabel("Rank")
        ax.set_ylabel("Relative Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        plt.tight_layout()
        out_path = dataset_dir / f"rfc_fitted_{scale_name}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {dataset_name}/{out_path.name}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for static RFC analysis."""
    parser = argparse.ArgumentParser(description="Static RFC analysis (CSVs + graphs + fitted curves)")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Dataset/attachments pairs: <dataset>=<attachments_csv_path>",
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        required=True,
        help="Subfolder name under the output root for this run (e.g. concept or batch label; dataset is not part of the path name)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results/rfc_study)",
    )
    return parser.parse_args()


def main() -> None:
    """Run static RFC analysis end-to-end."""
    args = parse_args()

    # Resolve output root with a simple local default.
    base_output_dir = Path(args.output_dir) if args.output_dir else Path("results") / "rfc_study"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = base_output_dir / args.analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_pairs = parse_dataset_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    all_rank_freqs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    correlations_data: List[dict] = []
    print(f"Processing {len(input_pairs)} dataset(s)...")

    for dataset_name, attachments_path in input_pairs:
        print(f"\nProcessing {dataset_name}...")
        if not attachments_path.exists():
            print(f"  Warning: attachments file not found: {attachments_path}")
            continue

        try:
            attachments_df = load_attachments(attachments_path)
        except ValueError as exc:
            print(f"  Warning: {exc}")
            continue

        ranks, frequencies = extract_rank_frequencies(attachments_df)
        counts = extract_frequency_counts(attachments_df)
        all_rank_freqs[dataset_name] = (ranks, frequencies)
        print(f"  Loaded {len(attachments_df)} attachments and found {len(ranks)} ranked nodes")

        # Fit all three simple models.
        linear_a, linear_b, linear_nll = fit_linear(ranks, frequencies)
        exp_a, exp_b, exp_nll = fit_exponential(ranks, frequencies)
        power_a, power_b, power_nll = fit_power_law(ranks, frequencies)
        estimated_power_law_exponent = estimate_power_law_exponent(counts)
        nll_values = {
            "Linear": linear_nll,
            "Exponential": exp_nll,
            "Power Law": power_nll,
        }
        finite_nll_values = {name: value for name, value in nll_values.items() if np.isfinite(value)}
        best_fit = min(finite_nll_values, key=finite_nll_values.get) if finite_nll_values else "None"
        best_nll = min(finite_nll_values.values()) if finite_nll_values else np.nan

        correlations_data.append(
            {
                "dataset": dataset_name,
                "linear_a": linear_a,
                "linear_b": linear_b,
                "linear_nll": linear_nll,
                "exponential_a": exp_a,
                "exponential_b": exp_b,
                "exponential_nll": exp_nll,
                "power_a": power_a,
                "power_b": power_b,
                "power_c": power_a,
                "power_alpha": power_b,
                "power_nll": power_nll,
                "estimated_power_law_exponent": estimated_power_law_exponent,
                "best_fit": best_fit,
                "best_nll": best_nll,
                "power_law_exponent_from_min_nll": float(power_b) if best_fit == "Power Law" else np.nan,
            }
        )

    if not all_rank_freqs:
        print("Error: no datasets processed successfully")
        raise SystemExit(1)

    # Build rank-frequency table with NaN padding for shorter datasets.
    max_rank = max(len(ranks) for ranks, _ in all_rank_freqs.values())
    rank_freq_data = {"rank": np.arange(1, max_rank + 1)}
    for dataset_name, (_, frequencies) in all_rank_freqs.items():
        padded = np.full(max_rank, np.nan)
        padded[: len(frequencies)] = frequencies
        rank_freq_data[dataset_name] = padded
    rank_freq_df = pd.DataFrame(rank_freq_data)
    rank_freq_df.to_csv(analysis_dir / "rank_frequencies.csv", index=False)

    # Build cumulative table for downstream curve-inspection workflows.
    cum_data = {"rank": rank_freq_df["rank"].values}
    for dataset_name in [c for c in rank_freq_df.columns if c != "rank"]:
        frequencies = rank_freq_df[dataset_name].values
        cum = np.nancumsum(np.nan_to_num(frequencies, nan=0.0))
        cum[np.isnan(frequencies)] = np.nan
        cum_data[dataset_name] = cum
    pd.DataFrame(cum_data).to_csv(analysis_dir / "cumulative_rank_frequencies.csv", index=False)

    correlations_df = pd.DataFrame(correlations_data)
    correlations_df.to_csv(analysis_dir / "static_analysis_summary.csv", index=False)
    create_all_plots(rank_freq_df, analysis_dir)
    for dataset_name, _ in input_pairs:
        if dataset_name in all_rank_freqs:
            create_dataset_plots(rank_freq_df, correlations_df, analysis_dir, dataset_name)

    print(f"\nAll static RFC outputs saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
