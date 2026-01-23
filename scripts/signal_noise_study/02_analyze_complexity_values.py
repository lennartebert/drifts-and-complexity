"""Analyze complexity values to compute relative noise under stability.

This script processes aggregate analysis results to compute relative noise metrics
for different complexity levels, noise levels, and window sizes during stable
(pre-drift) periods.

Produces both long-format (analysis-ready) and wide-format (presentation-ready)
outputs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuration
PATH_AGG = "results/signal_noise_study/aggregate_analysis.csv"
PATH_GEN_INFO = "data/synthetic/sudden_drifts/generation_info.csv"
PATH_OUT_WIDE = "results/signal_noise_study/relative_noise_wide.csv"
PATH_OUT_LONG = "results/signal_noise_study/relative_noise_long.csv"
PATH_SNR_LONG = "results/signal_noise_study/snr_long.csv"
PATH_SNR_WIDE = "results/signal_noise_study/snr_wide.csv"
PATH_SNR_BY_OP = "results/signal_noise_study/snr_by_operation.csv"
PATH_SNR_BY_EVOL = "results/signal_noise_study/snr_by_evolution_proportion.csv"
PATH_SNR_BY_NOISE = "results/signal_noise_study/snr_by_noise.csv"
PATH_NOISE_FACTOR_LONG = "results/signal_noise_study/noise_factor_importance_long.csv"
PATH_NOISE_FACTOR_WIDE = "results/signal_noise_study/noise_factor_importance_wide.csv"
choose_aggregation: Literal["mean", "median"] = "mean"
eps = 1e-9
MIN_OBS_FOR_FACTOR_ANALYSIS = 30

# Required columns in aggregate_analysis.csv
REQUIRED_COLS = [
    "Mean Value",
    "Sample CI Low",
    "Sample CI High",
    "log_id",
    "split_name",
    "window_size",
]

# Stable split name variants (normalized internally)
STABLE_SPLIT_NAMES = {"pre_drift", "pre-drift", "pre"}

# Noise probability to noise level mapping
NOISE_PROB_TO_LEVEL = {0.0: "0", 0.1: "low", 0.2: "high"}

# Complexity level mapping (from generation_info.csv)
COMPLEXITY_MAPPING = {"simple": "simple", "middle": "mid", "complex": "complex"}

# Change operation mapping (from generation_info.csv)
# Keep original names, only map combined case to "combined"
CHANGE_OPERATION_MAPPING = {
    "deletion": "deletion",
    "insertion": "insertion",
    "resequentialization": "resequentialization",
    "operator_replacement": "operator_replacement",
    "activity_replacement": "activity_replacement",
    "deletion, insertion, resequentialization, operator_replacement, activity_replacement": "combined",
}

# Change operation sort order for wide tables
CHANGE_OPERATION_ORDER = [
    "deletion",
    "insertion",
    "activity_replacement",
    "resequentialization",
    "operator_replacement",
    "combined",
]


def load_and_validate_input(path_agg: str) -> pd.DataFrame:
    """
    Load and validate the aggregate analysis CSV.

    Parameters
    ----------
    path_agg
        Path to aggregate_analysis.csv.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with required columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(path_agg, low_memory=False)

    # Auto-detect metric column
    metric_col = None
    for col in ["Metric", "metric", "metric_name"]:
        if col in df.columns:
            metric_col = col
            break

    if metric_col is None:
        raise ValueError(
            "Could not auto-detect metric column. "
            "Expected one of: 'Metric', 'metric', 'metric_name'"
        )

    # Verify required columns exist
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {path_agg}: {missing_cols}")

    # Rename metric column to standard name for consistency
    if metric_col != "Metric":
        df = df.rename(columns={metric_col: "Metric"})

    return df


def filter_to_stable_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to rows with stable (pre-drift) split names.

    Parameters
    ----------
    df
        Input DataFrame with split_name column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only stable regime rows.
    """
    # Normalize split_name values
    df = df.copy()
    df["split_name_normalized"] = df["split_name"].str.lower().str.replace("_", "-")

    # Filter to stable splits
    stable_mask = df["split_name_normalized"].isin(
        {name.replace("_", "-") for name in STABLE_SPLIT_NAMES}
    )
    df_stable = df[stable_mask].copy()

    # Drop the temporary normalized column
    df_stable = df_stable.drop(columns=["split_name_normalized"])

    return df_stable


def extract_log_number(log_id: str) -> int:
    """
    Extract log number from log_id (e.g., 'log_2647_1769101653.xes.gz' -> 2647).

    Parameters
    ----------
    log_id
        Log ID string in format 'log_{number}_{timestamp}.xes.gz'.

    Returns
    -------
    int
        Log number.
    """
    match = re.match(r"log_(\d+)_", log_id)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract log number from log_id: {log_id}")


def load_generation_info(path_gen_info: str) -> pd.DataFrame:
    """
    Load generation info CSV to map log_id to complexity, noise, evolution_proportion, and change_operation.

    Parameters
    ----------
    path_gen_info
        Path to generation_info.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with log_number, complexity, noise, evolution_proportion, and change_operation columns.
    """
    df = pd.read_csv(path_gen_info, sep=";")

    # Extract log number from log_id
    df["log_number"] = df["log_id"].apply(extract_log_number)

    # Map noise probability to noise level
    df["noise"] = df["Noisy_trace_prob"].map(NOISE_PROB_TO_LEVEL)

    # Map complexity
    df["complexity"] = df["Process_tree_complexity"].map(COMPLEXITY_MAPPING)

    # Map evolution proportion (keep as float)
    df["evolution_proportion"] = df["Process_tree_evolution_proportion"].astype(float)

    # Map change operation
    df["change_operation"] = df["Allowed_edit_operations"].map(CHANGE_OPERATION_MAPPING)

    # Select and rename columns
    df_mapping = df[
        [
            "log_number",
            "complexity",
            "noise",
            "evolution_proportion",
            "change_operation",
        ]
    ].copy()

    # Check for unmapped values
    if df_mapping["noise"].isna().any():
        unmapped_probs = df[df_mapping["noise"].isna()]["Noisy_trace_prob"].unique()
        raise ValueError(
            f"Unmapped noise probabilities found: {unmapped_probs}. "
            f"Expected values: {list(NOISE_PROB_TO_LEVEL.keys())}"
        )

    if df_mapping["complexity"].isna().any():
        unmapped_complexities = df[df_mapping["complexity"].isna()][
            "Process_tree_complexity"
        ].unique()
        raise ValueError(
            f"Unmapped complexity values found: {unmapped_complexities}. "
            f"Expected values: {list(COMPLEXITY_MAPPING.keys())}"
        )

    if df_mapping["change_operation"].isna().any():
        unmapped_ops = df[df_mapping["change_operation"].isna()][
            "Allowed_edit_operations"
        ].unique()
        raise ValueError(
            f"Unmapped change operations found: {unmapped_ops}. "
            f"Expected values: {list(CHANGE_OPERATION_MAPPING.keys())}"
        )

    return df_mapping


def normalize_split_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize split_name values to canonical "pre_drift" and "post_drift".

    Parameters
    ----------
    df
        DataFrame with split_name column.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized split_name column.
    """
    df = df.copy()

    # Normalize split_name
    df["split_name_normalized"] = df["split_name"].str.lower().str.replace("_", "-")

    # Map to canonical names
    pre_variants = {"pre-drift", "pre_drift", "pre"}
    post_variants = {"post-drift", "post_drift", "post"}

    def normalize_split(split: str) -> str:
        if split in pre_variants:
            return "pre_drift"
        elif split in post_variants:
            return "post_drift"
        else:
            return split  # Keep unknown values as-is for error detection

    df["split_name"] = df["split_name_normalized"].apply(normalize_split)
    df = df.drop(columns=["split_name_normalized"])

    return df


def enrich_with_experimental_factors(
    df: pd.DataFrame, gen_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Add complexity and noise columns based on log_id mapping.

    Parameters
    ----------
    df
        DataFrame with log_id column.
    gen_info
        DataFrame mapping log_number to complexity and noise.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with complexity and noise columns.
        Only includes rows where log_id could be mapped.

    Raises
    ------
    ValueError
        If no log_ids can be mapped (all are unmapped).
    """
    df = df.copy()

    # Extract log number from log_id
    df["log_number"] = df["log_id"].apply(extract_log_number)

    # Merge with generation info on log_number
    df_enriched = df.merge(gen_info, on="log_number", how="left")

    # Drop the temporary log_number column
    df_enriched = df_enriched.drop(columns=["log_number"])

    # Check for unmapped evolution_proportion and change_operation
    if "evolution_proportion" in df_enriched.columns:
        if df_enriched["evolution_proportion"].isna().any():
            unmapped_log_ids = df_enriched[df_enriched["evolution_proportion"].isna()][
                "log_id"
            ].unique()
            raise ValueError(
                f"Cannot map evolution_proportion for {len(unmapped_log_ids)} log_ids: "
                f"{unmapped_log_ids[:5].tolist() if len(unmapped_log_ids) > 5 else unmapped_log_ids.tolist()}"
            )

    if "change_operation" in df_enriched.columns:
        if df_enriched["change_operation"].isna().any():
            unmapped_log_ids = df_enriched[df_enriched["change_operation"].isna()][
                "log_id"
            ].unique()
            raise ValueError(
                f"Cannot map change_operation for {len(unmapped_log_ids)} log_ids: "
                f"{unmapped_log_ids[:5].tolist() if len(unmapped_log_ids) > 5 else unmapped_log_ids.tolist()}"
            )

    # Check for unmapped log_ids
    unmapped = df_enriched[
        df_enriched["complexity"].isna() | df_enriched["noise"].isna()
    ]
    if not unmapped.empty:
        unmapped_log_ids = unmapped["log_id"].unique()
        n_unmapped_rows = len(unmapped)
        n_total_rows = len(df_enriched)
        print(
            f"Warning: Cannot map {len(unmapped_log_ids)} log_ids "
            f"({n_unmapped_rows}/{n_total_rows} rows) to complexity/noise. "
            f"These will be excluded from analysis."
        )
        print(
            f"  Sample unmapped log_ids: "
            f"{unmapped_log_ids[:5].tolist() if len(unmapped_log_ids) > 5 else unmapped_log_ids.tolist()}"
        )

    # Filter to only mapped rows
    df_mapped = df_enriched[
        df_enriched["complexity"].notna() & df_enriched["noise"].notna()
    ].copy()

    if len(df_mapped) == 0:
        raise ValueError(
            "No log_ids could be mapped to complexity/noise. "
            "Cannot proceed with analysis."
        )

    print(
        f"  Proceeding with {len(df_mapped)} rows from "
        f"{df_mapped['log_id'].nunique()} unique log_ids."
    )

    return df_mapped


def compute_per_log_relative_noise(
    df: pd.DataFrame, aggregation: Literal["mean", "median"] = "mean"
) -> pd.DataFrame:
    """
    Compute relative noise per log (IQR / center).

    Parameters
    ----------
    df
        DataFrame with Mean Value, Sample CI Low, Sample CI High columns.
    aggregation
        Aggregation method for center: "mean" or "median".

    Returns
    -------
    pd.DataFrame
        DataFrame with added iqr, center, and rel_noise_log columns.
    """
    df = df.copy()

    # Compute IQR
    df["iqr"] = df["Sample CI High"] - df["Sample CI Low"]

    # Compute center based on aggregation method
    if aggregation == "mean":
        df["center"] = df["Mean Value"]
    elif aggregation == "median":
        if "Median Value" not in df.columns:
            raise ValueError(
                "choose_aggregation='median' requires 'Median Value' column, "
                "which is not present in the input data."
            )
        df["center"] = df["Median Value"]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Compute relative noise
    df["rel_noise_log"] = df["iqr"] / (df["center"].abs() + eps)

    # Flag invalid rows
    invalid_mask = (
        df["center"].isna()
        | df["iqr"].isna()
        | (df["iqr"] < 0)
        | (df["rel_noise_log"] < 0)
    )

    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        print(
            f"Warning: Found {n_invalid} rows with invalid values (NaN center, negative IQR, etc.)"
        )
        print(f"  Dropping {n_invalid} invalid rows.")

    df_valid = df[~invalid_mask].copy()

    return df_valid, n_invalid


def aggregate_relative_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate relative noise across logs by metric, complexity, noise, window_size.

    Parameters
    ----------
    df
        DataFrame with rel_noise_log, Metric, complexity, noise, window_size columns.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with aggregated rel_noise and n_logs.
    """
    # Group by experimental factors
    grouped = df.groupby(["Metric", "complexity", "noise", "window_size"])

    # Aggregate
    agg_result = grouped.agg(
        rel_noise=("rel_noise_log", "median"),
        n_logs=("rel_noise_log", "count"),
        rel_noise_iqr_logs=(
            "rel_noise_log",
            lambda x: x.quantile(0.75) - x.quantile(0.25),
        ),
    ).reset_index()

    return agg_result


def partial_r2(full_model, reduced_model) -> float:
    """
    Compute partial R² using SSE-based formula.

    Parameters
    ----------
    full_model
        Fitted OLS model with all predictors.
    reduced_model
        Fitted OLS model with one predictor removed.

    Returns
    -------
    float
        Partial R² value, clipped to [0, 1].
    """
    sse_full = full_model.ssr
    sse_reduced = reduced_model.ssr

    if sse_reduced == 0:
        return 1.0

    partial_r2_val = (sse_reduced - sse_full) / sse_reduced
    return np.clip(partial_r2_val, 0.0, 1.0)


def compute_partial_r2_components(df_m: pd.DataFrame) -> dict:
    """
    Compute partial R² for each factor (complexity, noise, window_size).

    Parameters
    ----------
    df_m
        DataFrame filtered to a single metric with columns:
        rel_noise_log, window_size, complexity, noise.

    Returns
    -------
    dict
        Dictionary with keys: complexity, noise, window_size, residual, n_obs, r2_full.
        Values are normalized shares (percentages) that sum to 100.
    """
    # Prepare data
    df_clean = df_m[["rel_noise_log", "window_size", "complexity", "noise"]].copy()

    # Create log-transformed outcome
    df_clean["y"] = np.log(df_clean["rel_noise_log"])

    # Drop rows with non-finite y
    df_clean = df_clean[np.isfinite(df_clean["y"])].copy()

    n_obs = len(df_clean)

    if n_obs < MIN_OBS_FOR_FACTOR_ANALYSIS:
        # Return zeros if insufficient data (residual as proportion, not percentage)
        return {
            "complexity": 0.0,
            "noise": 0.0,
            "window_size": 0.0,
            "residual": 1.0,  # 100% as proportion
            "n_obs": n_obs,
            "r2_full": 0.0,
        }

    # Fit full model
    try:
        model_full = smf.ols(
            "y ~ np.log(window_size) + C(complexity) + C(noise)",
            data=df_clean,
        ).fit()
        r2_full = model_full.rsquared
    except Exception as e:
        print(f"Warning: Failed to fit full model: {e}")
        return {
            "complexity": 0.0,
            "noise": 0.0,
            "window_size": 0.0,
            "residual": 1.0,  # 100% as proportion
            "n_obs": n_obs,
            "r2_full": 0.0,
        }

    # Fit reduced models (drop one factor at a time)
    partial_r2_vals = {}

    # Drop complexity
    try:
        model_no_complexity = smf.ols(
            "y ~ np.log(window_size) + C(noise)",
            data=df_clean,
        ).fit()
        partial_r2_vals["complexity"] = partial_r2(model_full, model_no_complexity)
    except Exception:
        partial_r2_vals["complexity"] = 0.0

    # Drop noise
    try:
        model_no_noise = smf.ols(
            "y ~ np.log(window_size) + C(complexity)",
            data=df_clean,
        ).fit()
        partial_r2_vals["noise"] = partial_r2(model_full, model_no_noise)
    except Exception:
        partial_r2_vals["noise"] = 0.0

    # Drop window_size
    try:
        model_no_window = smf.ols(
            "y ~ C(complexity) + C(noise)",
            data=df_clean,
        ).fit()
        partial_r2_vals["window_size"] = partial_r2(model_full, model_no_window)
    except Exception:
        partial_r2_vals["window_size"] = 0.0

    # Compute residual share (unexplained variance)
    residual_raw = max(0.0, 1.0 - r2_full)

    # Normalize partial R² values and residual to sum to 100%
    # Partial R² values are unique contributions and may not sum to model R²
    # For a clean 100% breakdown, normalize: total = sum(partial_R²) + residual, shares = each / total
    total_all = (
        partial_r2_vals["complexity"]
        + partial_r2_vals["noise"]
        + partial_r2_vals["window_size"]
        + residual_raw
    )

    if total_all > 0:
        # Normalize each component by dividing by total
        normalized = {
            "complexity": partial_r2_vals["complexity"] / total_all,
            "noise": partial_r2_vals["noise"] / total_all,
            "window_size": partial_r2_vals["window_size"] / total_all,
        }
        residual_share = residual_raw / total_all
    else:
        # Fallback: if total is zero, assign all to residual
        normalized = {
            "complexity": 0.0,
            "noise": 0.0,
            "window_size": 0.0,
        }
        residual_share = 1.0

    return {
        "complexity": normalized["complexity"],
        "noise": normalized["noise"],
        "window_size": normalized["window_size"],
        "residual": residual_share,
        "n_obs": n_obs,
        "r2_full": r2_full,
    }


def run_noise_factor_importance(
    df_relnoise_log: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run factor importance analysis for relative noise across all metrics.

    Parameters
    ----------
    df_relnoise_log
        DataFrame with per-log relative noise, including columns:
        Metric, rel_noise_log, window_size, complexity, noise.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Long and wide format DataFrames with factor importance results.
    """
    results = []

    for metric in df_relnoise_log["Metric"].unique():
        df_metric = df_relnoise_log[df_relnoise_log["Metric"] == metric].copy()

        # Check required columns
        required_cols = ["rel_noise_log", "window_size", "complexity", "noise"]
        missing_cols = [col for col in required_cols if col not in df_metric.columns]
        if missing_cols:
            print(f"Warning: Metric {metric} missing columns {missing_cols}, skipping.")
            continue

        # Compute partial R² components
        components = compute_partial_r2_components(df_metric)

        results.append(
            {
                "Metric": metric,
                "complexity": components["complexity"] * 100,
                "noise": components["noise"] * 100,
                "window_size": components["window_size"] * 100,
                "residual": components["residual"] * 100,
                "n_obs": components["n_obs"],
                "r2_full": components["r2_full"],
            }
        )

    # Create wide format
    df_wide = pd.DataFrame(results)
    df_wide = df_wide.set_index("Metric")

    # Create long format by melting
    df_long = df_wide.reset_index().melt(
        id_vars=["Metric", "n_obs", "r2_full"],
        value_vars=["complexity", "noise", "window_size", "residual"],
        var_name="factor",
        value_name="percent",
    )

    # Add raw partial R² if needed (optional, for reference)
    # For now, we'll just keep the normalized percentages

    return df_long, df_wide


def print_factor_importance_summary(
    df_long: pd.DataFrame, df_wide: pd.DataFrame
) -> None:
    """
    Print summary statistics for factor importance analysis.

    Parameters
    ----------
    df_long
        Long-format DataFrame with factor importance.
    df_wide
        Wide-format DataFrame with factor importance.
    """
    print("\n" + "=" * 60)
    print("NOISE FACTOR IMPORTANCE SUMMARY")
    print("=" * 60)

    # Average share across metrics
    print("\n1. Average factor importance across all metrics:")
    factor_means = df_wide[["complexity", "noise", "window_size", "residual"]].mean()
    for factor, mean_pct in factor_means.items():
        print(f"   {factor:15s}: {mean_pct:6.2f}%")

    # Top factor per metric
    print("\n2. Dominant factor per metric (highest share):")
    for metric in df_wide.index:
        row = df_wide.loc[metric]
        factors = ["complexity", "noise", "window_size", "residual"]
        top_factor = max(factors, key=lambda f: row[f])
        top_value = row[top_factor]
        print(f"   {metric:40s}: {top_factor:15s} ({top_value:6.2f}%)")

    # Metrics with low n_obs
    low_n = df_wide[df_wide["n_obs"] < MIN_OBS_FOR_FACTOR_ANALYSIS]
    if len(low_n) > 0:
        print(
            f"\n3. Warning: {len(low_n)} metrics have n_obs < {MIN_OBS_FOR_FACTOR_ANALYSIS}:"
        )
        for metric in low_n.index:
            print(f"   {metric}: n_obs = {low_n.loc[metric, 'n_obs']}")

    # Sanity checks
    print("\n4. Sanity checks:")
    window_mean = df_wide["window_size"].mean()
    if window_mean < 20:
        print(
            f"   ⚠ Warning: Average window_size importance ({window_mean:.2f}%) is lower than expected"
        )
    else:
        print(
            f"   ✓ Average window_size importance: {window_mean:.2f}% (expected to be high)"
        )

    noise_mean = df_wide["noise"].mean()
    print(f"   ✓ Average noise importance: {noise_mean:.2f}%")

    print("\n" + "=" * 60)


def create_wide_format_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Create wide-format table with 3-level column hierarchy.

    Parameters
    ----------
    df_long
        Long-format DataFrame with Metric, complexity, noise, window_size, rel_noise.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with MultiIndex columns (complexity, noise, window_size).
    """
    # Pivot to wide format
    df_wide = df_long.pivot_table(
        index="Metric",
        columns=["complexity", "noise", "window_size"],
        values="rel_noise",
        aggfunc="first",  # Should be unique per (Metric, complexity, noise, window_size)
    )

    # Sort columns in logical order
    complexity_order = ["simple", "mid", "complex"]
    noise_order = ["0", "low", "high"]

    # Get existing column levels
    if isinstance(df_wide.columns, pd.MultiIndex):
        complexity_level = list(df_wide.columns.get_level_values(0).unique())
        noise_level = list(df_wide.columns.get_level_values(1).unique())
        window_sizes = sorted(df_wide.columns.get_level_values(2).unique())

        # Reorder complexity and noise
        complexity_sorted = [c for c in complexity_order if c in complexity_level]
        complexity_sorted.extend(
            [c for c in complexity_level if c not in complexity_order]
        )

        noise_sorted = [n for n in noise_order if n in noise_level]
        noise_sorted.extend([n for n in noise_level if n not in noise_order])

        # Create sorted column MultiIndex
        sorted_columns = pd.MultiIndex.from_product(
            [complexity_sorted, noise_sorted, window_sizes],
            names=["complexity", "noise", "window_size"],
        )

        # Select only columns that exist in the original DataFrame
        existing_columns = [col for col in sorted_columns if col in df_wide.columns]

        # Reorder columns
        df_wide = df_wide[existing_columns]

    return df_wide


def save_outputs(df_long: pd.DataFrame, df_wide: pd.DataFrame) -> None:
    """
    Save long and wide format DataFrames to CSV files.

    Parameters
    ----------
    df_long
        Long-format DataFrame to save.
    df_wide
        Wide-format DataFrame to save.
    """
    # Ensure output directory exists
    Path(PATH_OUT_LONG).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_OUT_WIDE).parent.mkdir(parents=True, exist_ok=True)

    # Save long format
    df_long.to_csv(PATH_OUT_LONG, index=False)
    print(f"Saved long-format output to {PATH_OUT_LONG}")

    # Save wide format (preserve MultiIndex)
    df_wide.to_csv(PATH_OUT_WIDE)
    print(f"Saved wide-format output to {PATH_OUT_WIDE}")


def perform_sanity_checks(df_long: pd.DataFrame, n_invalid: int) -> None:
    """
    Perform sanity checks and print summary statistics.

    Parameters
    ----------
    df_long
        Long-format aggregated DataFrame.
    n_invalid
        Number of invalid rows that were dropped.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Check 1: rel_noise should generally decrease as window_size increases
    print("\n1. Checking: rel_noise decreases with increasing window_size")
    for metric in df_long["Metric"].unique():
        metric_df = df_long[df_long["Metric"] == metric]
        for complexity in metric_df["complexity"].unique():
            for noise in metric_df["noise"].unique():
                subset = metric_df[
                    (metric_df["complexity"] == complexity)
                    & (metric_df["noise"] == noise)
                ].sort_values("window_size")

                if len(subset) > 1:
                    rel_noise_values = subset["rel_noise"].values
                    # Check if generally decreasing (allow some exceptions)
                    decreasing = all(
                        rel_noise_values[i] >= rel_noise_values[i + 1] * 0.9
                        for i in range(len(rel_noise_values) - 1)
                    )
                    if not decreasing:
                        print(
                            f"   ⚠ Warning: {metric} ({complexity}, {noise}) "
                            f"does not show clear decrease with window_size"
                        )

    # Check 2: rel_noise should generally increase with higher noise level
    print("\n2. Checking: rel_noise increases with higher noise level")
    for metric in df_long["Metric"].unique():
        metric_df = df_long[df_long["Metric"] == metric]
        for complexity in metric_df["complexity"].unique():
            for window_size in metric_df["window_size"].unique():
                subset = metric_df[
                    (metric_df["complexity"] == complexity)
                    & (metric_df["window_size"] == window_size)
                ]

                # Map noise levels to numeric for comparison
                noise_order_map = {"0": 0, "low": 1, "high": 2}
                subset = subset.copy()
                subset["noise_numeric"] = subset["noise"].map(noise_order_map)
                subset = subset.sort_values("noise_numeric")

                if len(subset) > 1:
                    rel_noise_values = subset["rel_noise"].values
                    # Check if generally increasing (allow some exceptions)
                    increasing = all(
                        rel_noise_values[i] <= rel_noise_values[i + 1] * 1.1
                        for i in range(len(rel_noise_values) - 1)
                    )
                    if not increasing:
                        print(
                            f"   ⚠ Warning: {metric} ({complexity}, window_size={window_size}) "
                            f"does not show clear increase with noise level"
                        )

    # Summary statistics
    print("\n3. Summary Statistics")
    print(f"   Number of invalid rows dropped: {n_invalid}")
    print(f"   Number of metrics: {df_long['Metric'].nunique()}")
    print(f"   Number of complexity levels: {df_long['complexity'].nunique()}")
    print(f"   Number of noise levels: {df_long['noise'].nunique()}")
    print(f"   Number of window sizes: {df_long['window_size'].nunique()}")
    print(f"   Total cells (metric × complexity × noise × window_size): {len(df_long)}")

    n_logs_stats = df_long["n_logs"].describe()
    print(f"\n   n_logs per cell:")
    print(f"     Min: {n_logs_stats['min']:.0f}")
    print(f"     Median: {n_logs_stats['50%']:.0f}")
    print(f"     Max: {n_logs_stats['max']:.0f}")

    print("\n" + "=" * 60)


def compute_snr_per_log(
    df: pd.DataFrame, choose_aggregation: Literal["mean", "median"] = "mean"
) -> pd.DataFrame:
    """
    Compute SNR per log by joining pre_drift and post_drift rows.

    Parameters
    ----------
    df
        DataFrame with pre_drift and post_drift rows, enriched with design factors.
    choose_aggregation
        Aggregation method (not used for SNR, but kept for consistency).

    Returns
    -------
    pd.DataFrame
        Long DataFrame with SNR per log, including signal, iqr_pre, and snr_log.
    """
    # Filter to pre_drift and post_drift only
    df_split = df[df["split_name"].isin(["pre_drift", "post_drift"])].copy()

    # Select needed columns
    required_cols = [
        "Metric",
        "log_id",
        "window_size",
        "split_name",
        "Mean Value",
        "Sample CI Low",
        "Sample CI High",
        "complexity",
        "noise",
        "evolution_proportion",
        "change_operation",
    ]

    missing_cols = [col for col in required_cols if col not in df_split.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for SNR computation: {missing_cols}"
        )

    df_split = df_split[required_cols].copy()

    # Create pre_drift and post_drift frames
    df_pre = df_split[df_split["split_name"] == "pre_drift"].copy()
    df_post = df_split[df_split["split_name"] == "post_drift"].copy()

    # Compute IQR_pre in df_pre
    df_pre["iqr_pre"] = df_pre["Sample CI High"] - df_pre["Sample CI Low"]

    # Rename columns to avoid collisions
    df_pre = df_pre.rename(
        columns={
            "Mean Value": "center_pre",
            "Sample CI Low": "ci_low_pre",
            "Sample CI High": "ci_high_pre",
        }
    )
    df_post = df_post.rename(
        columns={
            "Mean Value": "center_post",
            "Sample CI Low": "ci_low_post",
            "Sample CI High": "ci_high_post",
        }
    )

    # Select columns for join
    join_cols = ["Metric", "log_id", "window_size"]
    pre_cols = join_cols + [
        "center_pre",
        "iqr_pre",
        "complexity",
        "noise",
        "evolution_proportion",
        "change_operation",
    ]
    post_cols = join_cols + ["center_post"]

    df_pre_join = df_pre[pre_cols].copy()
    df_post_join = df_post[post_cols].copy()

    # Inner join on metric, log_id, window_size
    df_joined = df_pre_join.merge(
        df_post_join, on=join_cols, how="inner", suffixes=("", "_post")
    )

    # Verify that complexity, noise, evolution_proportion, change_operation match
    # (They should since they come from the same log_id)
    if len(df_joined) == 0:
        raise ValueError(
            "No matching pre_drift/post_drift pairs found. "
            "Check that both splits exist for each (metric, log_id, window_size)."
        )

    # Compute signal and SNR
    df_joined["signal"] = (df_joined["center_post"] - df_joined["center_pre"]).abs()
    df_joined["snr_log"] = df_joined["signal"] / (df_joined["iqr_pre"] + eps)

    # Select final columns
    result_cols = [
        "Metric",
        "log_id",
        "complexity",
        "noise",
        "evolution_proportion",
        "change_operation",
        "window_size",
        "center_pre",
        "center_post",
        "iqr_pre",
        "signal",
        "snr_log",
    ]

    df_snr = df_joined[result_cols].copy()

    # Check for missing joins
    n_pre = len(df_pre)
    n_post = len(df_post)
    n_joined = len(df_snr)
    n_expected = min(n_pre, n_post)  # Approximate

    if n_joined < n_expected * 0.9:  # Allow 10% missing
        print(
            f"Warning: Only {n_joined} pre/post pairs found "
            f"(expected ~{n_expected} based on {n_pre} pre and {n_post} post rows)"
        )

    return df_snr


def aggregate_snr_cells(df_snr_log: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SNR across logs by experimental factors.

    Parameters
    ----------
    df_snr_log
        DataFrame with snr_log per log.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with aggregated SNR per cell.
    """
    # Group by experimental factors
    grouped = df_snr_log.groupby(
        [
            "Metric",
            "evolution_proportion",
            "change_operation",
            "complexity",
            "noise",
            "window_size",
        ]
    )

    # Aggregate
    agg_result = grouped.agg(
        snr=("snr_log", "median"),
        n_logs=("snr_log", "count"),
        snr_iqr_logs=("snr_log", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ).reset_index()

    return agg_result


def pivot_snr_wide(df_snr_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Create wide-format SNR table with 5-level column hierarchy.

    Parameters
    ----------
    df_snr_cells
        Long-format DataFrame with aggregated SNR.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with MultiIndex columns.
    """
    # Pivot to wide format
    df_wide = df_snr_cells.pivot_table(
        index="Metric",
        columns=[
            "evolution_proportion",
            "change_operation",
            "complexity",
            "noise",
            "window_size",
        ],
        values="snr",
        aggfunc="first",
    )

    # Sort columns in logical order
    if isinstance(df_wide.columns, pd.MultiIndex):
        # Get unique values for each level
        evolution_props = sorted(df_wide.columns.get_level_values(0).unique())
        change_ops = list(df_wide.columns.get_level_values(1).unique())
        complexities = list(df_wide.columns.get_level_values(2).unique())
        noises = list(df_wide.columns.get_level_values(3).unique())
        window_sizes = sorted(df_wide.columns.get_level_values(4).unique())

        # Reorder change_operation
        change_ops_sorted = [op for op in CHANGE_OPERATION_ORDER if op in change_ops]
        change_ops_sorted.extend(
            [op for op in change_ops if op not in CHANGE_OPERATION_ORDER]
        )

        # Reorder complexity
        complexity_order = ["simple", "mid", "complex"]
        complexities_sorted = [c for c in complexity_order if c in complexities]
        complexities_sorted.extend(
            [c for c in complexities if c not in complexity_order]
        )

        # Reorder noise
        noise_order = ["0", "low", "high"]
        noises_sorted = [n for n in noise_order if n in noises]
        noises_sorted.extend([n for n in noises if n not in noise_order])

        # Create sorted column MultiIndex
        sorted_columns = pd.MultiIndex.from_product(
            [
                evolution_props,
                change_ops_sorted,
                complexities_sorted,
                noises_sorted,
                window_sizes,
            ],
            names=[
                "evolution_proportion",
                "change_operation",
                "complexity",
                "noise",
                "window_size",
            ],
        )

        # Select only columns that exist in the original DataFrame
        existing_columns = [col for col in sorted_columns if col in df_wide.columns]

        # Reorder columns
        df_wide = df_wide[existing_columns]

    return df_wide


def aggregate_snr_by_operation(df_snr_log: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SNR by metric and change_operation (pooling across other factors).

    Parameters
    ----------
    df_snr_log
        DataFrame with snr_log per log.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric × change_operation.
    """
    # Group by metric and change_operation
    grouped = df_snr_log.groupby(["Metric", "change_operation"])

    # Aggregate
    agg_result = grouped.agg(
        median_snr=("snr_log", "median"),
        n_logs=("snr_log", "count"),
    ).reset_index()

    # Pivot to wide format
    df_wide = agg_result.pivot_table(
        index="Metric",
        columns="change_operation",
        values="median_snr",
        aggfunc="first",
    )

    # Reorder columns according to CHANGE_OPERATION_ORDER
    existing_ops = [op for op in CHANGE_OPERATION_ORDER if op in df_wide.columns]
    existing_ops.extend(
        [op for op in df_wide.columns if op not in CHANGE_OPERATION_ORDER]
    )
    df_wide = df_wide[existing_ops]

    return df_wide


def aggregate_snr_by_evolution_proportion(df_snr_log: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SNR by metric and evolution_proportion, fixing change_operation to "combined".

    Parameters
    ----------
    df_snr_log
        DataFrame with snr_log per log.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric × evolution_proportion.
    """
    # Filter to combined change_operation only
    df_combined = df_snr_log[df_snr_log["change_operation"] == "combined"].copy()

    if len(df_combined) == 0:
        raise ValueError(
            "No data found with change_operation='combined'. "
            "Cannot create snr_by_evolution_proportion table."
        )

    # Group by metric and evolution_proportion
    grouped = df_combined.groupby(["Metric", "evolution_proportion"])

    # Aggregate
    agg_result = grouped.agg(
        median_snr=("snr_log", "median"),
        n_logs=("snr_log", "count"),
    ).reset_index()

    # Pivot to wide format
    df_wide = agg_result.pivot_table(
        index="Metric",
        columns="evolution_proportion",
        values="median_snr",
        aggfunc="first",
    )

    # Sort columns by evolution_proportion (ascending)
    evolution_props = sorted(df_wide.columns)
    df_wide = df_wide[evolution_props]

    return df_wide


def aggregate_snr_by_noise(df_snr_log: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SNR by metric and noise, fixing change_operation to "combined".

    Parameters
    ----------
    df_snr_log
        DataFrame with snr_log per log.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric × noise.
    """
    # Filter to combined change_operation only
    df_combined = df_snr_log[df_snr_log["change_operation"] == "combined"].copy()

    if len(df_combined) == 0:
        raise ValueError(
            "No data found with change_operation='combined'. "
            "Cannot create snr_by_noise table."
        )

    # Group by metric and noise
    grouped = df_combined.groupby(["Metric", "noise"])

    # Aggregate
    agg_result = grouped.agg(
        median_snr=("snr_log", "median"),
        n_logs=("snr_log", "count"),
    ).reset_index()

    # Pivot to wide format
    df_wide = agg_result.pivot_table(
        index="Metric",
        columns="noise",
        values="median_snr",
        aggfunc="first",
    )

    # Sort columns by noise level (0, low, high)
    noise_order = ["0", "low", "high"]
    existing_noises = [n for n in noise_order if n in df_wide.columns]
    existing_noises.extend([n for n in df_wide.columns if n not in noise_order])
    df_wide = df_wide[existing_noises]

    return df_wide


def perform_snr_sanity_checks(
    df_snr_log: pd.DataFrame, df_snr_cells: pd.DataFrame
) -> None:
    """
    Perform sanity checks for SNR analysis.

    Parameters
    ----------
    df_snr_log
        Long-format DataFrame with SNR per log.
    df_snr_cells
        Long-format DataFrame with aggregated SNR per cell.
    """
    print("\n" + "=" * 60)
    print("SNR SANITY CHECKS")
    print("=" * 60)

    # Check: Report missing pre/post pairs
    print("\nChecking: Pre/post drift pair completeness")
    df_pre = df_snr_log.groupby(["Metric", "log_id", "window_size"]).size()
    total_combinations = len(df_snr_log.groupby(["Metric", "log_id", "window_size"]))
    print(
        f"   Total (metric, log_id, window_size) combinations with SNR: {total_combinations}"
    )

    # Summary statistics
    print("\nSummary Statistics")
    print(f"   Number of metrics: {df_snr_cells['Metric'].nunique()}")
    print(
        f"   Number of evolution_proportions: {df_snr_cells['evolution_proportion'].nunique()}"
    )
    print(
        f"   Number of change_operations: {df_snr_cells['change_operation'].nunique()}"
    )
    print(f"   Number of complexity levels: {df_snr_cells['complexity'].nunique()}")
    print(f"   Number of noise levels: {df_snr_cells['noise'].nunique()}")
    print(f"   Number of window sizes: {df_snr_cells['window_size'].nunique()}")
    print(f"   Total cells (metric × δ × O × C × N × W): {len(df_snr_cells)}")

    n_logs_stats = df_snr_cells["n_logs"].describe()
    print(f"\n   n_logs per cell:")
    print(f"     Min: {n_logs_stats['min']:.0f}")
    print(f"     Median: {n_logs_stats['50%']:.0f}")
    print(f"     Max: {n_logs_stats['max']:.0f}")

    snr_stats = df_snr_cells["snr"].describe()
    print(f"\n   SNR per cell:")
    print(f"     Min: {snr_stats['min']:.4f}")
    print(f"     Median: {snr_stats['50%']:.4f}")
    print(f"     Max: {snr_stats['max']:.4f}")

    print("\n" + "=" * 60)


def main() -> None:
    """Main entry point for the analysis."""
    print("=" * 60)
    print("RELATIVE NOISE ANALYSIS")
    print("=" * 60)
    print("Loading and validating input...")
    df = load_and_validate_input(PATH_AGG)

    print("Filtering to stable regime...")
    df_stable = filter_to_stable_regime(df)

    print("Loading generation info for log_id mapping...")
    gen_info = load_generation_info(PATH_GEN_INFO)

    print("Enriching with experimental factors...")
    df_enriched = enrich_with_experimental_factors(df_stable, gen_info)

    print("Computing per-log relative noise...")
    df_with_noise, n_invalid = compute_per_log_relative_noise(
        df_enriched, aggregation=choose_aggregation
    )

    print("Aggregating relative noise across logs...")
    df_long = aggregate_relative_noise(df_with_noise)

    print("Creating wide-format table...")
    df_wide = create_wide_format_table(df_long)

    print("Saving outputs...")
    save_outputs(df_long, df_wide)

    print("Performing sanity checks...")
    perform_sanity_checks(df_long, n_invalid)

    print("\n" + "=" * 60)
    print("NOISE FACTOR IMPORTANCE ANALYSIS")
    print("=" * 60)
    print("Computing factor importance for relative noise...")
    df_factor_long, df_factor_wide = run_noise_factor_importance(df_with_noise)

    print("Saving factor importance outputs...")
    # Ensure output directory exists
    Path(PATH_NOISE_FACTOR_LONG).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_NOISE_FACTOR_WIDE).parent.mkdir(parents=True, exist_ok=True)

    # Save long format
    df_factor_long.to_csv(PATH_NOISE_FACTOR_LONG, index=False)
    print(f"Saved factor importance long-format output to {PATH_NOISE_FACTOR_LONG}")

    # Save wide format
    df_factor_wide.to_csv(PATH_NOISE_FACTOR_WIDE)
    print(f"Saved factor importance wide-format output to {PATH_NOISE_FACTOR_WIDE}")

    print("Printing factor importance summary...")
    print_factor_importance_summary(df_factor_long, df_factor_wide)

    print("\n" + "=" * 60)
    print("SNR ANALYSIS")
    print("=" * 60)
    print("Loading and normalizing split names...")
    df_full = load_and_validate_input(PATH_AGG)
    df_normalized = normalize_split_name(df_full)

    print(
        "Enriching with design factors (including evolution_proportion and change_operation)..."
    )
    df_snr_enriched = enrich_with_experimental_factors(df_normalized, gen_info)

    print("Computing SNR per log...")
    df_snr_log = compute_snr_per_log(df_snr_enriched, choose_aggregation)

    print("Aggregating SNR across logs...")
    df_snr_cells = aggregate_snr_cells(df_snr_log)

    print("Creating wide-format SNR table...")
    df_snr_wide = pivot_snr_wide(df_snr_cells)

    print("Aggregating SNR by operation...")
    df_snr_by_op = aggregate_snr_by_operation(df_snr_log)

    print("Aggregating SNR by evolution_proportion (change_operation=combined)...")
    df_snr_by_evol = aggregate_snr_by_evolution_proportion(df_snr_log)

    print("Aggregating SNR by noise (change_operation=combined)...")
    df_snr_by_noise = aggregate_snr_by_noise(df_snr_log)

    print("Saving SNR outputs...")
    # Ensure output directory exists
    Path(PATH_SNR_LONG).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_SNR_WIDE).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_SNR_BY_OP).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_SNR_BY_EVOL).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH_SNR_BY_NOISE).parent.mkdir(parents=True, exist_ok=True)

    # Save SNR long format (use df_snr_cells, not df_snr_log, for consistency with relative_noise)
    df_snr_cells.to_csv(PATH_SNR_LONG, index=False)
    print(f"Saved SNR long-format output to {PATH_SNR_LONG}")

    # Save SNR wide format
    df_snr_wide.to_csv(PATH_SNR_WIDE)
    print(f"Saved SNR wide-format output to {PATH_SNR_WIDE}")

    # Save SNR by operation
    df_snr_by_op.to_csv(PATH_SNR_BY_OP)
    print(f"Saved SNR by operation output to {PATH_SNR_BY_OP}")

    # Save SNR by evolution_proportion
    df_snr_by_evol.to_csv(PATH_SNR_BY_EVOL)
    print(f"Saved SNR by evolution_proportion output to {PATH_SNR_BY_EVOL}")

    # Save SNR by noise
    df_snr_by_noise.to_csv(PATH_SNR_BY_NOISE)
    print(f"Saved SNR by noise output to {PATH_SNR_BY_NOISE}")

    print("Performing SNR sanity checks...")
    perform_snr_sanity_checks(df_snr_log, df_snr_cells)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
