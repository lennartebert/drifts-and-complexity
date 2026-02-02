"""Analyze complexity values to compute relative noise under stability.

This script processes aggregate analysis results to compute relative noise metrics
for different complexity levels, noise levels, and window sizes during stable
(pre-drift) periods.

Produces both long-format (analysis-ready) and wide-format (presentation-ready)
outputs.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.constants import (
    DIMENSIONS_ORDER,
    METRIC_DIMENSION_MAP,
    METRIC_NAMES_TO_LATEX_MAP,
)

# Configuration
PATH_AGG = "results/signal_noise_study/aggregate_analysis.csv"
PATH_GEN_INFO = "data/synthetic/sudden_drifts/generation_info.csv"
DIR_CSV = "results/signal_noise_study/csvs"
DIR_LATEX = "results/signal_noise_study/latex"
PATH_OUT_WIDE = f"{DIR_CSV}/relative_noise_wide.csv"
PATH_OUT_LONG = f"{DIR_CSV}/relative_noise_long.csv"
PATH_SNR_LONG = f"{DIR_CSV}/snr_long.csv"
PATH_SNR_WIDE = f"{DIR_CSV}/snr_wide.csv"
PATH_SNR_BY_OP = f"{DIR_CSV}/snr_by_operation.csv"
PATH_SNR_BY_EVOL = f"{DIR_CSV}/snr_by_evolution_proportion.csv"
PATH_SNR_BY_NOISE = f"{DIR_CSV}/snr_by_noise.csv"
PATH_SIGNAL_RELCHANGE_BY_OP = f"{DIR_CSV}/signal_relchange_by_operation.csv"
PATH_SIGNAL_RELCHANGE_BY_EVOL = (
    f"{DIR_CSV}/signal_relchange_by_evolution_proportion.csv"
)
PATH_SIGNAL_RELCHANGE_BY_NOISE = f"{DIR_CSV}/signal_relchange_by_noise.csv"
PATH_SIGNAL_SNR_BY_OP = f"{DIR_CSV}/signal_snr_by_operation.csv"
PATH_SIGNAL_SNR_BY_EVOL = f"{DIR_CSV}/signal_snr_by_evolution_proportion.csv"
PATH_SIGNAL_SNR_BY_NOISE = f"{DIR_CSV}/signal_snr_by_noise.csv"
PATH_NOISE_FACTOR_LONG = f"{DIR_CSV}/noise_factor_importance_long.csv"
PATH_NOISE_FACTOR_WIDE = f"{DIR_CSV}/noise_factor_importance_wide.csv"
PATH_NOISE_FACTOR_LONG_CHANGE = f"{DIR_CSV}/noise_factor_importance_change_long.csv"
PATH_NOISE_FACTOR_WIDE_CHANGE = f"{DIR_CSV}/noise_factor_importance_change_wide.csv"
PATH_NOISE_ABS_MEDIAN = f"{DIR_CSV}/noise_abs_median.csv"
PATH_NOISE_RELCI = f"{DIR_CSV}/noise_relci.csv"
PATH_NOISE_CHANGE_DUE_TO_NOISE = f"{DIR_CSV}/noise_change_due_to_noise.csv"
choose_aggregation: Literal["mean", "median"] = "mean"
eps = 0  # was 1e-9; set to 0 to surface NA when denominator is zero
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

# Ordered numeric encoding for factor analysis (simple < mid < complex; 0 < low < high)
COMPLEXITY_ORD = {"simple": 0, "mid": 1, "complex": 2}
NOISE_ORD = {"0": 0, "low": 1, "high": 2}

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

# Metric order: by dimension (DIMENSIONS_ORDER) then by order in METRIC_DIMENSION_MAP
_METRIC_ORDER = [
    m for d in DIMENSIONS_ORDER for m, dim in METRIC_DIMENSION_MAP.items() if dim == d
]


def _add_dimension_and_sort_long(
    df: pd.DataFrame, metric_col: str = "Metric"
) -> pd.DataFrame:
    """
    Add dimension as first column and sort rows by dimension then metric order.

    Parameters
    ----------
    df
        DataFrame with a metric column (e.g. Metric).
    metric_col
        Name of the column containing metric names.

    Returns
    -------
    pd.DataFrame
        Copy with dimension column first and rows sorted.
    """
    df = df.copy()
    df["dimension"] = df[metric_col].map(METRIC_DIMENSION_MAP).fillna("Other")
    dim_order = list(DIMENSIONS_ORDER) + ["Other"]
    df["_dim_ord"] = df["dimension"].map(
        lambda x: dim_order.index(x) if x in dim_order else len(dim_order)
    )
    metric_order = {m: i for i, m in enumerate(_METRIC_ORDER)}
    df["_met_ord"] = df[metric_col].map(
        lambda m: metric_order.get(m, len(metric_order))
    )
    df = df.sort_values(["_dim_ord", "_met_ord"]).drop(columns=["_dim_ord", "_met_ord"])
    cols = ["dimension", metric_col] + [
        c for c in df.columns if c not in ("dimension", metric_col)
    ]
    return df[cols]


def _add_dimension_and_sort_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dimension as first index level and sort by dimension then metric order.

    Parameters
    ----------
    df
        DataFrame with index = Metric (single level).

    Returns
    -------
    pd.DataFrame
        Copy with MultiIndex (dimension, Metric) sorted.
    """
    df = df.copy()
    dimension = df.index.map(METRIC_DIMENSION_MAP).fillna("Other")
    df.index = pd.MultiIndex.from_arrays(
        [dimension, df.index], names=["dimension", "Metric"]
    )
    dim_order = list(DIMENSIONS_ORDER) + ["Other"]
    desired_index = []
    for d in dim_order:
        for m in _METRIC_ORDER:
            if (d, m) in df.index:
                desired_index.append((d, m))
    for idx in df.index:
        if idx not in desired_index:
            desired_index.append(idx)
    return df.reindex(desired_index)


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
        DataFrame with log_number, complexity, noise, evolution_proportion,
        change_operation, and log_seed columns.
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

    # Log seed (from generation_info) for factor importance
    df["log_seed"] = df["Log_seed"]

    # Select and rename columns
    df_mapping = df[
        [
            "log_number",
            "complexity",
            "noise",
            "evolution_proportion",
            "change_operation",
            "log_seed",
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


def add_change_vs_baseline_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-log change vs. no-noise baseline (relative absolute deviation).

    For each (Metric, complexity, window_size), baseline is the median of
    center over rows where noise == "0". Then change_vs_baseline_log =
    |center - baseline| / (|baseline| + eps). For noise == "0", set to 0.

    Parameters
    ----------
    df
        DataFrame with Metric, complexity, noise, window_size, center.

    Returns
    -------
    pd.DataFrame
        Copy of df with added column change_vs_baseline_log.
    """
    df = df.copy()
    baseline = (
        df[df["noise"] == "0"]
        .groupby(["Metric", "complexity", "window_size"])["center"]
        .median()
        .reset_index()
        .rename(columns={"center": "baseline"})
    )
    df = df.merge(baseline, on=["Metric", "complexity", "window_size"], how="left")
    df["change_vs_baseline_log"] = np.where(
        df["noise"] == "0",
        0.0,
        (df["center"] - df["baseline"]).abs() / (df["baseline"].abs() + eps),
    )
    df = df.drop(columns=["baseline"])
    return df


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

    agg_result = _add_dimension_and_sort_long(agg_result)
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


def compute_partial_r2_components(
    df_m: pd.DataFrame,
    *,
    outcome_col: str = "rel_noise_log",
    outcome_log_shift: float = 0.0,
) -> dict:
    """
    Compute partial R² for each factor (complexity, noise, window_size, log_seed).
    Complexity and noise are treated as ordered (numeric); log_id is not a factor.
    Includes complexity × window_size interaction (effect of window size may depend on process complexity).

    Parameters
    ----------
    df_m
        DataFrame filtered to a single metric with columns:
        outcome_col, window_size, complexity, noise, log_seed.
    outcome_col
        Name of the outcome column (e.g. rel_noise_log or change_vs_baseline_log).
    outcome_log_shift
        Added to outcome before log so log(outcome + shift); use a small value
        when outcome can be 0 (e.g. change_vs_baseline_log).

    Returns
    -------
    dict
        Dictionary with keys: complexity, noise, window_size, log_seed, residual, n_obs, r2_full.
        Values are normalized shares (percentages) that sum to 100.
    """
    required = [outcome_col, "window_size", "complexity", "noise", "log_seed"]
    df_clean = df_m[[c for c in required if c in df_m.columns]].copy()
    if outcome_col not in df_clean.columns or "log_seed" not in df_clean.columns:
        return {
            "complexity": 0.0,
            "noise": 0.0,
            "window_size": 0.0,
            "log_seed": 0.0,
            "residual": 1.0,
            "n_obs": 0,
            "r2_full": 0.0,
        }

    # Ordered numeric encoding for complexity and noise (simple<mid<complex; 0<low<high)
    df_clean["complexity_ord"] = df_clean["complexity"].map(COMPLEXITY_ORD)
    df_clean["noise_ord"] = df_clean["noise"].map(NOISE_ORD)
    df_clean = df_clean[
        df_clean["complexity_ord"].notna() & df_clean["noise_ord"].notna()
    ].copy()

    # Create log-transformed outcome
    df_clean["y"] = np.log(df_clean[outcome_col] + outcome_log_shift)

    # Drop rows with non-finite y
    df_clean = df_clean[np.isfinite(df_clean["y"])].copy()

    n_obs = len(df_clean)

    if n_obs < MIN_OBS_FOR_FACTOR_ANALYSIS:
        # Return zeros if insufficient data (residual as proportion, not percentage)
        return {
            "complexity": 0.0,
            "noise": 0.0,
            "window_size": 0.0,
            "log_seed": 0.0,
            "residual": 1.0,  # 100% as proportion
            "n_obs": n_obs,
            "r2_full": 0.0,
        }

    # Fit full model (complexity and noise as ordered numeric; complexity × window_size interaction)
    try:
        model_full = smf.ols(
            "y ~ np.log(window_size) + complexity_ord + noise_ord + np.log(window_size):complexity_ord + C(log_seed)",
            data=df_clean,
        ).fit()
        r2_full = model_full.rsquared
    except Exception as e:
        print(f"Warning: Failed to fit full model: {e}")
        return {
            "complexity": 0.0,
            "noise": 0.0,
            "window_size": 0.0,
            "log_seed": 0.0,
            "residual": 1.0,  # 100% as proportion
            "n_obs": n_obs,
            "r2_full": 0.0,
        }

    # Fit reduced models (drop one factor at a time)
    partial_r2_vals = {}

    # Drop complexity
    try:
        model_no_complexity = smf.ols(
            "y ~ np.log(window_size) + noise_ord + C(log_seed)",
            data=df_clean,
        ).fit()
        partial_r2_vals["complexity"] = partial_r2(model_full, model_no_complexity)
    except Exception:
        partial_r2_vals["complexity"] = 0.0

    # Drop noise (keep complexity × window_size interaction)
    try:
        model_no_noise = smf.ols(
            "y ~ np.log(window_size) + complexity_ord + np.log(window_size):complexity_ord + C(log_seed)",
            data=df_clean,
        ).fit()
        partial_r2_vals["noise"] = partial_r2(model_full, model_no_noise)
    except Exception:
        partial_r2_vals["noise"] = 0.0

    # Drop window_size
    try:
        model_no_window = smf.ols(
            "y ~ complexity_ord + noise_ord + C(log_seed)",
            data=df_clean,
        ).fit()
        partial_r2_vals["window_size"] = partial_r2(model_full, model_no_window)
    except Exception:
        partial_r2_vals["window_size"] = 0.0

    # Drop log_seed (keep complexity × window_size interaction)
    try:
        model_no_log_seed = smf.ols(
            "y ~ np.log(window_size) + complexity_ord + noise_ord + np.log(window_size):complexity_ord",
            data=df_clean,
        ).fit()
        partial_r2_vals["log_seed"] = partial_r2(model_full, model_no_log_seed)
    except Exception:
        partial_r2_vals["log_seed"] = 0.0

    # Compute residual share (unexplained variance)
    residual_raw = max(0.0, 1.0 - r2_full)

    # Normalize partial R² values and residual to sum to 100%
    total_all = (
        partial_r2_vals["complexity"]
        + partial_r2_vals["noise"]
        + partial_r2_vals["window_size"]
        + partial_r2_vals["log_seed"]
        + residual_raw
    )

    if total_all > 0:
        normalized = {
            "complexity": partial_r2_vals["complexity"] / total_all,
            "noise": partial_r2_vals["noise"] / total_all,
            "window_size": partial_r2_vals["window_size"] / total_all,
            "log_seed": partial_r2_vals["log_seed"] / total_all,
        }
        residual_share = residual_raw / total_all
    else:
        normalized = {
            "complexity": 0.0,
            "noise": 0.0,
            "window_size": 0.0,
            "log_seed": 0.0,
        }
        residual_share = 1.0

    return {
        "complexity": normalized["complexity"],
        "noise": normalized["noise"],
        "window_size": normalized["window_size"],
        "log_seed": normalized["log_seed"],
        "residual": residual_share,
        "n_obs": n_obs,
        "r2_full": r2_full,
    }


def run_noise_factor_importance(
    df_relnoise_log: pd.DataFrame,
    *,
    outcome_col: str = "rel_noise_log",
    outcome_log_shift: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run factor importance analysis for a per-log outcome across all metrics.

    Parameters
    ----------
    df_relnoise_log
        DataFrame with per-log outcome and factors, including columns:
        Metric, outcome_col, window_size, complexity, noise, log_seed.
    outcome_col
        Name of the outcome column (e.g. rel_noise_log or change_vs_baseline_log).
    outcome_log_shift
        Added to outcome before log; use a small value when outcome can be 0.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Long and wide format DataFrames with factor importance results.
    """
    results = []

    required_cols = [outcome_col, "window_size", "complexity", "noise", "log_seed"]
    for metric in df_relnoise_log["Metric"].unique():
        df_metric = df_relnoise_log[df_relnoise_log["Metric"] == metric].copy()

        missing_cols = [col for col in required_cols if col not in df_metric.columns]
        if missing_cols:
            print(f"Warning: Metric {metric} missing columns {missing_cols}, skipping.")
            continue

        components = compute_partial_r2_components(
            df_metric,
            outcome_col=outcome_col,
            outcome_log_shift=outcome_log_shift,
        )

        results.append(
            {
                "Metric": metric,
                "complexity": components["complexity"] * 100,
                "noise": components["noise"] * 100,
                "window_size": components["window_size"] * 100,
                "log_seed": components["log_seed"] * 100,
                "residual": components["residual"] * 100,
                "n_obs": components["n_obs"],
                "r2_full": components["r2_full"],
            }
        )

    # Create wide format
    df_wide = pd.DataFrame(results)
    df_wide = df_wide.set_index("Metric")
    df_wide = _add_dimension_and_sort_wide(df_wide)

    # Create long format by melting
    df_long = df_wide.reset_index().melt(
        id_vars=["dimension", "Metric", "n_obs", "r2_full"],
        value_vars=["complexity", "noise", "window_size", "log_seed", "residual"],
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
    factor_means = df_wide[
        ["complexity", "noise", "window_size", "log_seed", "residual"]
    ].mean()
    for factor, mean_pct in factor_means.items():
        print(f"   {factor:15s}: {mean_pct:6.2f}%")

    # Top factor per metric
    print("\n2. Dominant factor per metric (highest share):")
    for dim, metric in df_wide.index:
        row = df_wide.loc[(dim, metric)]
        factors = ["complexity", "noise", "window_size", "log_seed", "residual"]
        top_factor = max(factors, key=lambda f: row[f])
        top_value = row[top_factor]
        print(f"   {metric:40s}: {top_factor:15s} ({top_value:6.2f}%)")

    # Metrics with low n_obs
    low_n = df_wide[df_wide["n_obs"] < MIN_OBS_FOR_FACTOR_ANALYSIS]
    if len(low_n) > 0:
        print(
            f"\n3. Warning: {len(low_n)} metrics have n_obs < {MIN_OBS_FOR_FACTOR_ANALYSIS}:"
        )
        for dim, metric in low_n.index:
            print(f"   {metric}: n_obs = {low_n.loc[(dim, metric), 'n_obs']}")

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


def _pivot_noise_wide(
    df_long: pd.DataFrame,
    value_column: str,
    *,
    drop_no_noise_columns: bool = False,
) -> pd.DataFrame:
    """
    Pivot long noise table to wide (dimension, Metric) x (complexity, noise, window_size).

    Parameters
    ----------
    df_long
        Long-format DataFrame with dimension, Metric, complexity, noise, window_size,
        and one value column (e.g. rel_noise or value).
    value_column
        Name of the column to use as cell values.
    drop_no_noise_columns
        If True, drop columns where noise level is "0".

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with MultiIndex columns (complexity, noise, window_size).
    """
    df_wide = df_long.pivot_table(
        index=["dimension", "Metric"],
        columns=["complexity", "noise", "window_size"],
        values=value_column,
        aggfunc="first",
    )
    dim_order = list(DIMENSIONS_ORDER) + ["Other"]
    desired_index = []
    for d in dim_order:
        for m in _METRIC_ORDER:
            if (d, m) in df_wide.index:
                desired_index.append((d, m))
    for idx in df_wide.index:
        if idx not in desired_index:
            desired_index.append(idx)
    df_wide = df_wide.reindex(desired_index)

    if drop_no_noise_columns and isinstance(df_wide.columns, pd.MultiIndex):
        noise_level = df_wide.columns.get_level_values(1)
        cols_keep = [i for i, n in enumerate(noise_level) if n != "0"]
        df_wide = df_wide.iloc[:, cols_keep]

    complexity_order = ["simple", "mid", "complex"]
    noise_order = ["0", "low", "high"]
    if isinstance(df_wide.columns, pd.MultiIndex):
        complexity_level = list(df_wide.columns.get_level_values(0).unique())
        noise_level = list(df_wide.columns.get_level_values(1).unique())
        window_sizes = sorted(df_wide.columns.get_level_values(2).unique())
        complexity_sorted = [c for c in complexity_order if c in complexity_level]
        complexity_sorted.extend(
            [c for c in complexity_level if c not in complexity_order]
        )
        noise_sorted = [n for n in noise_order if n in noise_level]
        noise_sorted.extend([n for n in noise_level if n not in noise_order])
        sorted_columns = pd.MultiIndex.from_product(
            [complexity_sorted, noise_sorted, window_sizes],
            names=["complexity", "noise", "window_size"],
        )
        existing_columns = [col for col in sorted_columns if col in df_wide.columns]
        df_wide = df_wide[existing_columns]
    return df_wide


def _aggregate_abs_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate absolute median of center per (Metric, complexity, noise, window_size).

    Parameters
    ----------
    df
        DataFrame with Metric, complexity, noise, window_size, center.

    Returns
    -------
    pd.DataFrame
        Long table with Metric, complexity, noise, window_size, value.
    """
    agg = (
        df.groupby(["Metric", "complexity", "noise", "window_size"])["center"]
        .median()
        .reset_index()
    )
    agg = agg.rename(columns={"center": "value"})
    return agg


def _aggregate_relci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate relative CI = (CI_high - CI_low) / center per cell; median across logs.

    Parameters
    ----------
    df
        DataFrame with Metric, complexity, noise, window_size, center, Sample CI Low/High.

    Returns
    -------
    pd.DataFrame
        Long table with Metric, complexity, noise, window_size, value.
    """
    d = df.copy()
    d["rel_ci"] = (d["Sample CI High"] - d["Sample CI Low"]) / (d["center"].abs() + eps)
    agg = (
        d.groupby(["Metric", "complexity", "noise", "window_size"])["rel_ci"]
        .median()
        .reset_index()
    )
    agg = agg.rename(columns={"rel_ci": "value"})
    return agg


def _aggregate_change_due_to_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustness = |median_current_noise - median_no_noise| / |median_no_noise|; only noise != "0".

    Parameters
    ----------
    df
        DataFrame with Metric, complexity, noise, window_size, center.

    Returns
    -------
    pd.DataFrame
        Long table with Metric, complexity, noise, window_size, value (only noise != "0").
    """
    median_cur = (
        df.groupby(["Metric", "complexity", "noise", "window_size"])["center"]
        .median()
        .reset_index()
    )
    median_no = (
        df[df["noise"] == "0"]
        .groupby(["Metric", "complexity", "window_size"])["center"]
        .median()
        .reset_index()
    )
    median_no = median_no.rename(columns={"center": "median_no_noise"})
    merged = median_cur.merge(
        median_no, on=["Metric", "complexity", "window_size"], how="inner"
    )
    merged["value"] = (merged["center"] - merged["median_no_noise"]).abs() / (
        merged["median_no_noise"].abs() + eps
    )
    merged = merged[merged["noise"] != "0"][
        ["Metric", "complexity", "noise", "window_size", "value"]
    ]
    return merged


def create_wide_format_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Create wide-format table with 3-level column hierarchy (relative noise).

    Parameters
    ----------
    df_long
        Long-format DataFrame with dimension, Metric, complexity, noise, window_size, rel_noise.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with MultiIndex columns (complexity, noise, window_size).
    """
    return _pivot_noise_wide(df_long, "rel_noise", drop_no_noise_columns=False)


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

    # Compute signal, SNR, and relative change
    df_joined["signal"] = (df_joined["center_post"] - df_joined["center_pre"]).abs()
    df_joined["snr_log"] = df_joined["signal"] / (df_joined["iqr_pre"] + eps)
    df_joined["rel_change_log"] = (
        df_joined["center_post"] - df_joined["center_pre"]
    ) / (df_joined["center_pre"].abs() + eps)

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
        "rel_change_log",
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

    agg_result = _add_dimension_and_sort_long(agg_result)
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
    # Pivot to wide format (index = dimension, Metric)
    df_wide = df_snr_cells.pivot_table(
        index=["dimension", "Metric"],
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

    # Sort index by dimension then metric order
    dim_order = list(DIMENSIONS_ORDER) + ["Other"]
    desired_index = []
    for d in dim_order:
        for m in _METRIC_ORDER:
            if (d, m) in df_wide.index:
                desired_index.append((d, m))
    for idx in df_wide.index:
        if idx not in desired_index:
            desired_index.append(idx)
    df_wide = df_wide.reindex(desired_index)

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


def _aggregate_by_group(
    df_log: pd.DataFrame,
    value_col: str,
    group_col: str,
    column_order: list[str] | None,
    *,
    filter_col: str | None = None,
    filter_val: object = None,
) -> pd.DataFrame:
    """
    Aggregate by (Metric, group_col): median(value_col), pivot to wide, add dimension and sort.

    Parameters
    ----------
    df_log
        Per-log DataFrame with Metric, value_col, and group_col (and optional filter_col).
    value_col
        Column to aggregate (e.g. snr_log or rel_change_log).
    group_col
        Column to use as wide columns (e.g. change_operation, evolution_proportion, noise).
    column_order
        Desired column order for the wide table; if None, use sorted(df_wide.columns).
    filter_col
        If set, filter to rows where df_log[filter_col] == filter_val before aggregating.
    filter_val
        Value for filter_col.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with index (dimension, Metric) and columns = group_col values.
    """
    df = df_log
    if filter_col is not None and filter_val is not None:
        df = df_log[df_log[filter_col] == filter_val].copy()
        if len(df) == 0:
            raise ValueError(
                f"No data found with {filter_col}={filter_val!r}. "
                f"Cannot create table grouped by {group_col}."
            )
    agg_result = df.groupby(["Metric", group_col])[value_col].median().reset_index()
    df_wide = agg_result.pivot_table(
        index="Metric",
        columns=group_col,
        values=value_col,
        aggfunc="first",
    )
    if column_order is not None:
        existing = [c for c in column_order if c in df_wide.columns]
        existing.extend([c for c in df_wide.columns if c not in column_order])
        df_wide = df_wide[existing]
    else:
        df_wide = df_wide[sorted(df_wide.columns)]
    return _add_dimension_and_sort_wide(df_wide)


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
    return _aggregate_by_group(
        df_snr_log,
        "snr_log",
        "change_operation",
        CHANGE_OPERATION_ORDER,
    )


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
    return _aggregate_by_group(
        df_snr_log,
        "snr_log",
        "evolution_proportion",
        None,
        filter_col="change_operation",
        filter_val="combined",
    )


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
    noise_order = ["0", "low", "high"]
    return _aggregate_by_group(
        df_snr_log,
        "snr_log",
        "noise",
        noise_order,
        filter_col="change_operation",
        filter_val="combined",
    )


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


# -----------------------------------------------------------------------------
# LaTeX helpers
# -----------------------------------------------------------------------------


def _escape_latex_cell(text: object) -> str:
    """
    Escape special LaTeX characters in header/cell strings.

    Escapes: _, #, %, &, {, }.

    Parameters
    ----------
    text
        Value to escape (converted to string).

    Returns
    -------
    str
        Escaped string safe for LaTeX.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    return s


def _apply_latex_metric_names(df: pd.DataFrame, index: bool) -> pd.DataFrame:
    """
    Return a copy of df with Metric names replaced by METRIC_NAMES_TO_LATEX_MAP.

    Parameters
    ----------
    df
        DataFrame with Metric column or index level "Metric".
    index
        If True, replace Metric in the index; if False, in the column.

    Returns
    -------
    pd.DataFrame
        Copy with LaTeX metric names; use escape=False when writing LaTeX.
    """
    out = df.copy()
    if index and isinstance(out.index, pd.MultiIndex) and "Metric" in out.index.names:
        level_pos = out.index.names.index("Metric")
        # set_levels expects the new level's unique values in same order as .levels[level_pos]
        unique_vals = out.index.levels[level_pos]
        tex_vals = [METRIC_NAMES_TO_LATEX_MAP.get(v, str(v)) for v in unique_vals]
        out.index = out.index.set_levels(tex_vals, level=level_pos)
    elif index and out.index.name == "Metric":
        tex = pd.Index(
            [METRIC_NAMES_TO_LATEX_MAP.get(v, str(v)) for v in out.index],
            name=out.index.name,
        )
        out.index = tex
    elif "Metric" in out.columns:
        out["Metric"] = out["Metric"].map(
            lambda x: METRIC_NAMES_TO_LATEX_MAP.get(x, str(x))
        )
    return out


def _latex_header_label(name: str | None) -> str | None:
    """Normalize header for LaTeX: log_id -> log\\_id (escaped); others underscore -> space."""
    if name is None:
        return None
    s = str(name)
    if s == "log_id":
        return "log\\_id"
    return s.replace("_", " ")


def _headers_underscores_to_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with index and column header labels normalized for LaTeX.

    Replaces underscores with spaces (e.g. window_size -> window size).
    log_id is rendered as log\\_id so the underscore is escaped in LaTeX.

    Parameters
    ----------
    df
        DataFrame to transform.

    Returns
    -------
    pd.DataFrame
        Copy with header labels normalized.
    """
    out = df.copy()
    # Index: names and level values
    if isinstance(out.index, pd.MultiIndex):
        new_names = [_latex_header_label(n) for n in out.index.names]
        out.index = out.index.set_names(new_names)
        for lev in range(out.index.nlevels):
            lvals = out.index.levels[lev]
            new_vals = [_latex_header_label(v) for v in lvals]
            out.index = out.index.set_levels(new_vals, level=lev)
    else:
        name = out.index.name
        new_name = _latex_header_label(name)
        out.index = pd.Index([_latex_header_label(v) for v in out.index], name=new_name)
    # Columns: names and level values
    if isinstance(out.columns, pd.MultiIndex):
        new_names = [_latex_header_label(n) for n in out.columns.names]
        out.columns = out.columns.set_names(new_names)
        for lev in range(out.columns.nlevels):
            lvals = out.columns.levels[lev]
            new_vals = [_latex_header_label(v) for v in lvals]
            out.columns = out.columns.set_levels(new_vals, level=lev)
    else:
        name = out.columns.name
        new_name = _latex_header_label(name)
        out.columns = pd.Index(
            [_latex_header_label(v) for v in out.columns], name=new_name
        )
    return out


def _inject_heatmap_cellcolor(
    latex: str,
    df: pd.DataFrame,
    n_index_cols: int,
    *,
    heatmap_max_percentile: float | None = None,
    heatmap_exclude_columns: list[str] | None = None,
) -> str:
    """
    Inject \\cellcolor{blue!NN} into numeric data cells (white=min, dark blue=max).

    Inf and NA values are excluded from the min/max scale and are not colored.

    Parameters
    ----------
    latex
        Full table LaTeX string (with \\midrule before body).
    df
        DataFrame that was used to generate the table (same row/column order).
    n_index_cols
        Number of index columns at the start of each row.
    heatmap_max_percentile
        If set (e.g. 95), use this percentile as the max for the color scale;
        values at or above it get the darkest color. If None, use the actual max.
    heatmap_exclude_columns
        Column names to exclude from coloring (e.g. n_obs, r2_full).

    Returns
    -------
    str
        Modified LaTeX with \\cellcolor in numeric data cells.
    """
    exclude = set(heatmap_exclude_columns or [])

    # Column names may be transformed (e.g. n_obs -> "n obs"); match both forms
    def _is_excluded(name: str) -> bool:
        return name in exclude or str(name).replace(" ", "_") in exclude

    numeric_cols = [
        j
        for j in range(len(df.columns))
        if pd.api.types.is_numeric_dtype(df.iloc[:, j])
        and not _is_excluded(df.columns[j])
    ]
    if not numeric_cols:
        return latex
    vals = df.iloc[:, numeric_cols].values.astype(float)
    valid = np.isfinite(vals)
    # Use only finite values for scale so inf/NA do not affect color range
    vals_finite = vals[valid]
    vmin = float(np.min(vals_finite)) if np.any(valid) else 0.0
    if heatmap_max_percentile is not None and np.any(valid):
        vmax = float(np.nanpercentile(vals_finite, heatmap_max_percentile))
    else:
        vmax = float(np.max(vals_finite)) if np.any(valid) else 1.0
    span = vmax - vmin
    if span <= 0:
        span = 1.0
    col_to_numeric_idx = {j: i for i, j in enumerate(numeric_cols)}

    lines = latex.split("\n")
    out_lines = []
    body_started = False
    row_idx = 0
    for line in lines:
        if "\\midrule" in line:
            body_started = True
            out_lines.append(line)
            continue
        if body_started and ("\\bottomrule" in line or "\\end{tabular}" in line):
            body_started = False
            out_lines.append(line)
            continue
        if body_started and row_idx < len(df):
            parts = line.split(" & ")
            if parts:
                new_parts = list(parts[:n_index_cols])
                for j in range(len(df.columns)):
                    idx = n_index_cols + j
                    cell = (
                        parts[idx].rstrip().rstrip("\\\\").strip()
                        if idx < len(parts)
                        else ""
                    )
                    if j in col_to_numeric_idx:
                        try:
                            v = float(df.iloc[row_idx, j])
                            if np.isfinite(v):
                                pct = int(5 + 95 * (v - vmin) / span)
                                pct = max(0, min(100, pct))  # values >= vmax get 100
                                cell = f"\\cellcolor{{blue!{pct}}}{{{cell}}}"
                        except (ValueError, TypeError):
                            pass
                    new_parts.append(cell)
                line = " & ".join(new_parts) + " \\\\"
            row_idx += 1
        out_lines.append(line)
    return "\n".join(out_lines)


def dataframe_to_latex_table(
    df: pd.DataFrame,
    filepath: str | Path,
    caption: str,
    label: str,
    *,
    decimals: int = 2,
    index: bool = True,
    use_latex_metric_names: bool = True,
    heatmap: bool = False,
    heatmap_max_percentile: float | None = None,
    heatmap_exclude_columns: list[str] | None = None,
) -> None:
    """
    Write a DataFrame to a LaTeX table file with caption, label, and tiny size.

    Handles both single-level and MultiIndex column headers. MultiIndex columns
    are rendered with multicolumn so multi-headers are correct. When
    use_latex_metric_names is True, metric names are replaced by
    METRIC_NAMES_TO_LATEX_MAP (and escape=False is used). When heatmap=True,
    data cells are colored from white (min) to dark blue (max).

    Parameters
    ----------
    df
        DataFrame to write.
    filepath
        Output path for the .tex file.
    caption
        Table caption.
    label
        LaTeX label (e.g. tab:signal-noise-relative-wide).
    decimals
        Number of decimal places for numeric cells.
    index
        If True, include the DataFrame index in the LaTeX table (e.g. dimension
        and Metric for wide tables). If False, omit the index (for long-format tables).
    use_latex_metric_names
        If True, replace metric names with METRIC_NAMES_TO_LATEX_MAP for LaTeX.
    heatmap
        If True, color data cells by value (min=white, max=dark blue).
    heatmap_max_percentile
        If set (e.g. 95), use this percentile as max for the color scale;
        values at or above it get the darkest color. If None, use actual max.
    heatmap_exclude_columns
        Column names to exclude from heatmap coloring (e.g. n_obs, r2_full).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if use_latex_metric_names:
        df = _apply_latex_metric_names(df, index)
    df = _headers_underscores_to_spaces(df)

    n_cols = len(df.columns)
    # Index: one or two columns (dimension, Metric); data columns: c
    if index:
        n_index_levels = (
            len(df.index.names)
            if isinstance(df.index, pd.MultiIndex)
            else (1 if df.index.name else 0)
        )
        if n_index_levels == 0:
            n_index_levels = 1
        col_fmt = "l" * n_index_levels + "c" * n_cols
    else:
        n_index_levels = 0
        col_fmt = "c" * n_cols

    is_multi = isinstance(df.columns, pd.MultiIndex)
    latex_kw: dict = {
        "caption": caption,
        "label": label,
        "position": "htbp",
        "column_format": col_fmt,
        "escape": not use_latex_metric_names,
        "float_format": f"%.{decimals}f",
        "na_rep": "",
        "index": index,
    }
    if is_multi:
        latex_kw["multicolumn"] = True
        latex_kw["multicolumn_format"] = "c"
        latex_kw["sparsify"] = True

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*DataFrame.to_latex.*",
        )
        out = df.to_latex(**latex_kw)
    if heatmap and len(df) > 0 and len(df.columns) > 0:
        out = _inject_heatmap_cellcolor(
            out,
            df,
            n_index_levels,
            heatmap_max_percentile=heatmap_max_percentile,
            heatmap_exclude_columns=heatmap_exclude_columns,
        )
    # Inject \tiny inside the table environment (after \centering) for tiny size
    if "\\centering" in out:
        out = out.replace("\\centering\n", "\\centering\n\\tiny\n")
    else:
        out = out.replace("\\begin{table}", "\\begin{table}\n\\tiny\n")
    filepath.write_text(out, encoding="utf-8")


def _write_latex_table(
    df: pd.DataFrame,
    stem: str,
    caption: str,
    label: str,
    *,
    heatmap: bool = True,
    index: bool = True,
    decimals: int = 2,
    **kwargs: object,
) -> None:
    """
    Write a wide table to DIR_LATEX/{stem}.tex using dataframe_to_latex_table.

    Parameters
    ----------
    df
        DataFrame to write.
    stem
        Basename for .tex file (no extension).
    caption
        Table caption.
    label
        LaTeX label (e.g. tab:signal-noise-relative-wide).
    heatmap
        If True, apply value-based cell coloring.
    index
        Whether to include index in the table.
    decimals
        Decimal places for numeric cells.
    **kwargs
        Passed through to dataframe_to_latex_table.
    """
    filepath = f"{DIR_LATEX}/{stem}.tex"
    dataframe_to_latex_table(
        df,
        filepath,
        caption=caption,
        label=label,
        decimals=decimals,
        index=index,
        heatmap=heatmap,
        **kwargs,
    )


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
    print("Adding per-log change vs. baseline...")
    df_with_noise = add_change_vs_baseline_log(df_with_noise)

    print("Aggregating relative noise across logs...")
    df_long = aggregate_relative_noise(df_with_noise)

    print("Creating wide-format table...")
    df_wide = create_wide_format_table(df_long)

    print("Saving outputs...")
    save_outputs(df_long, df_wide)

    # LaTeX tables for relative noise
    _write_latex_table(
        df_long,
        Path(PATH_OUT_LONG).stem,
        caption="Relative noise (long format).",
        label="tab:signal-noise-relative-long",
        index=False,
    )
    _write_latex_table(
        df_wide,
        Path(PATH_OUT_WIDE).stem,
        caption="Relative noise by complexity, noise, and window size.",
        label="tab:signal-noise-relative-wide",
        index=True,
    )

    print("Performing sanity checks...")
    perform_sanity_checks(df_long, n_invalid)

    # Tables 1–3: noise_abs_median, noise_relci, noise_change_due_to_noise (same layout as relative_noise_wide)
    _NOISE_TABLE_CONFIGS = [
        (
            "noise_abs_median",
            _aggregate_abs_median,
            False,
            "Absolute median metric value.",
            "tab:signal-noise-abs-median",
        ),
        (
            "noise_relci",
            _aggregate_relci,
            False,
            "Relative confidence interval (CI width / median).",
            "tab:signal-noise-relci",
        ),
        (
            "noise_change_due_to_noise",
            _aggregate_change_due_to_noise,
            True,
            "Robustness to noise (change vs. no-noise baseline).",
            "tab:signal-noise-change-due-to-noise",
        ),
    ]
    Path(PATH_NOISE_ABS_MEDIAN).parent.mkdir(parents=True, exist_ok=True)
    _NOISE_LATEX_WINDOW_SIZES = (50, 200)
    for stem, aggregator_fn, drop_no_noise, caption, label in _NOISE_TABLE_CONFIGS:
        df_noise_long = aggregator_fn(df_with_noise)
        df_noise_long = _add_dimension_and_sort_long(df_noise_long)
        df_noise_wide = _pivot_noise_wide(
            df_noise_long, "value", drop_no_noise_columns=drop_no_noise
        )
        path_csv = f"{DIR_CSV}/{stem}.csv"
        df_noise_wide.to_csv(path_csv)
        print(f"Saved {stem} to {path_csv}")
        # LaTeX: only window sizes 50 and 200; color scale capped at 95th percentile
        df_noise_wide_latex = df_noise_wide.loc[
            :,
            df_noise_wide.columns.get_level_values("window_size").isin(
                _NOISE_LATEX_WINDOW_SIZES
            ),
        ]
        _write_latex_table(
            df_noise_wide_latex,
            stem,
            caption=caption,
            label=label,
            index=True,
            heatmap_max_percentile=95.0,
        )

    print("\n" + "=" * 60)
    print("NOISE FACTOR IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Factor importance: relative CI (rel_noise_log)
    print("Computing factor importance for relative CIs (rel_noise_log)...")
    df_factor_long, df_factor_wide = run_noise_factor_importance(
        df_with_noise, outcome_col="rel_noise_log", outcome_log_shift=0.0
    )
    Path(PATH_NOISE_FACTOR_LONG).parent.mkdir(parents=True, exist_ok=True)
    df_factor_long.to_csv(PATH_NOISE_FACTOR_LONG, index=False)
    print(f"Saved factor importance (relci) long to {PATH_NOISE_FACTOR_LONG}")
    df_factor_wide.to_csv(PATH_NOISE_FACTOR_WIDE)
    print(f"Saved factor importance (relci) wide to {PATH_NOISE_FACTOR_WIDE}")
    _write_latex_table(
        df_factor_long,
        Path(PATH_NOISE_FACTOR_LONG).stem,
        caption="Noise factor importance (relative CI), long format.",
        label="tab:signal-noise-factor-importance-long",
        index=False,
    )
    _write_latex_table(
        df_factor_wide,
        Path(PATH_NOISE_FACTOR_WIDE).stem,
        caption="Noise factor importance (relative CI) by metric.",
        label="tab:signal-noise-factor-importance-wide",
        index=True,
        heatmap_exclude_columns=["n_obs", "r2_full"],
    )
    print("Factor importance (relative CI) summary:")
    print_factor_importance_summary(df_factor_long, df_factor_wide)

    # Factor importance: change vs. baseline (exclude noise=0: change is always 0 there)
    print("Computing factor importance for change vs. baseline...")
    df_factor_long_change, df_factor_wide_change = run_noise_factor_importance(
        df_with_noise[df_with_noise["noise"] != "0"],
        outcome_col="change_vs_baseline_log",
        outcome_log_shift=eps,
    )
    df_factor_long_change.to_csv(PATH_NOISE_FACTOR_LONG_CHANGE, index=False)
    print(f"Saved factor importance (change) long to {PATH_NOISE_FACTOR_LONG_CHANGE}")
    df_factor_wide_change.to_csv(PATH_NOISE_FACTOR_WIDE_CHANGE)
    print(f"Saved factor importance (change) wide to {PATH_NOISE_FACTOR_WIDE_CHANGE}")
    _write_latex_table(
        df_factor_long_change,
        Path(PATH_NOISE_FACTOR_LONG_CHANGE).stem,
        caption="Noise factor importance (change vs. baseline), long format.",
        label="tab:signal-noise-factor-importance-change-long",
        index=False,
    )
    _write_latex_table(
        df_factor_wide_change,
        Path(PATH_NOISE_FACTOR_WIDE_CHANGE).stem,
        caption="Noise factor importance (change vs. baseline) by metric.",
        label="tab:signal-noise-factor-importance-change-wide",
        index=True,
        heatmap_exclude_columns=["n_obs", "r2_full"],
    )
    print("Factor importance (change vs. baseline) summary:")
    print_factor_importance_summary(df_factor_long_change, df_factor_wide_change)

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

    # Tables 4–6: signal_relchange by operation, evolution_proportion, noise
    _SIGNAL_RELCHANGE_CONFIGS = [
        (
            "signal_relchange_by_operation",
            "change_operation",
            CHANGE_OPERATION_ORDER,
            None,
            None,
            "Median relative change pre-to-post by change operation.",
            "tab:signal-relchange-by-operation",
        ),
        (
            "signal_relchange_by_evolution_proportion",
            "evolution_proportion",
            None,
            "change_operation",
            "combined",
            "Median relative change pre-to-post by evolution proportion (change_operation=combined).",
            "tab:signal-relchange-by-evolution",
        ),
        (
            "signal_relchange_by_noise",
            "noise",
            ["0", "low", "high"],
            "change_operation",
            "combined",
            "Median relative change pre-to-post by noise (change_operation=combined).",
            "tab:signal-relchange-by-noise",
        ),
    ]
    for (
        stem,
        group_col,
        column_order,
        filter_col,
        filter_val,
        caption,
        label,
    ) in _SIGNAL_RELCHANGE_CONFIGS:
        df_rel = _aggregate_by_group(
            df_snr_log,
            "rel_change_log",
            group_col,
            column_order,
            filter_col=filter_col,
            filter_val=filter_val,
        )
        path_csv = f"{DIR_CSV}/{stem}.csv"
        df_rel.to_csv(path_csv)
        print(f"Saved {stem} to {path_csv}")
        _write_latex_table(df_rel, stem, caption=caption, label=label, index=True)

    # Tables 7–9: signal_snr (same DataFrames as snr_by_*; write to signal_snr_by_* paths)
    _SIGNAL_SNR_CONFIGS = [
        (
            PATH_SIGNAL_SNR_BY_OP,
            "signal_snr_by_operation",
            "SNR by change operation.",
            "tab:signal-snr-by-operation",
        ),
        (
            PATH_SIGNAL_SNR_BY_EVOL,
            "signal_snr_by_evolution_proportion",
            "SNR by evolution proportion (change\\_operation=combined).",
            "tab:signal-snr-by-evolution",
        ),
        (
            PATH_SIGNAL_SNR_BY_NOISE,
            "signal_snr_by_noise",
            "SNR by noise level (change\\_operation=combined).",
            "tab:signal-snr-by-noise",
        ),
    ]
    df_snr_by_tables = [df_snr_by_op, df_snr_by_evol, df_snr_by_noise]
    for (path_csv, stem, caption, label), df_snr_by in zip(
        _SIGNAL_SNR_CONFIGS, df_snr_by_tables
    ):
        df_snr_by.to_csv(path_csv)
        print(f"Saved {stem} to {path_csv}")
        _write_latex_table(df_snr_by, stem, caption=caption, label=label, index=True)

    # LaTeX tables for SNR
    _write_latex_table(
        df_snr_cells,
        Path(PATH_SNR_LONG).stem,
        caption="SNR per cell (long format).",
        label="tab:signal-noise-snr-long",
        index=False,
    )
    _write_latex_table(
        df_snr_wide,
        Path(PATH_SNR_WIDE).stem,
        caption="SNR by evolution proportion, change operation, complexity, noise, and window size.",
        label="tab:signal-noise-snr-wide",
        index=True,
    )
    _write_latex_table(
        df_snr_by_op,
        Path(PATH_SNR_BY_OP).stem,
        caption="SNR by change operation.",
        label="tab:signal-noise-snr-by-operation",
        index=True,
    )
    _write_latex_table(
        df_snr_by_evol,
        Path(PATH_SNR_BY_EVOL).stem,
        caption="SNR by evolution proportion (change\\_operation=combined).",
        label="tab:signal-noise-snr-by-evolution",
        index=True,
    )
    _write_latex_table(
        df_snr_by_noise,
        Path(PATH_SNR_BY_NOISE).stem,
        caption="SNR by noise level (change\\_operation=combined).",
        label="tab:signal-noise-snr-by-noise",
        index=True,
    )

    print("Performing SNR sanity checks...")
    perform_snr_sanity_checks(df_snr_log, df_snr_cells)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
