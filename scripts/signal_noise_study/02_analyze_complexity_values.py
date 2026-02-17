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
from typing import Any, Literal

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
PATH_SIGNAL_ABS_COHENS_D_BY_OP = f"{DIR_CSV}/signal_abs_cohens_d_by_operation.csv"
PATH_SIGNAL_ABS_COHENS_D_BY_EVOL = (
    f"{DIR_CSV}/signal_abs_cohens_d_by_evolution_proportion.csv"
)
PATH_SIGNAL_ABS_COHENS_D_BY_NOISE = f"{DIR_CSV}/signal_abs_cohens_d_by_noise.csv"
PATH_NOISE_FACTOR_LONG = f"{DIR_CSV}/noise_factor_importance_relci_long.csv"
PATH_NOISE_FACTOR_WIDE = f"{DIR_CSV}/noise_factor_importance_relci_wide.csv"
PATH_NOISE_FACTOR_LONG_CHANGE = f"{DIR_CSV}/noise_factor_importance_change_long.csv"
PATH_NOISE_FACTOR_WIDE_CHANGE = f"{DIR_CSV}/noise_factor_importance_change_wide.csv"
PATH_NOISE_FACTOR_LONG_STD = f"{DIR_CSV}/noise_factor_importance_std_long.csv"
PATH_NOISE_FACTOR_WIDE_STD = f"{DIR_CSV}/noise_factor_importance_std_wide.csv"
PATH_NOISE_FACTOR_LONG_CV = f"{DIR_CSV}/noise_factor_importance_cv_long.csv"
PATH_NOISE_FACTOR_WIDE_CV = f"{DIR_CSV}/noise_factor_importance_cv_wide.csv"
PATH_NOISE_ABS_MEDIAN = f"{DIR_CSV}/noise_abs_median.csv"
PATH_NOISE_RELCI = f"{DIR_CSV}/noise_relci.csv"
PATH_NOISE_STD_PRE = f"{DIR_CSV}/noise_std_pre.csv"
PATH_NOISE_CV_PRE = f"{DIR_CSV}/noise_cv_pre.csv"
PATH_NOISE_CHANGE_DUE_TO_NOISE = f"{DIR_CSV}/noise_change_due_to_noise.csv"
AGGREGATION: Literal["mean", "median"] = "median"
# Two-stage aggregation: within-seed first, then across-seed
WITHIN_SEED_AGGREGATION: Literal["mean", "median"] = "median"
ACROSS_SEED_AGGREGATION: Literal["mean", "median"] = "median"
eps = 0  # was 1e-9; set to 0 to surface NA when denominator is zero
MIN_OBS_FOR_FACTOR_ANALYSIS = 30
# SNR definition:
#   - "pooled_cohens_d": (mean_post - mean_pre) / pooled_std (signed, effects can cancel)
#   - "abs_cohens_d": |mean_post - mean_pre| / pooled_std (absolute, effects don't cancel)
#   - "iqr": signal / IQR_pre (legacy)
SNR_DEFINITION: Literal["pooled_cohens_d", "abs_cohens_d", "iqr"] = "abs_cohens_d"
# Regression-only: small positive shift to avoid log(0) dropping rows in factor-importance
# models for nonnegative outcomes (e.g. rel_noise_log, change_vs_baseline_log).
# This MUST NOT be used in the estimands/ratios themselves.
FACTOR_IMPORTANCE_LOG_SHIFT = 0

# LaTeX table positioning string (e.g., "H", "htbp", "!ht")
LATEX_TABLE_POSITION = "H"

# Header order for noise tables (column hierarchy in wide format)
NOISE_TABLE_HEADER_ORDER = ["Noise Level", "Model Complexity", "Window Size"]

# Debugging: set True to print why partial-R² observations are filtered out.
# This is intentionally noisy; keep False for normal runs.
DEBUG_R2_FILTERING = True
DEBUG_R2_MAX_EXAMPLES = 5


# --- Strict aggregation helpers (propagate NaN/inf instead of silently dropping) ---
def _strict_median(x: pd.Series) -> float:
    """Median that returns NaN if any NaN present, inf if any inf present."""
    return x.median(skipna=False)


def _strict_mean(x: pd.Series) -> float:
    """Mean that returns NaN if any NaN present, inf if any inf present."""
    return x.mean(skipna=False)


def _strict_std(x: pd.Series) -> float:
    """Std that returns NaN if any NaN present."""
    return x.std(skipna=False)


def _strict_iqr(x: pd.Series) -> float:
    """IQR that returns NaN if any NaN present, inf if any inf present."""
    if x.isna().any():
        return np.nan
    if np.isinf(x).any():
        return np.inf
    return x.quantile(0.75) - x.quantile(0.25)


# --- Tolerant aggregation helpers (skip NaN/inf, report filtering) ---
def _tolerant_median(x: pd.Series) -> float:
    """Median that skips NaN and inf values."""
    clean = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return np.nan
    return clean.median()


def _tolerant_mean(x: pd.Series) -> float:
    """Mean that skips NaN and inf values."""
    clean = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return np.nan
    return clean.mean()


def _count_nan_inf(x: pd.Series) -> int:
    """Count NaN and inf values in a series."""
    return x.isna().sum() + np.isinf(x).sum()


# Factor column names used across the analysis (title case, no underscores).
MODEL_COMPLEXITY_COL = "Model Complexity"
NOISE_LEVEL_COL = "Noise Level"
CHANGE_MAGNITUDE_COL = "Change Magnitude"
EDIT_OPERATIONS_COL = "Edit Operations"
WINDOW_SIZE_COL = "Window Size"
SEED_COL = "Seed"
LOG_ID_COL = "Log ID"
SPLIT_NAME_COL = "Split Name"
LOG_NUMBER_COL = "Log Number"

# Required columns in aggregate_analysis.csv: file column name -> analysis column name.
# All keys must exist in the file; columns are renamed to values. If "Sample Size"
# is present it is validated to equal Window Size and then dropped.
AGGREGATE_ANALYSIS_COL_MAP = {
    "Metric": "Metric",
    "Mean Value": "Mean Value",
    "Median Value": "Median Value",
    "Sample CI Low": "Sample CI Low",
    "Sample CI High": "Sample CI High",
    "Sample Std": "Sample Std",
    "log_id": LOG_ID_COL,
    "split_name": SPLIT_NAME_COL,
    "window_size": WINDOW_SIZE_COL,
}
# Optional: if present, must equal Window Size; then dropped.
AGGREGATE_ANALYSIS_VALIDATE_AND_DROP = "Sample Size"

# Required columns in generation_info.csv: file column name -> analysis column name.
# All keys must exist; output DataFrame has columns = values (with value mapping applied).
GEN_INFO_COL_MAP = {
    "log_id": LOG_ID_COL,
    "Noisy_trace_prob": NOISE_LEVEL_COL,
    "Process_tree_complexity": MODEL_COMPLEXITY_COL,
    "Process_tree_evolution_proportion": CHANGE_MAGNITUDE_COL,
    "Allowed_edit_operations": EDIT_OPERATIONS_COL,
    "Log_seed": SEED_COL,
}

# Noise probability to noise level mapping
NOISE_PROB_TO_LEVEL = {0.0: "None", 0.01: "Low", 0.02: "High"}

# Complexity level mapping (from generation_info.csv)
COMPLEXITY_MAPPING = {"simple": "Simple", "middle": "Middle", "complex": "Complex"}

# Ordered numeric encoding for factor analysis (Simple < Middle < Complex; None < Low < High)
COMPLEXITY_ORD = {"Simple": 0, "Middle": 1, "Complex": 2}
NOISE_ORD = {"None": 0, "Low": 1, "High": 2}

# Change operation mapping (from generation_info.csv)
# Combined case (all operations allowed) is mapped to "mixed".
CHANGE_OPERATION_MAPPING = {
    "deletion": "deletion",
    "insertion": "insertion",
    "resequentialization": "resequentialization",
    "operator_replacement": "operator_replacement",
    "activity_replacement": "activity_replacement",
    "deletion, insertion, resequentialization, operator_replacement, activity_replacement": "mixed",
}

# Change operation sort order for wide tables
CHANGE_OPERATION_ORDER = [
    "deletion",
    "insertion",
    "activity_replacement",
    "resequentialization",
    "operator_replacement",
    "mixed",
]

# Metric order: by dimension (DIMENSIONS_ORDER) then by order in METRIC_DIMENSION_MAP
_METRIC_ORDER = [
    m for d in DIMENSIONS_ORDER for m, dim in METRIC_DIMENSION_MAP.items() if dim == d
]

# Grouping keys for noise/relative-noise tables (with and without seed for two-stage aggregation)
# Order follows NOISE_TABLE_HEADER_ORDER: Noise Level, Model Complexity, Window Size
NOISE_TABLE_GROUP_KEYS = [
    "Metric",
    NOISE_LEVEL_COL,
    MODEL_COMPLEXITY_COL,
    WINDOW_SIZE_COL,
]
NOISE_TABLE_GROUP_KEYS_WITH_SEED = [*NOISE_TABLE_GROUP_KEYS, SEED_COL]

# Factor importance LaTeX: column headers with (%) for percent display
FACTOR_IMPORTANCE_PCT_HEADERS = {
    MODEL_COMPLEXITY_COL: "Model Complexity (%)",
    NOISE_LEVEL_COL: "Noise Level (%)",
    WINDOW_SIZE_COL: "Window Size (%)",
    SEED_COL: "Seed (%)",
    "Residual": "Residual (%)",
    "R2 Full": "R2 Full (%)",
}


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
    df["Dimension"] = df[metric_col].map(METRIC_DIMENSION_MAP).fillna("Other")
    dim_order = list(DIMENSIONS_ORDER) + ["Other"]
    df["_dim_ord"] = df["Dimension"].map(
        lambda x: dim_order.index(x) if x in dim_order else len(dim_order)
    )
    metric_order = {m: i for i, m in enumerate(_METRIC_ORDER)}
    df["_met_ord"] = df[metric_col].map(
        lambda m: metric_order.get(m, len(metric_order))
    )
    df = df.sort_values(["_dim_ord", "_met_ord"]).drop(columns=["_dim_ord", "_met_ord"])
    cols = ["Dimension", metric_col] + [
        c for c in df.columns if c not in ("Dimension", metric_col)
    ]
    return df[cols]


def _collapse_within_sigma(
    df: pd.DataFrame,
    *,
    group_keys_with_sigma: list[str],
    value_cols: list[str],
    context: str = "",
) -> pd.DataFrame:
    """
    Collapse potential multiplicity within seed using WITHIN_SEED_AGGREGATION.

    Aggregate within seed first (e.g. over nuisance multiplicity), then across seeds.
    Uses tolerant aggregation that skips NaN/inf values and reports filtering stats.

    Parameters
    ----------
    df
        Input DataFrame.
    group_keys_with_sigma
        Grouping keys that MUST include SEED_COL (log_seed).
    value_cols
        Value columns to aggregate (mean or median per WITHIN_SEED_AGGREGATION).
    context
        Optional context string for NaN reporting (e.g. function name).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per group_keys_with_sigma.

    Raises
    ------
    ValueError
        If SEED_COL is not present in group_keys_with_sigma or df.
    """
    if SEED_COL not in group_keys_with_sigma:
        raise ValueError(f"group_keys_with_sigma must include {SEED_COL!r}.")
    missing_cols = [
        c for c in [*group_keys_with_sigma, *value_cols] if c not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for within-seed collapse: {missing_cols}"
        )

    # Report NaN/inf stats before aggregation
    for col in value_cols:
        n_total = len(df)
        n_nan_inf = df[col].isna().sum() + np.isinf(df[col]).sum()
        if n_nan_inf > 0:
            pct = 100 * n_nan_inf / n_total
            ctx = f" [{context}]" if context else ""
            print(
                f"  [NaN/inf]{ctx} within-seed {col}: {n_nan_inf}/{n_total} ({pct:.2f}%) filtered"
            )

    grouped = df.groupby(group_keys_with_sigma, dropna=False, sort=False)
    agg_func = _tolerant_mean if WITHIN_SEED_AGGREGATION == "mean" else _tolerant_median
    agg_map = {c: agg_func for c in value_cols}
    out = grouped.agg(agg_map).reset_index()
    return out


def _median_over_sigma(
    df: pd.DataFrame,
    *,
    group_keys_without_sigma: list[str],
    value_col: str,
    out_col: str,
    n_col: str = "N Seed",
    context: str = "",
) -> pd.DataFrame:
    """
    Aggregate across seed as the final step using ACROSS_SEED_AGGREGATION.

    Uses tolerant aggregation that skips NaN/inf values and reports filtering stats.

    Parameters
    ----------
    df
        Input DataFrame containing SEED_COL and value_col.
    group_keys_without_sigma
        Grouping keys that must NOT include SEED_COL.
    value_col
        Column to aggregate across seed (mean or median per ACROSS_SEED_AGGREGATION).
    out_col
        Name of the output column containing the aggregated value.
    n_col
        Name of the output column containing the number of seed replications.
    context
        Optional context string for NaN reporting (e.g. function name).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns group_keys_without_sigma + [out_col, n_col].
    """
    if SEED_COL in group_keys_without_sigma:
        raise ValueError(f"group_keys_without_sigma must not include {SEED_COL!r}.")
    required = [*group_keys_without_sigma, SEED_COL, value_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for across-seed aggregation: {missing_cols}"
        )

    # Report NaN/inf stats before aggregation
    n_total = len(df)
    n_nan_inf = df[value_col].isna().sum() + np.isinf(df[value_col]).sum()
    if n_nan_inf > 0:
        pct = 100 * n_nan_inf / n_total
        ctx = f" [{context}]" if context else ""
        print(
            f"  [NaN/inf]{ctx} across-seed {value_col}: {n_nan_inf}/{n_total} ({pct:.2f}%) filtered"
        )

    grouped = df.groupby(group_keys_without_sigma, dropna=False, sort=False)
    agg_func = _tolerant_mean if ACROSS_SEED_AGGREGATION == "mean" else _tolerant_median
    out = grouped.agg(
        **{out_col: (value_col, agg_func), n_col: (SEED_COL, "nunique")}
    ).reset_index()
    return out


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
        [dimension, df.index], names=["Dimension", "Metric"]
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

    Uses AGGREGATE_ANALYSIS_COL_MAP to require and rename columns. If
    AGGREGATE_ANALYSIS_VALIDATE_AND_DROP is present, validates equality with
    window_size and drops it.

    Parameters
    ----------
    path_agg
        Path to aggregate_analysis.csv.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with analysis column names.

    Raises
    ------
    ValueError
        If required columns are missing or Sample Size != window_size.
    """
    df = pd.read_csv(path_agg, low_memory=False)
    missing = [k for k in AGGREGATE_ANALYSIS_COL_MAP if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path_agg}: {missing}")
    if AGGREGATE_ANALYSIS_VALIDATE_AND_DROP in df.columns:
        if not (
            df[AGGREGATE_ANALYSIS_VALIDATE_AND_DROP] == df["window_size"]
        ).all():  # file cols before rename
            raise ValueError(
                f"In {path_agg}: '{AGGREGATE_ANALYSIS_VALIDATE_AND_DROP}' must equal 'window_size'."
            )
        df = df.drop(columns=[AGGREGATE_ANALYSIS_VALIDATE_AND_DROP])
    df = df.rename(columns=AGGREGATE_ANALYSIS_COL_MAP)
    return df


def filter_to_stable_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to rows with stable (pre-drift) split names.

    Parameters
    ----------
    df
        Input DataFrame with SPLIT_NAME_COL.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only stable regime rows.
    """
    return df[df[SPLIT_NAME_COL] == "pre_drift"].copy()


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
    Load generation info CSV and return standardized analysis columns.

    Uses GEN_INFO_COL_MAP: all keys must exist in the file; output has columns
    = values with value mappings applied. Adds log_number from log_id.

    Parameters
    ----------
    path_gen_info
        Path to generation_info.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with log_id, log_number, complexity, noise, evolution_proportion,
        change_operation, log_seed.

    Raises
    ------
    ValueError
        If required columns are missing or any value mapping fails.
    """
    df = pd.read_csv(path_gen_info, sep=";")
    missing = [k for k in GEN_INFO_COL_MAP if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path_gen_info}: {missing}")

    out = pd.DataFrame()
    out[LOG_NUMBER_COL] = df["log_id"].apply(extract_log_number).values
    out[NOISE_LEVEL_COL] = df["Noisy_trace_prob"].map(NOISE_PROB_TO_LEVEL).values
    out[MODEL_COMPLEXITY_COL] = (
        df["Process_tree_complexity"].map(COMPLEXITY_MAPPING).values
    )
    out[CHANGE_MAGNITUDE_COL] = (
        df["Process_tree_evolution_proportion"].astype(float).values
    )
    out[EDIT_OPERATIONS_COL] = (
        df["Allowed_edit_operations"].map(CHANGE_OPERATION_MAPPING).values
    )
    out[SEED_COL] = df["Log_seed"].values

    if out[NOISE_LEVEL_COL].isna().any():
        unmapped = df.loc[out[NOISE_LEVEL_COL].isna(), "Noisy_trace_prob"].unique()
        raise ValueError(
            f"Unmapped noise probabilities in {path_gen_info}: {unmapped}. "
            f"Expected: {list(NOISE_PROB_TO_LEVEL.keys())}"
        )
    if out[MODEL_COMPLEXITY_COL].isna().any():
        unmapped = df.loc[
            out[MODEL_COMPLEXITY_COL].isna(), "Process_tree_complexity"
        ].unique()
        raise ValueError(
            f"Unmapped complexity values in {path_gen_info}: {unmapped}. "
            f"Expected: {list(COMPLEXITY_MAPPING.keys())}"
        )
    if out[EDIT_OPERATIONS_COL].isna().any():
        unmapped = df.loc[
            out[EDIT_OPERATIONS_COL].isna(), "Allowed_edit_operations"
        ].unique()
        raise ValueError(
            f"Unmapped change operations in {path_gen_info}: {unmapped}. "
            f"Expected: {list(CHANGE_OPERATION_MAPPING.keys())}"
        )
    return out


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
    df["Split Name Normalized"] = df[SPLIT_NAME_COL].str.lower().str.replace("_", "-")

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

    df[SPLIT_NAME_COL] = df["Split Name Normalized"].apply(normalize_split)
    df = df.drop(columns=["Split Name Normalized"])

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
    df[LOG_NUMBER_COL] = df[LOG_ID_COL].apply(extract_log_number)

    # Merge with generation info on log_number
    df_enriched = df.merge(gen_info, on=LOG_NUMBER_COL, how="left")

    # Drop the temporary log_number column
    df_enriched = df_enriched.drop(columns=[LOG_NUMBER_COL])

    # Check for unmapped evolution_proportion and change_operation
    if CHANGE_MAGNITUDE_COL in df_enriched.columns:
        if df_enriched[CHANGE_MAGNITUDE_COL].isna().any():
            unmapped_log_ids = df_enriched[df_enriched[CHANGE_MAGNITUDE_COL].isna()][
                LOG_ID_COL
            ].unique()
            raise ValueError(
                f"Cannot map evolution_proportion for {len(unmapped_log_ids)} log_ids: "
                f"{unmapped_log_ids[:5].tolist() if len(unmapped_log_ids) > 5 else unmapped_log_ids.tolist()}"
            )

    if EDIT_OPERATIONS_COL in df_enriched.columns:
        if df_enriched[EDIT_OPERATIONS_COL].isna().any():
            unmapped_log_ids = df_enriched[df_enriched[EDIT_OPERATIONS_COL].isna()][
                LOG_ID_COL
            ].unique()
            raise ValueError(
                f"Cannot map change_operation for {len(unmapped_log_ids)} log_ids: "
                f"{unmapped_log_ids[:5].tolist() if len(unmapped_log_ids) > 5 else unmapped_log_ids.tolist()}"
            )

    # Check for unmapped log_ids
    unmapped = df_enriched[
        df_enriched[MODEL_COMPLEXITY_COL].isna() | df_enriched[NOISE_LEVEL_COL].isna()
    ]
    if not unmapped.empty:
        unmapped_log_ids = unmapped[LOG_ID_COL].unique()
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
        df_enriched[MODEL_COMPLEXITY_COL].notna() & df_enriched[NOISE_LEVEL_COL].notna()
    ].copy()

    if len(df_mapped) == 0:
        raise ValueError(
            "No log_ids could be mapped to complexity/noise. "
            "Cannot proceed with analysis."
        )

    print(
        f"  Proceeding with {len(df_mapped)} rows from "
        f"{df_mapped[LOG_ID_COL].nunique()} unique log_ids."
    )

    return df_mapped


def remove_pre_drift_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove pre-drift duplicates across evolution-related design factors.

    In this dataset family, pre-drift segments are generated *before* any
    evolution/edit operations are applied, so pre-drift metric outputs should
    not depend on `evolution_proportion` or `change_operation`. However, the
    input tables can still contain multiple rows per seed-cell because multiple
    logs share the same replication seed (`log_seed`) while varying in those
    nuisance factors.

    This function collapses duplicates per seed-cell by taking medians of numeric
    columns, and drops `evolution_proportion` and `change_operation` to prevent
    accidental pooling/weighting across nuisance multiplicity in downstream
    analyses.

    Parameters
    ----------
    df
        Pre-drift DataFrame enriched with experimental factors, including
        `Metric`, `complexity`, `noise`, `window_size`, and SEED_COL.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with one row per (Metric, complexity, noise,
        window_size, seed). Keeps a representative `log_id` and a `log_ids` list
        for traceability.
    """
    if SEED_COL not in df.columns:
        raise ValueError(
            f"remove_pre_drift_duplicates requires {SEED_COL!r} to define seed-cells."
        )
    required = [*NOISE_TABLE_GROUP_KEYS_WITH_SEED, LOG_ID_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for pre-drift deduplication: {missing}"
        )

    # Ensure pre-drift only if split_name exists.
    if SPLIT_NAME_COL in df.columns:
        allowed_pre = {"pre_drift", "pre-drift", "pre"}
        split_vals = (
            df[SPLIT_NAME_COL].dropna().astype(str).str.lower().unique().tolist()
        )
        bad = [s for s in split_vals if s not in allowed_pre]
        if bad:
            raise ValueError(
                "remove_pre_drift_duplicates received non-pre rows. "
                f"Unexpected split_name values: {bad}."
            )

    keys = NOISE_TABLE_GROUP_KEYS_WITH_SEED
    grouped = df.groupby(keys, dropna=False, sort=False)

    # Aggregate numeric columns by median, excluding nuisance factors.
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if c not in keys and c not in {CHANGE_MAGNITUDE_COL}
    ]
    agg_map: dict[str, Any] = {c: _strict_median for c in numeric_cols}
    out = grouped.agg(agg_map).reset_index()

    # Keep a representative split_name if present.
    if SPLIT_NAME_COL in df.columns:
        split_first = grouped[SPLIT_NAME_COL].first().reset_index(name=SPLIT_NAME_COL)
        out = out.merge(split_first, on=keys, how="left")

    # Keep contributing log_ids for traceability and a representative log_id.
    log_ids = (
        grouped[LOG_ID_COL]
        .apply(lambda x: sorted(set(x.astype(str).tolist())))
        .reset_index(name="Log IDs")
    )
    out = out.merge(log_ids, on=keys, how="left")
    out[LOG_ID_COL] = out["Log IDs"].apply(
        lambda ids: ids[0] if isinstance(ids, list) and ids else None
    )

    # Drop nuisance factor columns if they survived (should not, but be explicit).
    out = out.drop(
        columns=[
            c for c in (CHANGE_MAGNITUDE_COL, EDIT_OPERATIONS_COL) if c in out.columns
        ]
    )
    return out


def compute_per_log_relative_noise(
    df: pd.DataFrame, aggregation: Literal["mean", "median"] = "mean"
) -> tuple[pd.DataFrame, int]:
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
    tuple[pd.DataFrame, int]
        (df_valid, n_invalid) where df_valid has added iqr, center, and
        rel_noise_log columns, and n_invalid is the number of dropped rows.
    """
    df = df.copy()

    # Compute IQR
    df["IQR"] = df["Sample CI High"] - df["Sample CI Low"]

    # Compute center based on aggregation method
    if aggregation == "mean":
        df["Center"] = df["Mean Value"]
    elif aggregation == "median":
        if "Median Value" not in df.columns:
            raise ValueError(
                "choose_aggregation='median' requires 'Median Value' column, "
                "which is not present in the input data."
            )
        df["Center"] = df["Median Value"]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Compute relative noise
    df["Relative Noise Log"] = df["IQR"] / (df["Center"].abs() + eps)

    # Flag invalid rows
    invalid_mask = (
        df["Center"].isna()
        | df["IQR"].isna()
        | (df["IQR"] < 0)
        | (df["Relative Noise Log"] < 0)
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

    Baseline is computed within the same seed (log_seed).

    For each (Metric, complexity, window_size, seed), baseline is the median of
    center over rows where noise == "None". Then change_vs_baseline_log =
    |center - baseline| / (|baseline| + eps). For noise == "None", set to 0.

    Parameters
    ----------
    df
        DataFrame with Metric, complexity, noise, window_size, center, and SEED_COL.

    Returns
    -------
    pd.DataFrame
        Copy of df with added column change_vs_baseline_log.
    """
    df = df.copy()
    if SEED_COL not in df.columns:
        raise ValueError(
            f"add_change_vs_baseline_log requires {SEED_COL!r} for baseline within seed."
        )
    baseline = (
        df[df[NOISE_LEVEL_COL] == "None"]
        .groupby(["Metric", MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL, SEED_COL])["Center"]
        .agg(_strict_median)
        .reset_index()
        .rename(columns={"Center": "Baseline"})
    )
    df = df.merge(
        baseline,
        on=["Metric", MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL, SEED_COL],
        how="left",
    )
    df["Change Vs Baseline Log"] = np.where(
        df[NOISE_LEVEL_COL] == "None",
        0.0,
        (df["Center"] - df["Baseline"]).abs() / (df["Baseline"].abs() + eps),
    )
    df = df.drop(columns=["Baseline"])
    return df


def aggregate_relative_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate relative noise across logs by metric, complexity, noise, window_size.

    Parameters
    ----------
    df
        Pre-drift DataFrame with rel_noise_log, Metric, complexity, noise, window_size,
        and SEED_COL.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with aggregated rel_noise and n_sigma.
        rel_noise is the median over seed of per-seed rel_noise_log values.
    """
    if SEED_COL not in df.columns:
        raise ValueError(
            f"aggregate_relative_noise requires {SEED_COL!r} for seed-last aggregation."
        )

    # 1) Collapse within seed to one rel_noise_log per (Metric, complexity, noise, window_size, seed)
    rel_noise_sigma = _collapse_within_sigma(
        df,
        group_keys_with_sigma=NOISE_TABLE_GROUP_KEYS_WITH_SEED,
        value_cols=["Relative Noise Log"],
        context="rel_noise",
    )

    # 2) Aggregate across seed last
    agg_result = _median_over_sigma(
        rel_noise_sigma,
        group_keys_without_sigma=NOISE_TABLE_GROUP_KEYS,
        value_col="Relative Noise Log",
        out_col="Relative Noise",
        n_col="N Seed",
        context="rel_noise",
    )

    # Optional dispersion across seed (IQR over seed-level values)
    grouped = rel_noise_sigma.groupby(NOISE_TABLE_GROUP_KEYS, dropna=False, sort=False)
    iqr_sigma = (
        grouped["Relative Noise Log"]
        .apply(_strict_iqr)
        .reset_index(name="Relative Noise IQR Seed")
    )
    agg_result = agg_result.merge(iqr_sigma, on=NOISE_TABLE_GROUP_KEYS, how="left")

    agg_result = _add_dimension_and_sort_long(agg_result)
    return agg_result


def partial_r2(full_model: Any, reduced_model: Any) -> float:
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
    return float(np.clip(partial_r2_val, 0.0, 1.0))


def compute_partial_r2_components(
    df_m: pd.DataFrame,
    *,
    outcome_col: str = "Relative Noise Log",
    outcome_log_shift: float = 0.0,
) -> dict:
    """
    Compute partial R² for each factor (complexity, noise, window_size, log_seed).
    Main effects only; no interactions (factors are never combined).

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
    required = [
        outcome_col,
        WINDOW_SIZE_COL,
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        SEED_COL,
    ]
    df_clean = df_m[[c for c in required if c in df_m.columns]].copy()
    if outcome_col not in df_clean.columns or SEED_COL not in df_clean.columns:
        return {
            MODEL_COMPLEXITY_COL: 0.0,
            NOISE_LEVEL_COL: 0.0,
            WINDOW_SIZE_COL: 0.0,
            SEED_COL: 0.0,
            "Residual": 1.0,
            "N Obs": 0,
            "R2 Full": 0.0,
        }

    # Ordered numeric encoding for complexity and noise (simple<mid<complex; 0<low<high)
    df_clean["Complexity Ord"] = df_clean[MODEL_COMPLEXITY_COL].map(COMPLEXITY_ORD)
    df_clean["Noise Ord"] = df_clean[NOISE_LEVEL_COL].map(NOISE_ORD)
    df_clean = df_clean[
        df_clean["Complexity Ord"].notna() & df_clean["Noise Ord"].notna()
    ].copy()

    # Create log-transformed outcome
    df_clean["y"] = np.log(df_clean[outcome_col] + outcome_log_shift)

    # Drop rows with non-finite y
    df_clean = df_clean[np.isfinite(df_clean["y"])].copy()

    # Snake-case aliases for OLS formula (formula uses identifiers, not title-case col names)
    df_clean["window_size"] = df_clean[WINDOW_SIZE_COL]
    df_clean["complexity_ord"] = df_clean["Complexity Ord"]
    df_clean["noise_ord"] = df_clean["Noise Ord"]
    df_clean["log_seed"] = df_clean[SEED_COL]

    n_obs = len(df_clean)

    if n_obs < MIN_OBS_FOR_FACTOR_ANALYSIS:
        # Return zeros if insufficient data (residual as proportion, not percentage)
        return {
            MODEL_COMPLEXITY_COL: 0.0,
            NOISE_LEVEL_COL: 0.0,
            WINDOW_SIZE_COL: 0.0,
            SEED_COL: 0.0,
            "Residual": 1.0,  # 100% as proportion
            "N Obs": n_obs,
            "R2 Full": 0.0,
        }

    # Full model: main effects only (no interactions)
    try:
        model_full = smf.ols(
            "y ~ np.log(window_size) + complexity_ord + noise_ord + C(log_seed)",
            data=df_clean,
        ).fit()
        r2_full = model_full.rsquared
    except Exception as e:
        print(f"Warning: Failed to fit full model: {e}")
        return {
            MODEL_COMPLEXITY_COL: 0.0,
            NOISE_LEVEL_COL: 0.0,
            WINDOW_SIZE_COL: 0.0,
            SEED_COL: 0.0,
            "Residual": 1.0,  # 100% as proportion
            "N Obs": n_obs,
            "R2 Full": 0.0,
        }

    # Reduced models: drop exactly one factor at a time (no combined terms)
    partial_r2_vals = {}

    try:
        model_no_complexity = smf.ols(
            "y ~ np.log(window_size) + noise_ord + C(log_seed)",
            data=df_clean,
        ).fit()
        partial_r2_vals[MODEL_COMPLEXITY_COL] = partial_r2(
            model_full, model_no_complexity
        )
    except Exception:
        partial_r2_vals[MODEL_COMPLEXITY_COL] = 0.0

    try:
        model_no_noise = smf.ols(
            "y ~ np.log(window_size) + complexity_ord + C(log_seed)",
            data=df_clean,
        ).fit()
        partial_r2_vals[NOISE_LEVEL_COL] = partial_r2(model_full, model_no_noise)
    except Exception:
        partial_r2_vals[NOISE_LEVEL_COL] = 0.0

    try:
        model_no_window = smf.ols(
            "y ~ complexity_ord + noise_ord + C(log_seed)",
            data=df_clean,
        ).fit()
        partial_r2_vals[WINDOW_SIZE_COL] = partial_r2(model_full, model_no_window)
    except Exception:
        partial_r2_vals[WINDOW_SIZE_COL] = 0.0

    try:
        model_no_log_seed = smf.ols(
            "y ~ np.log(window_size) + complexity_ord + noise_ord",
            data=df_clean,
        ).fit()
        partial_r2_vals[SEED_COL] = partial_r2(model_full, model_no_log_seed)
    except Exception:
        partial_r2_vals[SEED_COL] = 0.0

    # Compute residual share (unexplained variance)
    residual_raw = max(0.0, 1.0 - r2_full)

    # Normalize partial R² values and residual to sum to 100%
    total_all = (
        partial_r2_vals[MODEL_COMPLEXITY_COL]
        + partial_r2_vals[NOISE_LEVEL_COL]
        + partial_r2_vals[WINDOW_SIZE_COL]
        + partial_r2_vals[SEED_COL]
        + residual_raw
    )

    if total_all > 0:
        normalized = {
            MODEL_COMPLEXITY_COL: partial_r2_vals[MODEL_COMPLEXITY_COL] / total_all,
            NOISE_LEVEL_COL: partial_r2_vals[NOISE_LEVEL_COL] / total_all,
            WINDOW_SIZE_COL: partial_r2_vals[WINDOW_SIZE_COL] / total_all,
            SEED_COL: partial_r2_vals[SEED_COL] / total_all,
        }
        residual_share = residual_raw / total_all
    else:
        normalized = {
            MODEL_COMPLEXITY_COL: 0.0,
            NOISE_LEVEL_COL: 0.0,
            WINDOW_SIZE_COL: 0.0,
            SEED_COL: 0.0,
        }
        residual_share = 1.0

    return {
        MODEL_COMPLEXITY_COL: normalized[MODEL_COMPLEXITY_COL],
        NOISE_LEVEL_COL: normalized[NOISE_LEVEL_COL],
        WINDOW_SIZE_COL: normalized[WINDOW_SIZE_COL],
        SEED_COL: normalized[SEED_COL],
        "Residual": residual_share,
        "N Obs": n_obs,
        "R2 Full": r2_full,
    }


def run_noise_factor_importance(
    df_relnoise_log: pd.DataFrame,
    *,
    outcome_col: str = "Relative Noise Log",
    outcome_log_shift: float = 0.0,
    debug_filtering: bool = False,
    debug_max_examples: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run factor importance analysis for an outcome across all metrics.

    To avoid overweighting seed with multiple rows (e.g., nuisance multiplicity),
    this function collapses within seed first so the regression sees at most
    one observation per (Metric, complexity, noise, window_size, seed).

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

    def _debug_print_filtered_out(
        *,
        metric: str,
        df_metric_raw: pd.DataFrame,
        df_metric_sigma: pd.DataFrame,
        outcome_col: str,
        outcome_log_shift: float,
        max_examples: int,
    ) -> None:
        """
        Print which seed-cells are filtered out and why (mypy/OLS preprocessing).

        Notes
        -----
        This debug helper is intentionally verbose and only runs when requested.
        It reports counts and a small sample of affected keys. Keys are at the
        seed-cell level: (window_size, complexity, noise, log_seed).
        """
        keys = [WINDOW_SIZE_COL, MODEL_COMPLEXITY_COL, NOISE_LEVEL_COL, SEED_COL]
        # Attach contributing log_ids per seed-cell if available in the raw table.
        sigma_log_ids = None
        if LOG_ID_COL in df_metric_raw.columns:
            sigma_log_ids = (
                df_metric_raw.groupby(keys, dropna=False, sort=False)[LOG_ID_COL]
                .apply(lambda x: sorted(set(x.astype(str).tolist())))
                .reset_index(name="Log IDs")
            )

        d = df_metric_sigma.copy()
        if sigma_log_ids is not None:
            d = d.merge(sigma_log_ids, on=keys, how="left")

        # Map ordinals and compute y exactly like compute_partial_r2_components.
        d["Complexity Ord"] = d[MODEL_COMPLEXITY_COL].map(COMPLEXITY_ORD)
        d["Noise Ord"] = d[NOISE_LEVEL_COL].map(NOISE_ORD)

        reasons: dict[str, pd.Series] = {}
        reasons["unmapped_complexity_or_noise"] = (
            d["Complexity Ord"].isna() | d["Noise Ord"].isna()
        )
        reasons["outcome_nan"] = d[outcome_col].isna()
        reasons["outcome_nonfinite"] = ~np.isfinite(d[outcome_col].astype(float))
        # log() domain issues (after shift)
        log_arg = d[outcome_col].astype(float) + float(outcome_log_shift)
        reasons["log_arg_nonpositive"] = ~(log_arg > 0)  # includes NaN, 0, negative
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(log_arg)
        reasons["y_nonfinite"] = ~np.isfinite(y)

        # What ends up excluded from the regression input?
        excluded = reasons["unmapped_complexity_or_noise"] | reasons["y_nonfinite"]
        n_total = len(d)
        n_excl = int(excluded.sum())
        if n_excl == 0:
            return

        print("\n" + "-" * 60)
        print(f"R2 FILTER DEBUG — Metric: {metric}")
        print(f"  seed-cells before cleaning: {n_total}")
        print(f"  seed-cells excluded from model: {n_excl}")
        for reason, mask in reasons.items():
            n = int(mask.sum())
            if n == 0:
                continue
            print(f"  - {reason}: {n}")
            cols = keys + [outcome_col]
            if "Log IDs" in d.columns:
                cols = cols + ["Log IDs"]
            ex = d.loc[mask, cols].copy()
            if "Log IDs" in ex.columns:
                ex["Log IDs Preview"] = ex["Log IDs"].apply(
                    lambda ids: (
                        ids[:5]
                        + (["..."] if isinstance(ids, list) and len(ids) > 5 else [])
                        if isinstance(ids, list)
                        else ids
                    )
                )
                ex = ex.drop(columns=["Log IDs"], errors="ignore")
                ex = ex.assign(log_ids=ex["Log IDs Preview"]).drop(
                    columns=["Log IDs Preview"]
                )
            print(ex.head(max_examples).to_string(index=False))
        print("-" * 60)

    results = []

    # Collapse within seed to one observation per (Metric, complexity, noise, window_size, seed).
    df_in = df_relnoise_log
    if SEED_COL in df_in.columns:
        df_in = _collapse_within_sigma(
            df_in,
            group_keys_with_sigma=[
                "Metric",
                WINDOW_SIZE_COL,
                MODEL_COMPLEXITY_COL,
                NOISE_LEVEL_COL,
                SEED_COL,
            ],
            value_cols=[outcome_col],
        )

    required_cols = [
        outcome_col,
        WINDOW_SIZE_COL,
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        SEED_COL,
    ]
    for metric in df_in["Metric"].unique():
        df_metric = df_in[df_in["Metric"] == metric].copy()
        df_metric_raw = df_relnoise_log[df_relnoise_log["Metric"] == metric].copy()

        missing_cols = [col for col in required_cols if col not in df_metric.columns]
        if missing_cols:
            print(f"Warning: Metric {metric} missing columns {missing_cols}, skipping.")
            continue

        components = compute_partial_r2_components(
            df_metric,
            outcome_col=outcome_col,
            outcome_log_shift=outcome_log_shift,
        )

        if debug_filtering:
            _debug_print_filtered_out(
                metric=metric,
                df_metric_raw=df_metric_raw,
                df_metric_sigma=df_metric,
                outcome_col=outcome_col,
                outcome_log_shift=outcome_log_shift,
                max_examples=debug_max_examples,
            )

        results.append(
            {
                "Metric": metric,
                MODEL_COMPLEXITY_COL: components[MODEL_COMPLEXITY_COL] * 100,
                NOISE_LEVEL_COL: components[NOISE_LEVEL_COL] * 100,
                WINDOW_SIZE_COL: components[WINDOW_SIZE_COL] * 100,
                SEED_COL: components[SEED_COL] * 100,
                "Residual": components["Residual"] * 100,
                "N Obs": components["N Obs"],
                "R2 Full": components["R2 Full"] * 100,
            }
        )

    # Create wide format
    df_wide = pd.DataFrame(results)
    df_wide = df_wide.set_index("Metric")
    df_wide = _add_dimension_and_sort_wide(df_wide)

    # Create long format by melting
    df_long = df_wide.reset_index().melt(
        id_vars=["Dimension", "Metric", "N Obs", "R2 Full"],
        value_vars=[
            MODEL_COMPLEXITY_COL,
            NOISE_LEVEL_COL,
            WINDOW_SIZE_COL,
            SEED_COL,
            "Residual",
        ],
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
        [MODEL_COMPLEXITY_COL, NOISE_LEVEL_COL, WINDOW_SIZE_COL, SEED_COL, "Residual"]
    ].mean()
    for factor, mean_pct in factor_means.items():
        print(f"   {factor:15s}: {mean_pct:6.2f}%")

    # Top factor per metric
    print("\n2. Dominant factor per metric (highest share):")
    for dim, metric in df_wide.index:
        row = df_wide.loc[(dim, metric)]
        factors = [
            MODEL_COMPLEXITY_COL,
            NOISE_LEVEL_COL,
            WINDOW_SIZE_COL,
            SEED_COL,
            "Residual",
        ]
        top_factor = max(factors, key=lambda f: row[f])
        top_value = row[top_factor]
        print(f"   {metric:40s}: {top_factor:15s} ({top_value:6.2f}%)")

    # Metrics with low n_obs
    low_n = df_wide[df_wide["N Obs"] < MIN_OBS_FOR_FACTOR_ANALYSIS]
    if len(low_n) > 0:
        print(
            f"\n3. Warning: {len(low_n)} metrics have n_obs < {MIN_OBS_FOR_FACTOR_ANALYSIS}:"
        )
        for dim, metric in low_n.index:
            print(f"   {metric}: N Obs = {low_n.loc[(dim, metric), 'N Obs']}")

    # Sanity checks
    print("\n4. Sanity checks:")
    window_mean = df_wide[WINDOW_SIZE_COL].mean()
    if window_mean < 20:
        print(
            f"   [!] Warning: Average window_size importance ({window_mean:.2f}%) is lower than expected"
        )
    else:
        print(
            f"   [OK] Average window_size importance: {window_mean:.2f}% (expected to be high)"
        )

    noise_mean = df_wide[NOISE_LEVEL_COL].mean()
    print(f"   [OK] Average noise importance: {noise_mean:.2f}%")

    print("\n" + "=" * 60)


def _pivot_noise_wide(
    df_long: pd.DataFrame,
    value_column: str,
    *,
    drop_no_noise_columns: bool = False,
) -> pd.DataFrame:
    """
    Pivot long noise table to wide (dimension, Metric) x (noise, complexity, window_size).

    Column hierarchy follows NOISE_TABLE_HEADER_ORDER: Noise Level, Model Complexity, Window Size.

    Parameters
    ----------
    df_long
        Long-format DataFrame with dimension, Metric, complexity, noise, window_size,
        and one value column (e.g. rel_noise or value).
    value_column
        Name of the column to use as cell values.
    drop_no_noise_columns
        If True, drop columns where noise level is "None".

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with MultiIndex columns (noise, complexity, window_size).
    """
    # Column order follows NOISE_TABLE_HEADER_ORDER: Noise Level, Model Complexity, Window Size
    df_wide = df_long.pivot_table(
        index=["Dimension", "Metric"],
        columns=[NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
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
        # Noise level is now at index 0 (first level)
        noise_level = df_wide.columns.get_level_values(0)
        cols_keep = [i for i, n in enumerate(noise_level) if n != "None"]
        df_wide = df_wide.iloc[:, cols_keep]

    complexity_order = ["Simple", "Middle", "Complex"]
    noise_order = ["None", "Low", "High"]
    if isinstance(df_wide.columns, pd.MultiIndex):
        noise_level = list(df_wide.columns.get_level_values(0).unique())
        complexity_level = list(df_wide.columns.get_level_values(1).unique())
        window_sizes = sorted(df_wide.columns.get_level_values(2).unique())
        noise_sorted = [n for n in noise_order if n in noise_level]
        noise_sorted.extend([n for n in noise_level if n not in noise_order])
        complexity_sorted = [c for c in complexity_order if c in complexity_level]
        complexity_sorted.extend(
            [c for c in complexity_level if c not in complexity_order]
        )
        sorted_columns = pd.MultiIndex.from_product(
            [noise_sorted, complexity_sorted, window_sizes],
            names=[NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
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
        Pre-drift DataFrame with Metric, complexity, noise, window_size, center,
        and SEED_COL.

    Returns
    -------
    pd.DataFrame
        Long table with Metric, complexity, noise, window_size, value.
    """
    if SEED_COL not in df.columns:
        raise ValueError(
            f"_aggregate_abs_median requires {SEED_COL!r} for seed-last aggregation."
        )

    center_sigma = _collapse_within_sigma(
        df,
        group_keys_with_sigma=NOISE_TABLE_GROUP_KEYS_WITH_SEED,
        value_cols=["Center"],
    )
    agg = _median_over_sigma(
        center_sigma,
        group_keys_without_sigma=NOISE_TABLE_GROUP_KEYS,
        value_col="Center",
        out_col="Value",
        n_col="N Seed",
    )
    return agg


def _aggregate_relci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate relative CI = (CI_high - CI_low) / center per cell; median across logs.

    Parameters
    ----------
    df
        Pre-drift DataFrame with Metric, complexity, noise, window_size, center,
        Sample CI Low/High, and SEED_COL.

    Returns
    -------
    pd.DataFrame
        Long table with Metric, complexity, noise, window_size, value.
        value is the median over seed of per-seed relative CI values.
    """
    d = df.copy()
    if SEED_COL not in d.columns:
        raise ValueError(
            f"_aggregate_relci requires {SEED_COL!r} for seed-last aggregation."
        )

    # Division by zero produces inf when center is 0; these are filtered in aggregation
    d["Rel CI Row"] = (d["Sample CI High"] - d["Sample CI Low"]) / (
        d["Center"].abs() + eps
    )

    # 1) Collapse within seed to a single rel_ci per (Metric, complexity, noise, window_size, seed)
    rel_ci_sigma = _collapse_within_sigma(
        d,
        group_keys_with_sigma=NOISE_TABLE_GROUP_KEYS_WITH_SEED,
        value_cols=["Rel CI Row"],
        context="relci",
    )

    # 2) Aggregate across seed last
    agg = _median_over_sigma(
        rel_ci_sigma,
        group_keys_without_sigma=NOISE_TABLE_GROUP_KEYS,
        value_col="Rel CI Row",
        out_col="Value",
        n_col="N Seed",
        context="relci",
    )
    return agg


def _aggregate_change_due_to_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustness = |median_current_noise - median_no_noise| / |median_no_noise|; only noise != "None".

    Parameters
    ----------
    df
        Pre-drift DataFrame with Metric, complexity, noise, window_size, center, and SEED_COL.

    Returns
    -------
    pd.DataFrame
        Long table with Metric, complexity, noise, window_size, value (only noise != "None").
        value is the median over seed of per-seed robustness deviations.
    """
    if SEED_COL not in df.columns:
        raise ValueError(
            f"_aggregate_change_due_to_noise requires {SEED_COL!r} for seed-last aggregation."
        )

    # 1) Within-seed medians of center for each (Metric, complexity, noise, window_size, seed)
    c_pre_sigma = _collapse_within_sigma(
        df,
        group_keys_with_sigma=NOISE_TABLE_GROUP_KEYS_WITH_SEED,
        value_cols=["Center"],
        context="robustness_to_noise",
    )

    # 2) Within-seed noise=None baseline for each (Metric, complexity, window_size, seed)
    baseline = c_pre_sigma[c_pre_sigma[NOISE_LEVEL_COL] == "None"][
        ["Metric", MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL, SEED_COL, "Center"]
    ].rename(columns={"Center": "Center No Noise"})

    merged = c_pre_sigma.merge(
        baseline,
        on=["Metric", MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL, SEED_COL],
        how="inner",
    )

    # 3) Per-seed robustness deviation (exclude noise=None)
    # Division by zero produces inf when baseline is 0; these are filtered in aggregation
    merged["R Seed"] = (merged["Center"] - merged["Center No Noise"]).abs() / (
        merged["Center No Noise"].abs() + eps
    )
    merged = merged[merged[NOISE_LEVEL_COL] != "None"].copy()

    # 4) Aggregate across seed last
    out = _median_over_sigma(
        merged,
        group_keys_without_sigma=NOISE_TABLE_GROUP_KEYS,
        value_col="R Seed",
        out_col="Value",
        n_col="N Seed",
        context="robustness_to_noise",
    )
    return out


def _aggregate_std_pre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate standard deviation per (Metric, complexity, noise, window_size).

    Parameters
    ----------
    df
        Pre-drift DataFrame with Metric, complexity, noise, window_size, Sample Std,
        and SEED_COL.

    Returns
    -------
    pd.DataFrame
        Long table with Metric, complexity, noise, window_size, value.
        value is the median over seed of per-seed Sample Std values.
    """
    d = df.copy()
    if SEED_COL not in d.columns:
        raise ValueError(
            f"_aggregate_std_pre requires {SEED_COL!r} for seed-last aggregation."
        )
    if "Sample Std" not in d.columns:
        raise ValueError("_aggregate_std_pre requires 'Sample Std' column.")

    # 1) Collapse within seed to a single std per (Metric, complexity, noise, window_size, seed)
    std_sigma = _collapse_within_sigma(
        d,
        group_keys_with_sigma=NOISE_TABLE_GROUP_KEYS_WITH_SEED,
        value_cols=["Sample Std"],
    )

    # 2) Aggregate across seed last
    agg = _median_over_sigma(
        std_sigma,
        group_keys_without_sigma=NOISE_TABLE_GROUP_KEYS,
        value_col="Sample Std",
        out_col="Value",
        n_col="N Seed",
    )
    return agg


def _aggregate_cv_pre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate coefficient of variation (CV = std / |mean|) per cell; median across logs.

    Parameters
    ----------
    df
        Pre-drift DataFrame with Metric, complexity, noise, window_size, Center,
        Sample Std, and SEED_COL.

    Returns
    -------
    pd.DataFrame
        Long table with Metric, complexity, noise, window_size, value.
        value is the median over seed of per-seed CV values.
    """
    d = df.copy()
    if SEED_COL not in d.columns:
        raise ValueError(
            f"_aggregate_cv_pre requires {SEED_COL!r} for seed-last aggregation."
        )
    if "Sample Std" not in d.columns:
        raise ValueError("_aggregate_cv_pre requires 'Sample Std' column.")
    if "Center" not in d.columns:
        raise ValueError("_aggregate_cv_pre requires 'Center' column.")

    # Compute CV = std / |mean|; division by zero produces inf when center is 0
    d["CV Row"] = d["Sample Std"] / (d["Center"].abs() + eps)

    # 1) Collapse within seed to a single CV per (Metric, complexity, noise, window_size, seed)
    cv_sigma = _collapse_within_sigma(
        d,
        group_keys_with_sigma=NOISE_TABLE_GROUP_KEYS_WITH_SEED,
        value_cols=["CV Row"],
        context="cv_pre",
    )

    # 2) Aggregate across seed last
    agg = _median_over_sigma(
        cv_sigma,
        group_keys_without_sigma=NOISE_TABLE_GROUP_KEYS,
        value_col="CV Row",
        out_col="Value",
        n_col="N Seed",
        context="cv_pre",
    )
    return agg


def create_wide_format_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Create wide-format table with 3-level column hierarchy (relative noise).

    Column hierarchy follows NOISE_TABLE_HEADER_ORDER: Noise Level, Model Complexity, Window Size.

    Parameters
    ----------
    df_long
        Long-format DataFrame with dimension, Metric, complexity, noise, window_size, rel_noise.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with MultiIndex columns (noise, complexity, window_size).
    """
    return _pivot_noise_wide(df_long, "Relative Noise", drop_no_noise_columns=False)


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
        for complexity in metric_df[MODEL_COMPLEXITY_COL].unique():
            for noise in metric_df[NOISE_LEVEL_COL].unique():
                subset = metric_df[
                    (metric_df[MODEL_COMPLEXITY_COL] == complexity)
                    & (metric_df[NOISE_LEVEL_COL] == noise)
                ].sort_values(WINDOW_SIZE_COL)

                if len(subset) > 1:
                    rel_noise_values = subset["Relative Noise"].values
                    # Check if generally decreasing (allow some exceptions)
                    decreasing = all(
                        rel_noise_values[i] >= rel_noise_values[i + 1] * 0.9
                        for i in range(len(rel_noise_values) - 1)
                    )
                    if not decreasing:
                        print(
                            f"   [!] Warning: {metric} ({complexity}, {noise}) "
                            f"does not show clear decrease with window_size"
                        )

    # Check 2: rel_noise should generally increase with higher noise level
    print("\n2. Checking: rel_noise increases with higher noise level")
    for metric in df_long["Metric"].unique():
        metric_df = df_long[df_long["Metric"] == metric]
        for complexity in metric_df[MODEL_COMPLEXITY_COL].unique():
            for window_size in metric_df[WINDOW_SIZE_COL].unique():
                subset = metric_df[
                    (metric_df[MODEL_COMPLEXITY_COL] == complexity)
                    & (metric_df[WINDOW_SIZE_COL] == window_size)
                ]

                # Map noise levels to numeric for comparison
                noise_order_map = {"None": 0, "Low": 1, "High": 2}
                subset = subset.copy()
                subset["Noise Numeric"] = subset[NOISE_LEVEL_COL].map(noise_order_map)
                subset = subset.sort_values("Noise Numeric")

                if len(subset) > 1:
                    rel_noise_values = subset["Relative Noise"].values
                    # Check if generally increasing (allow some exceptions)
                    increasing = all(
                        rel_noise_values[i] <= rel_noise_values[i + 1] * 1.1
                        for i in range(len(rel_noise_values) - 1)
                    )
                    if not increasing:
                        print(
                            f"   [!] Warning: {metric} ({complexity}, window_size={window_size}) "
                            f"does not show clear increase with noise level"
                        )

    # Summary statistics
    print("\n3. Summary Statistics")
    print(f"   Number of invalid rows dropped: {n_invalid}")
    print(f"   Number of metrics: {df_long['Metric'].nunique()}")
    print(f"   Number of complexity levels: {df_long[MODEL_COMPLEXITY_COL].nunique()}")
    print(f"   Number of noise levels: {df_long[NOISE_LEVEL_COL].nunique()}")
    print(f"   Number of window sizes: {df_long[WINDOW_SIZE_COL].nunique()}")
    print(f"   Total cells (metric × complexity × noise × window_size): {len(df_long)}")

    n_col = "N Seed" if "N Seed" in df_long.columns else "N Logs"
    n_stats = df_long[n_col].describe()
    print(f"\n   {n_col} per cell:")
    print(f"     Min: {n_stats['min']:.0f}")
    print(f"     Median: {n_stats['50%']:.0f}")
    print(f"     Max: {n_stats['max']:.0f}")

    print("\n" + "=" * 60)


def compute_snr_per_log(
    df: pd.DataFrame,
    choose_aggregation: Literal["mean", "median"] = "mean",
    snr_definition: Literal["pooled_cohens_d", "abs_cohens_d", "iqr"] = "abs_cohens_d",
) -> pd.DataFrame:
    """
    Compute SNR per seed (log_seed) by joining pre_drift and post_drift rows.

    Parameters
    ----------
    df
        DataFrame with pre_drift and post_drift rows, enriched with design factors.
    choose_aggregation
        Aggregation method for center: "mean" uses Mean Value, "median" uses Median Value.
    snr_definition
        SNR definition to use:
        - "pooled_cohens_d": (mean_post - mean_pre) / sqrt((s_pre² + s_post²) / 2) (signed)
        - "abs_cohens_d": |mean_post - mean_pre| / sqrt((s_pre² + s_post²) / 2) (absolute)
        - "iqr": signal / IQR_pre (legacy)

    Returns
    -------
    pd.DataFrame
        Long DataFrame with SNR per seed (log_seed), including signal and snr_sigma.
    """
    center_col = "Median Value" if choose_aggregation == "median" else "Mean Value"
    if choose_aggregation == "median" and center_col not in df.columns:
        raise ValueError(
            "choose_aggregation='median' requires 'Median Value' column, "
            "which is not present in the input data."
        )

    # Filter to pre_drift and post_drift only
    df_split = df[df[SPLIT_NAME_COL].isin(["pre_drift", "post_drift"])].copy()

    # Select needed columns - include Sample Std for pooled Cohen's d
    required_cols = [
        "Metric",
        LOG_ID_COL,
        WINDOW_SIZE_COL,
        SPLIT_NAME_COL,
        center_col,
        "Sample CI Low",
        "Sample CI High",
        "Sample Std",
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        CHANGE_MAGNITUDE_COL,
        EDIT_OPERATIONS_COL,
        SEED_COL,
    ]

    missing_cols = [col for col in required_cols if col not in df_split.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for SNR computation: {missing_cols}"
        )

    df_split = df_split[required_cols].copy()

    # Create pre_drift and post_drift frames
    df_pre = df_split[df_split[SPLIT_NAME_COL] == "pre_drift"].copy()
    df_post = df_split[df_split[SPLIT_NAME_COL] == "post_drift"].copy()

    # Compute IQR_pre in df_pre (still used for legacy SNR if requested)
    df_pre["IQR Pre"] = df_pre["Sample CI High"] - df_pre["Sample CI Low"]

    # Rename columns to avoid collisions (use chosen center column)
    df_pre = df_pre.rename(
        columns={
            center_col: "Center Pre",
            "Sample CI Low": "ci_low_pre",
            "Sample CI High": "ci_high_pre",
            "Sample Std": "Std Pre",
        }
    )
    df_post = df_post.rename(
        columns={
            center_col: "Center Post",
            "Sample CI Low": "ci_low_post",
            "Sample CI High": "ci_high_post",
            "Sample Std": "Std Post",
        }
    )

    # Select columns for join
    join_cols = ["Metric", LOG_ID_COL, WINDOW_SIZE_COL]
    pre_cols = join_cols + [
        "Center Pre",
        "Std Pre",
        "IQR Pre",
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        CHANGE_MAGNITUDE_COL,
        EDIT_OPERATIONS_COL,
        SEED_COL,
    ]
    post_cols = join_cols + ["Center Post", "Std Post"]

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

    # Collapse within seed so each seed contributes at most once per cell.
    df_sigma = _collapse_within_sigma(
        df_joined,
        group_keys_with_sigma=[
            "Metric",
            SEED_COL,
            WINDOW_SIZE_COL,
            MODEL_COMPLEXITY_COL,
            NOISE_LEVEL_COL,
            CHANGE_MAGNITUDE_COL,
            EDIT_OPERATIONS_COL,
        ],
        value_cols=["Center Pre", "Center Post", "Std Pre", "Std Post", "IQR Pre"],
    )

    # Compute signal, SNR, and relative change (per seed)
    df_sigma["Relative Change Seed"] = (
        (df_sigma["Center Post"] - df_sigma["Center Pre"]).abs()
    ) / (df_sigma["Center Pre"].abs() + eps)
    df_sigma["Signal"] = (df_sigma["Center Post"] - df_sigma["Center Pre"]).abs()

    # Compute pooled std for Cohen's d variants
    pooled_std = np.sqrt((df_sigma["Std Pre"] ** 2 + df_sigma["Std Post"] ** 2) / 2)

    # Absolute Cohen's d: |mean_post - mean_pre| / pooled_std
    # This prevents effects from canceling out when aggregating across seeds
    # When pooled_std=0: inf if signal>0 (perfect detectability), 0 if signal=0
    signal_abs = (df_sigma["Center Post"] - df_sigma["Center Pre"]).abs()
    df_sigma["Abs Cohen D Seed"] = np.where(
        pooled_std == 0, np.where(signal_abs == 0, 0.0, np.inf), signal_abs / pooled_std
    )

    # Compute SNR based on definition
    # When denominator is 0: inf if signal>0 (perfect detectability), 0 if signal=0
    if snr_definition == "pooled_cohens_d":
        # Pooled Cohen's d (signed): (mean_post - mean_pre) / sqrt((s_pre² + s_post²) / 2)
        signal_signed = df_sigma["Center Post"] - df_sigma["Center Pre"]
        df_sigma["SNR Seed"] = np.where(
            pooled_std == 0,
            np.where(signal_signed == 0, 0.0, np.inf * np.sign(signal_signed)),
            signal_signed / pooled_std,
        )
    elif snr_definition == "abs_cohens_d":
        # Absolute Cohen's d: |mean_post - mean_pre| / sqrt((s_pre² + s_post²) / 2)
        # Effects don't cancel out when aggregating across seeds
        df_sigma["SNR Seed"] = np.where(
            pooled_std == 0,
            np.where(signal_abs == 0, 0.0, np.inf),
            signal_abs / pooled_std,
        )
    else:  # "iqr" - legacy definition
        iqr_pre = df_sigma["IQR Pre"]
        df_sigma["SNR Seed"] = np.where(
            iqr_pre == 0,
            np.where(df_sigma["Signal"] == 0, 0.0, np.inf),
            df_sigma["Signal"] / iqr_pre,
        )

    # Select final columns
    result_cols = [
        "Metric",
        SEED_COL,
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        CHANGE_MAGNITUDE_COL,
        EDIT_OPERATIONS_COL,
        WINDOW_SIZE_COL,
        "Center Pre",
        "Center Post",
        "Std Pre",
        "Std Post",
        "IQR Pre",
        "Signal",
        "SNR Seed",
        "Abs Cohen D Seed",
        "Relative Change Seed",
    ]

    df_snr = df_sigma[result_cols].copy()

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
    Aggregate SNR across seed by experimental factors.

    Parameters
    ----------
    df_snr_log
        DataFrame with snr_sigma per seed (log_seed).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with aggregated SNR per cell.
    """
    # Group by experimental factors
    grouped = df_snr_log.groupby(
        [
            "Metric",
            CHANGE_MAGNITUDE_COL,
            EDIT_OPERATIONS_COL,
            MODEL_COMPLEXITY_COL,
            NOISE_LEVEL_COL,
            WINDOW_SIZE_COL,
        ]
    )

    # Aggregate
    agg_result = grouped.agg(
        **{
            "SNR": ("SNR Seed", _strict_median),
            "N Seed": ("SNR Seed", "count"),
            "SNR IQR Seed": ("SNR Seed", _strict_iqr),
        }
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
        index=["Dimension", "Metric"],
        columns=[
            CHANGE_MAGNITUDE_COL,
            EDIT_OPERATIONS_COL,
            MODEL_COMPLEXITY_COL,
            NOISE_LEVEL_COL,
            WINDOW_SIZE_COL,
        ],
        values="SNR",
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
        complexity_order = ["Simple", "Middle", "Complex"]
        complexities_sorted = [c for c in complexity_order if c in complexities]
        complexities_sorted.extend(
            [c for c in complexities if c not in complexity_order]
        )

        # Reorder noise
        noise_order = ["None", "Low", "High"]
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
                CHANGE_MAGNITUDE_COL,
                EDIT_OPERATIONS_COL,
                MODEL_COMPLEXITY_COL,
                NOISE_LEVEL_COL,
                WINDOW_SIZE_COL,
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
        Per-seed DataFrame with Metric, value_col, and group_col (and optional filter_col).
    value_col
        Column to aggregate (e.g. snr_sigma or rel_change_sigma).
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
    if SEED_COL not in df.columns:
        raise ValueError(
            f"_aggregate_by_group requires {SEED_COL!r} to aggregate across seed last."
        )

    # Collapse within seed first so each seed contributes at most once per (Metric, group_col).
    per_sigma = _collapse_within_sigma(
        df,
        group_keys_with_sigma=["Metric", group_col, SEED_COL],
        value_cols=[value_col],
    )
    agg_result = _median_over_sigma(
        per_sigma,
        group_keys_without_sigma=["Metric", group_col],
        value_col=value_col,
        out_col=value_col,
        n_col="N Seed",
    )
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
        DataFrame with snr_sigma per seed.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric × change_operation.
    """
    return _aggregate_by_group(
        df_snr_log,
        "SNR Seed",
        EDIT_OPERATIONS_COL,
        CHANGE_OPERATION_ORDER,
    )


def aggregate_snr_by_evolution_proportion(df_snr_log: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SNR by metric and evolution_proportion, fixing Edit Operations to "mixed".

    Parameters
    ----------
    df_snr_log
        DataFrame with snr_sigma per seed.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric × evolution_proportion.
    """
    return _aggregate_by_group(
        df_snr_log,
        "SNR Seed",
        CHANGE_MAGNITUDE_COL,
        None,
        filter_col=EDIT_OPERATIONS_COL,
        filter_val="mixed",
    )


def aggregate_snr_by_noise(df_snr_log: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SNR by metric and noise, fixing Edit Operations to "mixed".

    Parameters
    ----------
    df_snr_log
        DataFrame with snr_sigma per seed.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric × noise.
    """
    noise_order = ["None", "Low", "High"]
    return _aggregate_by_group(
        df_snr_log,
        "SNR Seed",
        NOISE_LEVEL_COL,
        noise_order,
        filter_col=EDIT_OPERATIONS_COL,
        filter_val="mixed",
    )


def perform_snr_sanity_checks(
    df_snr_log: pd.DataFrame, df_snr_cells: pd.DataFrame
) -> None:
    """
    Perform sanity checks for SNR analysis.

    Parameters
    ----------
    df_snr_log
        Long-format DataFrame with SNR per seed (log_seed).
    df_snr_cells
        Long-format DataFrame with aggregated SNR per cell.
    """
    print("\n" + "=" * 60)
    print("SNR SANITY CHECKS")
    print("=" * 60)

    # Check: Report missing pre/post pairs
    print("\nChecking: SNR key uniqueness (per seed)")
    total_combinations = len(df_snr_log.groupby(["Metric", SEED_COL, WINDOW_SIZE_COL]))
    print(
        f"   Total (metric, {SEED_COL}, window_size) combinations with SNR: {total_combinations}"
    )
    dup_counts = (
        df_snr_log.groupby(["Metric", SEED_COL, WINDOW_SIZE_COL], dropna=False)
        .size()
        .reset_index(name="N Rows")
    )
    n_dups = int((dup_counts["N Rows"] > 1).sum())
    if n_dups > 0:
        print(
            f"   [!] Warning: Found {n_dups} duplicated (Metric, {SEED_COL}, window_size) keys in df_snr_log."
        )

    # Summary statistics
    print("\nSummary Statistics")
    print(f"   Number of metrics: {df_snr_cells['Metric'].nunique()}")
    print(
        f"   Number of evolution_proportions: {df_snr_cells[CHANGE_MAGNITUDE_COL].nunique()}"
    )
    print(
        f"   Number of change_operations: {df_snr_cells[EDIT_OPERATIONS_COL].nunique()}"
    )
    print(
        f"   Number of complexity levels: {df_snr_cells[MODEL_COMPLEXITY_COL].nunique()}"
    )
    print(f"   Number of noise levels: {df_snr_cells[NOISE_LEVEL_COL].nunique()}")
    print(f"   Number of window sizes: {df_snr_cells[WINDOW_SIZE_COL].nunique()}")
    print(f"   Total cells (metric × δ × O × C × N × W): {len(df_snr_cells)}")

    n_col = "N Seed" if "N Seed" in df_snr_cells.columns else "N Logs"
    n_stats = df_snr_cells[n_col].describe()
    print(f"\n   {n_col} per cell:")
    print(f"     Min: {n_stats['min']:.0f}")
    print(f"     Median: {n_stats['50%']:.0f}")
    print(f"     Max: {n_stats['max']:.0f}")

    snr_stats = df_snr_cells["SNR"].describe()
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
    """Normalize header for LaTeX: log_id -> log\\_id (escaped); % -> \\%; underscore -> space."""
    if name is None:
        return None
    s = str(name)
    if s == LOG_ID_COL:
        return "log\\_id"
    s = s.replace("_", " ")
    # Only escape % that is not already escaped (not preceded by backslash)
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    return s


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
    heatmap_vmin: float = 0.0,
    heatmap_vmax: float | None = None,
    heatmap_exclude_columns: list[str] | None = None,
) -> str:
    """
    Inject \\cellcolor{blue!NN} into numeric data cells (white=vmin, dark blue=vmax).

    Linear scale from heatmap_vmin to heatmap_vmax; values >= heatmap_vmax get
    the same darkest color. Inf and NA are excluded from coloring.

    Parameters
    ----------
    latex
        Full table LaTeX string (with \\midrule before body).
    df
        DataFrame that was used to generate the table (same row/column order).
    n_index_cols
        Number of index columns at the start of each row.
    heatmap_vmin
        Min value for the color scale. Values <= heatmap_vmin get the lightest
        color. Default is 0.
    heatmap_vmax
        Max value for the color scale. Color increases linearly from
        heatmap_vmin to heatmap_vmax; values >= heatmap_vmax get the darkest
        color.
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
    vmin = heatmap_vmin
    vmax = heatmap_vmax if heatmap_vmax is not None and heatmap_vmax > 0 else 1.0
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
                                # Linear vmin..vmax -> white..dark; values >= vmax same dark
                                t = min(1.0, max(0.0, (v - vmin) / (vmax - vmin)))
                                pct = int(5 + 95 * t)
                                pct = max(0, min(100, pct))
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
    heatmap_vmin: float = 0.0,
    heatmap_vmax: float | None = None,
    heatmap_exclude_columns: list[str] | None = None,
) -> None:
    """
    Write a DataFrame to a LaTeX table file with caption, label, and scriptsize.

    Handles both single-level and MultiIndex column headers. MultiIndex columns
    are rendered with multicolumn so multi-headers are correct. When
    use_latex_metric_names is True, metric names are replaced by
    METRIC_NAMES_TO_LATEX_MAP (and escape=False is used). When heatmap=True,
    data cells are colored linearly from heatmap_vmin (white) to heatmap_vmax
    (dark blue); values >= heatmap_vmax get the same darkest color.

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
        If True, color data cells by value (vmin=white, heatmap_vmax=dark blue).
    heatmap_vmin
        Min value for the linear color scale; values <= this get the lightest
        color. Default is 0.
    heatmap_vmax
        Max value for the linear color scale; values >= this get the darkest color.
        E.g. 1 for noise/SNR tables, 100 for factor importance (percent).
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
        "position": LATEX_TABLE_POSITION,
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
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            heatmap_exclude_columns=heatmap_exclude_columns,
        )
        # Build color-coding legend with actual vmin/vmax values
        eff_vmin = heatmap_vmin
        eff_vmax = heatmap_vmax if heatmap_vmax is not None and heatmap_vmax > 0 else 1.0

        def _fmt_val(v: float) -> str:
            return str(int(v)) if v == int(v) else f"{v:g}"

        vmin_display = _fmt_val(eff_vmin)
        vmax_display = _fmt_val(eff_vmax)
        legend = (
            "\n\\vspace{2pt}\n"
            "\\begin{minipage}{\\linewidth}\n"
            "\\centering\n"
            "\\tiny\n"
            "\\textit{Color coding:} \n"
            f"\\colorbox{{blue!5}}{{\\phantom{{00}}}} indicates {vmin_display}, \n"
            f"\\colorbox{{blue!100}}{{\\phantom{{00}}}} indicates values $\\geq {vmax_display}$.\n"
            "\\end{minipage}"
        )
        out = out.replace("\\end{tabular}", "\\end{tabular}" + legend)
    # Inject \scriptsize inside the table environment (after \centering)
    if "\\centering" in out:
        out = out.replace("\\centering\n", "\\centering\n\\scriptsize\n")
    else:
        out = out.replace("\\begin{table}", "\\begin{table}\n\\scriptsize\n")
    # Inject reduced column padding after \label
    out = re.sub(
        r"(\\label\{[^}]*\})\n",
        r"\1\n\\setlength{\\tabcolsep}{2pt}  % default is 6pt\n",
        out,
    )
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
    **kwargs: Any,
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
    df_predrift = filter_to_stable_regime(df)

    print("Loading generation info for log_id mapping...")
    gen_info = load_generation_info(PATH_GEN_INFO)

    print("Enriching with experimental factors...")
    df_predrift_enriched = enrich_with_experimental_factors(df_predrift, gen_info)

    print("Removing pre-drift duplicates...")
    n_before = len(df_predrift_enriched)
    n_logs_before = df_predrift_enriched[LOG_ID_COL].nunique()
    df_predrift_enriched = remove_pre_drift_duplicates(df_predrift_enriched)
    print(
        f"  Reduced from {n_before} to {len(df_predrift_enriched)} rows "
        f"({n_logs_before} -> {df_predrift_enriched[LOG_ID_COL].nunique()} representative log_ids)."
    )

    print("Computing per-log relative noise...")
    df_with_noise, n_invalid = compute_per_log_relative_noise(
        df_predrift_enriched, aggregation=AGGREGATION
    )
    print("Adding per-log change vs. baseline...")
    df_with_noise = add_change_vs_baseline_log(df_with_noise)

    # Add Std Pre and CV Pre columns for factor importance analysis
    df_with_noise["Std Pre"] = df_with_noise["Sample Std"]
    df_with_noise["CV Pre"] = df_with_noise["Sample Std"] / (
        df_with_noise["Center"].abs() + eps
    )

    print("Aggregating relative noise across logs...")
    df_long = aggregate_relative_noise(df_with_noise)

    print("Creating wide-format table...")
    df_wide = create_wide_format_table(df_long)

    print("Saving outputs...")
    save_outputs(df_long, df_wide)

    # LaTeX tables for relative noise (heatmap scale 0..1)
    _write_latex_table(
        df_long,
        Path(PATH_OUT_LONG).stem,
        caption="Relative noise (long format).",
        label="tab:signal-noise-relative-long",
        index=False,
        heatmap_vmax=1,
    )
    _write_latex_table(
        df_wide,
        Path(PATH_OUT_WIDE).stem,
        caption="Relative noise by complexity, noise, and window size.",
        label="tab:signal-noise-relative-wide",
        index=True,
        heatmap_vmax=1,
    )

    print("Performing sanity checks...")
    perform_sanity_checks(df_long, n_invalid)

    # Tables 1–5: noise_abs_median, noise_relci, noise_std_pre, noise_cv_pre, noise_change_due_to_noise
    _NOISE_TABLE_CONFIGS = [
        (
            "noise_abs_median",
            _aggregate_abs_median,
            False,
            f"Absolute {AGGREGATION} metric value.",
            "tab:signal-noise-abs-median",
        ),
        (
            "noise_relci",
            _aggregate_relci,
            False,
            f"Relative confidence interval (CI width / {AGGREGATION}).",
            "tab:signal-noise-relci",
        ),
        (
            "noise_std_pre",
            _aggregate_std_pre,
            False,
            "Standard deviation (sample std).",
            "tab:signal-noise-std-pre",
        ),
        (
            "noise_cv_pre",
            _aggregate_cv_pre,
            False,
            f"Coefficient of variation (std / |{AGGREGATION}|).",
            "tab:signal-noise-cv-pre",
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
            df_noise_long, "Value", drop_no_noise_columns=drop_no_noise
        )
        path_csv = f"{DIR_CSV}/{stem}.csv"
        df_noise_wide.to_csv(path_csv)
        print(f"Saved {stem} to {path_csv}")
        # LaTeX: only window sizes 50 and 200; linear color scale 0..1
        df_noise_wide_latex = df_noise_wide.loc[
            :,
            df_noise_wide.columns.get_level_values(WINDOW_SIZE_COL).isin(
                _NOISE_LATEX_WINDOW_SIZES
            ),
        ]
        _write_latex_table(
            df_noise_wide_latex,
            stem,
            caption=caption,
            label=label,
            index=True,
            heatmap_vmax=1,
        )

    print("\n" + "=" * 60)
    print("NOISE FACTOR IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Factor importance: relative CI (rel_noise_log)
    print("Computing factor importance for relative CIs (rel_noise_log)...")
    df_factor_long, df_factor_wide = run_noise_factor_importance(
        df_with_noise,
        outcome_col="Relative Noise Log",
        # Regression-only: keep zero-noise cases where rel_noise_log can be exactly 0.
        # Without this (shift=0), zeros produce -inf and are dropped, biasing n_obs.
        outcome_log_shift=FACTOR_IMPORTANCE_LOG_SHIFT,
        debug_filtering=DEBUG_R2_FILTERING,
        debug_max_examples=DEBUG_R2_MAX_EXAMPLES,
    )
    Path(PATH_NOISE_FACTOR_LONG).parent.mkdir(parents=True, exist_ok=True)
    df_factor_long.to_csv(PATH_NOISE_FACTOR_LONG, index=False)
    print(f"Saved factor importance (relci) long to {PATH_NOISE_FACTOR_LONG}")
    df_factor_wide.to_csv(PATH_NOISE_FACTOR_WIDE)
    print(f"Saved factor importance (relci) wide to {PATH_NOISE_FACTOR_WIDE}")
    # Only wide table LaTeX for factor importance
    _write_latex_table(
        df_factor_wide.rename(
            columns={
                k: v
                for k, v in FACTOR_IMPORTANCE_PCT_HEADERS.items()
                if k in df_factor_wide.columns
            }
        ),
        "noise_factor_importance_relci",
        caption="Noise factor importance (relative CI) by metric.",
        label="tab:signal-noise-factor-importance-relci",
        index=True,
        heatmap_vmax=100,
        heatmap_exclude_columns=["N Obs", "R2 Full (%)"],
    )
    print("Factor importance (relative CI) summary:")
    print_factor_importance_summary(df_factor_long, df_factor_wide)

    # Factor importance: standard deviation (Std Pre)
    print("Computing factor importance for standard deviation (Std Pre)...")
    df_factor_long_std, df_factor_wide_std = run_noise_factor_importance(
        df_with_noise,
        outcome_col="Std Pre",
        outcome_log_shift=FACTOR_IMPORTANCE_LOG_SHIFT,
        debug_filtering=DEBUG_R2_FILTERING,
        debug_max_examples=DEBUG_R2_MAX_EXAMPLES,
    )
    df_factor_long_std.to_csv(PATH_NOISE_FACTOR_LONG_STD, index=False)
    print(f"Saved factor importance (std) long to {PATH_NOISE_FACTOR_LONG_STD}")
    df_factor_wide_std.to_csv(PATH_NOISE_FACTOR_WIDE_STD)
    print(f"Saved factor importance (std) wide to {PATH_NOISE_FACTOR_WIDE_STD}")
    # Only wide table LaTeX for factor importance
    _write_latex_table(
        df_factor_wide_std.rename(
            columns={
                k: v
                for k, v in FACTOR_IMPORTANCE_PCT_HEADERS.items()
                if k in df_factor_wide_std.columns
            }
        ),
        "noise_factor_importance_std",
        caption="Noise factor importance (standard deviation) by metric.",
        label="tab:signal-noise-factor-importance-std",
        index=True,
        heatmap_vmax=100,
        heatmap_exclude_columns=["N Obs", "R2 Full (%)"],
    )
    print("Factor importance (standard deviation) summary:")
    print_factor_importance_summary(df_factor_long_std, df_factor_wide_std)

    # Factor importance: coefficient of variation (CV Pre)
    print("Computing factor importance for coefficient of variation (CV Pre)...")
    df_factor_long_cv, df_factor_wide_cv = run_noise_factor_importance(
        df_with_noise,
        outcome_col="CV Pre",
        outcome_log_shift=FACTOR_IMPORTANCE_LOG_SHIFT,
        debug_filtering=DEBUG_R2_FILTERING,
        debug_max_examples=DEBUG_R2_MAX_EXAMPLES,
    )
    df_factor_long_cv.to_csv(PATH_NOISE_FACTOR_LONG_CV, index=False)
    print(f"Saved factor importance (cv) long to {PATH_NOISE_FACTOR_LONG_CV}")
    df_factor_wide_cv.to_csv(PATH_NOISE_FACTOR_WIDE_CV)
    print(f"Saved factor importance (cv) wide to {PATH_NOISE_FACTOR_WIDE_CV}")
    # Only wide table LaTeX for factor importance
    _write_latex_table(
        df_factor_wide_cv.rename(
            columns={
                k: v
                for k, v in FACTOR_IMPORTANCE_PCT_HEADERS.items()
                if k in df_factor_wide_cv.columns
            }
        ),
        "noise_factor_importance_cv",
        caption="Noise factor importance (coefficient of variation) by metric.",
        label="tab:signal-noise-factor-importance-cv",
        index=True,
        heatmap_vmax=100,
        heatmap_exclude_columns=["N Obs", "R2 Full (%)"],
    )
    print("Factor importance (coefficient of variation) summary:")
    print_factor_importance_summary(df_factor_long_cv, df_factor_wide_cv)

    # Factor importance: change vs. baseline (exclude noise=None: change is always 0 there)
    print("Computing factor importance for change vs. baseline...")
    df_factor_long_change, df_factor_wide_change = run_noise_factor_importance(
        df_with_noise[df_with_noise[NOISE_LEVEL_COL] != "None"],
        outcome_col="Change Vs Baseline Log",
        # Regression-only: keep zero-change cases by shifting before log-transform.
        # Without this (shift=0), zeros produce -inf and are dropped, biasing n_obs.
        outcome_log_shift=FACTOR_IMPORTANCE_LOG_SHIFT,
        debug_filtering=DEBUG_R2_FILTERING,
        debug_max_examples=DEBUG_R2_MAX_EXAMPLES,
    )
    df_factor_long_change.to_csv(PATH_NOISE_FACTOR_LONG_CHANGE, index=False)
    print(f"Saved factor importance (change) long to {PATH_NOISE_FACTOR_LONG_CHANGE}")
    df_factor_wide_change.to_csv(PATH_NOISE_FACTOR_WIDE_CHANGE)
    print(f"Saved factor importance (change) wide to {PATH_NOISE_FACTOR_WIDE_CHANGE}")
    # Only wide table LaTeX for factor importance
    _write_latex_table(
        df_factor_wide_change.rename(
            columns={
                k: v
                for k, v in FACTOR_IMPORTANCE_PCT_HEADERS.items()
                if k in df_factor_wide_change.columns
            }
        ),
        "noise_factor_importance_change",
        caption="Noise factor importance (change vs. baseline) by metric.",
        label="tab:signal-noise-factor-importance-change",
        index=True,
        heatmap_vmax=100,
        heatmap_exclude_columns=["N Obs", "R2 Full (%)"],
    )
    print("Factor importance (change vs. baseline) summary:")
    print_factor_importance_summary(df_factor_long_change, df_factor_wide_change)

    print("\n" + "=" * 60)
    print("SNR ANALYSIS")
    print("=" * 60)
    print(f"Using SNR definition: {SNR_DEFINITION}")
    print("Loading and normalizing split names...")
    df_full = load_and_validate_input(PATH_AGG)
    df_normalized = normalize_split_name(df_full)

    print(
        "Enriching with design factors (including evolution_proportion and change_operation)..."
    )
    df_snr_enriched = enrich_with_experimental_factors(df_normalized, gen_info)

    print("Computing SNR per log...")
    df_snr_log = compute_snr_per_log(
        df_snr_enriched, AGGREGATION, snr_definition=SNR_DEFINITION
    )

    print("Aggregating SNR across logs...")
    df_snr_cells = aggregate_snr_cells(df_snr_log)

    print("Creating wide-format SNR table...")
    df_snr_wide = pivot_snr_wide(df_snr_cells)

    print("Aggregating SNR by operation...")
    df_snr_by_op = aggregate_snr_by_operation(df_snr_log)

    print("Aggregating SNR by evolution_proportion (Edit Operations=mixed)...")
    df_snr_by_evol = aggregate_snr_by_evolution_proportion(df_snr_log)

    print("Aggregating SNR by noise (Edit Operations=mixed)...")
    df_snr_by_noise = aggregate_snr_by_noise(df_snr_log)

    print("Saving SNR outputs...")
    Path(DIR_CSV).mkdir(parents=True, exist_ok=True)
    _SNR_SAVE_CONFIGS = [
        (PATH_SNR_LONG, df_snr_cells, False, "SNR long-format"),
        (PATH_SNR_WIDE, df_snr_wide, True, "SNR wide"),
        (PATH_SNR_BY_OP, df_snr_by_op, True, "SNR by operation"),
        (PATH_SNR_BY_EVOL, df_snr_by_evol, True, "SNR by evolution proportion"),
        (PATH_SNR_BY_NOISE, df_snr_by_noise, True, "SNR by noise"),
    ]
    for path, frame, index, desc in _SNR_SAVE_CONFIGS:
        frame.to_csv(path, index=index)
        print(f"Saved {desc} to {path}")

    # Tables 4–6: signal_relchange by operation, evolution_proportion, noise
    _SIGNAL_RELCHANGE_CONFIGS = [
        (
            "signal_relchange_by_operation",
            EDIT_OPERATIONS_COL,
            CHANGE_OPERATION_ORDER,
            None,
            None,
            f"{AGGREGATION.capitalize()} relative change pre-to-post by change operation.",
            "tab:signal-relchange-by-operation",
        ),
        (
            "signal_relchange_by_evolution_proportion",
            CHANGE_MAGNITUDE_COL,
            None,
            EDIT_OPERATIONS_COL,
            "mixed",
            f"{AGGREGATION.capitalize()} relative change pre-to-post by evolution proportion (Edit Operations=mixed).",
            "tab:signal-relchange-by-evolution",
        ),
        (
            "signal_relchange_by_noise",
            NOISE_LEVEL_COL,
            ["None", "Low", "High"],
            EDIT_OPERATIONS_COL,
            "mixed",
            f"{AGGREGATION.capitalize()} relative change pre-to-post by noise (Edit Operations=mixed).",
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
            "Relative Change Seed",
            group_col,
            column_order,
            filter_col=filter_col,
            filter_val=filter_val,
        )
        path_csv = f"{DIR_CSV}/{stem}.csv"
        df_rel.to_csv(path_csv)
        print(f"Saved {stem} to {path_csv}")
        _write_latex_table(
            df_rel, stem, caption=caption, label=label, index=True, heatmap_vmax=1
        )

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
            "SNR by evolution proportion (Edit Operations=mixed).",
            "tab:signal-snr-by-evolution",
        ),
        (
            PATH_SIGNAL_SNR_BY_NOISE,
            "signal_snr_by_noise",
            "SNR by noise level (Edit Operations=mixed).",
            "tab:signal-snr-by-noise",
        ),
    ]
    df_snr_by_tables = [df_snr_by_op, df_snr_by_evol, df_snr_by_noise]
    for (path_csv, stem, caption, label), df_snr_by in zip(
        _SIGNAL_SNR_CONFIGS, df_snr_by_tables
    ):
        df_snr_by.to_csv(path_csv)
        print(f"Saved {stem} to {path_csv}")
        _write_latex_table(
            df_snr_by,
            stem,
            caption=caption,
            label=label,
            index=True,
            heatmap_vmax=1,
        )

    # LaTeX tables for SNR (heatmap scale 0..1)
    _write_latex_table(
        df_snr_cells,
        Path(PATH_SNR_LONG).stem,
        caption="SNR per cell (long format).",
        label="tab:signal-noise-snr-long",
        index=False,
        heatmap_vmax=1,
    )
    _write_latex_table(
        df_snr_wide,
        Path(PATH_SNR_WIDE).stem,
        caption="SNR by evolution proportion, change operation, complexity, noise, and window size.",
        label="tab:signal-noise-snr-wide",
        index=True,
        heatmap_vmax=1,
    )
    _write_latex_table(
        df_snr_by_op,
        Path(PATH_SNR_BY_OP).stem,
        caption="SNR by change operation.",
        label="tab:signal-noise-snr-by-operation",
        index=True,
        heatmap_vmax=1,
    )
    _write_latex_table(
        df_snr_by_evol,
        Path(PATH_SNR_BY_EVOL).stem,
        caption="SNR by evolution proportion (Edit Operations=mixed).",
        label="tab:signal-noise-snr-by-evolution",
        index=True,
        heatmap_vmax=1,
    )
    _write_latex_table(
        df_snr_by_noise,
        Path(PATH_SNR_BY_NOISE).stem,
        caption="SNR by noise level (Edit Operations=mixed).",
        label="tab:signal-noise-snr-by-noise",
        index=True,
        heatmap_vmax=1,
    )

    # Tables: Absolute Cohen's d by operation, evolution_proportion, noise
    # |Cohen's d| = |mean_post - mean_pre| / pooled_std (effects don't cancel out)
    _SIGNAL_ABS_COHENS_D_CONFIGS = [
        (
            "signal_abs_cohens_d_by_operation",
            EDIT_OPERATIONS_COL,
            CHANGE_OPERATION_ORDER,
            None,
            None,
            "Mean absolute Cohen's d pre-to-post by change operation.",
            "tab:signal-abs-cohens-d-by-operation",
        ),
        (
            "signal_abs_cohens_d_by_evolution_proportion",
            CHANGE_MAGNITUDE_COL,
            None,
            EDIT_OPERATIONS_COL,
            "mixed",
            "Mean absolute Cohen's d pre-to-post by evolution proportion (Edit Operations=mixed).",
            "tab:signal-abs-cohens-d-by-evolution",
        ),
        (
            "signal_abs_cohens_d_by_noise",
            NOISE_LEVEL_COL,
            ["None", "Low", "High"],
            EDIT_OPERATIONS_COL,
            "mixed",
            "Mean absolute Cohen's d pre-to-post by noise (Edit Operations=mixed).",
            "tab:signal-abs-cohens-d-by-noise",
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
    ) in _SIGNAL_ABS_COHENS_D_CONFIGS:
        df_abs_d = _aggregate_by_group(
            df_snr_log,
            "Abs Cohen D Seed",
            group_col,
            column_order,
            filter_col=filter_col,
            filter_val=filter_val,
        )
        path_csv = f"{DIR_CSV}/{stem}.csv"
        df_abs_d.to_csv(path_csv)
        print(f"Saved {stem} to {path_csv}")
        _write_latex_table(
            df_abs_d, stem, caption=caption, label=label, index=True, heatmap_vmax=1
        )

    print("Performing SNR sanity checks...")
    perform_snr_sanity_checks(df_snr_log, df_snr_cells)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        main()
    if caught_warnings:
        print(
            f"\n  {len(caught_warnings)} runtime warning(s) (e.g. invalid value in subtract)."
        )
