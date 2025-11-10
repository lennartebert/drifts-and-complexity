"""Comparison table generation for before vs after scenarios."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .helpers import APPLY_FPC, DEFAULT_N, DEFAULT_NPOP, TAU


def fisher_z_test_independent(
    r1: float,
    r2: float,
    n1: int,
    n2: int,
    N1: Optional[int] = None,
    N2: Optional[int] = None,
) -> tuple[float, float]:
    """
    Two-sided Fisher r-to-z test for the difference between two independent correlations.
    Supports optional finite population correction (FPC) per group.

    Args:
        r1: First correlation coefficient.
        r2: Second correlation coefficient.
        n1: Sample size for first correlation.
        n2: Sample size for second correlation.
        N1: Population size for first correlation (optional, for FPC).
        N2: Population size for second correlation (optional, for FPC).

    Returns:
        Tuple of (z_stat, p_value).
    """
    # Basic checks
    if (n1 is None or n2 is None) or (n1 < 4 or n2 < 4) or np.isnan(r1) or np.isnan(r2):
        return np.nan, np.nan

    # Guard |r|<1 to avoid atanh overflow
    eps = 1e-12
    if not (-1 + eps < r1 < 1 - eps) or not (-1 + eps < r2 < 1 - eps):
        return np.nan, np.nan

    # Fisher z transforms
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    # Finite population corrections (default to 1 if not applicable)
    def fpc(n: int, N: Optional[int]) -> float:
        if not APPLY_FPC or N is None or N <= n or N <= 1:
            return 1.0
        return math.sqrt((N - n) / (N - 1))

    fpc1 = fpc(n1, N1)
    fpc2 = fpc(n2, N2)

    # Standard error (independent correlations)
    # Var(z_r) ≈ 1/(n-3) under SRS; with FPC multiply by FPC^2
    se = math.sqrt((fpc1**2) / (n1 - 3) + (fpc2**2) / (n2 - 3))

    z_stat = (z1 - z2) / se
    p_val = 2 * norm.sf(abs(z_stat))
    return float(z_stat), float(p_val)


def build_and_save_comparison_csv(
    before_csv_path: str,
    after_csv_path: str,
    out_csv_path: str,
) -> str:
    """
    Build a comparison table by joining before and after master tables.

    This performs an outer join on (Metric, Basis, Log) and computes:
    - Shape diagnostics and chosen correlation (Policy B, shape-aware)
    - Deltas for chosen correlation
    - Fisher z-test for chosen correlation
    - All diagnostic columns with _before and _after suffixes

    Args:
        before_csv_path: Path to before scenario master table CSV.
        after_csv_path: Path to after scenario master table CSV.
        out_csv_path: Path to save the comparison CSV.

    Returns:
        Path to the saved CSV file.
    """

    # Read master tables and convert string floats back to numeric
    def read_and_convert_master_csv(path: str) -> Optional[pd.DataFrame]:
        if not Path(path).exists():
            return None
        df = pd.read_csv(path)
        # Convert float columns from strings back to floats
        float_cols = [
            "Pearson_Rho",
            "Spearman_Rho",
            "Pearson_P",
            "Spearman_P",
            "Delta_PearsonSpearman",
            "RelCI_50",
            "RelCI_250",
            "RelCI_500",
            "Plateau_n",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Convert n and N_pop to nullable int
        for col in ["n", "N_pop"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df

    before_df = read_and_convert_master_csv(before_csv_path)
    after_df = read_and_convert_master_csv(after_csv_path)

    # Handle case where only after exists
    if (before_df is None or before_df.empty) and (after_df is None or after_df.empty):
        raise ValueError(
            "At least one of before_csv_path or after_csv_path must exist."
        )

    if before_df is None:
        # Only after exists - create comparison with missing before
        assert after_df is not None  # type guard
        after_df = after_df.copy()
        # Rename after columns first
        rename_map = {
            "Pearson_Rho": "Pearson_Rho_after",
            "Spearman_Rho": "Spearman_Rho_after",
            "Pearson_P": "Pearson_P_after",
            "Spearman_P": "Spearman_P_after",
            "n": "n_after",
            "N_pop": "N_pop_after",
            "RelCI_50": "RelCI_50_after",
            "RelCI_250": "RelCI_250_after",
            "RelCI_500": "RelCI_500_after",
            "Plateau_n": "Plateau_n_after",
            "Delta_PearsonSpearman": "Delta_PearsonSpearman_after",
            "Shape": "Shape_after",
            "Preferred_Correlation": "Preferred_Correlation_after",
        }
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in after_df.columns}
        after_df = after_df.rename(columns=rename_map)
        # Require that before_csv_path exists as a file
        if not Path(before_csv_path).is_file():
            raise FileNotFoundError(
                f"before_csv_path does not exist: {before_csv_path}"
            )
        # Add before columns as NaN
        for col in [
            "Pearson_Rho",
            "Spearman_Rho",
            "Pearson_P",
            "Spearman_P",
            "n",
            "N_pop",
            "RelCI_50",
            "RelCI_250",
            "RelCI_500",
            "Plateau_n",
            "Delta_PearsonSpearman",
            "Shape",
            "Preferred_Correlation",
        ]:
            after_df[f"{col}_before"] = np.nan
        after_df["Row_Status"] = "Missing_in_before"
        # Add all comparison columns as NaN
        comparison_df = _add_comparison_columns(after_df)
        comparison_df = _reorder_comparison_columns(comparison_df)
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(out_csv_path, index=False, float_format="%.10f")
        return out_csv_path

    if after_df is None:
        # Only before exists - create comparison with missing after
        assert before_df is not None  # type guard
        before_df = before_df.copy()
        # Rename before columns first
        rename_map = {
            "Pearson_Rho": "Pearson_Rho_before",
            "Spearman_Rho": "Spearman_Rho_before",
            "Pearson_P": "Pearson_P_before",
            "Spearman_P": "Spearman_P_before",
            "n": "n_before",
            "N_pop": "N_pop_before",
            "RelCI_50": "RelCI_50_before",
            "RelCI_250": "RelCI_250_before",
            "RelCI_500": "RelCI_500_before",
            "Plateau_n": "Plateau_n_before",
            "Delta_PearsonSpearman": "Delta_PearsonSpearman_before",
            "Shape": "Shape_before",
            "Preferred_Correlation": "Preferred_Correlation_before",
        }
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in before_df.columns}
        before_df = before_df.rename(columns=rename_map)
        # Add after columns as NaN
        for col in [
            "Pearson_Rho",
            "Spearman_Rho",
            "Pearson_P",
            "Spearman_P",
            "n",
            "N_pop",
            "RelCI_50",
            "RelCI_250",
            "RelCI_500",
            "Plateau_n",
            "Delta_PearsonSpearman",
            "Shape",
            "Preferred_Correlation",
        ]:
            before_df[f"{col}_after"] = np.nan
        before_df["Row_Status"] = "Missing_in_after"
        # Add all comparison columns as NaN
        comparison_df = _add_comparison_columns(before_df)
        comparison_df = _reorder_comparison_columns(comparison_df)
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(out_csv_path, index=False, float_format="%.10f")
        return out_csv_path

    # Both exist - perform outer join
    # Prepare before columns
    before_df = before_df.copy()
    before_rename_map = {
        "Pearson_Rho": "Pearson_Rho_before",
        "Spearman_Rho": "Spearman_Rho_before",
        "Pearson_P": "Pearson_P_before",
        "Spearman_P": "Spearman_P_before",
        "n": "n_before",
        "N_pop": "N_pop_before",
        "RelCI_50": "RelCI_50_before",
        "RelCI_250": "RelCI_250_before",
        "RelCI_500": "RelCI_500_before",
        "Plateau_n": "Plateau_n_before",
        "Delta_PearsonSpearman": "Delta_PearsonSpearman_before",
        "Shape": "Shape_before",
        "Preferred_Correlation": "Preferred_Correlation_before",
    }
    # Only rename columns that exist
    before_rename_map = {
        k: v for k, v in before_rename_map.items() if k in before_df.columns
    }
    before_df = before_df.rename(columns=before_rename_map)

    # Prepare after columns
    after_df = after_df.copy()
    after_rename_map = {
        "Pearson_Rho": "Pearson_Rho_after",
        "Spearman_Rho": "Spearman_Rho_after",
        "Pearson_P": "Pearson_P_after",
        "Spearman_P": "Spearman_P_after",
        "n": "n_after",
        "N_pop": "N_pop_after",
        "RelCI_50": "RelCI_50_after",
        "RelCI_250": "RelCI_250_after",
        "RelCI_500": "RelCI_500_after",
        "Plateau_n": "Plateau_n_after",
        "Delta_PearsonSpearman": "Delta_PearsonSpearman_after",
        "Shape": "Shape_after",
        "Preferred_Correlation": "Preferred_Correlation_after",
    }
    # Only rename columns that exist
    after_rename_map = {
        k: v for k, v in after_rename_map.items() if k in after_df.columns
    }
    after_df = after_df.rename(columns=after_rename_map)

    # Outer join on (Metric, Basis, Log)
    merged = pd.merge(
        before_df,
        after_df,
        on=["Metric", "Basis", "Log"],
        how="outer",
        suffixes=("", "_after_dup"),
    )

    # Determine Row_Status
    def determine_row_status(row: pd.Series) -> str:
        has_before = pd.notna(row.get("Pearson_Rho_before", np.nan))
        has_after = pd.notna(row.get("Pearson_Rho_after", np.nan))
        if has_before and has_after:
            return "OK"
        elif has_before and not has_after:
            return "Missing_in_after"
        elif not has_before and has_after:
            return "Missing_in_before"
        else:
            return "Missing_in_both"

    merged["Row_Status"] = merged.apply(determine_row_status, axis=1)

    # Add comparison columns
    comparison_df = _add_comparison_columns(merged)

    # Add timestamp
    comparison_df["ts"] = datetime.now().isoformat()

    # Round float columns to 10 decimal places in DataFrame to avoid precision issues
    float_cols = [
        "Pearson_Rho_before",
        "Spearman_Rho_before",
        "Pearson_P_before",
        "Spearman_P_before",
        "Pearson_Rho_after",
        "Spearman_Rho_after",
        "Pearson_P_after",
        "Spearman_P_after",
        "Delta_PearsonSpearman_before",
        "Delta_PearsonSpearman_after",
        "Chosen_Rho_before",
        "Chosen_Rho_after",
        "Delta_Chosen_Rho",
        "Delta_Pearson",
        "Delta_Spearman",
        "Z_test_stat",
        "Z_test_p",
        "RelCI_50_before",
        "RelCI_250_before",
        "RelCI_500_before",
        "Plateau_n_before",
        "RelCI_50_after",
        "RelCI_250_after",
        "RelCI_500_after",
        "Plateau_n_after",
    ]
    for col in float_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(10)

    # Add Significant_Improvement column (before MEAN row computation and reordering)
    from utils.helpers import ALPHA

    def sig_impr(row: pd.Series) -> str:
        z_p = row.get("Z_test_p", np.nan)
        delta_rho = row.get("Delta_Chosen_Rho", np.nan)
        # Use pd.isna() instead of np.isnan() for better compatibility
        val = (
            pd.notna(z_p)
            and pd.notna(delta_rho)
            and np.isfinite(z_p)
            and np.isfinite(delta_rho)
            and float(z_p) < ALPHA
            and float(delta_rho) > 0
        )
        return "TRUE" if val else "FALSE"

    comparison_df["Significant_Improvement"] = comparison_df.apply(sig_impr, axis=1)

    # Compute MEAN rows per (Metric, Basis) - same logic as master_table.py
    # Filter out any existing MEAN rows to avoid double-averaging
    # Check both Row_Status and Log columns to filter MEAN rows
    per_log_df = comparison_df[
        (comparison_df["Row_Status"] != "MEAN") & (comparison_df["Log"] != "MEAN")
    ].copy()

    # Identify numeric columns for mean computation
    numeric_cols = [
        c for c in per_log_df.columns if pd.api.types.is_numeric_dtype(per_log_df[c])
    ]
    # Exclude identifier columns from numeric means
    exclude_from_mean = [
        "n_before",
        "n_after",
        "N_pop_before",
        "N_pop_after",
        "FPC_used",
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude_from_mean]

    # Group by Metric and Basis, calculating means for numeric columns
    mean_rows = []
    if len(per_log_df) > 0:
        means = (
            per_log_df.groupby(["Metric", "Basis"])[numeric_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        for _, row in means.iterrows():
            mean_row = row.to_dict()
            mean_row["Log"] = "MEAN"  # Mark as mean row
            mean_row["Row_Status"] = "MEAN"
            mean_row["ts"] = datetime.now().isoformat()

            # Set non-numeric columns appropriately
            # For string columns, use empty string or appropriate default
            for col in per_log_df.columns:
                if col not in mean_row:
                    if col in ["Chosen_Correlation", "z_type"]:
                        # String columns should be left empty for MEAN rows (no aggregation)
                        mean_row[col] = ""
                    elif col in [
                        "Shape_before",
                        "Shape_after",
                        "Preferred_Correlation_before",
                        "Preferred_Correlation_after",
                    ]:
                        # String columns should be left empty for MEAN rows (no aggregation)
                        mean_row[col] = ""
                    elif col in ["n_before", "n_after", "N_pop_before", "N_pop_after"]:
                        # Use mean for these (they're numeric but we want to include them)
                        group = per_log_df[
                            (per_log_df["Metric"] == row["Metric"])
                            & (per_log_df["Basis"] == row["Basis"])
                        ]
                        if len(group) > 0 and col in group.columns:
                            mean_row[col] = group[col].mean()
                        else:
                            mean_row[col] = np.nan
                    elif col == "FPC_used":
                        # Boolean - use most common
                        group = per_log_df[
                            (per_log_df["Metric"] == row["Metric"])
                            & (per_log_df["Basis"] == row["Basis"])
                        ]
                        if len(group) > 0 and col in group.columns:
                            mean_row[col] = (
                                group[col].mode()[0]
                                if len(group[col].mode()) > 0
                                else group[col].iloc[0]
                            )
                        else:
                            mean_row[col] = False
                    else:
                        mean_row[col] = ""

            # Compute Significant_Improvement for MEAN row as share (float) of TRUE values
            group = per_log_df[
                (per_log_df["Metric"] == row["Metric"])
                & (per_log_df["Basis"] == row["Basis"])
            ]
            if "Significant_Improvement" in group.columns and len(group) > 0:
                # Count TRUE values (as string "TRUE")
                true_count = (group["Significant_Improvement"] == "TRUE").sum()
                total_count = len(group)
                share = float(true_count) / total_count if total_count > 0 else 0.0
                mean_row["Significant_Improvement"] = share
            else:
                mean_row["Significant_Improvement"] = 0.0

            mean_rows.append(mean_row)

    # Add mean rows to comparison_df (before string conversion)
    if mean_rows:
        mean_df = pd.DataFrame(mean_rows)
        # Ensure all columns from comparison_df are present in mean_df
        for col in comparison_df.columns:
            if col not in mean_df.columns:
                # Set appropriate defaults for missing columns
                if col == "Row_Status":
                    mean_df[col] = "MEAN"
                elif col == "Log":
                    mean_df[col] = "MEAN"
                elif col == "Significant_Improvement":
                    # Preserve Significant_Improvement as float (should already be set in mean_row)
                    # If not set, default to 0.0
                    mean_df[col] = 0.0
                else:
                    mean_df[col] = ""
        # Explicitly set Row_Status and Log to "MEAN" for all mean rows
        mean_df["Row_Status"] = "MEAN"
        mean_df["Log"] = "MEAN"
        # Ensure Significant_Improvement is preserved as float (not overwritten)
        if "Significant_Improvement" in mean_df.columns:
            # Convert to float if it's somehow a string
            mean_df["Significant_Improvement"] = pd.to_numeric(
                mean_df["Significant_Improvement"], errors="coerce"
            ).fillna(0.0)
        # Reorder columns to match comparison_df
        mean_df = mean_df[comparison_df.columns]
        # Remove any existing MEAN rows before adding new ones to avoid duplicates
        comparison_df = comparison_df[
            (comparison_df["Row_Status"] != "MEAN") & (comparison_df["Log"] != "MEAN")
        ].copy()

        # Insert each MEAN row immediately after the last per-log row for its (Metric, Basis) group
        # Preserve the original metric order from the input
        # Create a list to hold the final rows in order
        final_rows = []
        current_group = None

        # Iterate through rows preserving the original order
        for idx, row in comparison_df.iterrows():
            metric = row["Metric"]
            basis = row["Basis"]
            log = row["Log"]
            group_key = (metric, basis)

            # If we're starting a new (Metric, Basis) group, insert MEAN row for previous group
            if current_group is not None and group_key != current_group:
                # Insert MEAN row for previous group before starting new group
                prev_mean = mean_df[
                    (mean_df["Metric"] == current_group[0])
                    & (mean_df["Basis"] == current_group[1])
                ]
                if len(prev_mean) > 0:
                    final_rows.append(prev_mean.iloc[0].to_dict())

            # Add the current row (only if it's not a MEAN row - MEAN rows are inserted separately)
            if log != "MEAN":
                final_rows.append(row.to_dict())

            current_group = group_key

        # Insert MEAN row for the last group
        if current_group is not None:
            last_mean = mean_df[
                (mean_df["Metric"] == current_group[0])
                & (mean_df["Basis"] == current_group[1])
            ]
            if len(last_mean) > 0:
                final_rows.append(last_mean.iloc[0].to_dict())

        # Convert back to DataFrame, preserving the original metric order
        comparison_df = pd.DataFrame(final_rows)

    # Reorder columns now that Significant_Improvement is added (before formatting)
    comparison_df = _reorder_comparison_columns(comparison_df)

    # Ensure Row_Status is "MEAN" for MEAN rows (in case it got overwritten)
    comparison_df.loc[comparison_df["Log"] == "MEAN", "Row_Status"] = "MEAN"

    # Format floats as strings for CSV output to ensure fixed format
    def format_float(x: Any) -> str:
        if pd.isna(x) or not np.isfinite(x):
            return ""
        return f"{float(x):.10f}"

    csv_df = comparison_df.copy()
    for col in float_cols:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(format_float)

    # Format Significant_Improvement: keep as float for MEAN rows, keep as string for per-log rows
    if "Significant_Improvement" in csv_df.columns:

        def format_sig_impr(row: pd.Series) -> str:
            # Check both Row_Status and Log to identify MEAN rows
            is_mean = (row.get("Row_Status") == "MEAN") or (row.get("Log") == "MEAN")
            if is_mean:
                # For MEAN rows, format as float with 10 decimals
                val = row["Significant_Improvement"]
                # Handle both float and string values
                if isinstance(val, str):
                    # If it's a string "TRUE" or "FALSE", we need to compute the share from per-log rows
                    if val == "TRUE":
                        val = 1.0
                    elif val == "FALSE":
                        val = 0.0
                    else:
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            val = np.nan
                if pd.notna(val) and np.isfinite(val):
                    return f"{float(val):.10f}"
                return ""
            else:
                # For per-log rows, keep as string ("TRUE" or "FALSE")
                return str(row["Significant_Improvement"])

        csv_df["Significant_Improvement"] = csv_df.apply(format_sig_impr, axis=1)

    # Ensure column order is correct after formatting (reorder again to be safe)
    csv_df = _reorder_comparison_columns(csv_df)

    # Save CSV
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_csv_path, index=False)

    return out_csv_path


def _add_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all comparison columns to the DataFrame."""
    df = df.copy()

    # Ensure n and N_pop are integers (convert from master tables)
    for col in ["n_before", "n_after", "N_pop_before", "N_pop_after"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: int(x) if pd.notna(x) and np.isfinite(x) else None
            )

    # Fill missing values with defaults for computation (only for Fisher z-test)
    n_before_filled = df["n_before"].fillna(DEFAULT_N)
    n_after_filled = df["n_after"].fillna(DEFAULT_N)
    N_pop_before_filled = df["N_pop_before"].fillna(DEFAULT_NPOP)
    N_pop_after_filled = df["N_pop_after"].fillna(DEFAULT_NPOP)

    # Chosen correlation: use Preferred_Correlation_after (Policy B: shape-aware)
    # If Preferred_Correlation_after is missing, fall back to computing from delta
    if "Preferred_Correlation_after" in df.columns:
        df["Chosen_Correlation"] = df["Preferred_Correlation_after"].fillna("Pearson")
    else:
        # Fallback: compute from delta if Preferred_Correlation_after is missing
        delta_after = df.get(
            "Delta_PearsonSpearman_after", pd.Series([np.nan] * len(df))
        )
        df["Chosen_Correlation"] = delta_after.apply(
            lambda x: "Spearman" if (pd.notna(x) and x > TAU) else "Pearson"
        )

    # Chosen rhos and deltas
    def get_chosen_rho_before(row: pd.Series) -> float:
        if row["Chosen_Correlation"] == "Pearson":
            val = row.get("Pearson_Rho_before", np.nan)
        else:
            val = row.get("Spearman_Rho_before", np.nan)
        if pd.notna(val) and np.isfinite(val):
            return float(val)
        return float(np.nan)

    def get_chosen_rho_after(row: pd.Series) -> float:
        if row["Chosen_Correlation"] == "Pearson":
            val = row.get("Pearson_Rho_after", np.nan)
        else:
            val = row.get("Spearman_Rho_after", np.nan)
        if pd.notna(val) and np.isfinite(val):
            return float(val)
        return float(np.nan)

    df["Chosen_Rho_before"] = df.apply(get_chosen_rho_before, axis=1)
    df["Chosen_Rho_after"] = df.apply(get_chosen_rho_after, axis=1)
    df["Delta_Chosen_Rho"] = df["Chosen_Rho_after"] - df["Chosen_Rho_before"]

    # Optional raw deltas
    df["Delta_Pearson"] = df["Pearson_Rho_after"] - df["Pearson_Rho_before"]
    df["Delta_Spearman"] = df["Spearman_Rho_after"] - df["Spearman_Rho_before"]

    # Fisher r-to-z (independent samples; optional FPC)
    z_stats = []
    z_ps = []
    for idx, row in df.iterrows():
        if (
            row["Row_Status"] == "OK"
            and pd.notna(row["Chosen_Rho_before"])
            and pd.notna(row["Chosen_Rho_after"])
        ):
            z, p = fisher_z_test_independent(
                r1=row["Chosen_Rho_before"],
                r2=row["Chosen_Rho_after"],
                n1=(
                    int(n_before_filled.iloc[idx])
                    if pd.notna(n_before_filled.iloc[idx])
                    else DEFAULT_N
                ),
                n2=(
                    int(n_after_filled.iloc[idx])
                    if pd.notna(n_after_filled.iloc[idx])
                    else DEFAULT_N
                ),
                N1=(
                    int(N_pop_before_filled.iloc[idx])
                    if pd.notna(N_pop_before_filled.iloc[idx])
                    else None
                ),
                N2=(
                    int(N_pop_after_filled.iloc[idx])
                    if pd.notna(N_pop_after_filled.iloc[idx])
                    else None
                ),
            )
            z_stats.append(z)
            z_ps.append(p)
        else:
            z_stats.append(np.nan)
            z_ps.append(np.nan)

    df["Z_test_stat"] = z_stats
    df["Z_test_p"] = z_ps
    df["z_type"] = df["Chosen_Correlation"]
    df["FPC_used"] = APPLY_FPC

    return df


def _reorder_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to match the specified schema."""
    # Define exact column order
    cols_order = [
        "Metric",
        "Basis",
        "Log",
        # Carry before/after rhos + sizes
        "Pearson_Rho_before",
        "Spearman_Rho_before",
        "Pearson_P_before",
        "Spearman_P_before",
        "n_before",
        "N_pop_before",
        "Pearson_Rho_after",
        "Spearman_Rho_after",
        "Pearson_P_after",
        "Spearman_P_after",
        "n_after",
        "N_pop_after",
        # Shape diagnostics
        "Delta_PearsonSpearman_before",
        "Shape_before",
        "Delta_PearsonSpearman_after",
        "Shape_after",
        # Preferred correlations
        "Preferred_Correlation_before",
        "Preferred_Correlation_after",
        # Chosen correlation
        "Chosen_Correlation",
        # Chosen rhos and deltas
        "Chosen_Rho_before",
        "Chosen_Rho_after",
        "Delta_Chosen_Rho",
        # Optional raw deltas
        "Delta_Pearson",
        "Delta_Spearman",
        # Fisher r-to-z
        "Z_test_stat",
        "Z_test_p",
        "z_type",
        "Significant_Improvement",
        "FPC_used",
        # Diagnostics (suffix-preserved)
        "RelCI_50_before",
        "RelCI_250_before",
        "RelCI_500_before",
        "Plateau_n_before",
        "RelCI_50_after",
        "RelCI_250_after",
        "RelCI_500_after",
        "Plateau_n_after",
        # Merge hygiene
        "Row_Status",
        "ts",
    ]

    # Keep only columns that exist
    cols_order = [c for c in cols_order if c in df.columns]
    # Add any remaining columns at the end (excluding _dup artifacts)
    remaining_cols = [
        c for c in df.columns if c not in cols_order and not c.endswith("_dup")
    ]
    cols_order = cols_order + remaining_cols

    return df[cols_order]
