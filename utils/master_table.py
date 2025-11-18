from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .constants import FLOAT_COLUMNS
from .helpers import TAU


def read_master_csv(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Read a master_table CSV file.

    Args:
        path: Path to the CSV file (can be str or Path object).

    Returns:
        DataFrame with column names as stored in CSV, or None if file not found.
    """
    # Convert Path to string if needed
    path_str = str(path) if isinstance(path, Path) else path

    # Check if file exists first
    if not os.path.exists(path_str):
        return None

    try:
        return pd.read_csv(path_str)
    except FileNotFoundError:
        return None


def combine_analysis_with_means(
    analysis_per_log: Dict[str, pd.DataFrame],
    *,
    ref_sizes: List[int] = [50, 250, 500],
    measure_basis_map: Optional[Dict[str, str]] = None,
    n: Optional[int] = None,
    N_pop: Optional[int] = None,
    metric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Combine analysis data from all logs and add mean rows.

    This function:
    1. Combines analysis dataframes from all logs
    2. Extracts per-log correlations, RelCI values, and plateau data
    3. Adds computed columns (Delta Pearson Spearman, Shape, Preferred Correlation, Plateau Reached)
    4. Calculates and inserts mean rows across logs

    Args:
        analysis_per_log: Dict mapping log names to analysis DataFrames with index (Metric, Sample Size).
        ref_sizes: List of reference sizes for RelCI columns.
        measure_basis_map: Optional dict mapping metric names to basis names.
        n: Sample size used for correlation.
        N_pop: Population size (for FPC).
        metric_columns: Optional list of metric names to include.

    Returns:
        DataFrame with columns: Metric, Basis, Log, Pearson Rho, Spearman Rho, Pearson P, Spearman P,
        n, N Pop, Delta Pearson Spearman, Shape, Preferred Correlation, RelCI columns, Plateau n,
        Plateau Found, Plateau Reached, Timestamp. Includes both per-log rows and MEAN rows.
    """
    from datetime import datetime

    rows = []

    # Get all unique logs
    all_logs = list(analysis_per_log.keys())

    # Get all unique metrics from all logs
    all_metrics_set = set()
    for analysis_df in analysis_per_log.values():
        analysis_reset = analysis_df.reset_index()
        if "Metric" in analysis_reset.columns:
            all_metrics_set.update(analysis_reset["Metric"].unique().tolist())

    # Preserve order from metric_columns if provided, otherwise sort alphabetically
    if metric_columns is not None:
        # Preserve the order from metric_columns, filtering to only include metrics that exist
        all_metrics = [m for m in metric_columns if m in all_metrics_set]
        # Add any metrics that exist but weren't in metric_columns (shouldn't happen, but be safe)
        remaining_metrics = sorted(
            [m for m in all_metrics_set if m not in metric_columns]
        )
        all_metrics = all_metrics + remaining_metrics
    else:
        all_metrics = sorted(list(all_metrics_set))

    # Process each metric-log combination
    for m in all_metrics:
        # Get basis for this metric
        basis = measure_basis_map.get(m, "Unknown") if measure_basis_map else "Unknown"

        for log in all_logs:
            if log not in analysis_per_log:
                continue

            analysis_df = analysis_per_log[log]
            analysis_reset = analysis_df.reset_index()

            # Get correlation values for this metric-log combination
            # Correlations are constant per Metric within each log, so get first row for this metric
            metric_rows = analysis_reset[analysis_reset["Metric"] == m]
            if len(metric_rows) == 0:
                continue

            # Get correlation values (constant per metric in each log)
            first_row = metric_rows.iloc[0]
            pearson_rho = (
                first_row.get("Pearson Rho")
                if "Pearson Rho" in first_row.index
                else None
            )
            spearman_rho = (
                first_row.get("Spearman Rho")
                if "Spearman Rho" in first_row.index
                else None
            )
            pearson_p = (
                first_row.get("Pearson P") if "Pearson P" in first_row.index else None
            )
            spearman_p = (
                first_row.get("Spearman P") if "Spearman P" in first_row.index else None
            )

            # Get RelCI values for this log from analysis_per_log
            relci = {}
            if "Sample CI Rel Width" in analysis_reset.columns:
                for ref_size in ref_sizes:
                    col_name = f"RelCI {ref_size}"
                    # Find row where Sample Size matches ref_size and Metric matches m
                    size_rows = analysis_reset[
                        (analysis_reset["Sample Size"] == ref_size)
                        & (analysis_reset["Metric"] == m)
                    ]
                    if len(size_rows) > 0:
                        relci[col_name] = size_rows["Sample CI Rel Width"].iloc[0]
                    else:
                        relci[col_name] = np.nan
            else:
                for ref_size in ref_sizes:
                    relci[f"RelCI {ref_size}"] = np.nan

            # Get plateau n and plateau found for this metric-log combination
            # These are constant per metric in each log
            plateau = (
                first_row.get("Plateau n") if "Plateau n" in first_row.index else np.nan
            )
            plateau_found = (
                first_row.get("Plateau Found")
                if "Plateau Found" in first_row.index
                else False
            )

            # Compute Delta Pearson Spearman
            delta_pearson_spearman = (
                abs(pearson_rho - spearman_rho)
                if (pearson_rho is not None and spearman_rho is not None)
                else np.nan
            )

            # Compute Shape and Preferred Correlation
            shape = "Linear" if delta_pearson_spearman <= TAU else "Asymptotic"
            preferred_correlation = "Pearson" if shape == "Linear" else "Spearman"

            # Format Plateau Reached column
            if plateau_found:
                if pd.notna(plateau) and np.isfinite(plateau):
                    # Format as integer if it's a whole number
                    plateau_int = (
                        int(plateau) if float(int(plateau)) == plateau else plateau
                    )
                    plateau_reached = f"Y ({plateau_int})"
                else:
                    plateau_reached = "Y"
            else:
                plateau_reached = "N"

            # Build row
            row = {
                "Metric": m,
                "Basis": basis,
                "Log": str(log),
                "Pearson Rho": pearson_rho,
                "Spearman Rho": spearman_rho,
                "Pearson P": pearson_p,
                "Spearman P": spearman_p,
                "n": int(n) if n is not None and pd.notna(n) else None,
                "N Pop": int(N_pop) if N_pop is not None and pd.notna(N_pop) else None,
                "Delta Pearson Spearman": delta_pearson_spearman,
                "Shape": shape,
                "Preferred Correlation": preferred_correlation,
                **relci,
                "Plateau n": plateau if np.isfinite(plateau) else np.nan,
                "Plateau Found": plateau_found,
                "Plateau Reached": plateau_reached,
                "Timestamp": datetime.now().isoformat(),
            }
            rows.append(row)

    if not rows:
        raise ValueError("No rows produced; check inputs and metric names.")

    # Build DataFrame with exact column order
    ref_cols = [f"RelCI {n}" for n in ref_sizes]
    cols_order = (
        [
            "Metric",
            "Basis",
            "Log",
            "Pearson Rho",
            "Spearman Rho",
            "Pearson P",
            "Spearman P",
            "n",
            "N Pop",
            "Delta Pearson Spearman",
            "Shape",
            "Preferred Correlation",
        ]
        + ref_cols
        + [
            "Plateau n",
            "Plateau Found",
            "Plateau Reached",
            "Timestamp",
        ]
    )

    table_df = pd.DataFrame(rows)
    table_df = table_df[[c for c in cols_order if c in table_df.columns]]

    # Round float columns to 10 decimal places in DataFrame to avoid precision issues
    for col in FLOAT_COLUMNS:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(10)

    # Calculate mean value rows
    numeric_cols = [
        c for c in table_df.columns if pd.api.types.is_numeric_dtype(table_df[c])
    ]
    mean_rows = []

    # Group by Metric and Basis, calculating means for numeric columns
    if len(table_df) > 0:
        means = (
            table_df.groupby(["Metric", "Basis"])[numeric_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        for _, row in means.iterrows():
            mean_row = row.to_dict()
            mean_row["Log"] = "MEAN"
            mean_row["Timestamp"] = datetime.now().isoformat()

            # Special handling for Plateau Found and Plateau n
            metric = mean_row["Metric"]
            basis = mean_row["Basis"]
            metric_basis_rows = table_df[
                (table_df["Metric"] == metric)
                & (table_df["Basis"] == basis)
                & (table_df["Log"] != "MEAN")
            ]

            # Special handling for Plateau Found: calculate share as "X / TOTAL"
            if (
                "Plateau Found" in metric_basis_rows.columns
                and len(metric_basis_rows) > 0
            ):
                # Count True values and total non-null values
                plateau_found_values = metric_basis_rows["Plateau Found"]
                # Filter out NaN values and count True values
                valid_values = plateau_found_values[plateau_found_values.notna()]
                total_count = len(valid_values)
                # Sum boolean values (True=1, False=0)
                found_count = int(valid_values.sum()) if total_count > 0 else 0
                mean_row["Plateau Found"] = (
                    f"{found_count} / {total_count}" if total_count > 0 else ""
                )
            else:
                mean_row["Plateau Found"] = ""

            # Special handling for Plateau Reached: for mean rows, use value from Plateau Found
            if "Plateau Found" in mean_row:
                mean_row["Plateau Reached"] = mean_row["Plateau Found"]
            else:
                mean_row["Plateau Reached"] = ""

            # Special handling for Plateau n: set to NaN if any value in the group is NaN
            if "Plateau n" in metric_basis_rows.columns and len(metric_basis_rows) > 0:
                plateau_n_values = metric_basis_rows["Plateau n"]
                # If any value is NaN, set mean to NaN
                if plateau_n_values.isna().any():
                    mean_row["Plateau n"] = np.nan

            # Set non-numeric columns appropriately
            for col in table_df.columns:
                if col not in mean_row:
                    # String columns should be left empty for MEAN rows (no aggregation)
                    mean_row[col] = ""

            mean_rows.append(mean_row)

    # Add mean rows to table_df, inserting each MEAN row immediately after its (Metric, Basis) group
    if mean_rows:
        mean_df = pd.DataFrame(mean_rows)
        # Ensure all columns from table_df are present in mean_df
        for col in table_df.columns:
            if col not in mean_df.columns:
                mean_df[col] = ""
        # Reorder columns to match table_df
        mean_df = mean_df[table_df.columns]

        # Insert each MEAN row immediately after the last per-log row for its (Metric, Basis) group
        # Preserve the original metric order from the input
        # Create a list to hold the final rows in order
        final_rows = []
        current_group = None

        # Iterate through rows preserving the original order
        for idx, row in table_df.iterrows():
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
        table_df = pd.DataFrame(final_rows)

    return table_df


def build_and_save_master_csv(
    combined_analysis_df: pd.DataFrame,
    out_csv_path: str,
) -> str:
    """Save a combined analysis DataFrame to CSV.

    Args:
        combined_analysis_df: DataFrame with all analysis data including per-log rows and MEAN rows.
        out_csv_path: Path to save the CSV file.

    Returns:
        Path to the saved CSV file.
    """
    from pathlib import Path

    if combined_analysis_df.empty:
        raise ValueError("combined_analysis_df is empty; cannot save CSV.")

    # Format floats as strings for CSV output to ensure fixed format
    def format_float(x: Any) -> str:
        if pd.isna(x) or not np.isfinite(x):
            return ""
        return f"{float(x):.10f}"

    csv_df = combined_analysis_df.copy()

    # Format float columns
    for col in FLOAT_COLUMNS:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(format_float)

    # Format n and N Pop as integers (or empty string if NaN)
    for col in ["n", "N Pop"]:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x) != "" else ""
            )

    # Save CSV
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_csv_path, index=False)

    return out_csv_path
