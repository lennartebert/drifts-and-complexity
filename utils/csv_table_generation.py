"""CSV table generation for master and comparison tables."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .comparison_table import (
    _add_comparison_columns,
    _reorder_comparison_columns,
)
from .comparison_table import build_comparison_table_csv as _build_comparison_table_csv
from .helpers import TAU
from .master_table import build_master_table_csv as _build_master_table_csv


def generate_master_csv(
    measures_per_log: Dict[str, pd.DataFrame],
    sample_ci_rel_width_per_log: Dict[str, pd.DataFrame],
    correlations_df: pd.DataFrame,
    plateau_df: pd.DataFrame,
    out_csv_path: str,
    metric_columns: Optional[List[str]] = None,
    ref_sizes: List[int] = [50, 250, 500],
    measure_basis_map: Optional[Dict[str, str]] = None,
    n: Optional[int] = None,
    N_pop: Optional[int] = None,
) -> str:
    """Generate master.csv for a scenario.

    This is a wrapper around build_master_table_csv that provides
    a consistent interface for CSV generation.

    Args:
        measures_per_log: Dict mapping log names to DataFrames with measures.
        sample_ci_rel_width_per_log: Dict mapping log names to DataFrames with RelCI values.
        correlations_df: DataFrame with columns: Metric, Pearson_Rho, Pearson_P, Spearman_Rho, Spearman_P.
        plateau_df: DataFrame with metrics as rows, logs as columns, values are plateau n (or NaN).
        out_csv_path: Path to save the CSV file.
        metric_columns: Optional list of metric names to include.
        ref_sizes: List of reference sizes for RelCI columns.
        measure_basis_map: Optional dict mapping metric names to basis names.
        n: Sample size used for correlation.
        N_pop: Population size (for FPC).

    Returns:
        Path to the saved CSV file.
    """
    return _build_master_table_csv(
        measures_per_log=measures_per_log,
        sample_ci_rel_width_per_log=sample_ci_rel_width_per_log,
        correlations_df=correlations_df,
        plateau_df=plateau_df,
        out_csv_path=out_csv_path,
        metric_columns=metric_columns,
        ref_sizes=ref_sizes,
        measure_basis_map=measure_basis_map,
        n=n,
        N_pop=N_pop,
    )


def generate_comparison_csv(
    before_csv_path: str,
    after_csv_path: str,
    out_csv_path: str,
) -> str:
    """Generate metrics_comparison.csv from before and after master CSVs.

    This extends build_comparison_table_csv to add delta columns for
    RelCI and Plateau_n values.

    Args:
        before_csv_path: Path to before scenario master table CSV.
        after_csv_path: Path to after scenario master table CSV.
        out_csv_path: Path to save the comparison CSV.

    Returns:
        Path to the saved CSV file.
    """
    # First build the comparison table using the existing function
    comparison_path = _build_comparison_table_csv(
        before_csv_path=before_csv_path,
        after_csv_path=after_csv_path,
        out_csv_path=out_csv_path,
    )

    # Read the comparison CSV to add delta columns
    df = pd.read_csv(comparison_path)

    # Convert string floats back to numeric for computation
    def convert_to_numeric(col: str) -> None:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert RelCI and Plateau columns to numeric
    for col in [
        "RelCI_50_before",
        "RelCI_250_before",
        "RelCI_500_before",
        "Plateau_n_before",
        "RelCI_50_after",
        "RelCI_250_after",
        "RelCI_500_after",
        "Plateau_n_after",
    ]:
        convert_to_numeric(col)

    # Compute delta columns
    df["RelCI_50_delta"] = df["RelCI_50_after"] - df["RelCI_50_before"]
    df["RelCI_250_delta"] = df["RelCI_250_after"] - df["RelCI_250_before"]
    df["RelCI_500_delta"] = df["RelCI_500_after"] - df["RelCI_500_before"]
    df["Plateau_n_delta"] = df["Plateau_n_after"] - df["Plateau_n_before"]

    # Round delta columns to 10 decimal places
    for col in [
        "RelCI_50_delta",
        "RelCI_250_delta",
        "RelCI_500_delta",
        "Plateau_n_delta",
    ]:
        if col in df.columns:
            df[col] = df[col].round(10)

    # Format delta columns as strings for CSV output
    def format_float(x: Any) -> str:
        if pd.isna(x) or not np.isfinite(x):
            return ""
        return f"{float(x):.10f}"

    for col in [
        "RelCI_50_delta",
        "RelCI_250_delta",
        "RelCI_500_delta",
        "Plateau_n_delta",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(format_float)

    # Reorder columns to include delta columns after the corresponding after columns
    # Get the current column order from _reorder_comparison_columns
    df = _reorder_comparison_columns(df)

    # Insert delta columns after their corresponding after columns
    cols = list(df.columns)

    # Find positions to insert delta columns
    delta_insertions = [
        ("RelCI_50_after", "RelCI_50_delta"),
        ("RelCI_250_after", "RelCI_250_delta"),
        ("RelCI_500_after", "RelCI_500_delta"),
        ("Plateau_n_after", "Plateau_n_delta"),
    ]

    for after_col, delta_col in delta_insertions:
        if after_col in cols and delta_col in cols:
            # Remove delta_col from its current position
            if delta_col in cols:
                cols.remove(delta_col)
            # Insert delta_col right after after_col
            after_idx = cols.index(after_col)
            cols.insert(after_idx + 1, delta_col)

    df = df[cols]

    # Save updated CSV
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    return out_csv_path
