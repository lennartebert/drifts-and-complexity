from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .helpers import TAU

# Centralized header mapping for all outputs
HEADER_MAP = {
    "metric": "Metric",
    "basis": "Basis",
    "log": "Log",
    "Pearson_rho": "Pearson_Rho",
    "Pearson_p": "Pearson_P",
    "Spearman_rho": "Spearman_Rho",
    "Spearman_p": "Spearman_P",
    "delta_rho": "Delta_Rho",
    "RelCI_50": "RelCI_50",
    "RelCI_250": "RelCI_250",
    "RelCI_500": "RelCI_500",
    "plateau_n": "Plateau_n",
    "Pearson_improvement": "Pearson_Improvement",
    "Spearman_improvement": "Spearman_Improvement",
    # Add more if needed
}


def read_master_csv(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Read a master_table CSV and normalize headers to canonical column names.

    This maps friendly output headers (from HEADER_MAP) back to the canonical
    internal column names used throughout the code (e.g., 'Metric' -> 'metric').

    Args:
        path: Path to the CSV file (can be str or Path object).

    Returns:
        DataFrame with normalized column names, or None if file not found.
    """
    # Convert Path to string if needed
    path_str = str(path) if isinstance(path, Path) else path

    # Check if file exists first
    if not os.path.exists(path_str):
        return None

    try:
        df = pd.read_csv(path_str)
        inv_header_map = {v: k for k, v in HEADER_MAP.items()}
        try:
            return df.rename(columns=inv_header_map)
        except Exception:
            return df
    except FileNotFoundError:
        return None


def _escape_latex(text: Any) -> str:
    """Escape underscores in text, preserving \textit{...} wrappers."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if s.startswith("\\textit{") and s.endswith("}"):
        return s
    return s.replace("_", "\\_")


def build_and_save_master_csv(
    measures_per_log: Dict[str, pd.DataFrame],
    sample_ci_rel_width_per_log: Dict[str, pd.DataFrame],
    correlations_df: pd.DataFrame,  # columns: Metric, Pearson_Rho, Pearson_P, Spearman_Rho, Spearman_P
    plateau_df: pd.DataFrame,  # rows=metrics, cols=logs -> plateau n (or NaN)
    out_csv_path: str,
    metric_columns: Optional[List[str]] = None,
    ref_sizes: List[int] = [50, 250, 500],
    measure_basis_map: Optional[Dict[str, str]] = None,
    n: Optional[int] = None,  # sample size used for correlation
    N_pop: Optional[int] = None,  # population size (for FPC)
) -> str:
    """Build a master table CSV with all metrics, logs, and shape diagnostics.

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
    from datetime import datetime
    from pathlib import Path

    rows = []

    # Get all unique metrics from correlations_df
    all_metrics = correlations_df["Metric"].unique().tolist()
    if metric_columns is not None:
        all_metrics = [m for m in all_metrics if m in metric_columns]

    # Get all unique logs from measures_per_log
    all_logs = list(measures_per_log.keys())

    # Process each metric-log combination
    for m in all_metrics:
        # Get basis for this metric
        basis = measure_basis_map.get(m, "Unknown") if measure_basis_map else "Unknown"

        for log in all_logs:
            # Get correlation values for this metric-log combination
            metric_corr = correlations_df[(correlations_df["Metric"] == m)]

            if len(metric_corr) == 0:
                continue

            # Get correlation values (should be same for all logs for a given metric)
            pearson_rho = (
                metric_corr["Pearson_Rho"].iloc[0]
                if "Pearson_Rho" in metric_corr.columns
                else None
            )
            spearman_rho = (
                metric_corr["Spearman_Rho"].iloc[0]
                if "Spearman_Rho" in metric_corr.columns
                else None
            )
            pearson_p = (
                metric_corr["Pearson_P"].iloc[0]
                if "Pearson_P" in metric_corr.columns
                else None
            )
            spearman_p = (
                metric_corr["Spearman_P"].iloc[0]
                if "Spearman_P" in metric_corr.columns
                else None
            )

            # Get RelCI values for this log
            # sample_ci_rel_width_df has: columns=['sample_size', ...metric_names...]
            # We need to find the row where sample_size == ref_size, then get the metric column value
            relci = {}
            if log in sample_ci_rel_width_per_log:
                relci_df = sample_ci_rel_width_per_log[log]
                if (
                    relci_df is not None
                    and not relci_df.empty
                    and "sample_size" in relci_df.columns
                ):
                    for ref_size in ref_sizes:
                        col_name = f"RelCI_{ref_size}"
                        # Find row where sample_size matches ref_size
                        size_rows = relci_df[relci_df["sample_size"] == ref_size]
                        if len(size_rows) > 0 and m in relci_df.columns:
                            # Get the metric value for this sample_size
                            relci[col_name] = (
                                size_rows[m].iloc[0] if len(size_rows) > 0 else np.nan
                            )
                        else:
                            relci[col_name] = np.nan
                else:
                    for ref_size in ref_sizes:
                        relci[f"RelCI_{ref_size}"] = np.nan
            else:
                for ref_size in ref_sizes:
                    relci[f"RelCI_{ref_size}"] = np.nan

            # Get plateau n for this metric-log combination
            if m in plateau_df.index and log in plateau_df.columns:
                plateau = plateau_df.loc[m, log]
            else:
                plateau = np.nan

            # Compute Delta_PearsonSpearman
            delta_pearson_spearman = (
                abs(pearson_rho - spearman_rho)
                if (pearson_rho is not None and spearman_rho is not None)
                else np.nan
            )

            # Compute Shape and Preferred_Correlation
            shape = "Linear" if delta_pearson_spearman <= TAU else "Asymptotic"
            preferred_correlation = "Pearson" if shape == "Linear" else "Spearman"

            # Build row
            row = {
                "Metric": m,
                "Basis": basis,
                "Log": str(log),
                "Pearson_Rho": pearson_rho,
                "Spearman_Rho": spearman_rho,
                "Pearson_P": pearson_p,
                "Spearman_P": spearman_p,
                "n": int(n) if n is not None and pd.notna(n) else None,
                "N_pop": int(N_pop) if N_pop is not None and pd.notna(N_pop) else None,
                "Delta_PearsonSpearman": delta_pearson_spearman,
                "Shape": shape,
                "Preferred_Correlation": preferred_correlation,
                **relci,
                "Plateau_n": plateau if np.isfinite(plateau) else np.nan,
                "ts": datetime.now().isoformat(),
            }
            rows.append(row)

    if not rows:
        raise ValueError("No rows produced; check inputs and metric names.")

    # Build DataFrame with exact column order
    ref_cols = [f"RelCI_{n}" for n in ref_sizes]
    cols_order = (
        [
            "Metric",
            "Basis",
            "Log",
            "Pearson_Rho",
            "Spearman_Rho",
            "Pearson_P",
            "Spearman_P",
            "n",
            "N_pop",
            "Delta_PearsonSpearman",
            "Shape",
            "Preferred_Correlation",
        ]
        + ref_cols
        + [
            "Plateau_n",
            "ts",
        ]
    )

    table_df = pd.DataFrame(rows)
    table_df = table_df[[c for c in cols_order if c in table_df.columns]]

    # Round float columns to 10 decimal places in DataFrame to avoid precision issues
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
            mean_row["ts"] = datetime.now().isoformat()

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

    # Format floats as strings for CSV output to ensure fixed format
    def format_float(x: Any) -> str:
        if pd.isna(x) or not np.isfinite(x):
            return ""
        return f"{float(x):.10f}"

    csv_df = table_df.copy()
    for col in float_cols:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(format_float)

    # Format n and N_pop as integers (or empty string if NaN)
    for col in ["n", "N_pop"]:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x) != "" else ""
            )

    # Save CSV
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_csv_path, index=False)

    return out_csv_path


def write_means_only_table_from_master_csv(
    master_csv_path: str,
    means_csv_path: str,
    means_tex_path: str,
    caption: str = "Assessment (Means Across Logs)",
    label: str = "tab:master_table_means",
) -> None:
    """
    Create a compact 'means-only' table from the full master CSV produced earlier.

    - Keeps 'metric' and 'basis'
    - Drops the 'log' column entirely
    - Includes only rows where _is_mean == True
    - Formats all numeric values with exactly 4 decimals (no scientific notation)
    - Escapes underscores in headers and cell contents
    - Wraps LaTeX tabular in a full table environment
    """
    df = read_master_csv(master_csv_path)

    # Guard: file must exist and be readable
    if df is None:
        raise ValueError(
            f"Master CSV not found at {master_csv_path}. Please generate with the full function first."
        )

    # Guard: must have the mean-flag column
    if "_is_mean" not in df.columns:
        raise ValueError(
            "Master CSV is missing '_is_mean'. Please generate with the full function first."
        )

    # Map capitalized headers back to canonical names for processing
    inv_header_map = {v: k for k, v in HEADER_MAP.items()}
    df = df.rename(columns=inv_header_map)

    # Filter to mean rows only
    m = df[df["_is_mean"] == True].copy()

    # Keep columns: metric, basis, (drop log), then the numeric/result columns in original order
    # Identify likely result columns
    result_cols = [
        c for c in m.columns if c not in {"metric", "basis", "log", "_is_mean"}
    ]
    # Rebuild ordered columns
    cols = ["metric", "basis"] + result_cols
    m = m[cols]

    # ---- Format values: always 4 decimals for numeric, no exponent ----
    def _fmt_num(x: Any) -> str:
        if pd.isna(x):
            return ""
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    numeric_cols = [c for c in m.columns if pd.api.types.is_numeric_dtype(m[c])]
    for c in numeric_cols:
        m[c] = m[c].map(_fmt_num)

    # ---- Bold rho and p in means LaTeX if p < 0.05 for both Pearson and Spearman ----
    for corr_type in ["Pearson", "Spearman"]:
        p_col = f"{corr_type}_p"
        if p_col in m.columns:
            # use numeric p values before formatting where possible
            p_numeric = pd.to_numeric(
                m[p_col].map(lambda x: x if x != "" else np.nan), errors="coerce"
            )
            sig_mask = p_numeric.apply(
                lambda v: False if pd.isna(v) else (float(v) < 0.05)
            )
            for idx, is_sig in sig_mask.items():
                if not is_sig:
                    continue
                for col in (f"{corr_type}_rho", p_col):
                    if col in m.columns:
                        val = m.at[idx, col]
                        if val not in (None, "", np.nan):
                            # val is already a formatted string (4 decimals) — wrap with \textbf{}
                            m.at[idx, col] = f"\\textbf{{{val}}}"

    # ---- Bold improvement columns in means LaTeX if share_improved > 0.05 ----
    # Note: share_improved columns are not in CSV, but improvement column contains
    # the formatted share value (4 decimals) for mean rows
    for corr_type in ["Pearson", "Spearman"]:
        improvement_col = f"{corr_type}_improvement"
        if improvement_col not in m.columns:
            continue
        # For mean rows, improvement column contains formatted share value (e.g., "0.1234")
        # Parse it back to check if > 0.05
        for idx in m.index:
            val = m.at[idx, improvement_col]
            if val not in (None, "", np.nan) and val != "":
                try:
                    # Try to parse as float (works for formatted share values like "0.1234")
                    share_val = float(val)
                    if share_val > 0.05:
                        m.at[idx, improvement_col] = f"\\textbf{{{val}}}"
                except (ValueError, TypeError):
                    # If parsing fails, it might be "Yes" (per-log row) or empty, skip
                    pass

    # ---- Escape underscores in headers and in string cells ----
    # Escape cell contents in non-numeric columns
    for c in m.columns:
        if c not in numeric_cols:
            m[c] = m[c].map(_escape_latex)

    # Apply central header mapping and escape for LaTeX
    m.columns = [_escape_latex(HEADER_MAP.get(c, c)) for c in m.columns]

    # ---- Save CSV (plain, un-italicized, same column order) ----
    # Read the master CSV (headers are likely mapped to HEADER_MAP); map back
    # to canonical names for processing, then re-apply HEADER_MAP for output.
    csv_plain = read_master_csv(master_csv_path)
    if csv_plain is None:
        raise ValueError(
            f"Master CSV not found at {master_csv_path}. Please generate with the full function first."
        )
    csv_means = csv_plain[csv_plain["_is_mean"] == True].copy()
    csv_means = csv_means.drop(columns=["log", "_is_mean"], errors="ignore")
    csv_cols = ["metric", "basis"] + [
        c for c in csv_means.columns if c not in {"metric", "basis"}
    ]
    csv_means = csv_means[csv_cols]
    # Re-apply header mapping so the output CSV has the friendly headers
    csv_means.columns = [HEADER_MAP.get(c, c) for c in csv_means.columns]
    csv_means.to_csv(means_csv_path, index=False)

    # ---- Build LaTeX table ----
    tabular_str = m.to_latex(index=False, escape=False, na_rep="")
    table_env = (
        "\\begin{table}[Ht]\n"
        "\\centering\n"
        f"\\caption{{{_escape_latex(caption)}}}\n"
        "\\tiny\n"
        f"{tabular_str}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    with open(means_tex_path, "w", encoding="utf-8") as f:
        f.write(table_env)


def write_latex_master_tables(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate LaTeX tables from master.csv.

    Produces two tables:
    - master_full_<scenario_key>.tex (all rows including MEAN)
    - master_means_<scenario_key>.tex (MEAN rows only)

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX files.
        scenario_key: Key for table labels (e.g., "test").
        scenario_title: Title for table captions (e.g., "Test Scenario").
    """

    # Read CSV
    df = pd.read_csv(master_csv_path)

    # Ensure MEAN rows are at the end for each (Metric, Basis) group
    # Create a sort key: 0 for regular rows, 1 for MEAN rows
    df["_sort_key"] = df["Log"].apply(lambda x: 1 if x == "MEAN" else 0)
    df = df.sort_values(by=["Metric", "Basis", "_sort_key", "Log"])
    df = df.drop(columns=["_sort_key"])

    # Format numeric columns
    def format_num(x: Any) -> str:
        if pd.isna(x) or x == "":
            return ""
        try:
            val = float(x)
            if not np.isfinite(val):
                return ""
            return f"{val:.4f}"
        except (ValueError, TypeError):
            return str(x)

    # Helper to create LaTeX table rows
    def create_table_rows(
        df_subset: pd.DataFrame, is_means_only: bool = False
    ) -> List[str]:
        rows = []
        prev_metric = None
        prev_basis = None

        for idx, row in df_subset.iterrows():
            metric = row["Metric"]
            basis = row["Basis"]
            log = row["Log"]
            is_mean = log == "MEAN"

            # Determine if we should repeat Metric and Basis
            # Only repeat if both Metric and Basis are the same as previous row
            repeat_metric = (
                (metric == prev_metric) if prev_metric is not None else False
            )
            repeat_basis = (basis == prev_basis) if prev_basis is not None else False

            # Format values
            pearson_rho = format_num(row.get("Pearson_Rho", ""))
            spearman_rho = format_num(row.get("Spearman_Rho", ""))
            delta_pearson_spearman = format_num(row.get("Delta_PearsonSpearman", ""))
            shape = str(row.get("Shape", "")) if not is_mean else ""
            preferred_correlation = (
                str(row.get("Preferred_Correlation", "")) if not is_mean else ""
            )

            # Escape LaTeX special characters
            metric_str = _escape_latex(metric) if not repeat_metric else ""
            basis_str = _escape_latex(basis) if not repeat_basis else ""
            log_str = "" if is_mean else _escape_latex(log)
            shape_str = _escape_latex(shape)
            preferred_correlation_str = _escape_latex(preferred_correlation)

            # Build row
            if is_means_only:
                # Means only table: Metric, Basis, Pearson_Rho, Spearman_Rho, Delta_PearsonSpearman
                row_str = f"{metric_str} & {basis_str} & {pearson_rho} & {spearman_rho} & {delta_pearson_spearman} \\\\"
            else:
                # Full table: Metric, Basis, Log, Pearson_Rho, Spearman_Rho, Delta_PearsonSpearman, Shape, Preferred_Correlation
                row_str = f"{metric_str} & {basis_str} & {log_str} & {pearson_rho} & {spearman_rho} & {delta_pearson_spearman} & {shape_str} & {preferred_correlation_str} \\\\"

            # Italicize MEAN rows
            if is_mean:
                row_str = f"\\textit{{{row_str}}}"

            rows.append(row_str)
            prev_metric = metric
            prev_basis = basis

        return rows

    # Create full table
    full_rows = create_table_rows(df, is_means_only=False)
    full_header = "Metric & Basis & Log & Pearson\\_Rho & Spearman\\_Rho & Delta\\_PearsonSpearman & Shape & Preferred\\_Correlation \\\\"

    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Master table — {_escape_latex(scenario_title)}}}
\\label{{tab:master_full_{scenario_key}}}
\\begin{{tabular}}{{l l l c c c l l}}
\\toprule
{full_header}
\\midrule
{chr(10).join(full_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Create means-only table
    means_df = df[df["Log"] == "MEAN"].copy()
    means_rows = create_table_rows(means_df, is_means_only=True)
    means_header = (
        "Metric & Basis & Pearson\\_Rho & Spearman\\_Rho & Delta\\_PearsonSpearman \\\\"
    )

    means_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Master means — {_escape_latex(scenario_title)}}}
\\label{{tab:master_means_{scenario_key}}}
\\begin{{tabular}}{{l l c c c}}
\\toprule
{means_header}
\\midrule
{chr(10).join(means_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Save files
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    full_path = out_path / f"master_full_{scenario_key}.tex"
    means_path = out_path / f"master_means_{scenario_key}.tex"

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(full_latex)

    with open(means_path, "w", encoding="utf-8") as f:
        f.write(means_latex)


def write_latex_comparison_tables(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate LaTeX tables from metrics_comparison.csv.

    Produces two tables:
    - comparison_full_<scenario_key>.tex (all rows including MEAN)
    - comparison_means_<scenario_key>.tex (MEAN rows only)

    Args:
        comparison_csv_path: Path to metrics_comparison.csv file.
        out_dir: Directory to save LaTeX files.
        scenario_key: Key for table labels (e.g., "test").
        scenario_title: Title for table captions (e.g., "Test Scenario").
    """

    # Read CSV
    df = pd.read_csv(comparison_csv_path)

    # Ensure MEAN rows are at the end for each (Metric, Basis) group
    # Create a sort key: 0 for regular rows, 1 for MEAN rows
    df["_sort_key"] = df["Log"].apply(lambda x: 1 if x == "MEAN" else 0)
    df = df.sort_values(by=["Metric", "Basis", "_sort_key", "Log"])
    df = df.drop(columns=["_sort_key"])

    # Format numeric columns
    def format_num(x: Any) -> str:
        if pd.isna(x) or x == "":
            return ""
        try:
            val = float(x)
            if not np.isfinite(val):
                return ""
            return f"{val:.4f}"
        except (ValueError, TypeError):
            return str(x)

    # Format Significant_Improvement
    def format_sig_improvement(val: Any, is_mean: bool) -> str:
        if pd.isna(val) or val == "":
            return ""
        if is_mean:
            # MEAN rows: convert float to percentage
            try:
                float_val = float(val)
                if np.isfinite(float_val):
                    return f"{float_val * 100:.1f}\\%"
            except (ValueError, TypeError):
                pass
            return ""
        else:
            # Per-log rows: convert TRUE/FALSE to Yes/No
            val_str = str(val).upper()
            if val_str == "TRUE":
                return "Yes"
            elif val_str == "FALSE":
                return "No"
            return str(val)

    # Helper to create LaTeX table rows
    def create_table_rows(
        df_subset: pd.DataFrame, is_means_only: bool = False
    ) -> List[str]:
        rows = []
        prev_metric = None
        prev_basis = None

        for idx, row in df_subset.iterrows():
            metric = row["Metric"]
            basis = row["Basis"]
            log = row["Log"]
            is_mean = log == "MEAN"

            # Determine if we should repeat Metric and Basis
            # Only repeat if both Metric and Basis are the same as previous row
            repeat_metric = (
                (metric == prev_metric) if prev_metric is not None else False
            )
            repeat_basis = (basis == prev_basis) if prev_basis is not None else False

            # Format values
            shape_before = str(row.get("Shape_before", "")) if not is_mean else ""
            shape_after = str(row.get("Shape_after", "")) if not is_mean else ""
            preferred_correlation_before = (
                str(row.get("Preferred_Correlation_before", "")) if not is_mean else ""
            )
            preferred_correlation_after = (
                str(row.get("Preferred_Correlation_after", "")) if not is_mean else ""
            )
            chosen_correlation = str(row.get("Chosen_Correlation", ""))
            chosen_rho_before = format_num(row.get("Chosen_Rho_before", ""))
            chosen_rho_after = format_num(row.get("Chosen_Rho_after", ""))
            delta_chosen_rho = format_num(row.get("Delta_Chosen_Rho", ""))
            z_test_p = format_num(row.get("Z_test_p", ""))
            significant_improvement = format_sig_improvement(
                row.get("Significant_Improvement", ""), is_mean
            )

            # Escape LaTeX special characters
            metric_str = _escape_latex(metric) if not repeat_metric else ""
            basis_str = _escape_latex(basis) if not repeat_basis else ""
            log_str = "" if is_mean else _escape_latex(log)
            shape_before_str = _escape_latex(shape_before)
            shape_after_str = _escape_latex(shape_after)
            preferred_correlation_before_str = _escape_latex(
                preferred_correlation_before
            )
            preferred_correlation_after_str = _escape_latex(preferred_correlation_after)
            chosen_correlation_str = _escape_latex(chosen_correlation)

            # Build row
            if is_means_only:
                # Means only table: Metric, Basis, Chosen_Rho_before, Chosen_Rho_after, Delta_Chosen_Rho, Significant_Improvement
                row_str = f"{metric_str} & {basis_str} & {chosen_rho_before} & {chosen_rho_after} & {delta_chosen_rho} & {significant_improvement} \\\\"
            else:
                # Full table: Metric, Basis, Log, Shape_before, Shape_after, Preferred_Correlation_before, Preferred_Correlation_after, Chosen_Correlation, Chosen_Rho_before, Chosen_Rho_after, Delta_Chosen_Rho, Z_test_p, Significant_Improvement
                row_str = f"{metric_str} & {basis_str} & {log_str} & {shape_before_str} & {shape_after_str} & {preferred_correlation_before_str} & {preferred_correlation_after_str} & {chosen_correlation_str} & {chosen_rho_before} & {chosen_rho_after} & {delta_chosen_rho} & {z_test_p} & {significant_improvement} \\\\"

            # Italicize MEAN rows
            if is_mean:
                row_str = f"\\textit{{{row_str}}}"

            rows.append(row_str)
            prev_metric = metric
            prev_basis = basis

        return rows

    # Create full table
    full_rows = create_table_rows(df, is_means_only=False)
    full_header = "Metric & Basis & Log & Shape\\_before & Shape\\_after & Preferred\\_Correlation\\_before & Preferred\\_Correlation\\_after & Chosen\\_Correlation & Chosen\\_Rho\\_before & Chosen\\_Rho\\_after & Delta\\_Chosen\\_Rho & Z\\_test\\_p & Significant\\_Improvement \\\\"

    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Comparison table — {_escape_latex(scenario_title)}}}
\\label{{tab:comparison_full_{scenario_key}}}
\\begin{{tabular}}{{l l l l l l l l c c c c c}}
\\toprule
{full_header}
\\midrule
{chr(10).join(full_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Create means-only table
    means_df = df[df["Log"] == "MEAN"].copy()
    means_rows = create_table_rows(means_df, is_means_only=True)
    means_header = "Metric & Basis & Chosen\\_Rho\\_before & Chosen\\_Rho\\_after & Delta\\_Chosen\\_Rho & Significant\\_Improvement \\\\"

    means_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Comparison means — {_escape_latex(scenario_title)}}}
\\label{{tab:comparison_means_{scenario_key}}}
\\begin{{tabular}}{{l l c c c c}}
\\toprule
{means_header}
\\midrule
{chr(10).join(means_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Save files
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    full_path = out_path / f"comparison_full_{scenario_key}.tex"
    means_path = out_path / f"comparison_means_{scenario_key}.tex"

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(full_latex)

    with open(means_path, "w", encoding="utf-8") as f:
        f.write(means_latex)
