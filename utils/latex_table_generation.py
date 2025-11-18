"""LaTeX table generation from CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .constants import (
    BASIS_ORDER,
    COLUMN_NAMES_TO_LATEX_MAP,
    METRIC_BASIS_MAP,
    METRIC_NAMES_TO_LATEX_MAP,
)


def _escape_latex(text: Any) -> str:
    """Escape special LaTeX characters in text, preserving \textit{...} wrappers.

    Escapes:
    - Underscores: _ → \_
    - Hash signs: # → \# (but not if already escaped as \#)
    - Percent signs: % → \% (but not if already escaped as \%)
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if s.startswith("\\textit{") and s.endswith("}"):
        return s
    # Escape special LaTeX characters, but avoid double-escaping
    # Replace in reverse order to avoid interfering with already-escaped sequences
    s = s.replace("_", "\\_")
    # Escape # and % only if not already escaped
    # Use a temporary marker to avoid double-escaping
    s = s.replace("\\#", "__TEMP_HASH__")
    s = s.replace("\\%", "__TEMP_PERCENT__")
    s = s.replace("#", "\\#")
    s = s.replace("%", "\\%")
    s = s.replace("__TEMP_HASH__", "\\#")
    s = s.replace("__TEMP_PERCENT__", "\\%")
    return s


def _format_num(x: Any, decimals: int = 4) -> str:
    """Format numeric value with specified decimal places."""
    if pd.isna(x) or x == "":
        return ""
    try:
        val = float(x)
        if not np.isfinite(val):
            return ""
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(x)


def _format_pvalue(x: Any) -> str:
    """Format p-value with <0.001 rule."""
    if pd.isna(x) or x == "":
        return ""
    try:
        val = float(x)
        if not np.isfinite(val):
            return ""
        if val < 0.001:
            return "<0.001"
        return f"{val:.3f}"
    except (ValueError, TypeError):
        return str(x)


def _format_boolean(x: Any) -> str:
    """Format boolean value as T or F."""
    if pd.isna(x) or x == "":
        return ""
    val_str = str(x).upper().strip()
    if val_str in ("TRUE", "T", "1", "YES", "Y"):
        return "T"
    elif val_str in ("FALSE", "F", "0", "NO", "N"):
        return "F"
    return str(x)


def _build_header_and_colspec(
    columns: List[str], metric_column_width: str = "100pt"
) -> tuple[str, str]:
    """Build LaTeX header and column specification from column list.

    Uses COLUMN_NAMES_TO_LATEX_MAP for header display names (not escaped).
    Column spec uses p{metric_column_width} for Metric column, appropriate types for others.

    Args:
        columns: List of CSV column names.
        metric_column_width: Width for the Metric column (e.g., "100pt").

    Returns:
        Tuple of (header_string, column_spec_string).
    """
    header_parts = []
    colspec_parts = []

    for col in columns:
        # Use COLUMN_NAMES_TO_LATEX_MAP if available, otherwise escape the column name
        if col in COLUMN_NAMES_TO_LATEX_MAP:
            header_part = COLUMN_NAMES_TO_LATEX_MAP[
                col
            ]  # Already formatted, don't escape
        else:
            header_part = col.replace(" ", "\\ ").replace("_", "\\_")
        header_parts.append(header_part)

        # Build column spec
        if col == "Metric":
            colspec_parts.append(f"p{{{metric_column_width}}}")
        elif col in [
            "Basis",
            "Log",
            "Shape",
            "Preferred Correlation",
            "Shape Before",
            "Shape After",
            "Preferred Correlation Before",
            "Preferred Correlation After",
            "Chosen Correlation",
        ]:
            colspec_parts.append("l")
        elif col.endswith(" P") or col == "Z Test P":
            colspec_parts.append("c")
        elif col in [
            "Pearson Rho",
            "Spearman Rho",
            "Delta Pearson Spearman",
            "Chosen Rho Before",
            "Chosen Rho After",
            "Abs Delta Chosen Rho",
            "Delta Pearson",
            "Delta Spearman",
            "Z Test Stat",
            "RelCI 50",
            "RelCI 250",
            "RelCI 500",
            "Plateau n",
            "Plateau Reached",
            "RelCI 50 Before",
            "RelCI 50 After",
            "RelCI 50 Delta",
            "RelCI 250 Before",
            "RelCI 250 After",
            "RelCI 250 Delta",
            "RelCI 500 Before",
            "RelCI 500 After",
            "RelCI 500 Delta",
            "Plateau n Before",
            "Plateau n After",
            "Plateau n Delta",
            "Significant Improvement",
        ]:
            colspec_parts.append("c")
        else:
            # Default to left-aligned for unknown columns
            colspec_parts.append("l")

    header = " & ".join(header_parts) + " \\\\"
    colspec = " ".join(colspec_parts)

    return header, colspec


def _sort_dataframe_by_basis(
    df: pd.DataFrame, order_by_basis: bool = True
) -> pd.DataFrame:
    """Sort dataframe by Basis (OC, PC, PD, CT), then metric order, then MEAN rows last.

    When order_by_basis is True, Basis is put first in the sort order.
    When order_by_basis is False, metric order comes first.

    Args:
        df: DataFrame to sort.
        order_by_basis: If True, sort by Basis first. If False, sort by metric order first.

    Returns:
        Sorted DataFrame.
    """
    df = df.copy()

    # Create basis order mapping
    basis_order_map = {basis: idx for idx, basis in enumerate(BASIS_ORDER)}
    df["_basis_order"] = df["Basis"].map(
        lambda x: basis_order_map.get(x, 999)
    )  # Unknown basis goes to end

    # Create metric order mapping to preserve CSV order
    unique_metrics = df["Metric"].unique()
    metric_order = {metric: idx for idx, metric in enumerate(unique_metrics)}
    df["_metric_order"] = df["Metric"].map(metric_order)

    # Sort key: MEAN rows at end for each group
    df["_sort_key"] = df["Log"].apply(lambda x: 1 if x == "MEAN" else 0)

    if order_by_basis:
        # Sort by Basis first, then metric order, then MEAN rows last
        df = df.sort_values(by=["_basis_order", "_metric_order", "_sort_key", "Log"])
    else:
        # Sort by metric order first, then Basis, then MEAN rows last
        df = df.sort_values(by=["_metric_order", "_basis_order", "_sort_key", "Log"])

    # Drop temporary columns
    df = df.drop(columns=["_basis_order", "_metric_order", "_sort_key"])

    return df


def _break_measure_name_for_makecell(measure_name: str) -> str:
    """Break measure name into multiple lines for \makecell.

    Splits on spaces to create a reasonable line break. Uses \\\\ for line breaks in \makecell.

    Args:
        measure_name: The measure name to break (already LaTeX-escaped).

    Returns:
        Measure name with \\\\ line breaks for \makecell (will render as \\ in LaTeX).
    """
    # Split on spaces and join with \\\\ for \makecell line breaks
    # Try to break at natural points (after 2-3 words)
    words = measure_name.split()
    if len(words) <= 3:
        # Short names don't need breaking
        return measure_name
    # Break after first 2-3 words
    mid_point = min(3, len(words) // 2 + 1)
    first_line = " ".join(words[:mid_point])
    second_line = " ".join(words[mid_point:])
    # Use \\\\ to produce \\ in LaTeX output (line break in \makecell)
    return f"{first_line}\\\\{second_line}"


def _create_table_rows(
    df_subset: pd.DataFrame,
    columns: List[str],
    is_means_only: bool = False,
    metric_column_width: str = "100pt",
) -> List[str]:
    """Create LaTeX table rows with collapsing Metric/Basis and italicizing MEAN rows.

    Uses \multirow and \makecell for measures that span multiple rows.

    Args:
        df_subset: DataFrame subset to process.
        columns: List of column names to include.
        is_means_only: If True, this is a means-only table (no italicization).
        metric_column_width: Width for the Metric column (e.g., "100pt").

    Returns:
        List of LaTeX row strings.
    """
    rows = []
    prev_metric = None
    prev_basis = None

    # Count rows per metric (for \multirow)
    metric_row_counts = {}
    if "Metric" in df_subset.columns:
        metric_row_counts = df_subset["Metric"].value_counts().to_dict()

    # Identify numeric and boolean columns (for formatting)
    numeric_cols = set()
    boolean_cols = set()
    for col in df_subset.columns:
        if pd.api.types.is_numeric_dtype(df_subset[col]):
            numeric_cols.add(col)
        elif pd.api.types.is_bool_dtype(df_subset[col]):
            boolean_cols.add(col)

    for idx, row in df_subset.iterrows():
        metric = row["Metric"]
        basis = row["Basis"]
        log = row["Log"]
        is_mean = log == "MEAN"

        # Determine if we should repeat Metric and Basis
        repeat_metric = (metric == prev_metric) if prev_metric is not None else False
        # Only collapse basis if it repeats in consecutive rows (same as previous)
        repeat_basis = (basis == prev_basis) if prev_basis is not None else False

        # Build row cells
        cells = []
        for col in columns:
            skip_escape = False  # Reset for each column
            if col == "Metric":
                # Use METRIC_NAMES_TO_LATEX_MAP if available, otherwise use original metric name
                metric_display = METRIC_NAMES_TO_LATEX_MAP.get(metric, metric)
                metric_display_escaped = _escape_latex(metric_display)

                if repeat_metric:
                    cell = ""
                else:
                    # Check if this measure spans multiple rows
                    num_rows = metric_row_counts.get(metric, 1)
                    if num_rows > 1:
                        # Use \multirow and \makecell
                        measure_broken = _break_measure_name_for_makecell(
                            metric_display_escaped
                        )
                        cell = f"\\multirow[t]{{{num_rows}}}{{{metric_column_width}}}{{\\makecell[l]{{{measure_broken}}}}}"
                        skip_escape = True  # Don't escape LaTeX commands
                    else:
                        # Single row, no need for \multirow
                        cell = metric_display_escaped
            elif col == "Basis":
                cell = _escape_latex(basis) if not repeat_basis else ""
            elif col == "Log":
                cell = "" if is_mean else _escape_latex(log)
            elif col == "Plateau Reached":
                # Special formatting for Plateau Reached column
                # Use the value directly from CSV (already formatted correctly)
                plateau_reached_val = row.get("Plateau Reached", "")
                if plateau_reached_val:
                    cell = str(plateau_reached_val)
                elif is_mean:
                    # Fallback for mean rows: use Plateau Found value if Plateau Reached is missing
                    plateau_found_val = row.get("Plateau Found", "")
                    cell = str(plateau_found_val) if plateau_found_val else ""
                else:
                    # Fallback for non-mean rows: compute from Plateau Found and Plateau n
                    plateau_found = row.get("Plateau Found", False)
                    plateau_n = row.get("Plateau n", np.nan)

                    # Convert plateau_found to boolean if needed
                    if isinstance(plateau_found, str):
                        plateau_found = plateau_found.upper().strip() in (
                            "TRUE",
                            "T",
                            "1",
                            "YES",
                            "Y",
                        )

                    if plateau_found:
                        # Format plateau_n if available
                        if pd.notna(plateau_n) and plateau_n != "":
                            try:
                                # Try to format as integer if it's a whole number
                                plateau_n_float = float(plateau_n)
                                if plateau_n_float == int(plateau_n_float):
                                    plateau_n_str = str(int(plateau_n_float))
                                else:
                                    plateau_n_str = f"{plateau_n_float:.0f}"
                                cell = f"Y ({plateau_n_str})"
                            except (ValueError, TypeError):
                                cell = "Y"
                        else:
                            cell = "Y"
                    else:
                        cell = "N"
                # Don't escape LaTeX for this formatted string
                skip_escape = True
            else:
                # Get value from row
                val = row.get(col, "")
                if pd.isna(val) or val == "":
                    cell = ""
                elif (
                    col.endswith("_p")
                    or col == "Z_test_p"
                    or col.endswith(" P")
                    or col == "Z Test P"
                ):
                    cell = _format_pvalue(val)
                elif (
                    col in boolean_cols
                    or col == "Significant_Improvement"
                    or col == "Significant Improvement"
                ):
                    # Format booleans as T/F
                    cell = _format_boolean(val)
                elif col in numeric_cols:
                    # Try to format as numeric - handle both numeric and string numeric values
                    try:
                        # First try to convert to float if it's a string
                        if isinstance(val, str) and val.strip() != "":
                            float_val = float(val)
                            cell = _format_num(float_val)
                        else:
                            cell = _format_num(val)
                    except (ValueError, TypeError):
                        # If conversion fails, treat as string
                        cell = str(val)
                else:
                    # String column - just escape LaTeX
                    cell = str(val)

                # Escape LaTeX for non-empty cells (unless already formatted or special column)
                if cell != "" and not skip_escape:
                    cell = _escape_latex(cell)

            # For MEAN rows, italicize each cell individually (only in full tables, not means-only)
            if is_mean and not is_means_only:
                if cell == "":
                    # Empty cell - use \textit{} to maintain column count
                    cell = "\\textit{}"
                else:
                    # Non-empty cell - wrap content in \textit{}
                    cell = f"\\textit{{{cell}}}"

            cells.append(cell)

        # Build row string
        row_str = " & ".join(cells) + " \\\\"
        rows.append(row_str)
        prev_metric = metric
        prev_basis = basis

    return rows


def _create_latex_table(
    rows: List[str],
    header: str,
    colspec: str,
    caption: str,
    label: str,
) -> str:
    """Create LaTeX table string with standardized formatting.

    Args:
        rows: List of LaTeX row strings.
        header: LaTeX header string.
        colspec: LaTeX column specification string.
        caption: Table caption.
        label: Table label.

    Returns:
        Complete LaTeX table string.
    """
    return f"""\\begin{{table}}[H]
\\centering
\\tiny
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{colspec}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def write_full_latex_table(
    csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
    include_columns: List[str],
    label_extension: str,
    caption_extension: str,
    filename: str,
    order_by_basis: bool = True,
) -> None:
    """Generate full LaTeX table from CSV file.

    Args:
        csv_path: Path to CSV file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
        include_columns: List of column names to include (whitelist).
        label_extension: Extension for table label (e.g., "master_correlation_full").
        caption_extension: Human-readable caption extension (e.g., "Correlation table").
        filename: Output filename (e.g., "master_correlation_full.tex").
        order_by_basis: If True, sort by Basis first and put Basis as first column.
    """
    df = pd.read_csv(csv_path)

    # Filter to only include specified columns (if they exist)
    available_columns = [col for col in include_columns if col in df.columns]
    if not available_columns:
        raise ValueError(f"No valid columns found in CSV: {include_columns}")

    # When ordering by basis, put Basis first in column order
    if (
        order_by_basis
        and "Basis" in available_columns
        and "Metric" in available_columns
    ):
        # Remove Basis and Metric from their current positions
        other_cols = [
            col for col in available_columns if col not in ["Basis", "Metric"]
        ]
        # Put Basis first, then Metric, then others
        available_columns = ["Basis", "Metric"] + other_cols

    # Sort dataframe
    df = _sort_dataframe_by_basis(df, order_by_basis=order_by_basis)

    # Create table rows
    rows = _create_table_rows(
        df, available_columns, is_means_only=False, metric_column_width="100pt"
    )

    # Build header and column spec
    header, colspec = _build_header_and_colspec(
        available_columns, metric_column_width="100pt"
    )

    # Create caption and label
    caption = f"{caption_extension} — {_escape_latex(scenario_title)}"
    label = f"tab:{label_extension}_{scenario_key}"

    # Generate LaTeX
    latex = _create_latex_table(rows, header, colspec, caption, label)

    # Write to file
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / filename, "w", encoding="utf-8") as f:
        f.write(latex)


def write_means_latex_table(
    csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
    include_columns: List[str],
    label_extension: str,
    caption_extension: str,
    filename: str,
    order_by_basis: bool = True,
) -> None:
    """Generate means-only LaTeX table from CSV file.

    Args:
        csv_path: Path to CSV file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
        include_columns: List of column names to include (whitelist).
        label_extension: Extension for table label (e.g., "master_correlation_means").
        caption_extension: Human-readable caption extension (e.g., "Correlation means").
        filename: Output filename (e.g., "master_correlation_means.tex").
        order_by_basis: If True, sort by Basis first and put Basis as first column.
    """
    df = pd.read_csv(csv_path)

    # Filter to MEAN rows only
    means_df = df[df["Log"] == "MEAN"].copy()

    # Filter to only include specified columns (if they exist)
    available_columns = [col for col in include_columns if col in means_df.columns]
    if not available_columns:
        raise ValueError(f"No valid columns found in CSV: {include_columns}")

    # When ordering by basis, put Basis first in column order
    if (
        order_by_basis
        and "Basis" in available_columns
        and "Metric" in available_columns
    ):
        # Remove Basis and Metric from their current positions
        other_cols = [
            col for col in available_columns if col not in ["Basis", "Metric"]
        ]
        # Put Basis first, then Metric, then others
        available_columns = ["Basis", "Metric"] + other_cols

    # Sort dataframe
    means_df = _sort_dataframe_by_basis(means_df, order_by_basis=order_by_basis)

    # Create table rows
    rows = _create_table_rows(
        means_df, available_columns, is_means_only=True, metric_column_width="100pt"
    )

    # Build header and column spec
    header, colspec = _build_header_and_colspec(
        available_columns, metric_column_width="100pt"
    )

    # Create caption and label
    caption = f"{caption_extension} — {_escape_latex(scenario_title)}"
    label = f"tab:{label_extension}_{scenario_key}"

    # Generate LaTeX
    latex = _create_latex_table(rows, header, colspec, caption, label)

    # Write to file
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / filename, "w", encoding="utf-8") as f:
        f.write(latex)


# ============================================================================
# Wrapper functions for common table types
# ============================================================================


def write_master_ci_plateau_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate full master CI/Plateau table.

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
    """
    columns = [
        "Metric",
        "Basis",
        "Log",
        "RelCI 50",
        "RelCI 250",
        "RelCI 500",
        "Plateau n",
    ]
    write_full_latex_table(
        csv_path=master_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="master_ci_plateau",
        caption_extension="CI/Plateau table",
        filename="master_ci_plateau.tex",
        order_by_basis=True,
    )


def write_master_ci_plateau_means_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate means-only master CI/Plateau table.

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
    """
    columns = [
        "Metric",
        "Basis",
        "Log",
        "RelCI 50",
        "RelCI 250",
        "RelCI 500",
        "Plateau n",
    ]
    # Hide Log
    columns = [col for col in columns if col not in ["Log"]]
    write_means_latex_table(
        csv_path=master_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="master_ci_plateau_means",
        caption_extension="CI/Plateau means",
        filename="master_ci_plateau_means.tex",
        order_by_basis=True,
    )


def write_comparison_correlation_table(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate full comparison correlation table.

    Args:
        comparison_csv_path: Path to metrics_comparison.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
    """
    columns = [
        "Metric",
        "Basis",
        "Log",
        "Shape_before",
        "Shape_after",
        "Preferred_Correlation_before",
        "Preferred_Correlation_after",
        "Chosen_Correlation",
        "Chosen_Rho_before",
        "Chosen_Rho_after",
        "Abs_Delta_Chosen_Rho",
        "Z_test_p",
        "Significant_Improvement",
    ]
    write_full_latex_table(
        csv_path=comparison_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="comparison_correlation_full",
        caption_extension="Comparison correlation table",
        filename="comparison_correlation_full.tex",
        order_by_basis=True,
    )


def write_comparison_correlation_means_table(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate means-only comparison correlation table.

    Args:
        comparison_csv_path: Path to metrics_comparison.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
    """
    columns = [
        "Metric",
        "Basis",
        "Log",
        "Shape_before",
        "Shape_after",
        "Preferred_Correlation_before",
        "Preferred_Correlation_after",
        "Chosen_Correlation",
        "Chosen_Rho_before",
        "Chosen_Rho_after",
        "Abs_Delta_Chosen_Rho",
        "Z_test_p",
        "Significant_Improvement",
    ]
    # Hide Log, Shape_before, Shape_after, Preferred_Correlation_before,
    # Preferred_Correlation_after, Chosen_Correlation
    columns = [
        col
        for col in columns
        if col
        not in [
            "Log",
            "Shape_before",
            "Shape_after",
            "Preferred_Correlation_before",
            "Preferred_Correlation_after",
            "Chosen_Correlation",
        ]
    ]
    write_means_latex_table(
        csv_path=comparison_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="comparison_correlation_means",
        caption_extension="Comparison correlation means",
        filename="comparison_correlation_means.tex",
        order_by_basis=True,
    )


def write_comparison_ci_plateau_table(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate full comparison CI/Plateau table.

    Args:
        comparison_csv_path: Path to metrics_comparison.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
    """
    columns = [
        "Metric",
        "Basis",
        "Log",
        "RelCI 50 Before",
        "RelCI 50 After",
        "RelCI 50 Delta",
        "RelCI 250 Before",
        "RelCI 250 After",
        "RelCI 250 Delta",
        "RelCI 500 Before",
        "RelCI 500 After",
        "RelCI 500 Delta",
        "Plateau n Before",
        "Plateau n After",
        "Plateau n Delta",
    ]
    write_full_latex_table(
        csv_path=comparison_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="comparison_ci_plateau",
        caption_extension="Comparison CI/Plateau table",
        filename="comparison_ci_plateau.tex",
        order_by_basis=True,
    )


def write_comparison_ci_plateau_means_table(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate means-only comparison CI/Plateau table.

    Args:
        comparison_csv_path: Path to metrics_comparison.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
    """
    columns = [
        "Metric",
        "Basis",
        "Log",
        "RelCI 50 Before",
        "RelCI 50 After",
        "RelCI 50 Delta",
        "RelCI 250 Before",
        "RelCI 250 After",
        "RelCI 250 Delta",
        "RelCI 500 Before",
        "RelCI 500 After",
        "RelCI 500 Delta",
        "Plateau n Before",
        "Plateau n After",
        "Plateau n Delta",
    ]
    # Hide Log
    columns = [col for col in columns if col not in ["Log"]]
    write_means_latex_table(
        csv_path=comparison_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="comparison_ci_plateau_means",
        caption_extension="Comparison CI/Plateau means",
        filename="comparison_ci_plateau_means.tex",
        order_by_basis=True,
    )


def write_summarized_master_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
    correlation: str = "Spearman",
) -> None:
    """Generate full summarized master table.

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
        correlation: Correlation type to display ("Pearson" or "Spearman"). Default is "Spearman".
    """
    if correlation not in ["Pearson", "Spearman"]:
        raise ValueError(
            f"correlation must be 'Pearson' or 'Spearman', got '{correlation}'"
        )

    rho_col = f"{correlation} Rho"
    p_col = f"{correlation} P"

    columns = [
        "Basis",
        "Metric",
        "Log",
        rho_col,
        p_col,
        "RelCI 50",
        "RelCI 250",
        "RelCI 500",
        "Plateau Reached",
    ]
    write_full_latex_table(
        csv_path=master_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="summarized_master_full",
        caption_extension="Summarized master table",
        filename="summarized_master_full.tex",
        order_by_basis=True,
    )


def write_summarized_means_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
    correlation: str = "Spearman",
) -> None:
    """Generate means-only summarized master table.

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
        correlation: Correlation type to display ("Pearson" or "Spearman"). Default is "Spearman".
    """
    if correlation not in ["Pearson", "Spearman"]:
        raise ValueError(
            f"correlation must be 'Pearson' or 'Spearman', got '{correlation}'"
        )

    rho_col = f"{correlation} Rho"
    p_col = f"{correlation} P"

    columns = [
        "Basis",
        "Metric",
        "Log",
        rho_col,
        p_col,
        "RelCI 50",
        "RelCI 250",
        "RelCI 500",
        "Plateau Reached",
    ]
    # Hide Log column
    columns = [col for col in columns if col not in ["Log"]]
    write_means_latex_table(
        csv_path=master_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="summarized_master_means",
        caption_extension="Summarized master means",
        filename="summarized_master_means.tex",
        order_by_basis=True,
    )


# ============================================================================
# Entry point functions
# ============================================================================


def write_master_latex_tables(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
    correlation: str = "Spearman",
) -> None:
    """Entry point: Generate all master LaTeX tables from master.csv.

    This generates:
    - Correlation tables (full + means)
    - CI/Plateau tables (full + means)
    - Summarized master tables (full + means)

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX files.
        scenario_key: Key for table labels (e.g., "test").
        scenario_title: Title for table captions (e.g., "Test Scenario").
        correlation: Correlation type to display ("Pearson" or "Spearman"). Default is "Spearman".
    """
    write_master_ci_plateau_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_master_ci_plateau_means_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_summarized_master_table(
        master_csv_path, out_dir, scenario_key, scenario_title, correlation=correlation
    )
    write_summarized_means_table(
        master_csv_path, out_dir, scenario_key, scenario_title, correlation=correlation
    )


def write_comparison_latex_tables(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Entry point: Generate all comparison LaTeX tables from metrics_comparison.csv.

    This generates:
    - Correlation tables (full + means)
    - CI/Plateau tables (full + means)

    Args:
        comparison_csv_path: Path to metrics_comparison.csv file.
        out_dir: Directory to save LaTeX files.
        scenario_key: Key for table labels (e.g., "test").
        scenario_title: Title for table captions (e.g., "Test Scenario").
    """
    write_comparison_correlation_table(
        comparison_csv_path, out_dir, scenario_key, scenario_title
    )
    write_comparison_correlation_means_table(
        comparison_csv_path, out_dir, scenario_key, scenario_title
    )
    write_comparison_ci_plateau_table(
        comparison_csv_path, out_dir, scenario_key, scenario_title
    )
    write_comparison_ci_plateau_means_table(
        comparison_csv_path, out_dir, scenario_key, scenario_title
    )


def generate_all_latex_tables(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
    correlation: str = "Spearman",
    comparison_csv_path: str | None = None,
) -> None:
    """Generate all LaTeX tables from master CSV and optionally comparison CSV.

    This function handles both master and comparison table generation with error handling.
    It will attempt to generate all tables and print warnings for any failures.

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX files.
        scenario_key: Key for table labels (e.g., "test").
        scenario_title: Title for table captions (e.g., "Test Scenario").
        correlation: Correlation type to use ("Spearman" or "Pearson"), default "Spearman".
        comparison_csv_path: Optional path to metrics_comparison.csv file.
    """
    from pathlib import Path

    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Generate master LaTeX tables
    try:
        write_master_latex_tables(
            master_csv_path=master_csv_path,
            out_dir=out_dir,
            scenario_key=scenario_key,
            scenario_title=scenario_title,
            correlation=correlation,
        )
        print(f"Master LaTeX tables saved to: {out_dir}")
    except Exception as e:
        print(f"[WARNING] Could not generate master LaTeX tables: {e}")

    # Generate comparison LaTeX tables if comparison CSV exists
    if comparison_csv_path is not None and Path(comparison_csv_path).exists():
        try:
            write_comparison_latex_tables(
                comparison_csv_path=comparison_csv_path,
                out_dir=out_dir,
                scenario_key=scenario_key,
                scenario_title=scenario_title,
            )
            print(f"Comparison LaTeX tables saved to: {out_dir}")
        except Exception as e:
            print(f"[WARNING] Could not generate comparison LaTeX tables: {e}")
