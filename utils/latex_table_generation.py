"""LaTeX table generation from CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .constants import BASIS_ORDER, COLUMN_NAME_MAP, METRIC_BASIS_MAP


def _escape_latex(text: Any) -> str:
    """Escape underscores in text, preserving \textit{...} wrappers."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if s.startswith("\\textit{") and s.endswith("}"):
        return s
    return s.replace("_", "\\_")


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


def _build_header_and_colspec(columns: List[str]) -> tuple[str, str]:
    """Build LaTeX header and column specification from column list.

    Uses COLUMN_NAME_MAP for header display names (not escaped).
    Column spec uses p{50pt} for Metric column, appropriate types for others.

    Args:
        columns: List of CSV column names.

    Returns:
        Tuple of (header_string, column_spec_string).
    """
    header_parts = []
    colspec_parts = []

    for col in columns:
        # Use COLUMN_NAME_MAP if available, otherwise escape the column name
        if col in COLUMN_NAME_MAP:
            header_part = COLUMN_NAME_MAP[col]  # Already formatted, don't escape
        else:
            header_part = col.replace("_", "\\_")
        header_parts.append(header_part)

        # Build column spec
        if col == "Metric":
            colspec_parts.append("p{50pt}")
        elif col in [
            "Basis",
            "Log",
            "Shape",
            "Preferred_Correlation",
            "Shape_before",
            "Shape_after",
            "Preferred_Correlation_before",
            "Preferred_Correlation_after",
            "Chosen_Correlation",
        ]:
            colspec_parts.append("l")
        elif col.endswith("_p") or col == "Z_test_p":
            colspec_parts.append("c")
        elif col in [
            "Pearson_Rho",
            "Spearman_Rho",
            "Delta_PearsonSpearman",
            "Chosen_Rho_before",
            "Chosen_Rho_after",
            "Abs_Delta_Chosen_Rho",
            "Delta_Pearson",
            "Delta_Spearman",
            "Z_test_stat",
            "RelCI_50",
            "RelCI_250",
            "RelCI_500",
            "Plateau_n",
            "RelCI_50_before",
            "RelCI_50_after",
            "RelCI_50_delta",
            "RelCI_250_before",
            "RelCI_250_after",
            "RelCI_250_delta",
            "RelCI_500_before",
            "RelCI_500_after",
            "RelCI_500_delta",
            "Plateau_n_before",
            "Plateau_n_after",
            "Plateau_n_delta",
            "Significant_Improvement",
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


def _create_table_rows(
    df_subset: pd.DataFrame,
    columns: List[str],
    is_means_only: bool = False,
) -> List[str]:
    """Create LaTeX table rows with collapsing Metric/Basis and italicizing MEAN rows.

    Args:
        df_subset: DataFrame subset to process.
        columns: List of column names to include.
        is_means_only: If True, this is a means-only table (no italicization).

    Returns:
        List of LaTeX row strings.
    """
    rows = []
    prev_metric = None
    prev_basis = None

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
            if col == "Metric":
                cell = _escape_latex(metric) if not repeat_metric else ""
            elif col == "Basis":
                cell = _escape_latex(basis) if not repeat_basis else ""
            elif col == "Log":
                cell = "" if is_mean else _escape_latex(log)
            else:
                # Get value from row
                val = row.get(col, "")
                if pd.isna(val) or val == "":
                    cell = ""
                elif col.endswith("_p") or col == "Z_test_p":
                    cell = _format_pvalue(val)
                elif col in boolean_cols or col == "Significant_Improvement":
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

                # Escape LaTeX for non-empty cells (unless already formatted via COLUMN_NAME_MAP)
                if cell != "":
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
    rows = _create_table_rows(df, available_columns, is_means_only=False)

    # Build header and column spec
    header, colspec = _build_header_and_colspec(available_columns)

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
    rows = _create_table_rows(means_df, available_columns, is_means_only=True)

    # Build header and column spec
    header, colspec = _build_header_and_colspec(available_columns)

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


def write_master_correlation_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate full master correlation table.

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
        "Pearson_Rho",
        "Spearman_Rho",
        "Delta_PearsonSpearman",
        "Shape",
        "Preferred_Correlation",
    ]
    write_full_latex_table(
        csv_path=master_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="master_correlation_full",
        caption_extension="Correlation table",
        filename="master_correlation_full.tex",
        order_by_basis=True,
    )


def write_master_correlation_means_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate means-only master correlation table.

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
        "Pearson_Rho",
        "Spearman_Rho",
        "Delta_PearsonSpearman",
        "Shape",
        "Preferred_Correlation",
    ]
    # Hide Log, Shape, Preferred_Correlation
    columns = [
        col for col in columns if col not in ["Log", "Shape", "Preferred_Correlation"]
    ]
    write_means_latex_table(
        csv_path=master_csv_path,
        out_dir=out_dir,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        include_columns=columns,
        label_extension="master_correlation_means",
        caption_extension="Correlation means",
        filename="master_correlation_means.tex",
        order_by_basis=True,
    )


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
        "RelCI_50",
        "RelCI_250",
        "RelCI_500",
        "Plateau_n",
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
        "RelCI_50",
        "RelCI_250",
        "RelCI_500",
        "Plateau_n",
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
        "RelCI_50_before",
        "RelCI_50_after",
        "RelCI_50_delta",
        "RelCI_250_before",
        "RelCI_250_after",
        "RelCI_250_delta",
        "RelCI_500_before",
        "RelCI_500_after",
        "RelCI_500_delta",
        "Plateau_n_before",
        "Plateau_n_after",
        "Plateau_n_delta",
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
        "RelCI_50_before",
        "RelCI_50_after",
        "RelCI_50_delta",
        "RelCI_250_before",
        "RelCI_250_after",
        "RelCI_250_delta",
        "RelCI_500_before",
        "RelCI_500_after",
        "RelCI_500_delta",
        "Plateau_n_before",
        "Plateau_n_after",
        "Plateau_n_delta",
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
) -> None:
    """Generate full summarized master table.

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
    """
    columns = [
        "Basis",
        "Metric",
        "Log",
        "Pearson_Rho",
        "Pearson_P",
        "RelCI_50",
        "RelCI_250",
        "RelCI_500",
        "Plateau_n",
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
) -> None:
    """Generate means-only summarized master table.

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX file.
        scenario_key: Key for table label.
        scenario_title: Title for table caption.
    """
    columns = [
        "Basis",
        "Metric",
        "Log",
        "Pearson_Rho",
        "Pearson_P",
        "RelCI_50",
        "RelCI_250",
        "RelCI_500",
        "Plateau_n",
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
    """
    write_master_correlation_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_master_correlation_means_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_master_ci_plateau_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_master_ci_plateau_means_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_summarized_master_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_summarized_means_table(master_csv_path, out_dir, scenario_key, scenario_title)


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
