"""LaTeX table generation from CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd


def _escape_latex(text: Any) -> str:
    """Escape underscores in text, preserving \textit{...} wrappers."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if s.startswith("\\textit{") and s.endswith("}"):
        return s
    return s.replace("_", "\\_")


def _format_num(x: Any, decimals: int = 3) -> str:
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


def _create_table_rows(
    df_subset: pd.DataFrame,
    columns: List[str],
    is_means_only: bool = False,
) -> List[str]:
    """Create LaTeX table rows with collapsing Metric/Basis and italicizing MEAN rows."""
    rows = []
    prev_metric = None
    prev_basis = None

    # Identify numeric columns (for formatting)
    numeric_cols = set()
    for col in df_subset.columns:
        if pd.api.types.is_numeric_dtype(df_subset[col]):
            numeric_cols.add(col)

    for idx, row in df_subset.iterrows():
        metric = row["Metric"]
        basis = row["Basis"]
        log = row["Log"]
        is_mean = log == "MEAN"

        # Determine if we should repeat Metric and Basis
        repeat_metric = (metric == prev_metric) if prev_metric is not None else False
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


def write_latex_master_correlation_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate full correlation table from master.csv."""
    df = pd.read_csv(master_csv_path)

    # Sort: MEAN rows at end for each (Metric, Basis) group
    # Preserve original metric order from CSV (don't sort alphabetically)
    df["_sort_key"] = df["Log"].apply(lambda x: 1 if x == "MEAN" else 0)
    # Create a metric order mapping to preserve CSV order
    unique_metrics = df["Metric"].unique()
    metric_order = {metric: idx for idx, metric in enumerate(unique_metrics)}
    df["_metric_order"] = df["Metric"].map(metric_order)
    df = df.sort_values(by=["_metric_order", "Basis", "_sort_key", "Log"])
    df = df.drop(columns=["_sort_key", "_metric_order"])

    # Columns: Metric, Basis, Log, Pearson_Rho, Spearman_Rho, Delta_PearsonSpearman, Shape, Preferred_Correlation
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

    rows = _create_table_rows(df, columns, is_means_only=False)
    header = "Metric & Basis & Log & Pearson\\_Rho & Spearman\\_Rho & Delta\\_PearsonSpearman & Shape & Preferred\\_Correlation \\\\"

    latex = f"""\\begin{{table}}[Htbp]
\\centering
\\tiny
\\caption{{Correlation table — {_escape_latex(scenario_title)}}}
\\label{{tab:master_correlation_full_{scenario_key}}}
\\begin{{tabular}}{{l l l c c c l l}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "master_correlation_full.tex", "w", encoding="utf-8") as f:
        f.write(latex)


def write_latex_master_correlation_means_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate means-only correlation table from master.csv."""
    df = pd.read_csv(master_csv_path)
    means_df = df[df["Log"] == "MEAN"].copy()

    # Columns: Same as full table (Metric, Basis, Log, Pearson_Rho, Spearman_Rho, Delta_PearsonSpearman, Shape, Preferred_Correlation)
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

    rows = _create_table_rows(means_df, columns, is_means_only=True)
    header = "Metric & Basis & Log & Pearson\\_Rho & Spearman\\_Rho & Delta\\_PearsonSpearman & Shape & Preferred\\_Correlation \\\\"

    latex = f"""\\begin{{table}}[Htbp]
\\centering
\\tiny
\\caption{{Correlation means — {_escape_latex(scenario_title)}}}
\\label{{tab:master_correlation_means_{scenario_key}}}
\\begin{{tabular}}{{l l l c c c l l}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "master_correlation_means.tex", "w", encoding="utf-8") as f:
        f.write(latex)


def write_latex_comparison_correlation_table(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate full correlation table from metrics_comparison.csv."""
    df = pd.read_csv(comparison_csv_path)

    # Sort: MEAN rows at end for each (Metric, Basis) group
    # Preserve original metric order from CSV (don't sort alphabetically)
    df["_sort_key"] = df["Log"].apply(lambda x: 1 if x == "MEAN" else 0)
    # Create a metric order mapping to preserve CSV order
    unique_metrics = df["Metric"].unique()
    metric_order = {metric: idx for idx, metric in enumerate(unique_metrics)}
    df["_metric_order"] = df["Metric"].map(metric_order)
    df = df.sort_values(by=["_metric_order", "Basis", "_sort_key", "Log"])
    df = df.drop(columns=["_sort_key", "_metric_order"])

    # Format Significant_Improvement
    def format_sig_improvement(val: Any, is_mean: bool) -> str:
        if pd.isna(val) or val == "":
            return ""
        if is_mean:
            try:
                float_val = float(val)
                if np.isfinite(float_val):
                    return f"{float_val * 100:.1f}\\%"
            except (ValueError, TypeError):
                pass
            return ""
        else:
            val_str = str(val).upper()
            if val_str == "TRUE":
                return "Yes"
            elif val_str == "FALSE":
                return "No"
            return str(val)

    # Columns: Metric, Basis, Log, Shape_before, Shape_after, Preferred_Correlation_before, Preferred_Correlation_after, Chosen_Correlation, Chosen_Rho_before, Chosen_Rho_after, Delta_Chosen_Rho, Z_test_p, Significant_Improvement
    rows = []
    prev_metric = None
    prev_basis = None

    for idx, row in df.iterrows():
        metric = row["Metric"]
        basis = row["Basis"]
        log = row["Log"]
        is_mean = log == "MEAN"

        repeat_metric = (metric == prev_metric) if prev_metric is not None else False
        repeat_basis = (basis == prev_basis) if prev_basis is not None else False

        metric_str = _escape_latex(metric) if not repeat_metric else ""
        basis_str = _escape_latex(basis) if not repeat_basis else ""
        log_str = "" if is_mean else _escape_latex(log)
        shape_before = _escape_latex(
            str(row.get("Shape_before", "")) if not is_mean else ""
        )
        shape_after = _escape_latex(
            str(row.get("Shape_after", "")) if not is_mean else ""
        )
        preferred_correlation_before = _escape_latex(
            str(row.get("Preferred_Correlation_before", "")) if not is_mean else ""
        )
        preferred_correlation_after = _escape_latex(
            str(row.get("Preferred_Correlation_after", "")) if not is_mean else ""
        )
        chosen_correlation = _escape_latex(str(row.get("Chosen_Correlation", "")))
        chosen_rho_before = _format_num(row.get("Chosen_Rho_before", ""))
        chosen_rho_after = _format_num(row.get("Chosen_Rho_after", ""))
        delta_chosen_rho = _format_num(row.get("Delta_Chosen_Rho", ""))
        z_test_p = _format_pvalue(row.get("Z_test_p", ""))
        significant_improvement = format_sig_improvement(
            row.get("Significant_Improvement", ""), is_mean
        )

        # Build cells list for MEAN row italicization
        cells_list = [
            metric_str,
            basis_str,
            log_str,
            shape_before,
            shape_after,
            preferred_correlation_before,
            preferred_correlation_after,
            chosen_correlation,
            chosen_rho_before,
            chosen_rho_after,
            delta_chosen_rho,
            z_test_p,
            significant_improvement,
        ]

        # For MEAN rows, italicize each cell individually
        if is_mean:
            cells_list = [
                f"\\textit{{{cell}}}" if cell != "" else "\\textit{}"
                for cell in cells_list
            ]

        row_str = " & ".join(cells_list) + " \\\\"
        rows.append(row_str)
        prev_metric = metric
        prev_basis = basis

    header = "Metric & Basis & Log & Shape\\_before & Shape\\_after & Preferred\\_Correlation\\_before & Preferred\\_Correlation\\_after & Chosen\\_Correlation & Chosen\\_Rho\\_before & Chosen\\_Rho\\_after & Delta\\_Chosen\\_Rho & Z\\_test\\_p & Significant\\_Improvement \\\\"

    latex = f"""\\begin{{table}}[Htbp]
\\centering
\\tiny
\\caption{{Comparison correlation table — {_escape_latex(scenario_title)}}}
\\label{{tab:comparison_correlation_full_{scenario_key}}}
\\begin{{tabular}}{{l l l l l l l l c c c c c}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "comparison_correlation_full.tex", "w", encoding="utf-8") as f:
        f.write(latex)


def write_latex_comparison_correlation_means_table(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate means-only correlation table from metrics_comparison.csv."""
    df = pd.read_csv(comparison_csv_path)
    means_df = df[df["Log"] == "MEAN"].copy()

    # Format Significant_Improvement as percentage
    def format_sig_improvement(val: Any, is_mean: bool = True) -> str:
        if pd.isna(val) or val == "":
            return ""
        try:
            float_val = float(val)
            if np.isfinite(float_val):
                return f"{float_val * 100:.1f}\\%"
        except (ValueError, TypeError):
            pass
        return ""

    # Columns: Same as full table (Metric, Basis, Log, Shape_before, Shape_after, Preferred_Correlation_before, Preferred_Correlation_after, Chosen_Correlation, Chosen_Rho_before, Chosen_Rho_after, Delta_Chosen_Rho, Z_test_p, Significant_Improvement)
    rows = []
    prev_metric = None
    prev_basis = None

    for idx, row in means_df.iterrows():
        metric = row["Metric"]
        basis = row["Basis"]
        log = row["Log"]
        is_mean = log == "MEAN"

        repeat_metric = (metric == prev_metric) if prev_metric is not None else False
        repeat_basis = (basis == prev_basis) if prev_basis is not None else False

        metric_str = _escape_latex(metric) if not repeat_metric else ""
        basis_str = _escape_latex(basis) if not repeat_basis else ""
        log_str = "" if is_mean else _escape_latex(log)
        shape_before = _escape_latex(
            str(row.get("Shape_before", "")) if not is_mean else ""
        )
        shape_after = _escape_latex(
            str(row.get("Shape_after", "")) if not is_mean else ""
        )
        preferred_correlation_before = _escape_latex(
            str(row.get("Preferred_Correlation_before", "")) if not is_mean else ""
        )
        preferred_correlation_after = _escape_latex(
            str(row.get("Preferred_Correlation_after", "")) if not is_mean else ""
        )
        chosen_correlation = _escape_latex(str(row.get("Chosen_Correlation", "")))
        chosen_rho_before = _format_num(row.get("Chosen_Rho_before", ""))
        chosen_rho_after = _format_num(row.get("Chosen_Rho_after", ""))
        delta_chosen_rho = _format_num(row.get("Delta_Chosen_Rho", ""))
        z_test_p = _format_pvalue(row.get("Z_test_p", ""))
        significant_improvement = format_sig_improvement(
            row.get("Significant_Improvement", ""), is_mean
        )

        # Build cells list (no italicization for means-only tables)
        cells_list = [
            metric_str,
            basis_str,
            log_str,
            shape_before,
            shape_after,
            preferred_correlation_before,
            preferred_correlation_after,
            chosen_correlation,
            chosen_rho_before,
            chosen_rho_after,
            delta_chosen_rho,
            z_test_p,
            significant_improvement,
        ]

        row_str = " & ".join(cells_list) + " \\\\"
        rows.append(row_str)
        prev_metric = metric
        prev_basis = basis

    header = "Metric & Basis & Log & Shape\\_before & Shape\\_after & Preferred\\_Correlation\\_before & Preferred\\_Correlation\\_after & Chosen\\_Correlation & Chosen\\_Rho\\_before & Chosen\\_Rho\\_after & Delta\\_Chosen\\_Rho & Z\\_test\\_p & Significant\\_Improvement \\\\"

    latex = f"""\\begin{{table}}[Htbp]
\\centering
\\tiny
\\caption{{Comparison correlation means — {_escape_latex(scenario_title)}}}
\\label{{tab:comparison_correlation_means_{scenario_key}}}
\\begin{{tabular}}{{l l l l l l l l c c c c c}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(
        out_path / "comparison_correlation_means.tex", "w", encoding="utf-8"
    ) as f:
        f.write(latex)


def write_latex_master_ci_plateau_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate CI/Plateau table from master.csv."""
    df = pd.read_csv(master_csv_path)

    # Sort: MEAN rows at end for each (Metric, Basis) group
    # Preserve original metric order from CSV (don't sort alphabetically)
    df["_sort_key"] = df["Log"].apply(lambda x: 1 if x == "MEAN" else 0)
    # Create a metric order mapping to preserve CSV order
    unique_metrics = df["Metric"].unique()
    metric_order = {metric: idx for idx, metric in enumerate(unique_metrics)}
    df["_metric_order"] = df["Metric"].map(metric_order)
    df = df.sort_values(by=["_metric_order", "Basis", "_sort_key", "Log"])
    df = df.drop(columns=["_sort_key", "_metric_order"])

    # Columns: Metric, Basis, Log, RelCI_50, RelCI_250, RelCI_500, Plateau_n
    columns = [
        "Metric",
        "Basis",
        "Log",
        "RelCI_50",
        "RelCI_250",
        "RelCI_500",
        "Plateau_n",
    ]

    rows = _create_table_rows(df, columns, is_means_only=False)
    header = "Metric & Basis & Log & RelCI\\_50 & RelCI\\_250 & RelCI\\_500 & Plateau\\_n \\\\"

    latex = f"""\\begin{{table}}[Htbp]
\\centering
\\tiny
\\caption{{CI/Plateau table — {_escape_latex(scenario_title)}}}
\\label{{tab:master_ci_plateau_{scenario_key}}}
\\begin{{tabular}}{{l l l c c c c}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "master_ci_plateau.tex", "w", encoding="utf-8") as f:
        f.write(latex)


def write_latex_master_ci_plateau_means_table(
    master_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate means-only CI/Plateau table from master.csv."""
    df = pd.read_csv(master_csv_path)
    means_df = df[df["Log"] == "MEAN"].copy()

    # Columns: Same as full table (Metric, Basis, Log, RelCI_50, RelCI_250, RelCI_500, Plateau_n)
    columns = [
        "Metric",
        "Basis",
        "Log",
        "RelCI_50",
        "RelCI_250",
        "RelCI_500",
        "Plateau_n",
    ]

    rows = _create_table_rows(means_df, columns, is_means_only=True)
    header = "Metric & Basis & Log & RelCI\\_50 & RelCI\\_250 & RelCI\\_500 & Plateau\\_n \\\\"

    latex = f"""\\begin{{table}}[Htbp]
\\centering
\\tiny
\\caption{{CI/Plateau means — {_escape_latex(scenario_title)}}}
\\label{{tab:master_ci_plateau_means_{scenario_key}}}
\\begin{{tabular}}{{l l l c c c c}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "master_ci_plateau_means.tex", "w", encoding="utf-8") as f:
        f.write(latex)


def write_latex_comparison_ci_plateau_table(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate CI/Plateau table from metrics_comparison.csv."""
    df = pd.read_csv(comparison_csv_path)

    # Sort: MEAN rows at end for each (Metric, Basis) group
    # Preserve original metric order from CSV (don't sort alphabetically)
    df["_sort_key"] = df["Log"].apply(lambda x: 1 if x == "MEAN" else 0)
    # Create a metric order mapping to preserve CSV order
    unique_metrics = df["Metric"].unique()
    metric_order = {metric: idx for idx, metric in enumerate(unique_metrics)}
    df["_metric_order"] = df["Metric"].map(metric_order)
    df = df.sort_values(by=["_metric_order", "Basis", "_sort_key", "Log"])
    df = df.drop(columns=["_sort_key", "_metric_order"])

    # Columns: Metric, Basis, Log, RelCI_50_before, RelCI_50_after, RelCI_50_delta, RelCI_250_before, RelCI_250_after, RelCI_250_delta, RelCI_500_before, RelCI_500_after, RelCI_500_delta, Plateau_n_before, Plateau_n_after, Plateau_n_delta
    rows = []
    prev_metric = None
    prev_basis = None

    for idx, row in df.iterrows():
        metric = row["Metric"]
        basis = row["Basis"]
        log = row["Log"]
        is_mean = log == "MEAN"

        repeat_metric = (metric == prev_metric) if prev_metric is not None else False
        repeat_basis = (basis == prev_basis) if prev_basis is not None else False

        metric_str = _escape_latex(metric) if not repeat_metric else ""
        basis_str = _escape_latex(basis) if not repeat_basis else ""
        log_str = "" if is_mean else _escape_latex(log)

        # Format CI and Plateau values
        relci_50_before = _format_num(row.get("RelCI_50_before", ""))
        relci_50_after = _format_num(row.get("RelCI_50_after", ""))
        relci_50_delta = _format_num(row.get("RelCI_50_delta", ""))
        relci_250_before = _format_num(row.get("RelCI_250_before", ""))
        relci_250_after = _format_num(row.get("RelCI_250_after", ""))
        relci_250_delta = _format_num(row.get("RelCI_250_delta", ""))
        relci_500_before = _format_num(row.get("RelCI_500_before", ""))
        relci_500_after = _format_num(row.get("RelCI_500_after", ""))
        relci_500_delta = _format_num(row.get("RelCI_500_delta", ""))
        plateau_n_before = _format_num(row.get("Plateau_n_before", ""))
        plateau_n_after = _format_num(row.get("Plateau_n_after", ""))
        plateau_n_delta = _format_num(row.get("Plateau_n_delta", ""))

        # Build cells list for MEAN row italicization
        cells_list = [
            metric_str,
            basis_str,
            log_str,
            relci_50_before,
            relci_50_after,
            relci_50_delta,
            relci_250_before,
            relci_250_after,
            relci_250_delta,
            relci_500_before,
            relci_500_after,
            relci_500_delta,
            plateau_n_before,
            plateau_n_after,
            plateau_n_delta,
        ]

        # For MEAN rows, italicize each cell individually
        if is_mean:
            cells_list = [
                f"\\textit{{{cell}}}" if cell != "" else "\\textit{}"
                for cell in cells_list
            ]

        row_str = " & ".join(cells_list) + " \\\\"
        rows.append(row_str)
        prev_metric = metric
        prev_basis = basis

    header = "Metric & Basis & Log & RelCI\\_50\\_before & RelCI\\_50\\_after & RelCI\\_50\\_delta & RelCI\\_250\\_before & RelCI\\_250\\_after & RelCI\\_250\\_delta & RelCI\\_500\\_before & RelCI\\_500\\_after & RelCI\\_500\\_delta & Plateau\\_n\\_before & Plateau\\_n\\_after & Plateau\\_n\\_delta \\\\"

    latex = f"""\\begin{{table}}[Htbp]
\\centering
\\tiny
\\caption{{Comparison CI/Plateau table — {_escape_latex(scenario_title)}}}
\\label{{tab:comparison_ci_plateau_{scenario_key}}}
\\begin{{tabular}}{{l l l c c c c c c c c c c c c c}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "comparison_ci_plateau.tex", "w", encoding="utf-8") as f:
        f.write(latex)


def write_latex_comparison_ci_plateau_means_table(
    comparison_csv_path: str,
    out_dir: str,
    scenario_key: str,
    scenario_title: str,
) -> None:
    """Generate means-only CI/Plateau table from metrics_comparison.csv."""
    df = pd.read_csv(comparison_csv_path)
    means_df = df[df["Log"] == "MEAN"].copy()

    # Columns: Same as full table (Metric, Basis, Log, RelCI_50_before, RelCI_50_after, RelCI_50_delta, RelCI_250_before, RelCI_250_after, RelCI_250_delta, RelCI_500_before, RelCI_500_after, RelCI_500_delta, Plateau_n_before, Plateau_n_after, Plateau_n_delta)
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

    rows = _create_table_rows(means_df, columns, is_means_only=True)
    header = "Metric & Basis & Log & RelCI\\_50\\_before & RelCI\\_50\\_after & RelCI\\_50\\_delta & RelCI\\_250\\_before & RelCI\\_250\\_after & RelCI\\_250\\_delta & RelCI\\_500\\_before & RelCI\\_500\\_after & RelCI\\_500\\_delta & Plateau\\_n\\_before & Plateau\\_n\\_after & Plateau\\_n\\_delta \\\\"

    latex = f"""\\begin{{table}}[Htbp]
\\centering
\\tiny
\\caption{{Comparison CI/Plateau means — {_escape_latex(scenario_title)}}}
\\label{{tab:comparison_ci_plateau_means_{scenario_key}}}
\\begin{{tabular}}{{l l l c c c c c c c c c c c c c}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "comparison_ci_plateau_means.tex", "w", encoding="utf-8") as f:
        f.write(latex)


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

    Args:
        master_csv_path: Path to master.csv file.
        out_dir: Directory to save LaTeX files.
        scenario_key: Key for table labels (e.g., "test").
        scenario_title: Title for table captions (e.g., "Test Scenario").
    """
    write_latex_master_correlation_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_latex_master_correlation_means_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_latex_master_ci_plateau_table(
        master_csv_path, out_dir, scenario_key, scenario_title
    )
    write_latex_master_ci_plateau_means_table(
        master_csv_path, out_dir, scenario_key, scenario_title
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
    write_latex_comparison_correlation_table(
        comparison_csv_path, out_dir, scenario_key, scenario_title
    )
    write_latex_comparison_correlation_means_table(
        comparison_csv_path, out_dir, scenario_key, scenario_title
    )
    write_latex_comparison_ci_plateau_table(
        comparison_csv_path, out_dir, scenario_key, scenario_title
    )
    write_latex_comparison_ci_plateau_means_table(
        comparison_csv_path, out_dir, scenario_key, scenario_title
    )
