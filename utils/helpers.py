from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .constants import COMPLEXITY_RESULTS_DIR


def to_naive_ts(x: Any) -> Optional[pd.Timestamp]:
    """Convert timestamp to naive (timezone-unaware) format.

    Args:
        x: Input timestamp (can be string, datetime, or None).

    Returns:
        Naive pandas Timestamp or None if input is None.
    """
    if x is None:
        return None
    ts = pd.to_datetime(x)
    try:
        return ts.tz_convert(None)
    except Exception:
        return ts


def save_complexity_csv(
    dataset_key: str, configuration_name: str, df: pd.DataFrame
) -> Path:
    """Save complexity results DataFrame to CSV file.

    Args:
        dataset_key: Name of the dataset.
        configuration_name: Name of the configuration.
        df: DataFrame containing complexity results.

    Returns:
        Path to the saved CSV file.
    """
    out_dir = COMPLEXITY_RESULTS_DIR / dataset_key / configuration_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "complexity.csv"
    df.to_csv(out, index=False)
    return out


def flatten_measurements(window_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten window measurements into a single DataFrame.

    Args:
        window_rows: List of dictionaries containing window data and measurements.

    Returns:
        DataFrame with flattened measurements.
    """
    rows = []
    for row in window_rows:
        base = {k: v for k, v in row.items() if k != "measurements"}
        base.update(row.get("measurements", {}))
        rows.append(base)
    return pd.DataFrame(rows)


def load_data_dictionary(
    path: Path, get_real: bool = True, get_synthetic: bool = False
) -> Dict[str, Any]:
    """Load a JSON data dictionary and filter entries by their 'type' field.

    Args:
        path: Path to the JSON file.
        get_real: Include entries where type == "real".
        get_synthetic: Include entries where type == "synthetic".

    Returns:
        A dict filtered to the requested types. If both flags are False, returns {}.
    """
    # Determine which types to keep
    allowed_types = set()
    if get_real:
        allowed_types.add("real")
    if get_synthetic:
        allowed_types.add("synthetic")

    with open(path, "r", encoding="utf-8") as f:
        data_dictionary: Dict[str, Any] = json.load(f)

    # If nothing is requested, return empty dict
    if not allowed_types:
        return {}

    # Keep only entries whose 'type' is in the allowed set
    return {k: v for k, v in data_dictionary.items() if v.get("type") in allowed_types}


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing YAML data, or empty dict if file is empty.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_dataframe_from_drift_detection_results(
    datasets: List[str], cp_configurations: List[str]
) -> pd.DataFrame:
    """Load drift detection results from CSV files and combine into DataFrame.

    Args:
        datasets: List of dataset names.
        cp_configurations: List of change point configuration names.

    Returns:
        DataFrame containing combined drift detection results.
    """
    results = []
    for dataset in datasets:
        for cp_configuration in cp_configurations:
            results_path = f"results/drift_detection/{dataset}/results_{dataset}_{cp_configuration}.csv"
            if not os.path.exists(results_path):
                continue

            results_df = pd.read_csv(results_path)

            # Be tolerant to missing columns
            for _, row in results_df.iterrows():
                calc_drift_id = row.get("calc_drift_id")
                if pd.isna(calc_drift_id) or str(calc_drift_id).lower() == "na":
                    break  # assume remaining rows are empty markers

                change_point = row.get("calc_change_index")
                change_moment = row.get("calc_change_moment")

                results.append(
                    {
                        "dataset": dataset,
                        "configuration": cp_configuration,
                        "change_point": change_point,
                        "change_moment": pd.to_datetime(change_moment, utc=True),
                    }
                )

    # convert results into dataframe
    if not results:
        return pd.DataFrame(
            columns=["dataset", "configuration", "change_point", "change_moment"]
        )

    out = pd.DataFrame(results)
    return out.reset_index(drop=True)


# correlation helpers
def get_correlations_for_dictionary(
    sample_metrics_per_log: Dict[str, pd.DataFrame],
    rename_dictionary_map: Optional[Dict[str, str]],
    metric_columns: List[str],
    base_column: str = "sample_size",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson correlations between sample size and metrics.

    Args:
        sample_metrics_per_log: Dictionary mapping log names to DataFrames with metrics.
        rename_dictionary_map: Optional mapping to rename log names in output.
        metric_columns: List of metric column names to analyze.
        base_column: Name of the base column for correlation (default: 'sample_size').

    Returns:
        Tuple of (correlation DataFrame, p-value DataFrame).
    """
    from scipy import stats

    rename_map = rename_dictionary_map

    r_results: Dict[str, Dict[str, float]] = {}
    p_results: Dict[str, Dict[str, float]] = {}

    for key, df in sample_metrics_per_log.items():
        col_tag = (
            key if rename_map is None else rename_map[key]
        )  # apply the rename map if available
        r_results[col_tag] = {}
        p_results[col_tag] = {}

        for col in df.columns:
            if col not in metric_columns:
                continue
            # drop missing values and infinite values pairwise
            tmp = df[[base_column, col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(tmp) < 2:
                r, p = float("nan"), float("nan")
            else:
                r, p = stats.pearsonr(tmp[base_column], tmp[col])
            r_results[col_tag][col] = r
            p_results[col_tag][col] = p

    # Create DataFrames
    corr_df = pd.DataFrame(r_results)
    pval_df = pd.DataFrame(p_results)

    # Enforce consistent row order
    corr_df = corr_df.reindex(metric_columns)
    pval_df = pval_df.reindex(metric_columns)

    print("Correlations:")
    print(corr_df)
    print()
    print("P-values:")
    print(pval_df)

    return corr_df, pval_df


########################################################
# plateau detection helpers
########################################################


def _detect_plateau_1d(
    sample_sizes: np.ndarray,
    centers: np.ndarray,
    rel_threshold: float,
    min_runs: int = 1,
    report: str = "current",  # or "next"
) -> float:
    """
    Return the window size where plateau is reached, or np.nan if not found.
    Plateau when |Δ|/prev < rel_threshold for 'min_runs' consecutive steps.
    """
    ss = np.asarray(sample_sizes, dtype=float)
    vals = np.asarray(centers, dtype=float)

    if len(ss) < 2 or len(ss) != len(vals):
        return np.nan

    consec = 0
    for i in range(len(ss) - 1):
        denom = vals[i]
        if not np.isfinite(denom) or denom == 0.0:
            consec = 0
            continue
        rel_change = abs(vals[i + 1] - vals[i]) / abs(denom)
        if np.isfinite(rel_change) and rel_change < rel_threshold:
            consec += 1
            if consec >= min_runs:
                n_report = ss[i] if report == "current" else ss[i + 1]
                # prefer int-looking numbers as int
                return int(n_report) if float(int(n_report)) == n_report else n_report
        else:
            consec = 0

    return np.nan


def detect_plateau_df(
    measures_per_log: Dict[str, pd.DataFrame],
    metric_columns: Optional[List[str]] = None,
    rel_threshold: float = 0.05,
    agg: str = "mean",
    min_runs: int = 1,
    report: str = "current",
) -> pd.DataFrame:
    """
    Compute plateau window size per (metric, log). Returns a matrix-like DataFrame:
    rows = metrics, columns = logs, values = plateau window size (float/int) or NaN.

    Parameters
    ----------
    measures_per_log : dict[log_name -> measures_df]
        measures_df must have ['sample_size','sample_id', <metric columns...>].
    metric_columns : list[str], optional
        List of metric column names to analyze. If None, extracts numeric columns
        excluding 'sample_size' and 'sample_id' from all DataFrames.
    rel_threshold : float
        Relative change threshold between consecutive centers.
    agg : {'mean','median'}
        Aggregation across samples at each window size.
    min_runs : int
        Require this many consecutive steps below threshold to accept plateau.
    report : {'current','next'}
        Report the current or next window size as the plateau location.
    """
    agg = agg.lower()
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'.")

    # Collect union of metric names to build a consistent row index
    if metric_columns is None:
        all_metrics: set[str] = set()
        for df in measures_per_log.values():
            id_cols = {"sample_size", "sample_id"}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_metrics.update([c for c in numeric_cols if c not in id_cols])
        metrics_sorted = sorted(all_metrics)
    else:
        metrics_sorted = sorted(set(metric_columns))

    plateau_by_log: dict[str, pd.Series] = {}

    for log, df in measures_per_log.items():
        metric_cols = [m for m in metrics_sorted if m in df.columns]
        if not metric_cols:
            continue
        g = df.groupby("sample_size", sort=True, as_index=True)
        centers = g[metric_cols].mean() if agg == "mean" else g[metric_cols].median()
        centers = centers.sort_index()

        values = {}
        x = centers.index.values
        for m in metric_cols:
            # Skip if metric was dropped during aggregation (e.g., all NaN values)
            if m not in centers.columns:
                continue
            y = centers[m].values
            plateau_n = _detect_plateau_1d(
                sample_sizes=x,
                centers=y,
                rel_threshold=rel_threshold,
                min_runs=min_runs,
                report=report,
            )
            values[m] = plateau_n
        # Fill missing metrics as NaN for consistent shape
        plateau_by_log[log] = pd.Series(values, index=metrics_sorted, dtype="float64")

    # Build matrix DataFrame: rows metrics, columns logs
    plateau_df = pd.DataFrame(plateau_by_log, index=metrics_sorted)
    plateau_df.index.name = None  # keep similar to your corr_df look
    return plateau_df


########################################################
# Master table builder


def create_master_table_before_from_corrs(
    measures_per_log: Dict[str, pd.DataFrame],
    sample_ci_rel_width_per_log: Dict[str, pd.DataFrame],
    corr_df: pd.DataFrame,  # rows=metrics, cols=logs -> rho
    pval_df: pd.DataFrame,  # rows=metrics, cols=logs -> p
    plateau_df: pd.DataFrame,  # rows=metrics, cols=logs -> plateau n (or NaN)
    out_csv_path: str,
    metric_columns: Optional[List[str]] = None,
    ref_sizes: Optional[List[int]] = None,
    measure_basis_map: Optional[Dict[str, str]] = None,
    pretty_plateau: bool = True,  # True -> convert NaN to '---'
) -> Tuple[str, str]:
    """
    Build the 'before' master table by *reading* correlations/p-values/plateau
    from precomputed DataFrames. Saves CSV and LaTeX.

    Output columns:
      log, metric, basis, rho_before, p_before, RelCI_<n1>, RelCI_<n2>, ..., plateau_n

    Notes
    -----
    - corr_df/pval_df/plateau_df are matrix-like: rows=metrics, columns=logs.
    - Table is sorted by metric, basis, then log. A per-(metric,basis) 'MEAN' row
      is appended *after* each group.
    """
    if ref_sizes is None:
        ref_sizes = [50, 250, 500]

    rows = []

    # Collect union of all metric columns if not provided
    if metric_columns is None:
        all_metrics: set[str] = set()
        for df in measures_per_log.values():
            id_cols = {"sample_size", "sample_id"}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_metrics.update([c for c in numeric_cols if c not in id_cols])
        metric_columns = sorted(all_metrics)

    # Sorting helpers
    metric_order = {metric: idx for idx, metric in enumerate(metric_columns)}
    basis_priority = ["ObsCount", "PopCount", "PopDist", "Concrete", "—"]
    basis_order = {b: i for i, b in enumerate(basis_priority)}

    # Build rows
    logs_sorted = sorted(measures_per_log.keys())
    for log in logs_sorted:
        rel_df = sample_ci_rel_width_per_log.get(log)
        if rel_df is None:
            raise ValueError(f"Missing relative CI width DF for log '{log}'.")
        rel_df = rel_df.set_index("sample_size").sort_index()

        # Filter metrics to present columns
        metric_cols = [m for m in metric_columns if m in measures_per_log[log].columns]
        if not metric_cols:
            continue

        for m in metric_cols:
            # rho/p
            rho = np.nan
            pval = np.nan
            if (m in corr_df.index) and (log in corr_df.columns):
                rho = corr_df.at[m, log]
            if (m in pval_df.index) and (log in pval_df.columns):
                pval = pval_df.at[m, log]

            # plateau
            plateau = np.nan
            if (m in plateau_df.index) and (log in plateau_df.columns):
                plateau = plateau_df.at[m, log]

            # Rel CI at requested sizes
            relci = {}
            for n in ref_sizes:
                val = rel_df[m].get(n, np.nan) if m in rel_df.columns else np.nan
                relci[f"RelCI_{n}"] = val

            basis = measure_basis_map.get(m, "—") if measure_basis_map else "—"

            rows.append(
                {
                    "log": log,
                    "metric": m,
                    "basis": basis,
                    "rho_before": rho,
                    "p_before": pval,
                    **relci,
                    "plateau_n": (
                        "---"
                        if (pretty_plateau and (not np.isfinite(plateau)))
                        else plateau
                    ),
                }
            )

    if not rows:
        raise ValueError("No rows produced; check inputs and metric names.")

    # Column order
    ref_cols = [f"RelCI_{n}" for n in ref_sizes]
    cols_order = (
        ["log", "metric", "basis", "rho_before", "p_before"] + ref_cols + ["plateau_n"]
    )

    table_df = pd.DataFrame(rows)
    table_df = table_df[[c for c in cols_order if c in table_df.columns]]

    # ---- Append per-(metric, basis) MEAN rows ----
    # Compute numeric means across logs for each (metric, basis)
    numeric_cols = ["rho_before", "p_before"] + ref_cols
    # Coerce plateau_n to numeric temporarily to avoid aggregation issues
    # (we don't average plateau_n; we will blank it in the MEAN row)
    tbl_num = table_df.copy()
    # Identify group means
    grp = tbl_num.groupby(["metric", "basis"], dropna=False)
    means = grp[numeric_cols].mean(numeric_only=True).reset_index()
    means.insert(0, "log", "MEAN")
    means["plateau_n"] = "---"  # leave blank/pretty for the mean row

    # Combine and sort
    table_df = pd.concat([table_df, means], ignore_index=True, sort=False)

    # Sorting keys: metric → basis → log (MEAN comes last)
    table_df["_metric_order"] = table_df["metric"].map(
        lambda x: metric_order.get(x, len(metric_columns))
    )
    table_df["_basis_order"] = table_df["basis"].map(
        lambda x: basis_order.get(x, len(basis_order))
    )
    # Ensure MEAN is last within each (metric,basis) group
    table_df["_log_order"] = (table_df["log"] == "MEAN").astype(int)

    table_df = (
        table_df.sort_values(
            ["_metric_order", "_basis_order", "_log_order", "log", "metric", "basis"]
        )
        .drop(columns=["_metric_order", "_basis_order", "_log_order"])
        .reset_index(drop=True)
    )

    # Save CSV
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    table_df.to_csv(out_csv_path, index=False)

    # Save LaTeX next to CSV
    out_tex_path = os.path.splitext(out_csv_path)[0] + ".tex"
    os.makedirs(os.path.dirname(out_tex_path) or ".", exist_ok=True)

    # Compact float formatting in LaTeX
    def _fmt(x: Any) -> str:
        try:
            if pd.isna(x):
                return ""
            if isinstance(x, (int, np.integer)) or (
                isinstance(x, float) and float(x).is_integer()
            ):
                return f"{int(x)}"
            return f"{float(x):.4g}"
        except Exception:
            return str(x)

    table_df.to_latex(
        out_tex_path,
        index=False,
        escape=True,
        na_rep="",
        formatters=None,
        float_format=_fmt,
    )

    return out_csv_path, out_tex_path


# LATEX helpers
# Create Latex output
def _stars(p: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p: P-value.

    Returns:
        String with significance stars (*, **, ***).
    """
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def corr_p_to_latex_stars(
    corr_df: pd.DataFrame, pval_df: pd.DataFrame, out_path: Path, label: str
) -> None:
    """Generate LaTeX table with correlation coefficients and significance stars.

    Args:
        corr_df: DataFrame with correlation coefficients.
        pval_df: DataFrame with p-values.
        out_path: Path to save the LaTeX file.
        label: LaTeX label for the table.
    """
    # keep P1..P4 order if present
    cols = [c for c in ["P1", "P2", "P3", "P4"] if c in corr_df.columns]
    corr = corr_df[cols].copy()
    pval = pval_df[cols].copy()
    corr, pval = corr.align(pval, join="outer", axis=0)

    # Build display DataFrame with "+/-r***" format
    disp = corr.copy().astype(object)
    for c in cols:
        out_col = []
        for r, p in zip(corr[c], pval[c]):
            if pd.isna(r):
                out_col.append("")
            else:
                out_col.append(f"{r:+.2f}{_stars(p)}")
        disp[c] = out_col

    latex_body = disp.to_latex(
        escape=True,
        na_rep="",
        index=True,
        column_format="l" + "c" * len(cols),
        bold_rows=False,
    )

    wrapped = rf"""
    \begin{{table}}[htbp]
    \label{{label}}
    \centering
    \caption{{Pearson correlation ($r$) between window size and each measure.}}
    \scriptsize
    \setlength{{\tabcolsep}}{{6pt}}
    \renewcommand{{\arraystretch}}{{1.15}}
    {latex_body}
    \vspace{{2pt}}
    \begin{{minipage}}{{0.95\linewidth}}\footnotesize
    Stars denote significance: $^*p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.
    \end{{minipage}}
    \end{{table}}
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(wrapped)
