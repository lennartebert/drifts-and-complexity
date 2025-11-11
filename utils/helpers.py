from __future__ import annotations

import json

ALPHA = 0.05  # Default significance level for Z-test
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
) -> pd.DataFrame:
    """Calculate both Pearson and Spearman correlations between sample size and metrics.

    Args:
        sample_metrics_per_log: Dictionary mapping log names to DataFrames with metrics.
        rename_dictionary_map: Optional mapping to rename log names in output.
        metric_columns: List of metric column names to analyze.
        base_column: Name of the base column for correlation (default: 'sample_size').

    Returns:
        Tuple of (pearson_r_df, pearson_p_df, spearman_r_df, spearman_p_df).
    """
    from scipy import stats

    rename_map = rename_dictionary_map

    results = []
    for col in metric_columns:
        # Aggregate across all logs
        pearson_r_list = []
        pearson_p_list = []
        spearman_r_list = []
        spearman_p_list = []
        for key, df in sample_metrics_per_log.items():
            if col not in df.columns:
                continue
            tmp = df[[base_column, col]].copy()
            tmp = tmp.replace([np.inf, -np.inf], np.nan)
            tmp[base_column] = pd.to_numeric(tmp[base_column], errors="coerce")
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.dropna()
            if len(tmp) < 2:
                pearson_r, pearson_p = float("nan"), float("nan")
                spearman_r, spearman_p = float("nan"), float("nan")
            else:
                x = tmp[base_column].to_numpy(dtype=float)
                y = tmp[col].to_numpy(dtype=float)
                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    pearson_r, pearson_p = float("nan"), float("nan")
                    spearman_r, spearman_p = float("nan"), float("nan")
                elif np.nanstd(x) == 0 or np.nanstd(y) == 0:
                    pearson_r, pearson_p = float("nan"), float("nan")
                    spearman_r, spearman_p = float("nan"), float("nan")
                else:
                    try:
                        pearson_r, pearson_p = stats.pearsonr(x, y)
                    except Exception:
                        pearson_r, pearson_p = float("nan"), float("nan")
                    try:
                        spearman_r, spearman_p = stats.spearmanr(x, y)
                    except Exception:
                        spearman_r, spearman_p = float("nan"), float("nan")
            pearson_r_list.append(pearson_r)
            pearson_p_list.append(pearson_p)
            spearman_r_list.append(spearman_r)
            spearman_p_list.append(spearman_p)

        # Aggregate across logs (mean, ignoring NaN)
        def nanmean(lst):
            arr = np.array(lst, dtype=float)
            return float(np.nanmean(arr)) if len(arr) > 0 else float("nan")

        results.append(
            {
                "Metric": col,
                "Pearson_Rho": nanmean(pearson_r_list),
                "Pearson_P": nanmean(pearson_p_list),
                "Spearman_Rho": nanmean(spearman_r_list),
                "Spearman_P": nanmean(spearman_p_list),
            }
        )
    df = pd.DataFrame(results)
    return df


import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

# Configuration constants for comparison table generation
TAU = 0.15  # threshold for |Pearson − Spearman|
APPLY_FPC = True  # whether to apply finite population correction
DEFAULT_N = 2000  # default sample size if missing
DEFAULT_NPOP = 10000  # default population size if missing


def compute_significant_improvement(
    measures_per_log: Dict[str, pd.DataFrame],  # AFTER fixes (contains rho for 'after')
    base_measures_per_log: Dict[
        str, pd.DataFrame
    ],  # BEFORE fixes (contains rho for 'before')
    include_metrics: List[str],
    n: int,  # sample size (e.g., len(SIZES) * SAMPLES_PER_SIZE)
    rho_col: str = "rho",  # correlation column name in BOTH inputs
    log_col: str = "log",
    metric_col: str = "metric",
    N: Optional[int] = None,  # population size (for FPC, optional)
    correlation_type: str = "Pearson",  # for column naming
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare bias before vs after using Fisher's z-test on correlations with FPC.

    Significant improvement rule:
        improved = (Z_test_p < alpha) AND (Abs_Delta_Chosen_Rho < 0)
        where Abs_Delta_Chosen_Rho = abs(rho_after) - abs(rho_before)
        (negative delta means improvement: correlation moved closer to 0)

    Parameters
    ----------
    measures_per_log : dict[log -> DataFrame]
        AFTER fixes. Must contain columns [metric_col, rho_col] (and optionally p).
    base_measures_per_log : dict[log -> DataFrame]
        BEFORE fixes. Must contain columns [metric_col, rho_col] (and optionally p).
    include_metrics : list[str]
        Metrics to compare.
    n : int
        Sample size (e.g., len(SIZES) * SAMPLES_PER_SIZE).
    rho_col : str
        Column name for correlation (same in both inputs).
    log_col, metric_col : str
        Identifier column names.
    N : int, optional
        Population size for finite population correction (FPC).
        If None, FPC is not applied.
    correlation_type : str
        Type of correlation ("Pearson" or "Spearman") for column naming.
    alpha : float
        Significance level for Fisher z-test.

    Returns
    -------
    per_log_df : DataFrame
        Columns: [log, metric, rho_before, rho_after, delta_abs_rho, n, z_change, p_change, improved]
        Column names are prefixed with correlation_type (e.g., "Pearson_rho_before").
    summary_df : DataFrame
        Columns: [metric, share_improved, n_logs]
    """
    DECIMALS = 4  # harmonize precision across inputs

    def fisher_z_diff_independent(
        r1: float,
        r2: float,
        n1: int,
        n2: int,
        N1: Optional[int] = None,
        N2: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Two-sided Fisher r-to-z test for the difference between two *independent* correlations.
        Supports optional finite population correction (FPC) per group.

        Returns (z_stat, p_value).
        """
        # Basic checks
        if (
            (n1 is None or n2 is None)
            or (n1 < 4 or n2 < 4)
            or np.isnan(r1)
            or np.isnan(r2)
        ):
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
            if N is None or N <= n or N <= 1:
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

    rows = []

    for log in sorted(measures_per_log.keys()):
        if log not in base_measures_per_log:
            continue

        df_after = measures_per_log[log].copy()
        df_before = base_measures_per_log[log].copy()

        # keep only requested metrics
        df_after = df_after[df_after[metric_col].isin(include_metrics)].copy()
        df_before = df_before[df_before[metric_col].isin(include_metrics)].copy()

        # ensure rho is numeric and harmonized to 4 decimals
        for df in (df_after, df_before):
            if rho_col not in df.columns:
                raise ValueError(f"Missing column '{rho_col}' in data for log '{log}'.")
            df[rho_col] = pd.to_numeric(df[rho_col], errors="coerce").round(DECIMALS)

        merged = (
            pd.merge(
                df_before[[metric_col, rho_col]].rename(
                    columns={rho_col: f"{correlation_type}_{rho_col}_before"}
                ),
                df_after[[metric_col, rho_col]].rename(
                    columns={rho_col: f"{correlation_type}_{rho_col}_after"}
                ),
                on=metric_col,
                how="inner",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(
                subset=[
                    f"{correlation_type}_{rho_col}_before",
                    f"{correlation_type}_{rho_col}_after",
                ]
            )
        )

        if merged.empty:
            continue

        # compute stats per metric
        merged[log_col] = log
        abs_before = merged[f"{correlation_type}_{rho_col}_before"].abs()
        abs_after = merged[f"{correlation_type}_{rho_col}_after"].abs()
        merged[f"{correlation_type}_delta_abs_rho"] = (abs_after - abs_before).round(
            DECIMALS
        )

        # Fisher z-test
        z_list = []
        p_list = []
        imp_list = []
        for m, rb, ra in zip(
            merged[metric_col],
            merged[f"{correlation_type}_{rho_col}_before"],
            merged[f"{correlation_type}_{rho_col}_after"],
        ):
            z, p = fisher_z_diff_independent(rb, ra, n, n, N, N)
            z_list.append(z)
            p_list.append(p)
            improved = (abs(ra) < abs(rb)) and (not np.isnan(p)) and (p < alpha)
            imp_list.append(improved)

        merged[f"{correlation_type}_z_change"] = z_list
        merged[f"{correlation_type}_p_change"] = p_list
        merged[f"{correlation_type}_improved"] = imp_list

        rows.append(
            merged[
                [
                    log_col,
                    metric_col,
                    f"{correlation_type}_{rho_col}_before",
                    f"{correlation_type}_{rho_col}_after",
                    f"{correlation_type}_delta_abs_rho",
                    f"{correlation_type}_z_change",
                    f"{correlation_type}_p_change",
                    f"{correlation_type}_improved",
                ]
            ]
        )

    if not rows:
        per_log_df = pd.DataFrame(
            columns=[
                log_col,
                metric_col,
                f"{correlation_type}_{rho_col}_before",
                f"{correlation_type}_{rho_col}_after",
                f"{correlation_type}_delta_abs_rho",
                f"{correlation_type}_z_change",
                f"{correlation_type}_p_change",
                f"{correlation_type}_improved",
            ]
        )
        summary_df = pd.DataFrame(
            columns=[metric_col, f"{correlation_type}_share_improved", "n_logs"]
        )
        return per_log_df, summary_df

    per_log_df = pd.concat(rows, ignore_index=True)

    # summary across logs per metric
    summary_df = per_log_df.groupby(metric_col, as_index=False).agg(
        **{
            f"{correlation_type}_share_improved": (
                f"{correlation_type}_improved",
                "mean",
            )
        },
        n_logs=(f"{correlation_type}_improved", "size"),
    )

    return per_log_df, summary_df


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
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    out_path: Path,
    label: str,
    correlation_type: str = "Pearson",
) -> None:
    """Generate LaTeX table with correlation coefficients and significance stars.

    Args:
        corr_df: DataFrame with correlation coefficients.
        pval_df: DataFrame with p-values.
        out_path: Path to save the LaTeX file.
        label: LaTeX label for the table.
        correlation_type: Type of correlation ("Pearson" or "Spearman").
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
    \label{{{label}}}
    \centering
    \caption{{{correlation_type} correlation ($r$) between window size and each measure.}}
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
