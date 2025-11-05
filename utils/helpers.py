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


from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_significant_improvement(
    measures_per_log: Dict[str, pd.DataFrame],  # AFTER fixes (contains rho for 'after')
    base_measures_per_log: Dict[
        str, pd.DataFrame
    ],  # BEFORE fixes (contains rho for 'before')
    include_metrics: List[str],
    rho_col: str = "rho",  # correlation column name in BOTH inputs
    log_col: str = "log",
    metric_col: str = "metric",
    num_window_sizes: Union[int, Dict[str, int], Dict[Tuple[str, str], int]] = 0,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare bias before vs after using Fisher's z-test on correlations.

    Significant improvement rule:
        improved = (abs(rho_after) < abs(rho_before)) AND (p_change < alpha)

    Parameters
    ----------
    measures_per_log : dict[log -> DataFrame]
        AFTER fixes. Must contain columns [metric_col, rho_col] (and optionally p).
    base_measures_per_log : dict[log -> DataFrame]
        BEFORE fixes. Must contain columns [metric_col, rho_col] (and optionally p).
    include_metrics : list[str]
        Metrics to compare.
    rho_col : str
        Column name for correlation (same in both inputs).
    log_col, metric_col : str
        Identifier column names.
    num_window_sizes : int | dict
        Number of window sizes (k) used per correlation.
        - If int: same k for all (log, metric).
        - If dict[str -> int]: per-log k (key = log).
        - If dict[(log, metric) -> int]: per-(log, metric) k.
    alpha : float
        Significance level for Fisher z-test.

    Returns
    -------
    per_log_df : DataFrame
        Columns: [log, metric, rho_before, rho_after, delta_abs_rho, k, z_change, p_change, improved]
    summary_df : DataFrame
        Columns: [metric, share_improved, n_logs]
    """
    DECIMALS = 4  # harmonize precision across inputs

    def fisher_z_test(r_before: float, r_after: float, k: int) -> Tuple[float, float]:
        """Two-sided Fisher z-test for difference between two correlations."""
        if (k is None) or (k < 4) or np.isnan(r_before) or np.isnan(r_after):
            return np.nan, np.nan
        # guard against |r| >= 1
        if not (-0.999999 < r_before < 0.999999) or not (
            -0.999999 < r_after < 0.999999
        ):
            return np.nan, np.nan
        z1 = np.arctanh(r_before)
        z2 = np.arctanh(r_after)
        se = np.sqrt(2.0 / (k - 3))
        z_stat = (z1 - z2) / se
        p_val = 2 * (1 - norm.cdf(abs(z_stat)))
        return float(z_stat), float(p_val)

    def get_k(log: str, metric: str) -> int:
        if isinstance(num_window_sizes, int):
            return num_window_sizes
        if isinstance(num_window_sizes, dict):
            # try (log, metric) key first
            if (log, metric) in num_window_sizes:
                return int(num_window_sizes[(log, metric)])
            # fall back to per-log
            if log in num_window_sizes:
                return int(num_window_sizes[log])
        return 0  # unknown

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
                    columns={rho_col: f"{rho_col}_before"}
                ),
                df_after[[metric_col, rho_col]].rename(
                    columns={rho_col: f"{rho_col}_after"}
                ),
                on=metric_col,
                how="inner",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[f"{rho_col}_before", f"{rho_col}_after"])
        )

        if merged.empty:
            continue

        # compute stats per metric
        merged[log_col] = log
        abs_before = merged[f"{rho_col}_before"].abs()
        abs_after = merged[f"{rho_col}_after"].abs()
        merged["delta_abs_rho"] = (abs_after - abs_before).round(DECIMALS)

        # Fisher z-test
        k_list = []
        z_list = []
        p_list = []
        imp_list = []
        for m, rb, ra in zip(
            merged[metric_col], merged[f"{rho_col}_before"], merged[f"{rho_col}_after"]
        ):
            k = get_k(str(log), str(m))
            z, p = fisher_z_test(rb, ra, k)
            k_list.append(k)
            z_list.append(z)
            p_list.append(p)
            improved = (abs(ra) < abs(rb)) and (not np.isnan(p)) and (p < alpha)
            imp_list.append(improved)

        merged["k"] = k_list
        merged["z_change"] = z_list
        merged["p_change"] = p_list
        merged["improved"] = imp_list

        rows.append(
            merged[
                [
                    log_col,
                    metric_col,
                    f"{rho_col}_before",
                    f"{rho_col}_after",
                    "delta_abs_rho",
                    "k",
                    "z_change",
                    "p_change",
                    "improved",
                ]
            ]
        )

    if not rows:
        per_log_df = pd.DataFrame(
            columns=[
                log_col,
                metric_col,
                f"{rho_col}_before",
                f"{rho_col}_after",
                "delta_abs_rho",
                "k",
                "z_change",
                "p_change",
                "improved",
            ]
        )
        summary_df = pd.DataFrame(columns=[metric_col, "share_improved", "n_logs"])
        return per_log_df, summary_df

    per_log_df = pd.concat(rows, ignore_index=True)

    # summary across logs per metric
    summary_df = per_log_df.groupby(metric_col, as_index=False).agg(
        share_improved=("improved", "mean"),
        n_logs=("improved", "size"),
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


########################################################
# Master table builder


# ----------------------------
# Helpers
# ----------------------------


def _escape_latex(text: Any) -> str:
    """Escape underscores in text, preserving \textit{...} wrappers."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if s.startswith("\\textit{") and s.endswith("}"):
        return s
    return s.replace("_", "\\_")


def _validate_improvement_inputs(
    improvement_per_log_df: Optional[pd.DataFrame],
    improvement_summary_df: Optional[pd.DataFrame],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Light schema checks + type normalization."""
    ipl = improvement_per_log_df
    ims = improvement_summary_df

    if ipl is not None:
        required = {"log", "metric", "improved"}
        missing = required - set(ipl.columns)
        if missing:
            raise ValueError(f"improvement_per_log_df missing columns: {missing}")
        # normalize dtypes
        ipl = ipl.copy()
        ipl["log"] = ipl["log"].astype(str)
        ipl["metric"] = ipl["metric"].astype(str)
        # coerce to bool (allow 0/1, 'True'/'False', etc.)
        ipl["improved"] = ipl["improved"].astype(bool)

    if ims is not None:
        required = {"metric", "share_improved"}
        missing = required - set(ims.columns)
        if missing:
            raise ValueError(f"improvement_summary_df missing columns: {missing}")
        ims = ims.copy()
        ims["metric"] = ims["metric"].astype(str)
        # numeric share in [0,1]
        ims["share_improved"] = pd.to_numeric(ims["share_improved"], errors="coerce")

    return ipl, ims


# ----------------------------
# 1) Build & save CSV (returns DataFrame too)
# ----------------------------


def build_master_table_csv(
    measures_per_log: Dict[str, pd.DataFrame],
    sample_ci_rel_width_per_log: Dict[str, pd.DataFrame],
    corr_df: pd.DataFrame,  # rows=metrics, cols=logs -> rho
    pval_df: pd.DataFrame,  # rows=metrics, cols=logs -> p
    plateau_df: pd.DataFrame,  # rows=metrics, cols=logs -> plateau n (or NaN)
    out_csv_path: str,
    metric_columns: Optional[List[str]] = None,
    ref_sizes: Optional[List[int]] = None,
    measure_basis_map: Optional[Dict[str, str]] = None,
    pretty_plateau: bool = True,
    improvement_per_log_df: Optional[
        pd.DataFrame
    ] = None,  # cols: log, metric, improved (bool)
    improvement_summary_df: Optional[
        pd.DataFrame
    ] = None,  # cols: metric, share_improved
) -> Tuple[str, pd.DataFrame]:
    """
    Assemble the master table and write CSV.
    Returns (csv_path, table_df_with_mean_rows).

    - Integrates per-log 'improved' if provided (merged by log+metric).
    - Integrates mean-row 'share_improved' if provided (merged by metric).
    - Writes both into a single CSV column 'improvement':
        * Per-log rows -> "Yes" if improved else ""
        * Mean rows    -> share_improved formatted to 4 decimals
    """
    if ref_sizes is None:
        ref_sizes = [50, 250, 500]

    improvement_per_log_df, improvement_summary_df = _validate_improvement_inputs(
        improvement_per_log_df, improvement_summary_df
    )

    rows = []

    # Collect union of all metric columns if not provided
    if metric_columns is None:
        all_metrics: set[str] = set()
        for df in measures_per_log.values():
            id_cols = {"sample_size", "sample_id"}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_metrics.update([c for c in numeric_cols if c not in id_cols])
        metric_columns = sorted(all_metrics)

    # Sort helpers
    metric_order = {metric: idx for idx, metric in enumerate(metric_columns)}
    basis_priority = ["ObsCount", "PopCount", "PopDist", "Concrete", "—"]
    basis_order = {b: i for i, b in enumerate(basis_priority)}

    logs_sorted = sorted(measures_per_log.keys())

    for log in logs_sorted:
        rel_df = sample_ci_rel_width_per_log.get(log)
        if rel_df is None:
            raise ValueError(f"Missing relative CI width DF for log '{log}'.")
        rel_df = rel_df.set_index("sample_size").sort_index()

        metric_cols = [m for m in metric_columns if m in measures_per_log[log].columns]
        if not metric_cols:
            continue

        for m in metric_cols:
            rho = np.nan
            pval = np.nan
            if (m in corr_df.index) and (log in corr_df.columns):
                rho = corr_df.at[m, log]
            if (m in pval_df.index) and (log in pval_df.columns):
                pval = pval_df.at[m, log]

            plateau = np.nan
            if (m in plateau_df.index) and (log in plateau_df.columns):
                plateau = plateau_df.at[m, log]

            relci = {}
            for n in ref_sizes:
                val = rel_df[m].get(n, np.nan) if m in rel_df.columns else np.nan
                relci[f"RelCI_{n}"] = val

            basis = measure_basis_map.get(m, "—") if measure_basis_map else "—"

            row = {
                "metric": m,
                "basis": basis,
                "log": str(log),
                "rho": rho,
                "p": pval,
                **relci,
                "plateau_n": (
                    "---"
                    if (pretty_plateau and (not np.isfinite(plateau)))
                    else plateau
                ),
            }

            # Merge per-log improvement flag if provided
            if improvement_per_log_df is not None:
                sub = improvement_per_log_df[
                    (improvement_per_log_df["log"] == str(log))
                    & (improvement_per_log_df["metric"] == str(m))
                ]
                if not sub.empty:
                    row["improved"] = bool(sub.iloc[0]["improved"])

            rows.append(row)

    if not rows:
        raise ValueError("No rows produced; check inputs and metric names.")

    # Build DataFrame in requested order
    ref_cols = [f"RelCI_{n}" for n in ref_sizes]
    base_cols = ["metric", "basis", "log", "rho", "p"] + ref_cols + ["plateau_n"]
    # keep originals internally (not necessarily in CSV)
    if any(("improved" in r) for r in rows):
        base_cols.append("improved")

    table_df = pd.DataFrame(rows)
    table_df = table_df[[c for c in base_cols if c in table_df.columns]]

    # ---- Mean rows per (metric, basis) ----
    numeric_cols = ["rho", "p"] + [c for c in ref_cols if c in table_df.columns]
    grp = table_df.groupby(["metric", "basis"], dropna=False)
    means = grp[numeric_cols].mean(numeric_only=True).reset_index()
    means.insert(2, "log", "")  # blank identifier on mean row
    means["_is_mean"] = True
    means["plateau_n"] = "---"

    # Merge share_improved on mean rows only
    if improvement_summary_df is not None:
        means = means.merge(
            improvement_summary_df[["metric", "share_improved"]],
            on="metric",
            how="left",
        )
        # carry the column in per-log rows too (will be blank there)
        table_df["share_improved"] = np.nan

    # Per-log 'improved' should not appear on mean rows
    means["improved"] = np.nan

    # Combine
    table_df["_is_mean"] = False
    table_df = pd.concat([table_df, means], ignore_index=True, sort=False)

    # ---- Create combined 'improvement' column ----
    # Per-log: "Yes" if improved else ""
    if "improved" in table_df.columns:
        table_df["improvement"] = table_df["improved"].apply(
            lambda v: "Yes" if (isinstance(v, (bool, np.bool_)) and v) else ""
        )
    else:
        table_df["improvement"] = ""

    # Mean rows: write share_improved with 4 decimals (overwriting per-log text in those rows)
    if "share_improved" in table_df.columns:
        mean_mask = table_df["_is_mean"] == True
        table_df.loc[mean_mask, "improvement"] = table_df.loc[
            mean_mask, "share_improved"
        ].apply(lambda x: ("" if pd.isna(x) else f"{float(x):.4f}"))

    # Sorting: metric → basis → log; mean rows last
    table_df["_metric_order"] = table_df["metric"].map(
        lambda x: metric_order.get(x, len(metric_columns))
    )
    table_df["_basis_order"] = table_df["basis"].map(
        lambda x: basis_order.get(x, len(basis_order))
    )
    table_df["_log_order"] = table_df["_is_mean"].astype(int)

    table_df = (
        table_df.sort_values(
            ["_metric_order", "_basis_order", "_log_order", "log", "metric", "basis"]
        )
        .drop(columns=["_metric_order", "_basis_order", "_log_order"])
        .reset_index(drop=True)
    )

    # Final CSV order: include only the combined column (not the separate ones)
    cols_order = (
        ["metric", "basis", "log", "rho", "p"] + ref_cols + ["plateau_n", "improvement"]
    )
    cols_order_csv = [c for c in cols_order if c in table_df.columns]
    # keep helper flag at end if present (for downstream LaTeX)
    if "_is_mean" in table_df.columns:
        cols_order_csv = cols_order_csv + ["_is_mean"]
    # (We keep 'improved' and 'share_improved' in the DataFrame but omit from CSV to avoid duplication)

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    table_df[cols_order_csv].to_csv(out_csv_path, index=False)

    # Return DataFrame (still contains 'improved'/'share_improved' internally if present)
    return out_csv_path, table_df[cols_order_csv]


# ----------------------------
# 2) Render LaTeX
# ----------------------------


def render_master_table_latex(
    table_df: pd.DataFrame,
    out_tex_path: str,
    caption: str = "Assessment of Measures before Applying Remedies",
    label: str = "tab:master_table_before",
) -> str:
    """
    Pretty LaTeX rendering for the master table DataFrame produced by build_master_table_csv(...).
    - Suppresses repeats for 'log' and 'basis' within (metric,basis) blocks
    - Mean rows italicized; metric/basis/log blanked on mean rows
    - Always 4 decimals for numeric; no scientific notation
    - Escapes underscores in headers and cell content
    - Wrapped in a full table environment
    """
    disp = table_df.copy()

    # Suppress repeats for 'log' and 'basis' within (metric,basis) blocks
    suppress_cols = [c for c in ["log", "basis"] if c in disp.columns]
    last_seen = {c: (None, None) for c in suppress_cols}
    if suppress_cols:
        for idx, row in disp.iterrows():
            group_key = (row["metric"], row["basis"])
            for c in suppress_cols:
                prev_val, prev_key = last_seen[c]
                if prev_key != group_key:
                    prev_val = None
                if row[c] == prev_val:
                    disp.at[idx, c] = ""
                last_seen[c] = (row[c], group_key)

    # Mean row: blank metric/basis/log completely
    mean_mask = (
        disp["_is_mean"] == True
        if "_is_mean" in disp.columns
        else pd.Series(False, index=disp.index)
    )
    for col in ["metric", "basis", "log"]:
        if col in disp.columns:
            disp.loc[mean_mask, col] = ""

    # ---- Fixed 4-decimal formatting for numerics ----
    ref_cols = [c for c in disp.columns if c.startswith("RelCI_")]
    numeric_cols_present = [
        c for c in disp.columns if c in (["rho", "p"] + ref_cols + ["share_improved"])
    ]

    def fmt_num(x: Any) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        try:
            return f"{float(x):.4f}"
        except Exception:
            return _escape_latex(x)

    for col in numeric_cols_present:
        disp[col] = disp[col].map(fmt_num)

    # Format 'improved' (bool) → Yes / ""
    if "improved" in disp.columns:
        disp["improved"] = disp["improved"].map(
            lambda v: (
                "Yes"
                if (isinstance(v, (bool, np.bool_)) and v)
                else (
                    ""
                    if (v is None or (isinstance(v, float) and pd.isna(v)))
                    else str(v)
                )
            )
        )
        # blank 'improved' in mean rows explicitly
        disp.loc[mean_mask, "improved"] = ""

    # Escape underscores in text columns
    for col in disp.columns:
        if col in numeric_cols_present or col in {"_is_mean"}:
            continue
        disp[col] = disp[col].map(_escape_latex)

    # Italicize entire mean row (after formatting/escaping)
    for col in disp.columns:
        if col == "_is_mean":
            continue
        disp.loc[mean_mask, col] = disp.loc[mean_mask, col].apply(
            lambda s: (f"\\textit{{{s}}}" if s not in (None, "", np.nan) else s)
        )

    # Drop helper
    if "_is_mean" in disp.columns:
        disp = disp.drop(columns=["_is_mean"])

    # Escape headers
    disp.columns = [_escape_latex(c) for c in disp.columns]

    # Build LaTeX tabular
    tabular_str = disp.to_latex(index=False, escape=False, na_rep="")

    # Wrap with table env
    os.makedirs(os.path.dirname(out_tex_path) or ".", exist_ok=True)
    table_env = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{_escape_latex(caption)}}}\n"
        "\\small\n"
        f"{tabular_str}\n"
        f"\\label{{{_escape_latex(label)}}}\n"
        "\\end{table}\n"
    )
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_env)

    return out_tex_path


# ----------------------------
# Convenience wrapper
# ----------------------------


def create_master_table(
    measures_per_log: Dict[str, pd.DataFrame],
    sample_ci_rel_width_per_log: Dict[str, pd.DataFrame],
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    plateau_df: pd.DataFrame,
    out_csv_path: str,
    metric_columns: Optional[List[str]] = None,
    ref_sizes: Optional[List[int]] = None,
    measure_basis_map: Optional[Dict[str, str]] = None,
    pretty_plateau: bool = True,
    caption: str = "Assessment of Measures before Applying Remedies",
    label: str = "tab:master_table_before",
    improvement_per_log_df: Optional[pd.DataFrame] = None,
    improvement_summary_df: Optional[pd.DataFrame] = None,
) -> Tuple[str, str]:
    """
    Build CSV first, then render LaTeX via the helper.
    """
    csv_path, table_df = build_master_table_csv(
        measures_per_log=measures_per_log,
        sample_ci_rel_width_per_log=sample_ci_rel_width_per_log,
        corr_df=corr_df,
        pval_df=pval_df,
        plateau_df=plateau_df,
        out_csv_path=out_csv_path,
        metric_columns=metric_columns,
        ref_sizes=ref_sizes,
        measure_basis_map=measure_basis_map,
        pretty_plateau=pretty_plateau,
        improvement_per_log_df=improvement_per_log_df,
        improvement_summary_df=improvement_summary_df,
    )

    out_tex_path = os.path.splitext(csv_path)[0] + ".tex"
    render_master_table_latex(
        table_df=table_df,
        out_tex_path=out_tex_path,
        caption=caption,
        label=label,
    )
    return csv_path, out_tex_path


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
    df = pd.read_csv(master_csv_path)

    # Guard: must have the mean-flag column
    if "_is_mean" not in df.columns:
        raise ValueError(
            "Master CSV is missing '_is_mean'. Please generate with the full function first."
        )

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
    def _fmt_num(x):
        if pd.isna(x):
            return ""
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    numeric_cols = [c for c in m.columns if pd.api.types.is_numeric_dtype(m[c])]
    for c in numeric_cols:
        m[c] = m[c].map(_fmt_num)

    # ---- Escape underscores in headers and in string cells ----
    def _esc(s: str) -> str:
        if s is None or s == "":
            return ""
        return str(s).replace("_", "\\_")

    # Escape cell contents in non-numeric columns
    for c in m.columns:
        if c not in numeric_cols:
            m[c] = m[c].map(_esc)

    # Escape headers
    m.columns = [_esc(c) for c in m.columns]

    # ---- Italicize the entire row (means) after formatting & escaping ----
    def _ital(s):
        return "" if s in ("", None) else f"\\textit{{{s}}}"

    for c in m.columns:
        m[c] = m[c].map(_ital)

    # ---- Save CSV (plain, un-italicized, same column order) ----
    # For CSV, we generally want plain text; write a plain version without italics.
    csv_plain = pd.read_csv(master_csv_path)
    csv_means = csv_plain[csv_plain["_is_mean"] == True].copy()
    csv_means = csv_means.drop(columns=["log", "_is_mean"], errors="ignore")
    csv_cols = ["metric", "basis"] + [
        c for c in csv_means.columns if c not in {"metric", "basis"}
    ]
    csv_means = csv_means[csv_cols]
    csv_means.to_csv(means_csv_path, index=False)

    # ---- Build LaTeX table ----
    tabular_str = m.to_latex(index=False, escape=False, na_rep="")
    table_env = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{_esc(caption)}}}\n"
        "\\small\n"
        f"{tabular_str}\n"
        f"\\label{{{_esc(label)}}}\n"
        "\\end{table}\n"
    )
    with open(means_tex_path, "w", encoding="utf-8") as f:
        f.write(table_env)


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
