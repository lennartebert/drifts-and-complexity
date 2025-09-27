import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import pandas as pd
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

def save_complexity_csv(dataset_key: str, configuration_name: str, df: pd.DataFrame) -> Path:
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

def load_data_dictionary(path: Path, get_real: bool = True, get_synthetic: bool = False) -> Dict[str, Any]:
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
    
def get_dataframe_from_drift_detection_results(datasets: List[str], cp_configurations: List[str]) -> pd.DataFrame:
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

                results.append({
                    "dataset": dataset,
                    "configuration": cp_configuration,
                    "change_point": change_point,
                    "change_moment": pd.to_datetime(change_moment, utc=True)
                })

    # convert results into dataframe
    if not results:
        return pd.DataFrame(columns=["dataset", "configuration", "change_point", "change_moment"])

    out = pd.DataFrame(results)
    return out.reset_index(drop=True)

# correlation helpers
def get_correlations_for_dictionary(
    sample_metrics_per_log: Dict[str, pd.DataFrame],
    rename_dictionary_map: Optional[Dict[str, str]],
    metric_columns: List[str],
    base_column: str = 'sample_size'
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
        col_tag = key if rename_map is None else rename_map[key]  # apply the rename map if available
        r_results[col_tag] = {}
        p_results[col_tag] = {}

        for col in df.columns:
            if col not in metric_columns:
                continue
            # drop missing values pairwise
            tmp = df[[base_column, col]].dropna()
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

def corr_p_to_latex_stars(corr_df: pd.DataFrame, pval_df: pd.DataFrame, out_path: Path, label: str) -> None:
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
        column_format="l" + "c"*len(cols),
        bold_rows=False
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
