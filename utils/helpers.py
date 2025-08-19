import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import json, yaml
import pandas as pd
from .constants import COMPLEXITY_RESULTS_DIR

def to_naive_ts(x):
    if x is None: return None
    ts = pd.to_datetime(x)
    try: return ts.tz_convert(None)
    except Exception: return ts

def save_complexity_csv(dataset_key: str, configuration_name: str, df: pd.DataFrame) -> Path:
    out_dir = COMPLEXITY_RESULTS_DIR / dataset_key / configuration_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "complexity.csv"
    df.to_csv(out, index=False)
    return out

def flatten_measurements(window_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in window_rows:
        base = {k: v for k, v in r.items() if k != "measurements"}
        base.update(r.get("measurements", {}))
        rows.append(base)
    return pd.DataFrame(rows)

def load_data_dictionary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    
def get_dataframe_from_drift_detection_results(datasets, cp_configurations):
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