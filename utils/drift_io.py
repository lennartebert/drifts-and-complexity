from pathlib import Path
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

def get_dataframe_from_drift_detection_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_xes_log(path: Path):
    log = xes_importer.apply(str(path))
    return sorted(log, key=lambda tr: tr[0]["time:timestamp"])

def drift_info_to_dict(df: pd.DataFrame) -> dict:
    # index by calc_change_id for easy lookup (keeps 'na' if present)
    return df.set_index("calc_change_id").to_dict(orient="index")

def only_real_change_points(drift_info_by_id: dict) -> dict:
    return {k: v for k, v in drift_info_by_id.items() if k != "na"}