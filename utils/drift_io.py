"""I/O utilities for drift detection results and event logs."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog, Trace


def get_dataframe_from_drift_detection_results(path: Path) -> pd.DataFrame:
    """Load drift detection results from CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing drift detection results.
    """
    return pd.read_csv(path)


def load_xes_log(path: Path, activity_key: str = "concept:name") -> List[Trace]:
    """Load XES event log and sort by timestamp.

    Args:
        path: Path to the XES file.
        activity_key: Event attribute key to treat as activity name.
            Defaults to "concept:name". If another key is provided, event labels
            are normalized by copying that key to "concept:name" for downstream
            PM4Py compatibility.

    Returns:
        List of traces sorted by first event timestamp.
    """
    log = xes_importer.apply(str(path))

    if not activity_key or not isinstance(activity_key, str):
        raise ValueError("activity_key must be a non-empty string")

    if activity_key != "concept:name":
        missing_count = 0
        for trace in log:
            for event in trace:
                if activity_key in event:
                    event["concept:name"] = event[activity_key]
                elif "concept:name" not in event:
                    missing_count += 1
        if missing_count:
            raise KeyError(
                f"Missing activity key '{activity_key}' in {missing_count} event(s) "
                f"for log: {path}"
            )

    return sorted(log, key=lambda tr: tr[0]["time:timestamp"])


def drift_info_to_dict(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Convert drift detection DataFrame to dictionary indexed by change point ID.

    Args:
        df: DataFrame with drift detection results.

    Returns:
        Dictionary indexed by calc_change_id.
    """
    # index by calc_change_id for easy lookup (keeps 'na' if present)
    return df.set_index("calc_change_id").to_dict(orient="index")


def only_real_change_points(
    drift_info_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Filter out non-real change points from drift info dictionary.

    Args:
        drift_info_by_id: Dictionary indexed by change point ID.

    Returns:
        Dictionary with only real change points (excluding 'na' entries).
    """
    return {k: v for k, v in drift_info_by_id.items() if k != "na"}
