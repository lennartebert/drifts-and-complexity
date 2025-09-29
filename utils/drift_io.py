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


def load_xes_log(path: Path) -> List[Trace]:
    """Load XES event log and sort by timestamp.

    Args:
        path: Path to the XES file.

    Returns:
        List of traces sorted by first event timestamp.
    """
    log = xes_importer.apply(str(path))
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
