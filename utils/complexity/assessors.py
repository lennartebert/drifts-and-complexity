"""Complexity assessment utilities for different windowing strategies."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pm4py.objects.log.obj import EventLog

from utils.complexity.metrics_adapters.metrics_adapter import get_adapters
from utils.helpers import save_complexity_csv
from utils.windowing.helpers import (
    split_log_into_fixed_comparable_windows,
    split_log_into_fixed_windows,
    split_log_into_windows_by_change_points,
)
from utils.windowing.window import Window


def _cfg_name(base_cfg: str, approach: str) -> str:
    """Generate configuration name by combining base config and approach.

    Args:
        base_cfg: Base configuration name.
        approach: Approach name.

    Returns:
        Combined configuration name.
    """
    return f"{base_cfg}__{approach}"


def _flatten_adapter_results(
    windows: List[Window],
    per_adapter: List[Tuple[str, Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]]],
    add_prefix: bool = True,
    include_adapter_name: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Flatten adapter results into a single dictionary per window.

    Args:
        windows: List of Window objects.
        per_adapter: List of (adapter_name, result_dict) where result_dict maps
                    window_id to (metrics_dict, info_dict).
        add_prefix: Whether to add "measure_" and "info_" prefixes.
        include_adapter_name: Whether to include adapter name in keys.

    Returns:
        Dictionary mapping window_id to flattened results.
    """
    out: Dict[str, Dict[str, Any]] = {w.id: {} for w in windows}

    for adapter_name, result in per_adapter:
        for w in windows:
            metrics, info = result[w.id]
            if add_prefix:
                # prefix "measure_" / "info_" and optionally namespace by adapter
                def _k(pfx: str, k: str) -> str:
                    if include_adapter_name:
                        return f"{pfx}{adapter_name}::{k}"
                    return f"{pfx}{k}"

                out[w.id].update({_k("measure_", k): v for k, v in metrics.items()})
                out[w.id].update({_k("info_", k): v for k, v in info.items()})
            else:
                # raw keys, no prefixes
                out[w.id].update(metrics)
                out[w.id].update(info)
    return out


def run_metric_adapters(
    windows: List[Window],
    adapter_names: Iterable[str],
    add_prefix: bool = True,
    include_adapter_name: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run metric adapters on windows and return flattened results.

    Args:
        windows: List of Window objects.
        adapter_names: Names of adapters to run.
        add_prefix: Whether to add "measure_" and "info_" prefixes.
        include_adapter_name: Whether to include adapter name in keys.

    Returns:
        Dictionary mapping window_id to flattened results.
    """
    per_adapter = []
    for adapter in get_adapters(adapter_names):
        per_adapter.append((adapter.name, adapter.compute_many(windows)))
    return _flatten_adapter_results(
        windows,
        per_adapter,
        add_prefix=add_prefix,
        include_adapter_name=include_adapter_name,
    )


def _materialize_rows(
    windows: List[Window], merged: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Convert windows and merged results into a DataFrame.

    Args:
        windows: List of Window objects.
        merged: Dictionary mapping window_id to results.

    Returns:
        DataFrame with window data and results.
    """
    rows = []
    for w in windows:
        rows.append({**w.to_dict(), **merged[w.id]})
    return pd.DataFrame(rows)


def assess_complexity_via_change_point_split(
    pm4py_log: EventLog,
    drift_info_by_id: Dict[str, Any],
    dataset_key: str,
    configuration_name: str,
    approach_name: str,
    adapter_names: Iterable[str],
    add_prefix: bool = True,
    include_adapter_name: bool = False,
) -> pd.DataFrame:
    """Assess complexity by splitting log at change points.

    Args:
        pm4py_log: Event log to analyze.
        drift_info_by_id: Dictionary mapping change point IDs to drift information.
        dataset_key: Name of the dataset.
        configuration_name: Name of the configuration.
        approach_name: Name of the approach.
        adapter_names: Names of metric adapters to use.
        add_prefix: Whether to add prefixes to metric names.
        include_adapter_name: Whether to include adapter name in metric names.

    Returns:
        DataFrame with complexity assessment results.
    """
    cp_info = {k: v for k, v in drift_info_by_id.items() if k != "na"}
    cps = (
        [
            (cp_info[i]["calc_change_index"], i, cp_info[i]["calc_change_type"])
            for i in sorted(cp_info.keys())
        ]
        if cp_info
        else []
    )
    windows = split_log_into_windows_by_change_points(pm4py_log, cps)

    merged = run_metric_adapters(
        windows,
        adapter_names,
        add_prefix=add_prefix,
        include_adapter_name=include_adapter_name,
    )
    df = _materialize_rows(windows, merged)
    save_complexity_csv(dataset_key, _cfg_name(configuration_name, approach_name), df)
    return df


def assess_complexity_via_fixed_sized_windows(
    pm4py_log: EventLog,
    window_size: int,
    offset: int,
    dataset_key: str,
    configuration_name: str,
    approach_name: str,
    adapter_names: Iterable[str],
    drift_info_by_id: Optional[Dict[str, Any]] = None,
    add_prefix: bool = True,
    include_adapter_name: bool = False,
) -> pd.DataFrame:
    """Assess complexity using fixed-sized windows.

    Args:
        pm4py_log: Event log to analyze.
        window_size: Size of each window.
        offset: Offset between windows.
        dataset_key: Name of the dataset.
        configuration_name: Name of the configuration.
        approach_name: Name of the approach.
        adapter_names: Names of metric adapters to use.
        drift_info_by_id: Optional drift information (unused in this method).
        add_prefix: Whether to add prefixes to metric names.
        include_adapter_name: Whether to include adapter name in metric names.

    Returns:
        DataFrame with complexity assessment results.
    """
    windows = split_log_into_fixed_windows(pm4py_log, window_size, offset)
    merged = run_metric_adapters(
        windows,
        adapter_names,
        add_prefix=add_prefix,
        include_adapter_name=include_adapter_name,
    )
    df = _materialize_rows(windows, merged)
    save_complexity_csv(dataset_key, _cfg_name(configuration_name, approach_name), df)
    return df


def assess_complexity_via_window_comparison(
    pm4py_log: EventLog,
    window_1_size: int,
    window_2_size: int,
    offset: int,
    step: int,
    dataset_key: str,
    configuration_name: str,
    approach_name: str,
    adapter_names: Iterable[str],
    add_prefix: bool = True,
    include_adapter_name: bool = False,
) -> pd.DataFrame:
    """Assess complexity by comparing pairs of windows.

    Args:
        pm4py_log: Event log to analyze.
        window_1_size: Size of first window in each pair.
        window_2_size: Size of second window in each pair.
        offset: Offset between windows.
        step: Step size for window positioning.
        dataset_key: Name of the dataset.
        configuration_name: Name of the configuration.
        approach_name: Name of the approach.
        adapter_names: Names of metric adapters to use.
        add_prefix: Whether to add prefixes to metric names.
        include_adapter_name: Whether to include adapter name in metric names.

    Returns:
        DataFrame with complexity assessment results including deltas.
    """
    pairs = split_log_into_fixed_comparable_windows(
        pm4py_log, window_1_size, window_2_size, offset, step
    )
    all_windows: List[Window] = [w for pair in pairs for w in pair]

    merged = run_metric_adapters(
        all_windows,
        adapter_names,
        add_prefix=add_prefix,
        include_adapter_name=include_adapter_name,
    )

    rows = []
    for pid, (w1, w2) in enumerate(pairs):
        w1_info = {f"w1_{k}": v for k, v in w1.to_dict().items()}
        w2_info = {f"w2_{k}": v for k, v in w2.to_dict().items()}
        w1_meas = {f"w1_{k}": v for k, v in merged[w1.id].items()}
        w2_meas = {f"w2_{k}": v for k, v in merged[w2.id].items()}

        # deltas for measure_* keys (still works because prefixing is centralized)
        delta: Dict[str, float] = {}
        for k, v1 in merged[w1.id].items():
            if not k.startswith("measure_"):
                continue
            if k in merged[w2.id]:
                v2 = merged[w2.id][k]
                try:
                    delta[f"delta_{k.removeprefix('measure_')}"] = float(v2) - float(v1)
                except (TypeError, ValueError):
                    pass

        rows.append(
            {"pair_id": pid, **w1_info, **w2_info, **w1_meas, **w2_meas, **delta}
        )

    df = pd.DataFrame(rows)
    save_complexity_csv(dataset_key, _cfg_name(configuration_name, approach_name), df)
    return df
