from typing import Dict, Optional
import pandas as pd
from .windowing.windowing import (
    split_log_into_windows_by_change_points,
    split_log_into_fixed_windows,
    split_log_into_fixed_comparable_windows,
)
from .helpers import to_naive_ts, flatten_measurements, save_complexity_csv
from .process_complexity_adapter import get_measures_for_traces

def _cfg_name(base_cfg: str, approach: str) -> str:
    return f"{base_cfg}__{approach}"

def assess_complexity_via_change_point_split(traces_sorted, drift_info_by_id: Dict, dataset_key: str, configuration_name: str, approach_name: str):
    real = {k: v for k, v in drift_info_by_id.items() if k != "na"}
    cps = [real[i]["calc_change_index"] for i in sorted(real.keys())] if real else []
    cp_map = {real[i]["calc_change_index"]: real[i] for i in sorted(real.keys())} if real else {}
    windows = split_log_into_windows_by_change_points(traces_sorted, cps, attach_change_points=True)

    results=[]
    for w in windows:
        end_cp_type = cp_map.get(w.end_change_point, {}).get("calc_change_type")
        window_change = "gradual" if end_cp_type == "gradual_end" else "stable"
        ms = get_measures_for_traces(w.traces)
        results.append({
            "window_id": w.id, "first_index": w.first_index, "last_index": w.last_index,
            "start_time": to_naive_ts(w.start_moment), "end_time": to_naive_ts(w.end_moment),
            "start_change_id": cp_map.get(w.start_change_point, {}).get("calc_change_id"),
            "end_change_id": cp_map.get(w.end_change_point, {}).get("calc_change_id"),
            "window_change": window_change, "traces_in_window": w.size, "measurements": ms
        })
    flat = flatten_measurements(results)
    cfg = _cfg_name(configuration_name, approach_name)
    save_complexity_csv(dataset_key, cfg, flat)
    return flat

def assess_complexity_via_fixed_sized_windows(traces_sorted, window_size:int, offset:int, dataset_key:str, configuration_name:str, approach_name:str, drift_info_by_id: Optional[Dict]=None):
    windows = split_log_into_fixed_windows(traces_sorted, window_size, offset)
    results=[]
    for w in windows:
        ms = get_measures_for_traces(w.traces)
        results.append({
            "window_id": w.id, "first_index": w.first_index, "last_index": w.last_index,
            "start_time": to_naive_ts(w.start_moment), "end_time": to_naive_ts(w.end_moment),
            "start_change_id": None, "end_change_id": None, "window_change": "stable",
            "traces_in_window": w.size, "measurements": ms
        })
    flat = flatten_measurements(results)
    cfg = _cfg_name(configuration_name, approach_name)
    save_complexity_csv(dataset_key, cfg, flat)
    return flat

def assess_complexity_via_window_comparison(traces_sorted, window_1_size:int, window_2_size:int, offset:int, step:int, dataset_key:str, configuration_name:str, approach_name:str):
    pairs = split_log_into_fixed_comparable_windows(traces_sorted, window_1_size, window_2_size, offset, step)
    rows=[]; pid=0
    for w1, w2 in pairs:
        m1 = get_measures_for_traces(w1.traces); m2 = get_measures_for_traces(w2.traces)
        row = {
            "pair_id": pid,
            "w1_id": w1.id, "w2_id": w2.id,
            "w1_first_index": w1.first_index, "w1_last_index": w1.last_index,
            "w2_first_index": w2.first_index, "w2_last_index": w2.last_index,
            "w1_start_time": to_naive_ts(w1.start_moment), "w1_end_time": to_naive_ts(w1.end_moment),
            "w2_start_time": to_naive_ts(w2.start_moment), "w2_end_time": to_naive_ts(w2.end_moment),
            "w1_traces_in_window": w1.size, "w2_traces_in_window": w2.size,
        }
        keys = sorted(set(m1)|set(m2))
        for k in keys:
            v1, v2 = m1.get(k), m2.get(k)
            row[f"w1_{k}"]=v1; row[f"w2_{k}"]=v2
            row[f"delta_{k}"]=None if (v1 is None or v2 is None) else (v2 - v1)
        rows.append(row); pid+=1
    df = pd.DataFrame(rows)
    cfg = _cfg_name(configuration_name, approach_name)
    save_complexity_csv(dataset_key, cfg, df)
    return df
