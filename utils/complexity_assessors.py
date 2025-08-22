from typing import Dict, Optional
import pandas as pd
from .windowing.windowing import (
    split_log_into_windows_by_change_points,
    split_log_into_fixed_windows,
    split_log_into_fixed_comparable_windows,
)
from utils.helpers import to_naive_ts, flatten_measurements, save_complexity_csv
from utils.process_complexity_adapter import get_measures_and_info_for_windows

def _cfg_name(base_cfg: str, approach: str) -> str:
    return f"{base_cfg}__{approach}"

def assess_complexity_via_change_point_split(pm4py_log, drift_info_by_id: Dict, dataset_key: str, configuration_name: str, approach_name: str):
    cp_info = {k: v for k, v in drift_info_by_id.items() if k != "na"}
    cps = [(cp_info[i]["calc_change_index"], i, cp_info[i]["calc_change_type"]) for i in sorted(cp_info.keys())] if cp_info else []
    windows = split_log_into_windows_by_change_points(pm4py_log, cps)

    # get measures per window
    measures_per_window_dict = get_measures_and_info_for_windows(windows)

    results=[]
    for w in windows:
        ms = measures_per_window_dict[w.id]
        window_info = w.to_dict()

        results.append({
            **ms, **window_info
        })
    flat = flatten_measurements(results)
    cfg = _cfg_name(configuration_name, approach_name)
    save_complexity_csv(dataset_key, cfg, flat)
    return flat

def assess_complexity_via_fixed_sized_windows(traces_sorted, window_size:int, offset:int, dataset_key:str, configuration_name:str, approach_name:str, drift_info_by_id: Optional[Dict]=None):
    windows = split_log_into_fixed_windows(traces_sorted, window_size, offset)
    # get measures per window
    measures_per_window_dict = get_measures_and_info_for_windows(windows)

    results=[]
    for w in windows:
        ms = measures_per_window_dict[w.id]
        window_info = w.to_dict()

        results.append({
            **ms, **window_info
        })
    flat = flatten_measurements(results)
    cfg = _cfg_name(configuration_name, approach_name)
    save_complexity_csv(dataset_key, cfg, flat)
    return flat

def assess_complexity_via_window_comparison(traces_sorted, window_1_size:int, window_2_size:int, offset:int, step:int, dataset_key:str, configuration_name:str, approach_name:str):
    pairs = split_log_into_fixed_comparable_windows(traces_sorted, window_1_size, window_2_size, offset, step)

    # get list with all windows (will need to separate them again later)
    all_windows = []
    for w1, w2 in pairs:
        all_windows.append(w1)
        all_windows.append(w2)

    # calcuate measures
    measures_per_window_dict = get_measures_and_info_for_windows(all_windows)

    rows=[]
    for pid, (w1, w2) in enumerate(pairs):
        w1_measures = {f'w1_{key}': value for key, value in measures_per_window_dict[w1.id].items()}
        w2_measures = {f'w2_{key}': value for key, value in measures_per_window_dict[w2.id].items()}

        w1_info = {f'w1_{key}': value for key, value in w1.to_dict().items()}
        w2_info = {f'w2_{key}': value for key, value in w2.to_dict().items()}
        
        # get delta measures
        delta_measures = {}
        for measure_name in w1_measures.keys():
            if measure_name.startswith('measure_'):
                delta_measure_name = 'delta_' + measure_name.removeprefix('measure_')
                delta_measure_value = w2_measures[measure_name] - w1_measures[measure_name]
                delta_measures[delta_measure_name] = delta_measure_value
        
        row = {
            "pair_id": pid,
            **w1_info,
            **w2_info,
            **w1_measures,
            **w2_measures,
            **delta_measures
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    cfg = _cfg_name(configuration_name, approach_name)
    save_complexity_csv(dataset_key, cfg, df)
    return df
