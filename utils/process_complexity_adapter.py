from pathlib import Path
import sys
from typing import Dict, Any, List

from utils.windowing.windowing import Window

# Make the forked dependency importable:
# repo layout assumption: this file sits next to the 'process-complexity' folder
THIS_DIR = Path(__file__).resolve().parent.parent
PROC_COMPLEXITY_DIR = THIS_DIR / "process-complexity"
if PROC_COMPLEXITY_DIR.exists():
    sys.path.insert(0, str(PROC_COMPLEXITY_DIR))

# Imports from Vidgof’s package
from Complexity import (
    generate_log,
    build_graph,
    perform_measurements,
    graph_complexity,
    log_complexity,
)

# Public entry point: get_measures_for_traces(traces)
#
# What it does:
#  1) Computes sample measures with accessible names.
#  2) Computes population upsampling (Chao1) on activities, DFG edges, trace variants.
#  3) Composes full-population measures and Pentland complexity.
#  4) Optionally (and privately) calls Vidgof’s module to add his metrics
#     (entropy etc.). Vidgof-specific objects (PA/LOG) are *not* reused
#     anywhere else—kept inside a private helper.
#
# All output keys are prefixed with "measure_" to match your plotting pipeline.

from typing import List, Dict, Any

import pandas as pd

# Our modules
from utils.population_estimates import estimate_populations
from utils.complexity_from_populations import measures_from_population_estimates


# ---------- Public API ----------

def get_measures_for_windows(windows: List[Window]):
    """
        Returns a dictionary of windows with their according measures and additional information.
    """

    # get vidgof measures for each window
    flat_measures_per_window_vidgof = {window.id: _vidgof_sample_measures(window.traces) for window in windows}

    # get full coverage populations and measures
    pop_df_dict_full_coverage = estimate_populations(
        windows = windows,
        species = ("activities", "dfg_edges", "trace_variants"),
        coverage_level = 1,
        estimator = "Chao1",
        q_orders = (0, 1, 2),
        nboot = 200,
        conf = 0.95,
    )
    flat_population_information_per_window_full_coverage = _get_flat_population_information_per_window(pop_df_dict_full_coverage, windows, pre_fix='full coverage_')
    flat_measures_per_window_full_coverage = _get_flat_measures_from_population_per_window(pop_df_dict_full_coverage, windows, 'full coverage')

    # get same coverage populations and measures
    pop_df_dict_same_coverage = estimate_populations(
        windows = windows,
        species = ("activities", "dfg_edges", "trace_variants"),
        coverage_level = None,
        estimator = "Chao1",
        q_orders = (0, 1, 2),
        nboot = 200,
        conf = 0.95,
    )
    flat_population_information_per_window_same_coverage = _get_flat_population_information_per_window(pop_df_dict_same_coverage, windows, pre_fix='same coverage_')
    flat_measures_per_window_same_coverage = _get_flat_measures_from_population_per_window(pop_df_dict_same_coverage, windows, 'same coverage')

    flat_output_dictionary = {}
    for window in windows:
        window_output_dict = {}

        window_output_dict.update({
            f'measure_{measure_name} (Vidgof)': value for measure_name, value in flat_measures_per_window_vidgof[window.id].items()
        })
        window_output_dict.update({
            f'info fc_{info_name} (Own)': value for info_name, value in flat_population_information_per_window_full_coverage[window.id].items()
        })
        window_output_dict.update({
            f'measure_{measure}': value for measure, value in flat_measures_per_window_full_coverage[window.id].items()
        })
        window_output_dict.update({
            f'info sc_{info_name} (Own)': value for info_name, value in flat_population_information_per_window_same_coverage[window.id].items()
        })
        window_output_dict.update({
            f'measure_{measure}': value for measure, value in flat_measures_per_window_same_coverage[window.id].items()
        })
        flat_output_dictionary[window.id] = window_output_dict

    return flat_output_dictionary

def _get_flat_population_information_per_window(pop_df_dict, windows, pre_fix='full coverage_'):
    flat_population_information_per_window = {}
    for window in windows:
        pop_df = pop_df_dict[window.id]
        flat_population_information_per_window[window.id] = {}
        # include same coverage population information in all return data (prefix: same_coverage)
        pop_columns = pop_df.columns
        for _, row in pop_df.iterrows():
            species = row['species']
            for column in pop_columns:
                if str(column) == 'species': continue
                flat_population_information_per_window[window.id][f'{pre_fix}{species}_{str(column)}'] = row[column]
    return flat_population_information_per_window

def _get_flat_measures_from_population_per_window(pop_df_dict, windows, coverage_string):
    population_measures_per_window = {}

    for window in windows:
        pop_df = pop_df_dict[window.id]
        population_measures_per_window[window.id] = measures_from_population_estimates(pop_df, population_column='q0', coverage_string=coverage_string)

    return population_measures_per_window

# def get_measures_for_traces(
#     traces,
# ) -> Dict[str, Any]:
#     """
#     Adapter that returns a flat dict of 'measure_*' keys.

#     Parameters
#     ----------
#     traces : pm4py Event log
#         Window/sample traces 

#     Returns
#     -------
#     dict[str, Any]
#     """
#     if not traces:
#         raise ValueError('No traces passed to measurement function')

#     all_return_data: Dict[str, Any] = {}

#     # 1) Population upsampling on species richness
#     pop_df: pd.DataFrame = estimate_populations(
#         traces,
#         species=("activities", "dfg_edges", "trace_variants"),
#         estimator="Chao1",
#     )

#     # include population information in all return data (prefix: population)
#     pop_columns = pop_df.columns
#     for _, row in pop_df.iterrows():
#         species = row['species']
#         for column in pop_columns:
#             if str(column) == 'species': continue
#             all_return_data[f'population_{species}_{str(column)}'] = row[column]

#     # 2) Compose accessible-named measures (sample + population + Pentland)
#     population_estimate_measures: Dict[str, Any] = measures_from_population_estimates(pop_df, population_column='S_hat_inf', coverage_string='Full coverage')

#     # add prefix and postfix to population estimate measures
#     all_return_data.update({
#         f'measure_{measure_name} (Own)': value for measure_name, value in population_estimate_measures.items()
#     })

#     # adapt prefix of measures:
#     vidgof_measures = _vidgof_sample_measures(traces)
#     all_return_data.update({
#             f'measure_{measure_name} (Vidgof)': value for measure_name, value in vidgof_measures.items()
#         })

#     return all_return_data


# ---------- Private: Vidgof integration (sandboxed) ----------

def _vidgof_sample_measures(traces: List[List[str]]) -> Dict[str, Any]:
    """
    Return measure keys with accessible names.
    """
  
    # Build Vidgof internals (kept private to this function)
    log = generate_log(traces, verbose=False)
    pa = build_graph(log, verbose=False, accepting=False)

    # Base measurements
    base = perform_measurements('all', log, traces, pa, quiet=True, verbose=False)

    # Entropies
    var_ent = graph_complexity(pa)  # (entropy, normalized_entropy)
    seq_ent = log_complexity(pa)    # (entropy, normalized_entropy)

    # Attach with explicit names
    base["Variant Entropy"] = var_ent[0]
    base["Normalized Variant Entropy"] = var_ent[1]
    base["Trace Entropy"] = seq_ent[0]
    base["Normalized Trace Entropy"] = seq_ent[1]

    # Trace length fields may be a dict; expand MIN/AVG/MAX
    if isinstance(base.get("Trace length"), dict):
        tl = base.pop("Trace length")
        base["Trace length min"] = tl.get("min")
        base["Trace length avg"] = tl.get("avg")
        base["Trace length max"] = tl.get("max")

    # Always include support (= number of traces in the window)
    base["Support"] = len(traces)

    measures_out = {f'{measure_name} (sample)': value for measure_name, value in base.items()}

    return measures_out
