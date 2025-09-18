from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import math, sys

_MAX_LOG10 = math.log10(sys.float_info.max)

def _pentland_complexity(e: int, v: int):
    exp10 = 0.08 * (1 + e - v)
    if exp10 > _MAX_LOG10:
        return np.inf # in case of overflow, write infinity
    return 10.0 ** exp10

def _structure(num_directly_follows_population, num_distinct_activities_population):
    # Günther 2009: "Definition 3.12 (Structure). The structure (ST ) of a log trace l is defined as the 
	# inverse relative amount of direct following relations in l, i.e. compared to the maximum 
	# amount of direct following relations possible"

    return float (1 - ((num_directly_follows_population) / (num_distinct_activities_population**2)))

def measures_from_population_estimates(
    pop_df: pd.DataFrame,
    population_column: str = 'S_hat_inf',
    coverage_string: str = 'Full Coverage'
) -> Dict[str, float]:
    measures: Dict[str, float] = {}

    def _s_hat(sp: str) -> float:
        row = pop_df.loc[pop_df["species"] == sp]
        return float(row[population_column].iloc[0]) if not row.empty else 0.0

    v_hat = _s_hat("activities")
    e_hat = _s_hat("dfg_edges")
    trace_variants_hat = _s_hat("trace_variants")

    measures[f"Number of Distinct Activities"] = float(v_hat)
    measures[f"Number of Distinct traces"] = float(trace_variants_hat)
    measures[f"Structure"] = _structure(e_hat, v_hat)
    measures[f"Estimated Number of Acyclic Paths"] = _pentland_complexity(
        e=int(round(e_hat)), v=int(round(v_hat))
    )

    # add coverage string to measure names, if not None
    if coverage_string is not None:
        measures = {
            f"{k} ({coverage_string})": v
            for k, v in measures.items()
        }
    
    return measures
