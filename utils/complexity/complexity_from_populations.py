from __future__ import annotations
from typing import Dict
import pandas as pd
import math, sys

_MAX_LOG10 = math.log10(sys.float_info.max)

def _pentland_complexity(e: int, v: int):
    exp10 = 0.08 * (1 + e - v)
    if exp10 > _MAX_LOG10:
        return None
    return 10.0 ** exp10

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

    measures[f"Distinct traces ({coverage_string})"] = float(trace_variants_hat)
    measures[f"Variety ({coverage_string})"] = float(v_hat)  # fixed missing "("
    measures[f"Pentland process complexity ({coverage_string})"] = _pentland_complexity(
        e=int(round(e_hat)), v=int(round(v_hat))
    )
    return measures
