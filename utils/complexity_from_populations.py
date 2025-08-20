# Compose measure outputs from:
#   - sample (computed directly from traces)
#   - full-population estimates (from population_upsampling.py)
#
# Implements:
#   Distinct traces                 (sample + full population estimate)
#   Variety (activities)            (sample + full population estimate)
#   Magnitude (events)              (sample + same value under population name)
#   Trace count (traces)            (sample + same value under population name)
#   Pentland’s Process Complexity   (sample + full population estimate)

from __future__ import annotations
from collections import Counter
from typing import List, Dict
import pandas as pd
import math, sys

# Precompute max log once at import time
_MAX_LOG10 = math.log10(sys.float_info.max)  # ca. 308.2547155599


def _count_dfg_edges(traces: List[List[str]]) -> int:
    """Number of distinct directly-follows edges in the sample."""
    edges = set()
    for t in traces:
        for i in range(len(t) - 1):
            edges.add((t[i], t[i + 1]))
    return len(edges)


def _count_activities(traces: List[List[str]]) -> int:
    """Number of distinct activities (vertices) in the sample."""
    return len({a for t in traces for a in t})


def _count_trace_variants(traces: List[List[str]]) -> int:
    """Number of distinct trace variants in the sample."""
    return len({tuple(t) for t in traces})


def _count_events(traces: List[List[str]]) -> int:
    """Total number of events (Magnitude) in the sample."""
    return sum(len(t) for t in traces)

def _pentland_complexity(e: int, v: int):
    """
    Pentland's process complexity: 10 ** (0.08 * (1 + e - v)).

    If the value would overflow double precision, return None
    """
    exp10 = 0.08 * (1 + e - v)

    # If exponent would overflow 10**exp10, short-circuit.
    if exp10 > _MAX_LOG10:
        return None

    val = 10.0 ** exp10
    return val


def measures_from_population_estimates(
    pop_df: pd.DataFrame,
    population_column: str = 'S_hat_inf',
    coverage_string: str = 'Full Coverage'
) -> Dict[str, float]:
    """
    Measure outputs from population estimates.

    Parameters
    ----------
    pop_df : pd.DataFrame
        Output of utils.population_upsampling. Must contain rows with
        species in {"activities","dfg_edges","trace_variants"} and column "S_hat_inf".

    Returns
    -------
    dict[str, float]
        Flat mapping of measure names to values.
    """
    measures: Dict[str, float] = {}

    # Helper to get s_obs and S_hat_inf by species (default to 0.0 if missing)
    def _s_hat(sp: str) -> float:
        row = pop_df.loc[pop_df["species"] == sp]
        return float(row[population_column].iloc[0]) if not row.empty else 0.0
    
    
    v_hat = _s_hat("activities")
    e_hat = _s_hat("dfg_edges")
    trace_variants_hat = _s_hat("trace_variants")

    # --- Distinct traces (variants) ---
    measures[f"Distinct traces ({coverage_string})"] = float(trace_variants_hat)

    # --- Variety (activities) ---
    measures[f"Variety {coverage_string})"] = float(v_hat)

    # --- Pentland’s Process Complexity ---
    measures[f"Pentland process complexity ({coverage_string})"] = _pentland_complexity(
        e=int(round(e_hat)), v=int(round(v_hat))
    )

    return measures
