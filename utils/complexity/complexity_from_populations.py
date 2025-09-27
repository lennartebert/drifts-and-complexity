"""Complexity measures derived from population estimates."""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import math, sys

_MAX_LOG10 = math.log10(sys.float_info.max)

def _pentland_complexity(e: int, v: int) -> float:
    """Calculate Pentland complexity measure.
    
    Args:
        e: Number of edges in the process graph.
        v: Number of vertices (activities) in the process graph.
        
    Returns:
        Pentland complexity value.
    """
    exp10 = 0.08 * (1 + e - v)
    if exp10 > _MAX_LOG10:
        return np.inf # in case of overflow, write infinity
    return 10.0 ** exp10

def _structure(num_directly_follows_population: float, num_distinct_activities_population: float) -> float:
    """Calculate structure measure based on Günther 2009.
    
    Definition 3.12 (Structure). The structure (ST) of a log trace l is defined as the 
    inverse relative amount of direct following relations in l, i.e. compared to the maximum 
    amount of direct following relations possible.
    
    Args:
        num_directly_follows_population: Number of direct follows relations in population.
        num_distinct_activities_population: Number of distinct activities in population.
        
    Returns:
        Structure measure value.
    """
    return float (1 - ((num_directly_follows_population) / (num_distinct_activities_population**2)))

def measures_from_population_estimates(
    pop_df: pd.DataFrame,
    population_column: str = 'S_hat_inf',
    coverage_string: str = 'Full Coverage'
) -> Dict[str, float]:
    """Extract complexity measures from population estimates DataFrame.
    
    Args:
        pop_df: DataFrame with population estimates for different species.
        population_column: Column name containing population estimates.
        coverage_string: String to append to measure names for coverage identification.
        
    Returns:
        Dictionary mapping measure names to values.
    """
    measures: Dict[str, float] = {}

    def _s_hat(sp: str) -> float:
        """Get population estimate for a species.
        
        Args:
            sp: Species name.
            
        Returns:
            Population estimate or 0.0 if not found.
        """
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
