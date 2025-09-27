"""Number of distinct activity transitions metric implementation (trace-based)."""

from __future__ import annotations
from typing import List, Set, Tuple
from pm4py.objects.log.obj import Trace
from pm4py.util import xes_constants as xes
from pm4py.algo.filtering.log.variants import variants_filter
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


def _observed_n_act_transitions(traces: List[Trace]) -> int:
    """Count distinct activity transitions (ordered pairs of consecutive activities)
    observed in the log. Uses one representative trace per variant for efficiency.
    
    Args:
        traces: List of PM4Py Trace objects.
        
    Returns:
        Number of distinct activity transitions.
    """
    variants = variants_filter.get_variants(traces)  # dict: variant -> list[trace]
    distinct_activity_transitions: Set[Tuple[str, str]] = set()

    for traces_in_variant in variants.values():
        if not traces_in_variant:
            continue
        trace = traces_in_variant[0]  # representative
        # Extract activity names, skipping events without the name key
        names = [ev.get(xes.DEFAULT_NAME_KEY) for ev in trace if xes.DEFAULT_NAME_KEY in ev]
        # Add all consecutive pairs
        for a, b in zip(names, names[1:]):
            if a is not None and b is not None:
                distinct_activity_transitions.add((a, b))

    return len(distinct_activity_transitions)


@register_metric("Number of Distinct Activity Transitions")
class NumberOfDistinctActivityTransitions(TraceMetric):
    """Trace-based count of distinct activity transitions (ordered pairs of consecutive activities)."""
    name = "Number of Distinct Activity Transitions"

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the number of distinct activity transitions.
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return

        number_distinct_activity_transitions = _observed_n_act_transitions(traces)
        measures.set(
            self.name,
            number_distinct_activity_transitions,
            hidden=False,
            meta={"basis": "traces"},
        )
