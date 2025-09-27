"""Number of distinct activities metric implementation."""

from __future__ import annotations
from typing import List
from pm4py.util import xes_constants as xes
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.objects.log.obj import Trace
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric

def _observed_n_acts(traces: List[Trace]) -> int:
    """Count distinct activities observed in traces.
    
    Args:
        traces: List of PM4Py Trace objects.
        
    Returns:
        Number of distinct activities.
    """
    vals = attributes_get.get_attribute_values(traces, xes.DEFAULT_NAME_KEY)
    return len(vals)

@register_metric("Number of Distinct Activities")
class NumberOfDistinctActivities(TraceMetric):
    """Trace-based count of distinct activities."""
    name = "Number of Distinct Activities"

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the number of distinct activities.
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return

        number_distinct_activities = _observed_n_acts(traces)

        measures.set(self.name, number_distinct_activities, hidden=False, meta={"basis": "traces"})
