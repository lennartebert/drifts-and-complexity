
"""Average distinct activities per trace metric implementation."""

from __future__ import annotations
from typing import List
from pm4py.objects.log.obj import Trace
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Average Distinct Activities per Trace")
class AverageDistinctActivitiesPerTrace(TraceMetric):
    """Average number of distinct activities per trace."""
    name = "Average Distinct Activities per Trace"
    requires: list[str] = []

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the average number of distinct activities per trace.
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return
        
        n_traces = len(traces)
        if n_traces == 0:
            value = 0.0
        else:
            total_distinct_activities = sum(
                len({event["concept:name"] for event in trace}) for trace in traces
            )
            value = total_distinct_activities / n_traces
        
        measures.set(self.name, value, hidden=False, meta={"basis": "traces"})
        