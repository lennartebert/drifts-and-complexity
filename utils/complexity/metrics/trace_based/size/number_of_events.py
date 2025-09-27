"""Number of events metric implementation."""

from __future__ import annotations
from typing import List
from pm4py.objects.log.obj import Trace
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Number of Events")
class NumberOfEvents(TraceMetric):
    """Total number of events in the sample (from sample traces)."""
    name = "Number of Events"

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the total number of events across all traces.
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return
        
        num_events = sum(len(tr) for tr in traces)
        
        measures.set(self.name, num_events, hidden=False, meta={"basis": "traces"})
