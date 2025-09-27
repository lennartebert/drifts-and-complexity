"""Number of traces metric implementation."""

from __future__ import annotations
from typing import List
from pm4py.objects.log.obj import Trace
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Number of Traces")
class NumberOfTraces(TraceMetric):
    """Total number of traces in the sample (from sample traces)."""
    name = "Number of Traces"

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the total number of traces.
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return
        
        number_of_traces = len(traces)
        
        measures.set(self.name, number_of_traces, hidden=False, meta={"basis": "traces"})
