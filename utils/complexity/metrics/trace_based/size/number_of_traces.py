from __future__ import annotations
from typing import Any, Iterable
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric
from utils.windowing.window import Window


@register_metric("Number of Traces")
class NumberOfTraces(TraceMetric):
    """Total number of traces in the sample (from sample traces)."""
    name = "Number of Traces"

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        
        number_of_traces = len(traces)
        
        measures.set(self.name, number_of_traces, hidden=False, meta={"basis": "traces"})
