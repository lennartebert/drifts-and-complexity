from __future__ import annotations

from typing import Any, Iterable

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Number of Events")
class NumberOfEvents(TraceMetric):
    """Total number of events in the sample (from sample traces)."""
    name = "Number of Events"

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        
        num_events = sum(len(tr) for tr in traces)
        
        measures.set(self.name, num_events, hidden=False, meta={"basis": "traces"})
