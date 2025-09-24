
from __future__ import annotations
from typing import Any, Iterable

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Average Distinct Activities per Trace")
class AverageDistinctActivitiesPerTrace(TraceMetric):
    name = "Average Distinct Activities per Trace"
    requires: list[str] = []

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        
        traces = list(traces)
        n_traces = len(traces)
        if n_traces == 0:
            value = 0.0
        else:
            total_distinct_activities = sum(
                len({event["concept:name"] for event in trace}) for trace in traces
            )
            value = total_distinct_activities / n_traces
        
        measures.set(self.name, value, hidden=False, meta={"basis": "traces"})
        