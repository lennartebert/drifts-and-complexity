
from __future__ import annotations
from collections import Counter
from typing import Iterable, Any, Tuple

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Percentage of Distinct Traces")
class PercentageOfDistinctTraces(TraceMetric):
    name = "Percentage of Distinct Traces"
    requires: list[str] = ['Number of Distinct Traces', 'Number of Traces']

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        number_of_distinct_traces = measures.get('Number of Distinct Traces')
        number_of_traces = measures.get('Number of Traces')
        if number_of_distinct_traces is None or number_of_traces is None:
            raise ValueError(
                "Required measures missing: "
                f"Number of Distinct Traces={number_of_distinct_traces}, "
                f"Number of Traces={number_of_traces}"
            )
        
        value = number_of_distinct_traces.value / number_of_traces.value
        measures.set(self.name, value, hidden=False, meta={"basis": "derived"}) # TODO may create a new metric class for this
