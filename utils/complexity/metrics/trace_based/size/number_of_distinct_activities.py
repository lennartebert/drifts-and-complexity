from __future__ import annotations
from typing import Any, Iterable
from pm4py.util import xes_constants as xes
from pm4py.statistics.attributes.log import get as attributes_get
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric
from utils.windowing.window import Window

def _observed_n_acts(traces: Iterable[Iterable[Any]]) -> int:
    vals = attributes_get.get_attribute_values(traces, xes.DEFAULT_NAME_KEY)
    return len(vals)

@register_metric("Number of Distinct Activities")
class NumberOfDistinctActivities(TraceMetric):
    """
    Trace-based count of distinct activities.
    """
    name = "Number of Distinct Activities"

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name):
            return

        number_distinct_activities = _observed_n_acts(traces)

        measures.set(self.name, number_distinct_activities, hidden=False, meta={"basis": "traces"})
