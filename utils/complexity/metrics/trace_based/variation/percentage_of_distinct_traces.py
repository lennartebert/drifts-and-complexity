
"""Percentage of distinct traces metric implementation."""

from __future__ import annotations
from typing import List
from pm4py.objects.log.obj import Trace
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Percentage of Distinct Traces")
class PercentageOfDistinctTraces(TraceMetric):
    """Percentage of distinct traces relative to total traces."""
    name = "Percentage of Distinct Traces"
    requires: list[str] = ['Number of Distinct Traces', 'Number of Traces']

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the percentage of distinct traces.
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
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
        measures.set(self.name, value, hidden=False, meta={"basis": "derived"})
