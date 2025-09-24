from __future__ import annotations

from typing import Any, Iterable

from pm4py.algo.filtering.log.variants import variants_filter

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


def _observed_n_variants(traces: Iterable[Iterable[Any]]) -> int:
    variants = variants_filter.get_variants(traces)
    return len(variants)

@register_metric("Number of Distinct Traces")
class NumberOfDistinctTraces(TraceMetric):
    """
    Population-aware count of distinct trace variants.
    Prefers window.population_distributions.trace_variants.count if present; else observed.
    """
    name = "Number of Distinct Traces"

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        
        number_of_distinct_traces = _observed_n_variants(traces)

        measures.set(self.name, number_of_distinct_traces, hidden=False, meta={"basis": "traces"})
