"""Number of distinct traces metric implementation (trace-based)."""

from __future__ import annotations

from typing import List

from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import Trace

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


def _observed_n_variants(traces: List[Trace]) -> int:
    """Count distinct trace variants in the traces.

    Args:
        traces: List of PM4Py Trace objects.

    Returns:
        Number of distinct trace variants.
    """
    variants = variants_filter.get_variants(traces)
    return len(variants)


@register_metric("Number of Distinct Traces")
class NumberOfDistinctTraces(TraceMetric):
    """Population-aware count of distinct trace variants.

    Prefers window.population_distributions.trace_variants.count if present; else observed.
    """

    name = "Number of Distinct Traces"

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the number of distinct traces.

        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return

        number_of_distinct_traces = _observed_n_variants(traces)

        measures.set(
            self.name, number_of_distinct_traces, hidden=False, meta={"basis": "traces"}
        )
