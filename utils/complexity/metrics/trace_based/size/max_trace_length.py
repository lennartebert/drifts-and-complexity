from __future__ import annotations

from typing import Any, Iterable

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.size.trace_lengh_stats import \
    TraceLengthStats
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Max. Trace Length")
class MaxTraceLength(TraceMetric):
    """Maximum trace length in the sample traces (revealed)."""
    name = "Max. Trace Length"

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        TraceLengthStats().compute(traces, measures)
        measures.reveal(self.name)
