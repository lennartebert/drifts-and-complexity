"""Minimum trace length metric implementation."""

from __future__ import annotations
from typing import List
from pm4py.objects.log.obj import Trace
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.size.trace_lengh_stats import \
    TraceLengthStats
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Min. Trace Length")
class MinTraceLength(TraceMetric):
    """Minimum trace length in the sample traces (revealed)."""
    name = "Min. Trace Length"

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the minimum trace length.
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        TraceLengthStats().compute(traces, measures)
        measures.reveal(self.name)
