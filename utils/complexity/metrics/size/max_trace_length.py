from __future__ import annotations
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.size.trace_lengh_stats import TraceLengthStats
from utils.windowing.window import Window

@register_metric("Max. Trace Length")
class MaxTraceLength(Metric):
    """Maximum trace length in the sample traces (revealed)."""
    name = "Max. Trace Length"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        TraceLengthStats().compute(window, measures)
        measures.reveal(self.name)
