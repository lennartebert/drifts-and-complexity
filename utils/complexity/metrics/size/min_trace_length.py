from __future__ import annotations
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.size.trace_lengh_stats import TraceLengthStats
from utils.windowing.window import Window

@register_metric("Min. Trace Length")
class MinTraceLength(Metric):
    """Minimum trace length in the sample traces (revealed)."""
    name = "Min. Trace Length"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        # ensure dependency is computed (hidden)
        TraceLengthStats().compute(window, measures)
        # reveal owned measure
        measures.reveal(self.name)
