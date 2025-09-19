from __future__ import annotations
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window

def _num_traces(window: "Window") -> int:
    return len(window.traces)

@register_metric("Number of Traces")
class NumberOfTraces(Metric):
    """Total number of traces in the sample (from sample traces)."""
    name = "Number of Traces"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        measures.set(self.name, _num_traces(window), hidden=False, meta={"basis": "observation count"})
