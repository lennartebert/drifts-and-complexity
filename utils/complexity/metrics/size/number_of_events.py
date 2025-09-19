from __future__ import annotations
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window

def _num_events(window: "Window") -> int:
    return sum(len(tr) for tr in window.traces)


@register_metric("Number of Events")
class NumberOfEvents(Metric):
    """Total number of events in the sample (from sample traces)."""
    name = "Number of Events"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        measures.set(self.name, _num_events(window), hidden=False, meta={"bases": "observations"})
