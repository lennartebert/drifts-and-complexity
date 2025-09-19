
from __future__ import annotations
from typing import Iterable

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window

def _count_distinct_acts(trace: Iterable[dict]) -> int:
    acts = []
    for ev in trace:
        acts.append(ev.get("concept:name", ev.get("activity", ev.get("Activity", None))))
    return len(set(acts))

@register_metric("Average Distinct Activities per Trace")
class AverageDistinctActivitiesPerTrace(Metric):
    name = "Average Distinct Activities per Trace"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        counts = [_count_distinct_acts(tr) for tr in window.traces]
        value = (sum(counts) / len(counts)) if counts else float("nan")
        measures.set(self.name, value, hidden=False, meta={"basis": "population distribution"})
