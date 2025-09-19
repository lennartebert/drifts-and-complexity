
from __future__ import annotations
from collections import Counter
from typing import Iterable, Any, Tuple

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window

def _sequence(trace: Iterable[dict]) -> Tuple[Any, ...]:
    seq = []
    for ev in trace:
        seq.append(ev.get("concept:name", ev.get("activity", ev.get("Activity", None))))
    return tuple(seq)

@register_metric("Percentage of Distinct Traces")
class PercentageOfDistinctTraces(Metric):
    name = "Percentage of Distinct Traces"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name):
            return
        variants = Counter(_sequence(tr) for tr in window.traces)
        n_traces = sum(variants.values())
        value = (len(variants) / n_traces) if n_traces > 0 else float("nan")
        measures.set(self.name, value, hidden=False, meta={"basis": "population count"})
