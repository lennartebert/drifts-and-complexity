
from __future__ import annotations
import math
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window

from typing import Sequence, Any, Dict, Tuple, Iterable, List, Set

def _acts(ev): return ev.get("concept:name", ev.get("activity", ev.get("Activity", None)))

def _seq(trace: Iterable[dict]) -> List[Any]:
    return [_acts(ev) for ev in trace]

def _df_pairs(seq: Sequence[Any]):
    return [(seq[i], seq[i+1]) for i in range(len(seq)-1)]


def _global_df(window: "Window"):
    counts = {}
    for tr in window.traces:
        s = _seq(tr)
        for a,b in _df_pairs(s):
            counts[(a,b)] = counts.get((a,b), 0) + 1
    return counts

@register_metric("Estimated Number of Acyclic Paths")
class EstimatedNumberOfAcyclicPaths(Metric):
    name = "Estimated Number of Acyclic Paths"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name): return
        df = _global_df(window)
        e = len(df)
        acts = set()
        for (a,b) in df.keys():
            acts.add(a); acts.add(b)
        v = len(acts)
        value = 10 ** (0.08 * (1 + e - v))
        measures.set(self.name, float(value), hidden=False, meta={"formula": "10**(0.08*(1+e-v))", "basis": "population count"})
