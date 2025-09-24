
from __future__ import annotations
import math
from typing import Iterable, Any

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric

def _acts(ev): return ev.get("concept:name", ev.get("activity", ev.get("Activity", None)))
def _seq(trace): return [_acts(ev) for ev in trace]
def _df_pairs(seq): return [(seq[i], seq[i+1]) for i in range(len(seq)-1)]

@register_metric("Deviation from Random")
class DeviationFromRandom(TraceMetric):
    name = "Deviation from Random"
    requires: list[str] = []

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name): return
        counts = {}
        for tr in traces:
            s = _seq(tr)
            for a,b in _df_pairs(s):
                counts[(a,b)] = counts.get((a,b), 0) + 1
        total = sum(counts.values())
        if total == 0:
            measures.set(self.name, float("nan"), hidden=False, meta={"note": "no transitions"})
            return
        acts = set()
        for (a,b) in counts.keys():
            acts.add(a); acts.add(b)
        V = len(acts)
        if V <= 1:
            measures.set(self.name, 0.0, hidden=False, meta={"note": "V<=1"})
            return
        p_u = 1.0 / (V * V)
        l2_seen = 0.0
        prob_seen_total = 0.0
        for cnt in counts.values():
            p = cnt / total
            prob_seen_total += p
            l2_seen += (p - p_u)**2
        k_unseen = V*V - len(counts)
        l2_unseen = k_unseen * (0.0 - p_u)**2
        d = math.sqrt(l2_seen + l2_unseen)
        dmax = math.sqrt(1.0 - 1.0/(V*V))
        value = 1.0 - (d / dmax if dmax > 0 else 0.0)
        measures.set(self.name, float(value), hidden=False, meta={"ref": "Pentland-style, normalized L2 to uniform", "basis": "traces"})
