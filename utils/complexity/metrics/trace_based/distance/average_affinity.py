
"""Average affinity metric implementation."""

from __future__ import annotations
from typing import List, Tuple
from collections import Counter
from pm4py.objects.log.obj import Trace
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric

def _sequence(trace: Trace) -> Tuple[str, ...]:
    """Extract activity sequence from trace.
    
    Args:
        trace: PM4Py Trace object.
        
    Returns:
        Tuple of activity names.
    """
    return tuple(ev.get("concept:name", ev.get("activity", ev.get("Activity", None))) for ev in trace)

@register_metric("Average Affinity")
class AverageAffinity(TraceMetric):
    """Average affinity between traces using weighted Jaccard similarity."""
    name = "Average Affinity"
    requires: list[str] = []

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute average affinity between all trace pairs.
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name): 
            return
        var_counts = Counter(_sequence(tr) for tr in traces)
        if sum(var_counts.values()) < 2:
            measures.set(self.name, float("nan"), hidden=False, meta={"note": "requires >=2 traces"})
            return
        # activity sets per variant
        var_acts = {v: set(v) for v in var_counts.keys()}
        variants = list(var_counts.items())
        n = sum(c for _, c in variants)

        total = 0.0
        for i in range(len(variants)):
            v1, c1 = variants[i]
            s1 = var_acts[v1]
            for j in range(i, len(variants)):
                v2, c2 = variants[j]
                s2 = var_acts[v2]
                if i == j:
                    total += 1.0 * c1 * (c1 - 1)
                else:
                    inter = len(s1 & s2); uni = len(s1 | s2)
                    sim = 0.0 if uni == 0 else (inter / uni)
                    total += sim * c1 * c2 * 2
        denom = n * (n - 1)
        value = total / denom if denom > 0 else float("nan")
        measures.set(self.name, float(value), hidden=False, meta={"definition": "weighted Jaccard over activity sets", "basis": "traces"})
