
from __future__ import annotations
import math
from typing import Any, Tuple, List, Dict
from collections import Counter

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window

import Levenshtein


def _encode_trace_activities_as_single_chars(pm4py_log):
    # collect all activities
    acts = []
    for trace in pm4py_log:
        acts.extend(ev["concept:name"] for ev in trace)
    uniq = {a: i for i, a in enumerate(dict.fromkeys(acts))}
    # map to private-use Unicode (safe single code points)
    base = 0xE000  # BMP Private Use Area (ca. 6.4k symbols)
    if len(uniq) + base > 0xF8FF:
        # fall back to supplementary PUA if needed
        base = 0xF0000
    enc = {a: chr(base + i) for a, i in uniq.items()}

    encoded = ["".join(enc[ev["concept:name"]] for ev in trace) for trace in pm4py_log]
    return encoded

@register_metric("Average Edit Distance")
class AverageEditDistance(Metric):
    name = "Average Edit Distance"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        traces = _encode_trace_activities_as_single_chars(window.traces)
        n = len(traces)
        if n < 2:
            measures.set(self.name, float("nan"), hidden=False, meta={"note": "requires >=2 traces"})
            return

        counts = Counter(traces)
        unique = list(counts.items())
        total_pairs = math.comb(n, 2)

        total = 0
        for i in range(len(unique)):
            s1, c1 = unique[i]
            for j in range(i+1, len(unique)):
                s2, c2 = unique[j]
                d = Levenshtein.distance(s1, s2)  # char-level == activity-level due to mapping
                total += d * c1 * c2

        value = total / total_pairs
        measures.set(self.name, float(value), hidden=False, meta={"distance": "activity-level Levenshtein", "basis": "traces"})
        