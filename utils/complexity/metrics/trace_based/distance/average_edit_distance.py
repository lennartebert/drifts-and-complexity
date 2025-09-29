"""Average edit distance metric implementation."""

from __future__ import annotations

import math
from collections import Counter
from typing import List

import Levenshtein
from pm4py.objects.log.obj import Trace

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


def _encode_trace_activities_as_single_chars(traces: List[Trace]) -> List[str]:
    """Encode trace activities as single characters for edit distance computation.

    Args:
        traces: List of PM4Py Trace objects.

    Returns:
        List of encoded trace strings.
    """
    # collect all activities
    acts: List[str] = []
    for trace in traces:
        acts.extend(ev["concept:name"] for ev in trace)
    uniq = {a: i for i, a in enumerate(dict.fromkeys(acts))}
    # map to private-use Unicode (safe single code points)
    base = 0xE000  # BMP Private Use Area (ca. 6.4k symbols)
    if len(uniq) + base > 0xF8FF:
        # fall back to supplementary PUA if needed
        base = 0xF0000
    enc = {a: chr(base + i) for a, i in uniq.items()}

    encoded = ["".join(enc[ev["concept:name"]] for ev in trace) for trace in traces]
    return encoded


@register_metric("Average Edit Distance")
class AverageEditDistance(TraceMetric):
    """Average edit distance between all pairs of traces using Levenshtein distance."""

    name = "Average Edit Distance"
    requires: list[str] = []

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the average edit distance between all trace pairs.

        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        traces = _encode_trace_activities_as_single_chars(traces)
        n = len(traces)
        if n < 2:
            measures.set(
                self.name,
                float("nan"),
                hidden=False,
                meta={"note": "requires >=2 traces"},
            )
            return

        counts = Counter(traces)
        unique = list(counts.items())
        total_pairs = math.comb(n, 2)

        total = 0
        for i in range(len(unique)):
            s1, c1 = unique[i]
            for j in range(i + 1, len(unique)):
                s2, c2 = unique[j]
                d = Levenshtein.distance(
                    s1, s2
                )  # char-level == activity-level due to mapping
                total += d * c1 * c2

        value = total / total_pairs
        measures.set(
            self.name,
            float(value),
            hidden=False,
            meta={"distance": "activity-level Levenshtein", "basis": "traces"},
        )
