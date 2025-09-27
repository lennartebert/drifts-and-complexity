"""Trace length statistics computation."""

from __future__ import annotations
from typing import List
from pm4py.objects.log.obj import Trace
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric

class TraceLengthStats(TraceMetric):
    """Hidden one-pass aggregator that computes Min/Avg/Max trace length and stores
    them as hidden measures. Public metrics reveal only their own names.
    """
    name = "_TraceLengthStats"

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute trace length statistics (min, avg, max).
        
        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metrics.
        """
        needed = ("Min. Trace Length", "Avg. Trace Length", "Max. Trace Length")
        if all(measures.has(n) for n in needed):
            return

        lens = [len(tr) for tr in traces]
        if lens:
            mn = float(min(lens))
            avg = float(sum(lens) / len(lens))
            mx = float(max(lens))
        else:
            mn = avg = mx = 0.0

        measures.set("Min. Trace Length", mn,  hidden=True, meta={"basis": "observation distribution"})
        measures.set("Avg. Trace Length", avg, hidden=True, meta={"basis": "observation distribution"})
        measures.set("Max. Trace Length", mx,  hidden=True, meta={"basis": "observation distribution"})
