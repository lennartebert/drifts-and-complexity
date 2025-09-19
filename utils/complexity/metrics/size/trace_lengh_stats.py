from __future__ import annotations
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.windowing.window import Window

def _lens(window: "Window"):
    return [len(tr) for tr in window.traces]

class TraceLengthStats(Metric):
    """
    Hidden one-pass aggregator that computes Min/Avg/Max trace length and stores
    them as hidden measures. Public metrics reveal only their own names.
    """
    name = "_TraceLengthStats"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        needed = ("Min. Trace Length", "Avg. Trace Length", "Max. Trace Length")
        if all(measures.has(n) for n in needed):
            return

        lens = _lens(window)
        if lens:
            mn = float(min(lens))
            avg = float(sum(lens) / len(lens))
            mx = float(max(lens))
        else:
            mn = avg = mx = 0.0

        measures.set("Min. Trace Length", mn,  hidden=True, meta={"basis": "observation distribution"})
        measures.set("Avg. Trace Length", avg, hidden=True, meta={"basis": "observation distribution"})
        measures.set("Max. Trace Length", mx,  hidden=True, meta={"basis": "observation distribution"})
