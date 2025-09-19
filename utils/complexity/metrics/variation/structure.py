
from __future__ import annotations
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


@register_metric("Structure")
class Structure(Metric):
    # Günther 2009: "Definition 3.12 (Structure). The structure (ST ) of a log trace l is defined as the 
	# inverse relative amount of direct following relations in l, i.e. compared to the maximum 
	# amount of direct following relations possible"
    name = "Structure"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name): return
        vals = []
        for tr in window.traces:
            s = _seq(tr)
            if len(s) < 2:
                vals.append(0.0)
                continue
            df = len(set(_df_pairs(s)))
            denom = max(1, len(s) - 1)
            vals.append(1.0 - (df / denom))
        value = (sum(vals)/len(vals)) if vals else float("nan")
        measures.set(self.name, value, hidden=False, meta={"definition": "1 - distinct DF / max(|l|-1,1)", "basis": "population count"})
