
from __future__ import annotations
from typing import Tuple, Any
from collections import Counter
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


from typing import Sequence, Any, Dict, Tuple, Iterable, List, Set

def _acts(ev): return ev.get("concept:name", ev.get("activity", ev.get("Activity", None)))

def _seq(trace: Iterable[dict]) -> List[Any]:
    return [_acts(ev) for ev in trace]

def _df_pairs(seq: Sequence[Any]):
    return [(seq[i], seq[i+1]) for i in range(len(seq)-1)]


@register_metric("Number of Ties in Paths to Goal")
class NumberOfTiesInPathsToGoal(TraceMetric):
    # pentland-haerem taks complexity is defined as "... all possible paths to each goal of the task and by summing up the number of ties making up these paths..." (Haerem et al. 2015)
	# ideally, this would be implemented as all paths to the goal through the observed adjacency matrix
	# However, this would lead to too long run times
	# The previous implementation by Vidgof counted the number of child nodes of all leaf nodes in an EPA. However, this does not correctly count shorter variants that are contained in the prefix of larger variants.
	# Hence, we resort to counting the activity transitions in all unique variants.

    name = "Number of Ties in Paths to Goal"
    requires: list[str] = []


    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name): return
        variants = Counter(tuple(_seq(tr)) for tr in traces)
        total_ties = 0
        for var_seq in variants.keys():
            total_ties += max(len(var_seq) - 1, 0)
        measures.set(self.name, int(total_ties), hidden=False, meta={"note": "sum of transitions across unique variants", "basis": "traces"})
