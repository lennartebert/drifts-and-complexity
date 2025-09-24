
from __future__ import annotations
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


@register_metric("Structure")
class Structure(TraceMetric):
    # Günther 2009: "Definition 3.12 (Structure). The structure (ST ) of a log trace l is defined as the 
	# inverse relative amount of direct following relations in l, i.e. compared to the maximum 
	# amount of direct following relations possible"
    name = "Structure"
    requires: list[str] = ['Number of Distinct Activities', 'Number of Distinct Activity Transitions']

    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None:
        if measures.has(self.name): return

        number_of_distinct_activities = measures.get('Number of Distinct Activities')
        number_of_activity_transitions = measures.get('Number of Distinct Activity Transitions')
        if number_of_distinct_activities is None or number_of_activity_transitions is None:
            raise ValueError(
                "Required measures missing: "
                f"Number of Distinct Activities={number_of_distinct_activities}, "
                f"Number of Distinct Activity Transitions={number_of_activity_transitions}"
            )
        
        value = 1 - number_of_activity_transitions.value / number_of_distinct_activities.value**2
        
        measures.set(self.name, value, hidden=False, meta={"definition": "1 - distinct DF / (distinct activities)**2", "basis": "derived"})
