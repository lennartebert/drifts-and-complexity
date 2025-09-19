from __future__ import annotations
from pm4py.util import xes_constants as xes
from pm4py.statistics.attributes.log import get as attributes_get
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window

def _observed_n_acts(window: "Window") -> int:
    vals = attributes_get.get_attribute_values(window.traces, xes.DEFAULT_NAME_KEY)
    return len(vals)

@register_metric("Number of Distinct Activities")
class NumberOfDistinctActivities(Metric):
    """
    Population-aware count of distinct activities.
    Prefers window.population_distributions.activities.count if present; else observed.
    """
    name = "Number of Distinct Activities"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name):
            return

        pd = getattr(window, "population_distributions", None)
        n_acts = None
        if pd is not None:
            n_acts = getattr(getattr(pd, "activities", object()), "count", None)
            if n_acts is not None:
                measures.set(self.name, float(n_acts), hidden=False, meta={"basis": "population count"})
                return

        measures.set(self.name, float(_observed_n_acts(window)), hidden=False, meta={"basis": "observation count"})
