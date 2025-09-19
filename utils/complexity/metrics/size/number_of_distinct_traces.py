from __future__ import annotations
from pm4py.algo.filtering.log.variants import variants_filter
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window

def _observed_n_variants(window: "Window") -> int:
    variants = variants_filter.get_variants(window.traces)
    return len(variants)

@register_metric("Number of Distinct Traces")
class NumberOfDistinctTraces(Metric):
    """
    Population-aware count of distinct trace variants.
    Prefers window.population_distributions.trace_variants.count if present; else observed.
    """
    name = "Number of Distinct Traces"

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name):
            return

        pd = getattr(window, "population_distributions", None)
        n_vars = None
        if pd is not None:
            n_vars = getattr(getattr(pd, "trace_variants", object()), "count", None)
            if n_vars is not None:
                measures.set(self.name, float(n_vars), hidden=False, meta={"bases": "population"})
                return

        measures.set(self.name, float(_observed_n_variants(window)), hidden=False, meta={"bases": "observations"})
