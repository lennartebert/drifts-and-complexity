from __future__ import annotations
from typing import Dict, List, Tuple, Any
from utils.windowing.windowing import Window
from .base import ComplexityAdapter, Metrics, Info
from utils.population.estimators import estimate_populations_inext
from utils.complexity.complexity_from_populations import measures_from_population_estimates

class PopulationInextAdapter(ComplexityAdapter):
    """iNEXT-backed population metrics. Supports coverage auto-alignment, nboot, conf."""
    name = "population_inext"

    def __init__(self, q_orders=(0, 1, 2), nboot=200, conf=0.95, same_coverage=False):
        self.q_orders = q_orders
        self.nboot = nboot
        self.conf = conf
        self.same_coverage = same_coverage

    def _info_flat(self, pop_df, pre: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for _, row in pop_df.iterrows():
            sp = row["species"]
            for col, val in row.items():
                if col in ("species", "window_id"):
                    continue
                out[f"{pre}{sp}_{col}"] = val
        return out

    def compute_many(self, windows: List[Window]) -> Dict[str, Tuple[Metrics, Info]]:
        pop = None

        if not self.same_coverage:
            # full coverage (1.0)
            pop = estimate_populations_inext(
                windows=windows,
                species=("activities", "dfg_edges", "trace_variants"),
                coverage_level=1.0,
                q_orders=self.q_orders,
                nboot=self.nboot,
                conf=self.conf,
            )
        else:
            # same coverage (auto)
            pop = estimate_populations_inext(
                windows=windows,
                species=("activities", "dfg_edges", "trace_variants"),
                coverage_level=None,
                q_orders=self.q_orders,
                nboot=self.nboot,
                conf=self.conf,
            )

        out: Dict[str, Tuple[Metrics, Info]] = {}
        for w in windows:
            df = pop[w.id]
            # df_sc = pop_sc[w.id]
            coverage_string = 'full coverage' if not self.same_coverage else 'same_coverage'
            measures = measures_from_population_estimates(df, population_column="q0", coverage_string=coverage_string)
            info = {**self._info_flat(df, "")}
            out[w.id] = ({**measures}, info)
        return out

    def compute_one(self, window: Window) -> Tuple[Metrics, Info]:
        return self.compute_many([window])[window.id]
