from __future__ import annotations
from typing import Dict, List, Tuple, Any
from utils.windowing.windowing import Window
from .base import ComplexityAdapter, Metrics, Info
from utils.population.estimators import estimate_populations_simple
from utils.complexity.complexity_from_populations import measures_from_population_estimates

class PopulationSimpleAdapter(ComplexityAdapter):
    """
    Own implementation (Chao1-like). Only supports coverage_level=1.0.
    No nboot/conf, no same-coverage alignment.
    """
    name = "population_simple"

    def compute_many(self, windows: List[Window]) -> Dict[str, Tuple[Metrics, Info]]:
        pop_fc = estimate_populations_simple(
            windows=windows,
            species=("activities", "dfg_edges", "trace_variants"),
            coverage_level=1.0,   # required by this simple impl
        )

        out: Dict[str, Tuple[Metrics, Info]] = {}
        for w in windows:
            df_fc = pop_fc[w.id]
            metrics = measures_from_population_estimates(df_fc, population_column="q0", coverage_string="full coverage")
            # expose all columns as info (already only full coverage)
            info: Dict[str, Any] = {}
            for _, row in df_fc.iterrows():
                sp = row["species"]
                for col, val in row.items():
                    if col in ("species", "window_id"):
                        continue
                    info[f"fc_{sp}_{col}"] = val
            out[w.id] = (metrics, info)
        return out

    def compute_one(self, window: Window) -> Tuple[Metrics, Info]:
        return self.compute_many([window])[window.id]
