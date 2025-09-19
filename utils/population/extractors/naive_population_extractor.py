from __future__ import annotations
from collections import Counter
from typing import Dict, Tuple, List

from pm4py.util import xes_constants as xes
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter

from utils.windowing.helpers import Window
from utils.population.population_distributions import PopulationDistributions, PopulationDistribution
from utils.population.extractors.population_extractor import PopulationExtractor

# ---- observed abundance counters (PM4Py) ----
def _counts_activities(log) -> Counter[str]:
    vals = attributes_get.get_attribute_values(log, xes.DEFAULT_NAME_KEY)
    return Counter(vals)

def _counts_dfg_edges(log) -> Counter[str]:
    dfg = dfg_discovery.apply(log)
    return Counter({f"{a}>{b}": c for (a, b), c in dfg.items()})

def _counts_trace_variants(log) -> Counter[Tuple[str, ...]]:
    varmap = variants_filter.get_variants(log)
    out = Counter()
    for trs in varmap.values():
        key = tuple(ev[xes.DEFAULT_NAME_KEY] for ev in trs[0])
        out[key] = len(trs)
    return out

def _build_naive_distribution_from_counts(counts: Counter) -> PopulationDistribution:
    """
    Build a distribution assuming full coverage (C_hat=1), no unseen mass.
    Observed probabilities are empirical relative frequencies (sum to 1).
    """
    labels: List = list(counts.keys())
    obs = list(counts.values())
    N = int(sum(obs))
    if N > 0:
        probs = [c / N for c in obs]
    else:
        probs = [0.0 for _ in obs]
    return PopulationDistribution(
        observed_labels=labels, observed_probs=probs, unseen_count=0, p0=0.0, n_samples=N
    )


class NaivePopulationExtractor(PopulationExtractor):
    """Naive estimator: assumes sample == population (iNEXT: Ĉ=1, p₀=0)."""

    def apply(self, window: Window) -> Window:
        log = window.traces

        # Build distributions for all species
        pd_acts = _build_naive_distribution_from_counts(_counts_activities(log))
        pd_dfg  = _build_naive_distribution_from_counts(_counts_dfg_edges(log))
        pd_vars = _build_naive_distribution_from_counts(_counts_trace_variants(log))

        window.population_distributions = PopulationDistributions(
            activities=pd_acts, dfg_edges=pd_dfg, trace_variants=pd_vars
        )

        return window
