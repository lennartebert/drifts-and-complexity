"""
Population extraction helpers using Chao1 (abundance, iNEXT-aligned).

This module turns *observed* abundances (activities, DFG edges, trace variants)
from a PM4Py event log into *asymptotic richness* population models using the
bias-corrected Chao1 estimator (iNEXT style). It exposes a
`Chao1PopulationExtractor` that augments a `Window` with:

- `population_distributions`: for each domain, a species-inventory population
  sized to S_obs + S0 (S0 = ceil(f0_hat)), i.e., an integerized proxy of
  S_hat = S_obs + f0_hat (asymptotic richness). The distribution uses presence
  counts (1 per species, including unseen placeholders `unseen_i`).
- (Optional downstream) you can draw bootstrap replicates via
  `create_chao1_bootstrapped_population_distribution` to compute CI(S_hat).

Notes
-----
- This module is **abundance-only (Chao1)**. Inputs must be integer, non-negative
  counts. If you need incidence (Chao2), build per-trace presence counters and
  use an incidence pipeline.
- For *variants*, abundance equals number of traces per variant; this coincides
  with incidence, so Chao1 is fine with n_reference = #traces.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.util import xes_constants as xes

from utils.population.population_distribution import (
    create_chao1_population_distribution,
)
from utils.population.population_distributions import PopulationDistributions

# ---- observed abundance counters (PM4Py) ----


def _counts_activities(log: List) -> Counter[str]:
    """
    Count activity **occurrences** in the event log (abundance data).
    Returns Counter(activity -> total occurrences across the window).
    """
    vals = attributes_get.get_attribute_values(log, xes.DEFAULT_NAME_KEY)
    return Counter(vals)


def _counts_dfg_edges(log: List) -> Counter[str]:
    """
    Count directly-follows (DFG) **edge occurrences** in the log (abundance).
    Keys encoded as "A>B".
    """
    dfg = dfg_discovery.apply(log)
    return Counter({f"{a}>{b}": c for (a, b), c in dfg.items()})


def _counts_trace_variants(log: List) -> Counter[Tuple[str, ...]]:
    """
    Count trace variants by sequence (abundance = #traces per variant).
    Keys are tuples of activity labels.
    """
    varmap = variants_filter.get_variants(log)
    out: Counter[Tuple[str, ...]] = Counter()
    for trs in varmap.values():
        key = tuple(ev[xes.DEFAULT_NAME_KEY] for ev in trs[0])
        out[key] = len(trs)  # abundance equals number of traces for that variant
    return out


class Chao1PopulationExtractor:
    """
    Population extractor that augments a `Window` with **Chao1 (abundance)**
    asymptotic richness distributions for activities, DFG edges, and variants.

    Behavior
    --------
    - If `window.population_distributions` exists, re-fit from their `.observed` counters.
    - Otherwise, compute observed abundance counters from the PM4Py log.
    - For each domain, build a `PopulationDistribution` via iNEXT-style Chao1:
      species-inventory of size `S_obs + S0` with `s_asymptotic = S_hat` stored.
    """

    def apply(self, window: "Window") -> "Window":
        """
        Apply Chao1 abundance modeling to the given window.

        Returns the same window with `population_distributions` populated.
        """
        if window.population_distributions is not None:
            PD = window.population_distributions
            activity_count_vector = PD.activities.observed
            dfg_count_vector = PD.dfg_edges.observed
            vars_count_vector = PD.trace_variants.observed
        else:
            log = window.traces
            activity_count_vector = _counts_activities(log)
            dfg_count_vector = _counts_dfg_edges(log)
            vars_count_vector = _counts_trace_variants(log)

        # Build S_hat-focused population distributions (abundance, Chao1)
        pd_acts = create_chao1_population_distribution(activity_count_vector)
        pd_dfg = create_chao1_population_distribution(dfg_count_vector)
        pd_vars = create_chao1_population_distribution(vars_count_vector)

        window.population_distributions = PopulationDistributions(
            activities=pd_acts,
            dfg_edges=pd_dfg,
            trace_variants=pd_vars,
        )
        return window
