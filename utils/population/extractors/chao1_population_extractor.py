"""
Population extraction helpers using Chao1 / sample coverage adjustments.

This module turns *observed* abundances (activities, directly-follows edges, and
trace variants) in a PM4Py event log into *estimated population models* with
(i) an estimated number of unseen species/items and (ii) estimated sample
coverage, following the Chao1 family of estimators. It then exposes a
`Chao1PopulationExtractor` that augments a `Window` with:

- `population_distributions`: an iNEXT-like mixture consisting of
  observed labels with reweighted probabilities that sum to the estimated
  coverage Ĉ, plus an aggregate unseen mass p0 = 1 - Ĉ and a count of
  unseen categories M_unseen.
- `population_counts`: deterministic richness estimates derived from the
  fitted distributions (Ŝ = S_obs + M_unseen for Chao1).

References (for context):
- Chao, A. (1984, 1987). Estimating the number of classes in a population.
- Chao & Jost (2012–2015). Coverage-based rarefaction and extrapolation.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.util import xes_constants as xes

# Chao1 functions are now handled by create_chao1_population_distribution
from utils.population.extractors.population_extractor import PopulationExtractor
from utils.population.population_distribution import (
    PopulationDistribution,
    create_chao1_population_distribution,
)
from utils.population.population_distributions import PopulationDistributions
from utils.windowing.helpers import Window


# ---- observed abundance counters (PM4Py) ----
def _counts_activities(log: List) -> Counter[str]:
    """
    Count activity occurrences in the given PM4Py event log.

    Parameters
    ----------
    log : List[Trace]
        Event log of traces and events.

    Returns
    -------
    Counter[str]
        A Counter mapping activity label -> observed frequency (abundance).
    """
    # pm4py utility returns a dict of {activity: count}; wrap in Counter for convenience
    vals = attributes_get.get_attribute_values(log, xes.DEFAULT_NAME_KEY)
    return Counter(vals)


def _counts_dfg_edges(log: List) -> Counter[str]:
    """
    Count directly-follows (DFG) edge occurrences in the log.

    The DFG is discovered from the log and edge keys are encoded as "A>B".

    Parameters
    ----------
    log : List[Trace]

    Returns
    -------
    Counter[str]
        A Counter mapping "src>tgt" -> frequency.
    """
    dfg = dfg_discovery.apply(log)
    return Counter({f"{a}>{b}": c for (a, b), c in dfg.items()})


def _counts_trace_variants(log: List) -> Counter[Tuple[str, ...]]:
    """
    Count trace variants by their activity sequences.

    Variant keys are represented as tuples of activity labels.

    Parameters
    ----------
    log : List[Trace]

    Returns
    -------
    Counter[Tuple[str, ...]]
        A Counter mapping variant tuple -> frequency (number of traces with that variant).
    """
    varmap = variants_filter.get_variants(log)
    out: Counter[Tuple[str, ...]] = Counter()
    # `varmap` is {variant_id: [trace1, trace2, ...]}; we reconstruct a canonical tuple key
    for trs in varmap.values():
        # use the first representative trace to obtain the activity sequence
        key = tuple(ev[xes.DEFAULT_NAME_KEY] for ev in trs[0])
        out[key] = len(trs)
    return out


# ---- Chao / coverage helpers ----
# Chao1 functions now imported from utils.population.chao1


# Coverage estimation now handled by chao1_coverage_estimate


def _population_distribution_to_counter(pd: PopulationDistribution) -> Counter:
    """
    Convert a PopulationDistribution back to a Counter.

    This function simply returns the observed counter from the population distribution.
    This is useful when you need to work with the original count-based representation
    after population modeling.

    Parameters
    ----------
    pd : PopulationDistribution
        The population distribution containing observed categories.

    Returns
    -------
    Counter
        A Counter mapping observed categories to their counts.
    """
    return Counter(pd.observed)


# _build_chao_distribution_from_counts removed - use create_chao1_population_distribution directly


class Chao1PopulationExtractor(PopulationExtractor):
    """
    Population extractor that augments a `Window` with Chao1-based estimates.

    Behavior
    --------
    - If `window.population_distributions` is already present, it is reused.
    - Otherwise, observed abundances are computed for:
        * Activities (event labels)
        * DFG edges ("A>B")
        * Trace variants (tuples of activity labels)
      and each is converted into a `PopulationDistribution` using Chao1 and
      coverage estimates.
    - `window.population_counts` is then set using the deterministic counts
      obtained from the fitted distributions (Ŝ = S_obs + M_unseen).

    Notes
    -----
    The distribution encodes both observed mass (Ĉ) and unseen mass (p0),
    enabling downstream procedures (e.g., coverage-standardized computations
    or extrapolations).
    """

    def apply(self, window: Window) -> Window:
        """
        Apply the Chao1 population modeling to the given window.

        Parameters
        ----------
        window : Window
            A time or trace window containing a PM4Py event log or a population distribution.

        Returns
        -------
        Window
            The same window, mutated to include:
            - `population_distributions` (activities, dfg_edges, trace_variants)
            - `population_counts` (richness estimates for the same three domains)
        """

        if window.population_distributions is not None:
            # perform chao estimation on the population distribution
            PD = window.population_distributions
            activity_count_vector = _population_distribution_to_counter(PD.activities)
            dfg_count_vector = _population_distribution_to_counter(PD.dfg_edges)
            vars_count_vector = _population_distribution_to_counter(PD.trace_variants)

            # Extract n_samples from existing distributions
            n_samples_acts = PD.activities.n_samples
            n_samples_dfg = PD.dfg_edges.n_samples
            n_samples_vars = PD.trace_variants.n_samples
        else:
            # perform chao estimation on the traces
            log = window.traces
            activity_count_vector = _counts_activities(log)
            dfg_count_vector = _counts_dfg_edges(log)
            vars_count_vector = _counts_trace_variants(log)

            # Use actual trace count as n_samples
            n_samples_acts = len(log)
            n_samples_dfg = len(log)
            n_samples_vars = len(log)

        # Build from traces once (observed abundances -> iNEXT-like model)
        pd_acts = create_chao1_population_distribution(
            activity_count_vector, n_samples_acts
        )
        pd_dfg = create_chao1_population_distribution(dfg_count_vector, n_samples_dfg)
        pd_vars = create_chao1_population_distribution(
            vars_count_vector, n_samples_vars
        )
        PD = PopulationDistributions(
            activities=pd_acts, dfg_edges=pd_dfg, trace_variants=pd_vars
        )
        window.population_distributions = PD

        return window
