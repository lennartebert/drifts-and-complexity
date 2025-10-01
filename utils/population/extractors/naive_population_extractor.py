"""Naive population extractor that assumes sample equals population."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.util import xes_constants as xes

from utils.population.extractors.population_extractor import PopulationExtractor
from utils.population.population_distributions import (
    PopulationDistribution,
    PopulationDistributions,
)

# Window imported as string annotation to avoid circular import


# ---- observed abundance counters (PM4Py) ----
def _counts_activities(log: List) -> Counter[str]:
    """Count activity occurrences in the event log.

    Args:
        log: List of PM4Py Trace objects.

    Returns:
        Counter mapping activity names to frequencies.
    """
    vals = attributes_get.get_attribute_values(log, xes.DEFAULT_NAME_KEY)
    return Counter(vals)


def _counts_dfg_edges(log: List) -> Counter[str]:
    """Count directly-follows graph edge occurrences.

    Args:
        log: List of PM4Py Trace objects.

    Returns:
        Counter mapping "source>target" edge strings to frequencies.
    """
    dfg = dfg_discovery.apply(log)
    return Counter({f"{a}>{b}": c for (a, b), c in dfg.items()})


def _counts_trace_variants(log: List) -> Counter[Tuple[str, ...]]:
    """Count trace variant occurrences by activity sequence.

    Args:
        log: List of PM4Py Trace objects.

    Returns:
        Counter mapping activity sequence tuples to frequencies.
    """
    varmap = variants_filter.get_variants(log)
    out: Counter[Tuple[str, ...]] = Counter()
    for trs in varmap.values():
        key = tuple(ev[xes.DEFAULT_NAME_KEY] for ev in trs[0])
        out[key] = len(trs)
    return out


def _build_naive_distribution_from_counts(counts: Counter) -> PopulationDistribution:
    """Build a distribution assuming full coverage (C_hat=1), no unseen mass.

    This is the naive assumption that the sample perfectly represents the population.
    No unseen species are modeled.

    Args:
        counts: Counter with observed frequencies.

    Returns:
        PopulationDistribution with full coverage assumption.
    """
    N = int(sum(counts.values()))
    return PopulationDistribution(
        observed=counts,
        population=counts,  # No unseen species
        population_count=len(counts),
        unseen_count=None,  # No unseen species
        p0=None,  # No unseen mass
        n_samples=N,
    )


class NaivePopulationExtractor(PopulationExtractor):
    """Naive estimator: assumes sample == population (iNEXT: Ĉ=1, p₀=0).

    This extractor makes the simplest possible assumption that the observed
    sample perfectly represents the underlying population. It sets coverage
    to 1.0 and unseen mass to 0.0, meaning all observed categories are
    assumed to be the complete set of categories in the population.

    This is appropriate when you have high confidence that your sample
    captures the full diversity of the process, or when you want to
    establish a baseline for comparison with more sophisticated estimators.

    Examples:
        >>> extractor = NaivePopulationExtractor()
        >>> window = extractor.apply(window)
        >>> # window.population_distributions now contains naive estimates
    """

    def apply(self, window: "Window") -> "Window":
        """Apply naive population extraction to a window.

        Args:
            window: Window containing traces to analyze.

        Returns:
            The same window object with population_distributions populated
            using naive (full coverage) assumptions.

        Raises:
            ValueError: If window has no traces.
        """
        log = window.traces

        if not log:
            raise ValueError("Window must contain traces for population extraction")

        # Build distributions for all species
        pd_acts = _build_naive_distribution_from_counts(_counts_activities(log))
        pd_dfg = _build_naive_distribution_from_counts(_counts_dfg_edges(log))
        pd_vars = _build_naive_distribution_from_counts(_counts_trace_variants(log))

        window.population_distributions = PopulationDistributions(
            activities=pd_acts, dfg_edges=pd_dfg, trace_variants=pd_vars
        )

        return window
