"""
Population distribution data class and factory functions.

This module provides a pure data class for representing population distributions
and factory functions for creating them in different scenarios (bootstrap, non-bootstrap, etc.).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.population.chao1 import (
    chao1_unseen_estimation_inext,
)


@dataclass(frozen=True)
class PopulationDistribution:
    """
    Pure data class representing a population distribution.

    This is a final, immutable data class that holds all the information
    needed to represent a population distribution and work directly with
    iNEXT bootstrap functions. All behavior is handled by factory functions
    and external utilities.

    Fields
    ------
    observed : Counter
        Observed categories with their counts (e.g., activities, DFG edges, variants).
        Keys can be any hashable type (e.g., strings, tuples for variants).
    population : Counter
        Full population counter including observed categories and unseen species.
        If unseen_count or p0 are None, equals the observed counter.
        Unseen species are labeled as "unseen_1", "unseen_2", etc.
    count : int
        Total number of categories represented in population counter.
    unseen_count : Optional[int]
        Number of *unseen* categories M (e.g., round(Ŝ - S_obs)) to be represented
        as equally likely mass within p0. If None, no unseen categories are modeled.
    p0 : Optional[float]
        Unseen probability mass (1 - Ĉ), i.e., mass not covered by observed categories.
        Must be in [0, 1] if provided. If None, no unseen mass is modeled.
    n_samples : int
        Reference sample size (e.g., the size of the sample from which the estimates were
        derived). Kept for traceability/meta-data.
    """

    observed: Counter
    population: Counter
    count: int
    unseen_count: Optional[int] = None
    p0: Optional[float] = None
    n_samples: int = 0

    def __post_init__(self):
        """Validate the population distribution data."""
        if self.n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if self.p0 is not None and not (0.0 <= self.p0 <= 1.0):
            raise ValueError("p0 must be in [0, 1] if provided")
        if self.unseen_count is not None and self.unseen_count < 0:
            raise ValueError("unseen_count must be non-negative if provided")


# Factory Functions


def create_naive_population_distribution(
    observed: Counter, n_samples: int
) -> PopulationDistribution:
    """
    Create a naive population distribution without unseen species modeling.

    Parameters
    ----------
    observed : Counter
        Observed abundance counts.
    n_samples : int
        Reference sample size.

    Returns
    -------
    PopulationDistribution
        Population distribution with no unseen species modeling.
    """
    return PopulationDistribution(
        observed=observed,
        population=observed,  # No unseen species
        count=len(observed),
        unseen_count=None,
        p0=None,
        n_samples=n_samples,
    )


def create_chao1_population_distribution(
    observed: Counter, n_samples: int
) -> PopulationDistribution:
    """
    Create a population distribution with Chao1 unseen species estimation.

    Parameters
    ----------
    observed : Counter
        Observed abundance counts.
    n_samples : int
        Reference sample size.

    Returns
    -------
    PopulationDistribution
        Population distribution with Chao1 unseen species modeling.
    """
    f0_hat, S0 = chao1_unseen_estimation_inext(observed)
    p0 = _compute_unseen_probability_mass(observed, f0_hat)

    # Build population counter with unseen species
    population = Counter(observed)
    if f0_hat > 0 and S0 > 0 and p0 > 0:
        unseen_mass_per_species = p0 / S0
        unseen_count_per_species = int(unseen_mass_per_species * n_samples)

        for i in range(1, S0 + 1):
            population[f"unseen_{i}"] = unseen_count_per_species

    return PopulationDistribution(
        observed=observed,
        population=population,
        count=len(population),
        unseen_count=S0 if f0_hat > 0 else None,
        p0=p0 if f0_hat > 0 else None,
        n_samples=n_samples,
    )


def create_bootstrap_population_distribution(
    observed: Counter, n_samples: int
) -> PopulationDistribution:
    """
    Create a population distribution for bootstrap sampling.

    This creates a distribution with only observed species (no unseen modeling)
    for use in bootstrap replicates where we want to recompute metrics from
    the sampled counts.

    Parameters
    ----------
    observed : Counter
        Observed abundance counts from bootstrap sampling.
    n_samples : int
        Reference sample size.

    Returns
    -------
    PopulationDistribution
        Population distribution for bootstrap replicates.
    """
    return PopulationDistribution(
        observed=observed,
        population=observed,  # No unseen species for bootstrap replicates
        count=len(observed),
        unseen_count=None,
        p0=None,
        n_samples=n_samples,
    )


# Helper Functions


def _compute_unseen_probability_mass(observed: Counter, f0_hat: float) -> float:
    """
    Compute unseen probability mass using Chao1 methodology.

    Parameters
    ----------
    observed : Counter
        Observed abundance counts.
    f0_hat : float
        Expected number of unseen species.

    Returns
    -------
    float
        Unseen probability mass (p0).
    """
    n = sum(observed.values())
    if n == 0:
        return 0.0

    f1 = sum(1 for c in observed.values() if c == 1)
    coverage = max(0.0, 1.0 - (f1 / n))
    return 1.0 - coverage


# Utility Functions


def get_population_counter(pop_dist: PopulationDistribution) -> Counter:
    """
    Get the full population counter including observed and unseen species.

    Parameters
    ----------
    pop_dist : PopulationDistribution
        Population distribution.

    Returns
    -------
    Counter
        Full population counter.
    """
    result = Counter(pop_dist.observed)

    if (
        pop_dist.unseen_count is not None
        and pop_dist.unseen_count > 0
        and pop_dist.p0 is not None
        and pop_dist.p0 > 0
    ):
        # Add unseen species with equal probability mass
        unseen_mass_per_species = pop_dist.p0 / pop_dist.unseen_count
        unseen_count_per_species = int(unseen_mass_per_species * pop_dist.n_samples)

        for i in range(1, pop_dist.unseen_count + 1):
            result[f"unseen_{i}"] = unseen_count_per_species

    return result


def get_count(pop_dist: PopulationDistribution) -> int:
    """
    Get the total number of categories in the population.

    Parameters
    ----------
    pop_dist : PopulationDistribution
        Population distribution.

    Returns
    -------
    int
        Total number of categories.
    """
    return len(get_population_counter(pop_dist))


def to_dict(pop_dist: PopulationDistribution) -> Dict:
    """
    Convert population distribution to dictionary representation.

    Parameters
    ----------
    pop_dist : PopulationDistribution
        Population distribution.

    Returns
    -------
    Dict
        Dictionary representation.
    """
    return {
        "observed": dict(pop_dist.observed),
        "unseen_count": pop_dist.unseen_count,
        "p0": pop_dist.p0,
        "n_samples": pop_dist.n_samples,
        "population": dict(get_population_counter(pop_dist)),
        "count": get_count(pop_dist),
    }


def get_inext_parameters(pop_dist: PopulationDistribution) -> Dict:
    """
    Get iNEXT-specific parameters for a population distribution.

    Parameters
    ----------
    pop_dist : PopulationDistribution
        Population distribution.

    Returns
    -------
    Dict
        Dictionary containing f0_hat, S0, coverage, and unseen mass.
    """
    f0_hat, S0 = chao1_unseen_estimation_inext(pop_dist.observed)
    a, b, w = _compute_inext_unseen_probability_mass(pop_dist.observed, f0_hat)

    return {
        "f0_hat": f0_hat,
        "S0": S0,
        "coverage": 1.0
        - (
            a / sum(pop_dist.observed.values())
            if sum(pop_dist.observed.values()) > 0
            else 0.0
        ),
        "unseen_mass": a,
        "shrinkage_weight": w,
        "shrinkage_denominator": b,
    }


def get_labels_and_probabilities(
    pop_dist: PopulationDistribution,
) -> Tuple[List[str], List[float]]:
    """
    Builds the exact probability vector p (observed + unseen) without integer rounding.

    Parameters
    ----------
    pop_dist : PopulationDistribution
        The population distribution object.

    Returns
    -------
    labels : List[str]
        Labels for all species (observed + unseen).
    probs : List[float]
        Probability vector for all species (sums to 1.0).
    """
    labels: List = []
    probs: List[float] = []

    obs_items = list(pop_dist.observed.items())
    obs_sum = float(sum(v for _, v in obs_items))
    p0 = float(pop_dist.p0) if pop_dist.p0 is not None else 0.0
    observed_mass = 1.0 - p0

    if obs_sum > 0.0 and observed_mass > 0.0:
        for k, v in obs_items:
            labels.append(k)
            probs.append(observed_mass * (float(v) / obs_sum))
    else:
        for k, _ in obs_items:
            labels.append(k)
            probs.append(0.0)

    if pop_dist.unseen_count is not None and pop_dist.unseen_count > 0 and p0 > 0.0:
        unseen_each = p0 / float(pop_dist.unseen_count)
        for i in range(1, pop_dist.unseen_count + 1):
            labels.append(f"unseen_{i}")
            probs.append(unseen_each)

    # Normalize to 1.0 defensively and correct small fp drift on the last entry
    s = float(sum(probs))
    if s > 0.0:
        probs = [p / s for p in probs]
    if probs:
        # Adjust last element to make exact sum 1.0
        tail = 1.0 - float(sum(probs[:-1]))
        probs[-1] = tail
    return labels, probs
