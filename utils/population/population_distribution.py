"""
Population distribution data class and factory functions.

This module defines a compact, immutable data container for iNEXT-style
population distributions and thin factory wrappers that construct instances
using Chao1 (iNEXT-aligned) or a naive pass-through.

Design
------
- ``n_reference`` is the size of the original observed sample used to fit the
  estimator (iNEXT reference size ``n``).
- ``n_population`` is the size at which the fitted population is represented
  (often equal to ``n_reference``; may differ for coverage-targeted builds).
- ``observed`` may be abundances or weights; factories will internally coerce
  to an abundance vector of size ``n_reference`` for iNEXT computations.
- ``population`` is an integerized histogram at ``n_population`` (may include
  keys named ``unseen_1``, ``unseen_2``, ... if unseen mass is modeled).
"""

from __future__ import annotations

import random
from collections import Counter

# =============================
# utils/population/population_distributions.py (revised dataclass)
# =============================
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Data class
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationDistribution:
    """
    Minimal record focused on **asymptotic richness S_hat** (iNEXT/Chao1).

    - ``population`` is a **species-inventory**: presence counts (1 per species),
      including unseen placeholders (``unseen_i``). Thus ``sum(population)`` equals
      the integerized asymptotic richness proxy (``S_obs + S0``).
    - ``s_asymptotic`` stores the non-integer S_hat = S_obs + f0_hat.

    Fields
    ------
    observed : Counter
        Original histogram provided by the caller (abundances or weights).
    population : Counter
        Presence-only inventory: 1 per observed species + 1 per unseen placeholder.
    n_reference : int
        Reference sample size used to fit the estimator (iNEXT ``n``).
    n_population : int
        Interpreted here as the **number of species represented** in population
        (``S_obs + S0``). Kept for backward compatibility with downstream code that
        expects an "n_*" size.
    observed_count : int
        S_obs (sample richness at the reference).
    population_count : int
        Number of species represented in ``population`` (``S_obs + S0``).
    unseen_count : Optional[int]
        S0 = ceil(f0_hat); number of unseen placeholders included.
    p0 : Optional[float]
        Unseen probability mass at the reference sample (from EstiBootComm).
    coverage_observed / coverage_population : Optional[float]
        Kept for compatibility; typically ``None`` in S_hat mode.
    f0_hat : Optional[float]
        Bias-corrected Chao1 unseen species estimate used for S_hat.
    s_asymptotic : Optional[float]
        S_hat = S_obs + f0_hat (non-integer).
    """

    observed: Counter
    population: Counter
    n_reference: int
    n_population: int
    observed_count: int
    population_count: int
    unseen_count: Optional[int] = None
    p0: Optional[float] = None
    coverage_observed: Optional[float] = None
    coverage_population: Optional[float] = None
    f0_hat: Optional[float] = None
    s_asymptotic: Optional[float] = None


# iNEXT/Chao1 wrappers
# Note: Implemented as thin wrappers around chao1_helpers so this module remains
# a lightweight schema + facade.


def create_chao1_population_distribution(
    observed: Counter,
    rng: Optional[random.Random] = None,
) -> PopulationDistribution:
    """
    Create an iNEXT-aligned Chao1 population distribution.

    This delegates to ``utils.population.chao1_helpers.create_chao1_population_distribution``
    so the math and iNEXT-aligned behavior live in one place.

    Parameters
    ----------
    observed : Counter
        Observed counts (or weights). If not summing to ``n_reference``, the helper
        will coerce to abundances for iNEXT computations.
    rng: Optional[random.Random] = None
        Random number generator for reproducibility.

    Returns
    -------
    PopulationDistribution
        Fitted population distribution, including unseen mass and coverages.
    """
    from utils.population.chao1_helpers import (
        create_chao1_bootstrapped_population_distribution as _chao_pd_boot,
    )
    from utils.population.chao1_helpers import (
        create_chao1_population_distribution as _chao_pd,
    )

    return _chao_pd(observed=observed, rng=rng)


def create_chao1_bootstrap_population_distribution(
    base_pd: PopulationDistribution, B: int = 200, rng: Optional[random.Random] = None
) -> List[PopulationDistribution]:
    """
    Generate bootstrap replicates of a Chao1 population distribution.

    This function delegates to
    ``utils.population.chao1_helpers.create_chao1_bootstrapped_population_distribution``
    to produce B bootstrap samples of the population distribution, each fitted
    to a resampled version of the observed data.

    Parameters
    ----------
    base_pd : PopulationDistribution
        The base (fitted) population distribution to bootstrap from.
    B : int, optional
        Number of bootstrap replicates to generate (default is 200).
    rng : Optional[random.Random], optional
        Random number generator for reproducibility.

    Returns
    -------
    List[PopulationDistribution]
        List of bootstrapped population distributions.
    """
    from utils.population.chao1_helpers import (
        create_chao1_bootstrapped_population_distribution as _chao_pd_boot,
    )
    from utils.population.chao1_helpers import (
        create_chao1_population_distribution as _chao_pd,
    )

    return _chao_pd_boot(base_pd=base_pd, B=B, rng=rng)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def to_dict(pop_dist: PopulationDistribution) -> Dict:
    """
    Convert a ``PopulationDistribution`` to a serializable dictionary.

    The dictionary summarizes observed and population histograms, key sizes,
    coverages, and unseen parameters.
    """
    return {
        "n_reference": pop_dist.n_reference,
        "n_population": pop_dist.n_population,
        "observed": {
            "num_categories": pop_dist.observed_count,
            "total_count": int(sum(pop_dist.observed.values())),
            "sample_keys": list(dict(pop_dist.observed).keys())[:3],
        },
        "population": {
            "num_categories": pop_dist.population_count,
            "total_count": int(sum(pop_dist.population.values())),
            "sample_keys": list(dict(pop_dist.population).keys())[:3],
        },
        "unseen_count": pop_dist.unseen_count,
        "p0": pop_dist.p0,
        "coverage": {
            "observed": pop_dist.coverage_observed,
            "population": pop_dist.coverage_population,
        },
    }
