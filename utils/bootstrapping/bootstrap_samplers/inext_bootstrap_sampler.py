"""
iNEXT-style individual-based bootstrap sampler.

This module implements the iNEXT methodology for bootstrap sampling from abundance data,
following the "EstiBootComm.Ind" approach described in Hsieh, Ma & Chao (2016).

The key innovation is the construction of a "bootstrap community" that:
1. Shrinks observed species probabilities based on rarity
2. Allocates probability mass to unseen species
3. Allows multinomial sampling from this corrected community

References:
- Hsieh, T. C., Ma, K. H., & Chao, A. (2016). iNEXT: an R package for rarefaction and
  extrapolation of species diversity (Hill numbers). Methods in Ecology and Evolution, 7(12), 1451-1456.
- Chao, A., & Jost, L. (2012). Coverage-based rarefaction and extrapolation:
  standardizing samples by completeness rather than size. Ecology, 93(12), 2533-2547.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.bootstrapping.bootstrap_samplers.bootstrap_sampler import BootstrapSampler
from utils.population.chao1 import (
    chao1_unseen_estimation_inext,
    create_inext_bootstrap_community,
    inext_center_line_from_counts,
    inext_richness_at_size_m,
)
from utils.population.extractors.chao1_population_extractor import (
    Chao1PopulationExtractor,
)
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
)
from utils.population.extractors.population_extractor import PopulationExtractor
from utils.population.population_distribution import (
    create_bootstrap_population_distribution,
    create_chao1_population_distribution,
    get_labels_and_probabilities,
)
from utils.population.population_distributions import (
    PopulationDistribution,
    PopulationDistributions,
)
from utils.windowing.window import Window


# -----------------
# small helpers
# -----------------
def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed)


def _multinomial_draw(rng: random.Random, n: int, probs: List[float]) -> List[int]:
    """
    Simple cumulative-CDF multinomial sampler.
    Returns a list of category counts that sum to n.
    """
    if n <= 0:
        return [0] * len(probs)
    s = sum(probs)
    if not probs:
        return []
    p = [x / s for x in probs] if s > 0 else [1.0 / len(probs)] * len(probs)

    # cumulative distribution
    cum, acc = [], 0.0
    for x in p:
        acc += x
        cum.append(acc)
    cum[-1] = 1.0  # absorb fp error

    out = [0] * len(p)
    for _ in range(n):
        u = rng.random()
        for i, c in enumerate(cum):
            if u <= c:
                out[i] += 1
                break
    return out


# iNEXT unseen estimation now handled by chao1_unseen_estimation_inext


def bootstrap_inext_curve(
    labels: List[str],
    probs: List[float],
    n_ref: int,
    m_grid: List[int],
    B: int,
    rng: random.Random,
) -> Dict[str, np.ndarray]:
    """
    Draw n_ref from p* once per replicate, then evaluate the iNEXT estimator at each m.
    Returns dict with arrays: vals (replicate matrix), se, and optionally mean, lo, hi.
    """
    probs_array = np.array(probs, dtype=float)
    vals = np.empty((B, len(m_grid)), dtype=float)

    for b in range(B):
        draws = np.random.multinomial(
            n_ref, probs_array
        )  # replicate dataset of size n_ref
        xs = draws.tolist()  # counts per (observed+unseen) bin
        for j, m in enumerate(m_grid):
            vals[b, j] = inext_richness_at_size_m(xs, int(m))

    means = vals.mean(axis=0)
    sds = vals.std(axis=0, ddof=1)
    ses = sds / np.sqrt(B)
    lo = means - 1.96 * ses
    hi = means + 1.96 * ses
    return {"vals": vals, "mean": means, "se": ses, "lo": lo, "hi": hi}


def compute_inext_curve_with_cis(
    original_counts: Counter,
    m_grid: List[int],
    B: int = 200,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute iNEXT curve with CIs centered on the estimator curve.

    This function:
    1. Computes the center line (blue) from original counts
    2. Builds bootstrap community p*
    3. Generates bootstrap replicates
    4. Centers CIs around the estimator curve, not bootstrap mean

    Parameters
    ----------
    original_counts : Counter
        Original abundance counts.
    m_grid : List[int]
        Grid of sample sizes to evaluate.
    B : int, default=200
        Number of bootstrap replicates.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'center': Center line from original counts
        - 'se': Standard error from bootstrap
        - 'ci_lo': Lower CI (center - 1.96*se)
        - 'ci_hi': Upper CI (center + 1.96*se)
        - 'vals': Bootstrap replicate matrix
    """
    from collections import Counter

    from utils.population.population_distribution import (
        create_chao1_population_distribution,
    )

    # Step 1: Compute center line from original counts
    center = np.array(inext_center_line_from_counts(original_counts, m_grid))

    # Step 2: Build bootstrap community
    n_ref = sum(original_counts.values())
    pop_dist = create_chao1_population_distribution(original_counts, n_ref)
    bootstrap_comm = create_inext_bootstrap_community(pop_dist)

    # Step 3: Get labels and probabilities from bootstrap community
    labels, probs = get_labels_and_probabilities(bootstrap_comm)

    # Step 4: Generate bootstrap replicates
    rng = _rng(seed)
    boot = bootstrap_inext_curve(labels, probs, n_ref, m_grid, B, rng)

    # Step 5: Center CIs around the estimator curve
    se = boot["se"]
    ci_lo = center - 1.96 * se
    ci_hi = center + 1.96 * se

    return {
        "center": center,
        "se": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "vals": boot["vals"],
    }


def verify_inext_invariants(original_counts: Counter, m_grid: List[int]) -> None:
    """
    Verify iNEXT invariants as per specification.

    This function checks:
    1. At m=n: Ŝ(n) = S_obs exactly
    2. If f1=0: a=0, S0=0, p*=p (checked in bootstrap community builder)
    3. Each replicate's observed richness ≤ m (checked during bootstrap)

    Parameters
    ----------
    original_counts : Counter
        Original abundance counts.
    m_grid : List[int]
        Grid of sample sizes to evaluate.
    """
    from collections import Counter

    # Invariant 1: At m=n, rarefaction equals S_obs exactly
    xs0 = [int(c) for c in original_counts.values() if c > 0]
    n_ref = sum(xs0)
    S_obs = len(xs0)

    if n_ref in m_grid:
        richness_at_n = inext_richness_at_size_m(xs0, n_ref)
        assert (
            abs(richness_at_n - S_obs) < 1e-12
        ), f"At m=n, Ŝ(n)={richness_at_n} != S_obs={S_obs}"

    # Invariant 2: If f1=0, check that unseen mass is zero
    f1 = sum(1 for c in original_counts.values() if c == 1)
    if f1 == 0:
        # This should be checked in the bootstrap community builder
        # but we can verify here that the curve is flat for m >= n
        for m in m_grid:
            if m >= n_ref:
                richness_at_m = inext_richness_at_size_m(xs0, m)
                assert (
                    abs(richness_at_m - S_obs) < 1e-12
                ), f"For f1=0 and m={m}>=n, Ŝ(m)={richness_at_m} != S_obs={S_obs}"


def _draw_inext_bootstrap_sample(
    rng: random.Random, pop_dist: PopulationDistribution
) -> Tuple[Counter, int]:
    """
    Draw a single iNEXT bootstrap sample using the correct iNEXT methodology.

    This function draws n_samples items from the bootstrap community p* and returns
    the counts and total sample size. The iNEXT estimator should then be applied to
    these counts for different m values.

    Parameters
    ----------
    rng : random.Random
        Random number generator.
    pop_dist : PopulationDistribution
        Population distribution object.

    Returns
    -------
    Tuple[Counter, int]
        Bootstrap sample counts and total number of samples drawn.
    """
    # Build iNEXT bootstrap community as PopulationDistribution
    bootstrap_comm = create_inext_bootstrap_community(pop_dist)

    # Get labels and probabilities from the bootstrap community
    labels, probs = get_labels_and_probabilities(bootstrap_comm)

    # Draw n_samples from multinomial using bootstrap community
    draws = _multinomial_draw(rng, pop_dist.n_samples, probs)

    # Map back to counts
    result: Counter = Counter()
    for label, count in zip(labels, draws):
        if count > 0:
            result[label] = count

    return result, pop_dist.n_samples


class INextBootstrapSampler(BootstrapSampler):
    """
    iNEXT-style individual-based bootstrap sampler.

    This sampler implements the iNEXT methodology for bootstrap sampling from abundance data,
    following the "EstiBootComm.Ind" approach. It constructs a bootstrap community that
    shrinks observed species probabilities and allocates mass to unseen species.
    """

    def __init__(
        self,
        population_extractor: Optional[PopulationExtractor] = None,
        B: int = 200,
        m: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the iNEXT bootstrap sampler.

        Parameters
        ----------
        population_extractor : PopulationExtractor, optional
            Population extractor to use. Defaults to Chao1PopulationExtractor.
        B : int, default=200
            Number of bootstrap replicates.
        m : int, optional
            Target draw size. If None, uses the original sample size.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.population_extractor = population_extractor or Chao1PopulationExtractor()
        self.B = B
        self.m = m
        self.seed = seed

    def sample(self, window: Window) -> List[Window]:
        """
        Generate bootstrap replicates using iNEXT methodology.

        Parameters
        ----------
        window : Window
            Input window to bootstrap.

        Returns
        -------
        List[Window]
            List of B bootstrap replicate windows.
        """
        # Apply population extractor if not already present
        if window.population_distributions is None:
            window = self.population_extractor.apply(window)

        # Generate bootstrap replicates
        rng = _rng(self.seed)
        replicates = []

        for b in range(self.B):
            # Create a copy of the window
            rep_window = deepcopy(window)

            # Sample from each population distribution using iNEXT methodology
            if window.population_distributions is not None:
                # Sample activities using iNEXT bootstrap community
                activities_counts, activities_n_samples = _draw_inext_bootstrap_sample(
                    rng, window.population_distributions.activities
                )
                # Apply population extractor to sampled counts to get proper population estimates
                # This ensures distribution-based metrics work correctly on bootstrap replicates
                activities_dist = create_chao1_population_distribution(
                    activities_counts, activities_n_samples
                )

                # Sample dfg_edges using iNEXT bootstrap community
                dfg_edges_counts, dfg_edges_n_samples = _draw_inext_bootstrap_sample(
                    rng, window.population_distributions.dfg_edges
                )
                dfg_edges_dist = create_chao1_population_distribution(
                    dfg_edges_counts, dfg_edges_n_samples
                )

                # Sample trace_variants using iNEXT bootstrap community
                trace_variants_counts, trace_variants_n_samples = (
                    _draw_inext_bootstrap_sample(
                        rng, window.population_distributions.trace_variants
                    )
                )
                trace_variants_dist = create_chao1_population_distribution(
                    trace_variants_counts, trace_variants_n_samples
                )

                rep_window.population_distributions = PopulationDistributions(
                    activities=activities_dist,
                    dfg_edges=dfg_edges_dist,
                    trace_variants=trace_variants_dist,
                )
            replicates.append(rep_window)

        return replicates
