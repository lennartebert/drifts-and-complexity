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

from utils.bootstrapping.bootstrap_samplers.bootstrap_sampler import BootstrapSampler
from utils.population.extractors.chao1_population_extractor import (
    Chao1PopulationExtractor,
)
from utils.population.extractors.population_extractor import PopulationExtractor
from utils.population.population_distribution import (
    create_chao1_bootstrap_population_distribution,
)
from utils.population.population_distributions import (
    PopulationDistributions,
)
from utils.windowing.window import Window


# -----------------
# small helpers
# -----------------
def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed)


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
        seed : int, optional
            Random seed for reproducibility.
        """
        self.population_extractor = population_extractor or Chao1PopulationExtractor()
        self.B = B
        self.seed = seed

    def sample(self, window: Window) -> List["Window"]:
        """
        Generate iNEXT-style bootstrap replicate windows from a given window.

        For each population distribution in the input window (activities, dfg_edges,
        trace_variants), this method generates B bootstrap replicates using the
        iNEXT-aligned Chao1 bootstrap procedure. Each replicate window contains
        the same traces as the original window, but with population distributions
        replaced by the corresponding bootstrap replicate.

        Parameters
        ----------
        window : Window
            The input window containing traces and population distributions.

        Returns
        -------
        List[Window]
            A list of B new Window objects, each with bootstrapped population
            distributions for activities, dfg_edges, and trace_variants.

        Raises
        ------
        ValueError
            If the input window does not contain population distributions and
            population extraction fails.
        """

        # Ensure population distributions are present
        if window.population_distributions is None:
            window = self.population_extractor.apply(window)

        rng = _rng(self.seed)

        # Pull observed & reference sizes
        PDs = window.population_distributions

        # Activities
        act_reps = create_chao1_bootstrap_population_distribution(
            PDs.activities, B=self.B, rng=rng
        )

        # DFG edges
        dfg_reps = create_chao1_bootstrap_population_distribution(
            PDs.dfg_edges, B=self.B, rng=rng
        )

        # Trace variants
        var_reps = create_chao1_bootstrap_population_distribution(
            PDs.trace_variants, B=self.B, rng=rng
        )

        # Build replicate windows
        replicates: List["Window"] = []
        n_traces = len(window.traces)

        for b in range(self.B):
            rep_window = deepcopy(window)
            rep_window.population_distributions = PopulationDistributions(
                activities=act_reps[b],
                dfg_edges=dfg_reps[b],
                trace_variants=var_reps[b],
            )

            # also update the traces
            # Sample traces with replacement
            sampled_traces = [
                window.traces[rng.randint(0, n_traces - 1)] for _ in range(n_traces)
            ]

            # Update the replicate window with sampled traces
            rep_window.traces = sampled_traces
            replicates.append(rep_window)

        return replicates
