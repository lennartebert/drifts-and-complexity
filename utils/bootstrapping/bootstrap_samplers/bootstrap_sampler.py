"""Bootstrap sampler for re-sampling traces."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import List, Optional

from utils.windowing.window import Window


class BootstrapSampler:
    """Concrete bootstrap sampler that re-samples traces.

    This bootstrap sampler generates multiple replicate samples from a single window
    by randomly sampling traces with replacement. Population distributions are
    deleted from bootstrap replicates to ensure they are recomputed from the
    sampled traces using the population extractor.

    Parameters
    ----------
    B : int, default=200
        Number of bootstrap replicates to generate.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, B: int = 200, seed: Optional[int] = None):
        """Initialize the bootstrap sampler.

        Parameters
        ----------
        B : int, default=200
            Number of bootstrap replicates to generate.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.B = B
        self.seed = seed

    def sample(self, window: Window) -> List[Window]:
        """Generate bootstrap replicates by re-sampling traces.

        Args:
            window: Input window to resample from.

        Returns:
            List of Window objects representing bootstrap replicates. Each replicate
            contains randomly sampled traces (with replacement) and has its
            population_distributions set to None.

        Raises:
            ValueError: If window is invalid or has no traces.
        """
        if window.traces is None or len(window.traces) == 0:
            raise ValueError("Window must have traces to bootstrap sample from")

        # Set up random number generator
        rng = random.Random(self.seed)

        # Generate bootstrap replicates
        replicates = []
        n_traces = len(window.traces)

        for _ in range(self.B):
            # Create a copy of the window
            rep_window = deepcopy(window)

            # Sample traces with replacement
            sampled_traces = [
                window.traces[rng.randint(0, n_traces - 1)] for _ in range(n_traces)
            ]

            # Update the replicate window with sampled traces
            rep_window.traces = sampled_traces

            # Delete population distributions to ensure they are recomputed
            rep_window.population_distributions = None

            replicates.append(rep_window)

        return replicates
