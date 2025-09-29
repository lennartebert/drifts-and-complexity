"""Abstract base class for population extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.windowing.window import Window


class PopulationExtractor(ABC):
    """Abstract strategy for inferring population data from a Window.

    Population extractors analyze traces in a window to estimate population-level
    distributions and statistics. This enables downstream metrics to work with
    both observed data and estimated population parameters.

    Contract:
        - Input: Window with .traces
        - Side effects: set window.population_distributions and window population_counts
        - Return: the same Window (for chaining)

    Examples:
        >>> extractor = NaivePopulationExtractor()
        >>> window_with_pop = extractor.apply(window)
        >>> print(window_with_pop.population_distributions.activities.count)
    """

    @abstractmethod
    def apply(self, window: Window) -> Window:
        """Apply population extraction to a window.

        Args:
            window: Window containing traces to analyze.

        Returns:
            The same window object with population_distributions populated.

        Raises:
            ValueError: If window has no traces or traces are invalid.
        """
        ...
