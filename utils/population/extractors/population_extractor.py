from __future__ import annotations
from abc import ABC, abstractmethod
from utils.windowing.helpers import Window

class PopulationExtractor(ABC):
    """
    Abstract strategy for inferring population data from a Window.

    Contract:
      - Input:  Window with .traces
      - Side effects: set window.population_distributions and window population_counts
      - Return: the same Window (for chaining)
    """
    @abstractmethod
    def apply(self, window: Window) -> Window:
        ...
