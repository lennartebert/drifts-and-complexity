from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from utils.windowing.window import Window


class BootstrapSampler(ABC):
    @abstractmethod
    def sample(self, window: Window) -> List[Window]:
        """
        Input:  Window
        Output: list[Window] (bootstrap replicates)

        Note: This *does not* compute Chao on the bootstrap; it only resamples
        abundance counts and optionally traces. The results should be plugged into
        the population extractor / metric pipeline afterwards.
        """
        ...