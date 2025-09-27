"""Abstract base class for bootstrap samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from utils.windowing.window import Window


class BootstrapSampler(ABC):
    """Abstract base class for bootstrap sampling strategies.
    
    Bootstrap samplers generate multiple replicate samples from a single window
    to enable statistical analysis of metric uncertainty. Different sampling
    strategies can be used depending on the analysis requirements.
    
    Note: This *does not* compute Chao on the bootstrap; it only resamples
    abundance counts and optionally traces. The results should be plugged into
    the population extractor / metric pipeline afterwards.
    """
    
    @abstractmethod
    def sample(self, window: Window) -> List[Window]:
        """Generate bootstrap replicates from a window.
        
        Args:
            window: Input window to resample from.
            
        Returns:
            List of Window objects representing bootstrap replicates.
            
        Raises:
            ValueError: If window is invalid or sampling parameters are incorrect.
        """
        ...