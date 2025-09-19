from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Optional, Iterable, List

# -------------------------
# Normalizer interface
# -------------------------
class Normalizer(ABC):
    """
    Base interface for metric normalizers.
    Subclasses must implement `apply(metrics) -> Dict[str, Optional[float]]`,
    returning a (possibly partial) dict of metric_name -> normalized_value.
    """
    @abstractmethod
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
        ...