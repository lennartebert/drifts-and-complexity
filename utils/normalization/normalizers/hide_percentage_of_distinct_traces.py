from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Optional, Iterable, List

from utils.normalization.normalizers.normalizer import Normalizer

class HidePercentageOfDistinctTraces(Normalizer):
    """Optionally null out the percentage after deriving the count."""
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
        return {"Percentage of Distinct Traces": None}