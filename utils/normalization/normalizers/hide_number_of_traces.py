from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Optional, Iterable, List

from utils.normalization.normalizers.normalizer import Normalizer

class HideNumberOfTraces(Normalizer):
    """Do not report Number of Traces (sets it to None)."""
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
        return {"Number of Traces": None}