from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Optional, Iterable, List

from utils.normalization.normalizers.normalizer import Normalizer

# helper
def _safe_log(x: float) -> Optional[float]:
    return math.log(x) if (x is not None and isinstance(x, (int, float)) and x > 0) else None

class NormalizeLZComplexity(Normalizer):
    """
    Lempel-Ziv normalization (Kaspar & Schuster, 1987):
        LZ' = LZ / ( N / (log N / log V) )
    where:
        N = Number of Events
        V = Number of Distinct Activities
    """
    KEY = "Lempel-Ziv Complexity"

    def apply(self, metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
        lz = metrics.get(self.KEY)
        N = metrics.get("Number of Events")
        V = metrics.get("Number of Distinct Activities")
        val = None
        if lz is not None and N and V and N > 1 and V > 1:
            lnN = _safe_log(float(N))
            lnV = _safe_log(float(V))
            if lnN and lnV and lnV > 0:
                denom = float(N) / (lnN / lnV)
                if denom > 0:
                    val = float(lz) / denom
        return {self.KEY: val}