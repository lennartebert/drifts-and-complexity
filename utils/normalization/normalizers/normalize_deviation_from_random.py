from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Optional, Iterable, List

from utils.normalization.normalizers.normalizer import Normalizer

class NormalizeDeviationFromRandom(Normalizer):
    """
    Replication-invariant normalization of 'Deviation from Random':
        D' = 1 - (1 - D) / sqrt(1 - 1 / V^2)
    where D = 'Deviation from Random' and V = 'Number of Distinct Activities'.
    Clips to [0,1].
    """
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
        D = metrics.get("Deviation from Random")
        V = metrics.get("Number of Distinct Activities")
        val = None
        if D is not None and V and V > 1:
            denom_inner = 1.0 - 1.0 / (float(V) ** 2)
            if denom_inner > 0:
                denom = math.sqrt(denom_inner)
                if denom > 0:
                    val = 1.0 - (1.0 - float(D)) / denom
                    val = max(0.0, min(1.0, val))
        return {"Deviation from Random": val}
    