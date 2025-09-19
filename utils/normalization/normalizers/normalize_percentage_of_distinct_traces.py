from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Optional, Iterable, List

from utils.normalization.normalizers.normalizer import Normalizer

class NormalizePercentageOfDistinctTraces(Normalizer):
    """
    Convert 'Percentage of Distinct Traces' * Number of Traces -> count of distinct traces.
    Overwrites 'Percentage of Distinct Traces' with a count (0..N_traces).
    """
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
        p = metrics.get("Percentage of Distinct Traces")  # expected in [0,1] or [0,100]
        nt = metrics.get("Number of Traces")
        val = None
        if p is not None and nt and nt > 0:
            p01 = float(p) / 100.0 if p > 1.0 else float(p)  # auto-detect scale
            val = max(0.0, min(float(nt), p01 * float(nt)))
        return {"Percentage of Distinct Traces": val}