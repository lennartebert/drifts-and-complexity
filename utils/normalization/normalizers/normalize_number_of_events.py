from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Optional, Iterable, List

from utils.normalization.normalizers.normalizer import Normalizer

class NormalizeNumberOfEvents(Normalizer):
    """
    Normalize Number of Events by Number of Traces -> average trace length.
    Overwrites 'Number of Events'.
    """
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
        ne = metrics.get("Number of Events")
        nt = metrics.get("Number of Traces")
        val = None
        if ne is not None and nt and nt > 0:
            val = float(ne) / float(nt)
        return {"Number of Events": val}