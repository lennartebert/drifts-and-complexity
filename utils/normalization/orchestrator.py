from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Optional, Iterable, List

from utils.normalization.normalizers.hide_number_of_traces import HideNumberOfTraces
from utils.normalization.normalizers.normalize_deviation_from_random import NormalizeDeviationFromRandom
from utils.normalization.normalizers.normalize_lz_complexity import NormalizeLZComplexity
from utils.normalization.normalizers.normalize_number_of_events import NormalizeNumberOfEvents
from utils.normalization.normalizers.normalize_percentage_of_distinct_traces import NormalizePercentageOfDistinctTraces
from utils.normalization.normalizers.normalizer import Normalizer

# -------------------------
# Default pipeline (instances)
# -------------------------
DEFAULT_NORMALIZERS: List[Normalizer] = [
    NormalizeNumberOfEvents(),
    HideNumberOfTraces(),
    NormalizePercentageOfDistinctTraces(),
    NormalizeDeviationFromRandom(),
    NormalizeLZComplexity(),
]

def apply_normalizers(
    metrics: Mapping[str, float],
    normalizers: Iterable[Normalizer] | None,
) -> Dict[str, Optional[float]]:
    """
    Apply all normalizers to a metrics mapping and return an ordered dict-like result:
    1) Start from the original metrics order
    2) Overwrite with any normalized values having the same keys
    3) Append any additional keys produced by normalizers (normalized-only)

    Notes:
    - `metrics` is not modified; a new dict is returned.
    - Normalizers should return *only* the keys they modify/add.
    """
    # keep a shallow copy of original metrics to preserve order
    non_normalized = dict(metrics)
    normalized_accum: Dict[str, Optional[float]] = {}

    # apply all normalizers (later normalizers can overwrite earlier ones)
    for normalizer in (normalizers or []):
        normalized_accum.update(normalizer.apply(non_normalized))

    # 1) keep original order, using normalized values if present
    ordered: Dict[str, Optional[float]] = {
        k: normalized_accum.get(k, non_normalized[k]) for k in non_normalized
    }

    # 2) append any normalized-only keys
    for k, v in normalized_accum.items():
        if k not in ordered:
            ordered[k] = v

    return ordered

