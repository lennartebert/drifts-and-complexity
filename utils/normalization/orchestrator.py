from __future__ import annotations
from typing import List, Dict, Any

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalize_number_of_traces import NormalizeNumberOfTraces
from utils.normalization.normalizers.normalizer import Normalizer
from utils.normalization.normalizers.normalize_number_of_events import NormalizeNumberOfEvents
from utils.normalization.normalizers.hide_number_of_traces import HideNumberOfTraces
from utils.normalization.normalizers.normalize_percentage_of_distinct_traces import NormalizePercentageOfDistinctTraces
from utils.normalization.normalizers.hide_percentage_of_distinct_traces import HidePercentageOfDistinctTraces
from utils.normalization.normalizers.normalize_deviation_from_random import NormalizeDeviationFromRandom
from utils.normalization.normalizers.normalize_lz_complexity import NormalizeLZComplexity

# Default pipeline (instances, in order)
DEFAULT_NORMALIZERS: List[Normalizer] = [
    NormalizeNumberOfEvents(),
    NormalizeNumberOfTraces(),
    NormalizePercentageOfDistinctTraces(),
    NormalizePercentageOfDistinctTraces(),
    NormalizeDeviationFromRandom(),
    NormalizeLZComplexity(),
]


def apply_normalizers(store: MeasureStore, normalizers: List[Normalizer] | None = None) -> MeasureStore:
    """
    Apply all normalizers IN PLACE on the provided MeasureStore.
    Normalizers must not add new measures; only modify or hide existing ones.

    Returns a MeasureStore
    """
    if normalizers is None:
        return store
    for n in normalizers:
        n.apply(store)
    return store
