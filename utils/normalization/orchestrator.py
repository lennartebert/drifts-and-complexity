from __future__ import annotations

from typing import Any, Dict, List

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.hide_number_of_traces import HideNumberOfTraces
from utils.normalization.normalizers.hide_percentage_of_distinct_traces import (
    HidePercentageOfDistinctTraces,
)
from utils.normalization.normalizers.naive_normalizer import NaiveNormalizer
from utils.normalization.normalizers.normalize_deviation_from_random import (
    NormalizeDeviationFromRandom,
)
from utils.normalization.normalizers.normalize_lz_complexity import (
    NormalizeLZComplexity,
)
from utils.normalization.normalizers.normalize_number_of_events import (
    NormalizeNumberOfEvents,
)
from utils.normalization.normalizers.normalize_number_of_traces import (
    NormalizeNumberOfTraces,
)
from utils.normalization.normalizers.normalize_percentage_of_distinct_traces import (
    NormalizePercentageOfDistinctTraces,
)
from utils.normalization.normalizers.normalize_variant_entropy import (
    NormalizeVariantEntropy,
)
from utils.normalization.normalizers.normalizer import Normalizer
from utils.normalization.normalizers.set_to_none_number_of_events import (
    SetToNoneNumberOfEvents,
)
from utils.normalization.normalizers.set_to_none_number_of_traces import (
    SetToNoneNumberOfTraces,
)
from utils.normalization.normalizers.set_to_none_percentage_of_distinct_traces import (
    SetToNonePercentageOfDistinctTraces,
)

# Default pipeline (instances, in order)
DEFAULT_NORMALIZERS: List[Normalizer] = [
    SetToNoneNumberOfEvents(),
    SetToNoneNumberOfTraces(),
    SetToNonePercentageOfDistinctTraces(),
    NormalizeDeviationFromRandom(),
    NormalizeLZComplexity(),
    NormalizeVariantEntropy(),
]


def apply_normalizers(
    store: MeasureStore, normalizers: List[Normalizer] | None = None
) -> MeasureStore:
    """
    Apply all normalizers IN PLACE on the provided MeasureStore.
    Normalizers must not add new measures; only modify or hide existing ones.

    Always applies NaiveNormalizer at the end to ensure all measures have
    a normalized value (value_normalized = value if not already set).

    Returns a MeasureStore
    """
    if normalizers is not None:
        for n in normalizers:
            n.apply(store)

    # Always apply naive normalizer at the end to ensure all measures have normalized values
    naive_normalizer = NaiveNormalizer()
    naive_normalizer.apply(store)

    return store
