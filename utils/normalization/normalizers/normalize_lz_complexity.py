from __future__ import annotations

import math

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


def _safe_log(x: float) -> float | None:
    try:
        return math.log(x) if x is not None and x > 0 else None
    except Exception:
        return None


class NormalizeLZComplexity(Normalizer):
    """
    Lempel-Ziv normalization (Kaspar & Schuster, 1987):
        LZ' = LZ / ( N / (log N / log V) )
    where:
        LZ = 'Lempel-Ziv Complexity'
        N  = 'Number of Events'
        V  = 'Number of Distinct Activities'
    """

    KEY = "Lempel-Ziv Complexity"

    def apply(self, measures: MeasureStore) -> None:
        # If the base measure isn't present, do nothing
        if not measures.has(self.KEY):
            return

        # Retrieve the underlying measure object (so we can set .value_normalized)
        lz_measure = measures.get(self.KEY)
        if lz_measure is None:
            return

        lz = lz_measure.value

        # Require N and V explicitly
        nda_key = "Number of Distinct Activities"
        ne_key = "Number of Events"

        if not measures.has(ne_key):
            raise Exception(
                "Number of Events required to normalize Lempel–Ziv Complexity"
            )
        if not measures.has(nda_key):
            raise Exception(
                "Number of Distinct Activities required to normalize Lempel–Ziv Complexity"
            )

        N = measures.get_value(ne_key)
        V = measures.get_value(nda_key)

        # Guard rails
        if lz is None or N is None or V is None or N <= 1 or V <= 1:
            return

        lnN = _safe_log(float(N))
        lnV = _safe_log(float(V))
        if lnN is None or lnV is None or lnV <= 0:
            return

        denom = float(N) / (lnN / lnV)
        if denom <= 0:
            return

        # Compute normalized value and store it on the measure (do not overwrite .value)
        norm_val = float(lz) / denom
        lz_measure.value_normalized = norm_val

        # Set has_normalized to True
        lz_measure.has_normalized = True

        # Update meta
        prev_meta = lz_measure.meta or {}
        lz_measure.meta = {**prev_meta, "normalized_by": type(self).__name__}
