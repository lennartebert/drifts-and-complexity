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
        if not measures.has(self.KEY):
            return
        lz = measures.get_value(self.KEY)
        N = measures.get_value("Number of Events")
        V = measures.get_value("Number of Distinct Activities")
        if lz is None or N is None or V is None or N <= 1 or V <= 1:
            return

        lnN = _safe_log(float(N))
        lnV = _safe_log(float(V))
        if not lnN or not lnV or lnV <= 0:
            return

        denom = float(N) / (lnN / lnV)
        if denom <= 0:
            return

        new_val = float(lz) / denom
        prev_meta = (measures.get(self.KEY).meta if measures.get(self.KEY) else {})
        meta = {**prev_meta, "normalized_by": type(self).__name__}
        measures.set(self.KEY, new_val, hidden=False, meta=meta)
