from __future__ import annotations
import math

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizeDeviationFromRandom(Normalizer):
    """
    Replication-invariant normalization of 'Deviation from Random':
        D' = 1 - (1 - D) / sqrt(1 - 1 / V^2)
    where D = 'Deviation from Random' and V = 'Number of Distinct Activities'.
    Clips to [0,1].
    """

    KEY = "Deviation from Random"

    def apply(self, measures: MeasureStore) -> None:
        if not measures.has(self.KEY):
            return
        D = measures.get_value(self.KEY)
        V = measures.get_value("Number of Distinct Activities")
        if D is None or V is None or V <= 1:
            return

        denom_inner = 1.0 - 1.0 / (float(V) ** 2)
        if denom_inner <= 0:
            return
        denom = math.sqrt(denom_inner)
        if denom <= 0:
            return

        new_val = 1.0 - (1.0 - float(D)) / denom
        new_val = max(0.0, min(1.0, float(new_val)))

        prev_meta = (measures.get(self.KEY).meta if measures.get(self.KEY) else {})
        meta = {**prev_meta, "normalized_by": type(self).__name__}
        measures.set(self.KEY, new_val, hidden=False, meta=meta)
