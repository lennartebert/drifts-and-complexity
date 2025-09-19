from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizePercentageOfDistinctTraces(Normalizer):
    """
    Convert 'Percentage of Distinct Traces' into a COUNT by multiplying with
    'Number of Traces'. Overwrites 'Percentage of Distinct Traces'.
    - Accepts percentages in [0,1] or [0,100].
    - If either key is missing/invalid, no-ops.
    """

    KEY = "Percentage of Distinct Traces"

    def apply(self, measures: MeasureStore) -> None:
        if not measures.has(self.KEY):
            return
        pct = measures.get_value(self.KEY)
        nt = measures.get_value("Number of Traces")
        if pct is None or nt is None or nt <= 0:
            return

        p01 = float(pct) / 100.0 if float(pct) > 1.0 else float(pct)
        new_val = max(0.0, min(float(nt), p01 * float(nt)))

        prev_meta = (measures.get(self.KEY).meta if measures.get(self.KEY) else {})
        meta = {**prev_meta, "normalized_by": type(self).__name__}
        measures.set(self.KEY, new_val, hidden=False, meta=meta)
