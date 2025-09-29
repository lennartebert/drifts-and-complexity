from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class HidePercentageOfDistinctTraces(Normalizer):
    """
    Hide 'Percentage of Distinct Traces' after conversion (if present).
    """

    KEY = "Percentage of Distinct Traces"

    def apply(self, measures: MeasureStore) -> None:
        if not measures.has(self.KEY):
            return
        m = measures.get(self.KEY)
        prev_meta = m.meta if m else {}
        meta = {**prev_meta, "hidden_by": type(self).__name__}
        measures.set(self.KEY, m.value if m else None, hidden=True, meta=meta)
