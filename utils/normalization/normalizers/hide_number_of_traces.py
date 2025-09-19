from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class HideNumberOfTraces(Normalizer):
    """
    Hide 'Number of Traces' from visible outputs (keeps value, sets hidden=True).
    """

    KEY = "Number of Traces"

    def apply(self, measures: MeasureStore) -> None:
        if not measures.has(self.KEY):
            return
        m = measures.get(self.KEY)
        prev_meta = (m.meta if m else {})
        meta = {**prev_meta, "hidden_by": type(self).__name__}
        # retain current value, just mark hidden
        measures.set(self.KEY, m.value if m else None, hidden=True, meta=meta)
