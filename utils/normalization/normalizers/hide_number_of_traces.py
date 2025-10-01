from __future__ import annotations

from utils.complexity.measures.measure import Measure
from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class HideNumberOfTraces(Normalizer):
    """
    Hide 'Number of Traces' by setting normalized value to None and setting measure to hidden.
    """

    KEY = "Number of Traces"

    def apply(self, measures: MeasureStore) -> None:
        if not measures.has(self.KEY):
            return
        m = measures.get(self.KEY)
        if not isinstance(m, Measure):
            return

        # Set has_normalized to True
        m.has_normalized = True
        # Set normalized value to None
        m.value_normalized = None
        # Hide the measure
        m.hidden = True
        # Update meta
        m.meta = {**m.meta, "hidden_by": type(self).__name__}
