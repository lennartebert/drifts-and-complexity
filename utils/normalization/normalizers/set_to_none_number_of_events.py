from __future__ import annotations

from utils.complexity.measures.measure import Measure
from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class SetToNoneNumberOfEvents(Normalizer):
    """
    Set normalized value of 'Number of Events' to None but do not hide the measure.
    """

    KEY = "Number of Events"

    def apply(self, measures: MeasureStore) -> None:
        if not measures.has(self.KEY):
            return
        m = measures.get(self.KEY)
        if not isinstance(m, Measure):
            return

        # Set normalized value to None
        m.value_normalized = None
        # Set has_normalized to True
        m.has_normalized = True

        # Update meta
        m.meta = {**m.meta, "set_to_none_by": type(self).__name__}
