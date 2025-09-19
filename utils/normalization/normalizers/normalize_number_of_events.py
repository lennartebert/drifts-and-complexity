from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizeNumberOfEvents(Normalizer):
    """
    Normalize 'Number of Events' by 'Number of Traces' -> average trace length.

    Overwrites the existing 'Number of Events' measure with the normalized value.
    If either key is missing or invalid, does nothing.
    """

    KEY = "Number of Events"

    def apply(self, measures: MeasureStore) -> None:
        if not measures.has(self.KEY):
            return
        ne = measures.get_value(self.KEY)
        nt = measures.get_value("Number of Traces")
        if ne is None or nt is None or nt <= 0:
            return

        new_val = float(ne) / float(nt)
        prev_meta = (measures.get(self.KEY).meta if measures.get(self.KEY) else {})
        meta = {**prev_meta, "normalized_by": type(self).__name__}
        measures.set(self.KEY, new_val, hidden=False, meta=meta)
