from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizeNumberOfEvents(Normalizer):
    """
    Normalize 'Number of Events' by 'Number of Traces' -> average trace length.

    - Does nothing if the base 'Number of Events' measure is absent.
    - Raises if 'Number of Traces' is missing (required dependency).
    - Stores the normalized value in Measure.value_normalized (does not overwrite .value).
    """

    KEY = "Number of Events"

    def apply(self, measures: MeasureStore) -> None:
        # If the base measure isn't present, do nothing
        if not measures.has(self.KEY):
            return

        # Retrieve the underlying measure to set .value_normalized
        ne_measure = measures.get(self.KEY)
        if ne_measure is None:
            return

        ne = ne_measure.value
        nt_key = "Number of Traces"

        # Require Number of Traces explicitly
        if not measures.has(nt_key):
            raise Exception("Number of Traces required to normalize Number of Events")

        nt = measures.get_value(nt_key)

        # Guard rails: if values unusable, skip normalization
        if ne is None or nt is None or nt <= 0:
            return

        # Compute normalized value (average trace length)
        norm_val = float(ne) / float(nt)

        # Store normalized value without overwriting the raw value
        ne_measure.value_normalized = norm_val

        # Update meta
        prev_meta = ne_measure.meta or {}
        ne_measure.meta = {**prev_meta, "normalized_by": type(self).__name__}
