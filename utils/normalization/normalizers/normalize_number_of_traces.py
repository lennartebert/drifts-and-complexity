from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizeNumberOfTraces(Normalizer):
    """
    Normalize 'Number of Traces' to a constant 1
    (i.e., value_normalized = number_of_traces / number_of_traces).

    - Does nothing if the base measure is absent.
    - Stores 1.0 as the normalized value without overwriting .value.
    """

    KEY = "Number of Traces"

    def apply(self, measures: MeasureStore) -> None:
        # Require presence of the base measure
        if not measures.has(self.KEY):
            raise Exception("Number of Traces required to normalize Number of Traces")

        trace_measure = measures.get(self.KEY)
        if trace_measure is None:
            return

        # Always normalize to 1
        trace_measure.value_normalized = 1.0

        # Update meta
        prev_meta = trace_measure.meta or {}
        trace_measure.meta = {**prev_meta, "normalized_by": type(self).__name__}
