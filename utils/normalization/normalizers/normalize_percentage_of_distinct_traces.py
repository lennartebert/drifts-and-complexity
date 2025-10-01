from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizePercentageOfDistinctTraces(Normalizer):
    """
    Convert 'Percentage of Distinct Traces' into a COUNT by multiplying with
    'Number of Traces'.

    - Accepts percentages in [0,1] or [0,100].
    - Does nothing if the base measure is absent.
    - Raises if 'Number of Traces' is missing.
    - Stores the normalized COUNT in .value_normalized, keeps raw percentage in .value.
    """

    KEY = "Percentage of Distinct Traces"

    def apply(self, measures: MeasureStore) -> None:
        # Require presence of the base measure
        if not measures.has(self.KEY):
            return

        pct_measure = measures.get(self.KEY)
        if pct_measure is None:
            return

        pct = pct_measure.value

        nt_key = "Number of Traces"
        if not measures.has(nt_key):
            raise Exception(
                "Number of Traces required to normalize Percentage of Distinct Traces"
            )

        nt = measures.get_value(nt_key)

        # Guard rails
        if pct is None or nt is None or nt <= 0:
            return

        # Accept both [0,1] and [0,100] ranges
        p01 = float(pct) / 100.0 if float(pct) > 1.0 else float(pct)

        # Clamp to valid range
        norm_val = max(0.0, min(float(nt), p01 * float(nt)))

        # Store normalized value without overwriting raw percentage
        pct_measure.value_normalized = norm_val

        # Set has_normalized to True
        pct_measure.has_normalized = True

        # Update meta
        prev_meta = pct_measure.meta or {}
        pct_measure.meta = {**prev_meta, "normalized_by": type(self).__name__}
