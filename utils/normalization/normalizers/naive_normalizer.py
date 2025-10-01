from __future__ import annotations

from utils.complexity.measures.measure import Measure
from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NaiveNormalizer(Normalizer):
    """
    Default normalizer that sets value_normalized = value for any measure
    that doesn't already have a normalized value.

    This ensures all measures have a normalized value, making downstream
    code simpler by avoiding null checks.
    """

    def apply(self, measures: MeasureStore) -> None:
        """
        Apply naive normalization to all measures in the store.
        Only sets value_normalized if it's currently None and hasn't been
        explicitly set to None by another normalizer.
        """
        for measure_name in measures.keys():
            measure = measures.get(measure_name)
            if not isinstance(measure, Measure):
                continue

            # Only set normalized value if it's currently None AND
            # hasn't been explicitly set to None by another normalizer
            if (
                measure.value_normalized is None
                and measure.value is not None
                and "set_to_none_by" not in measure.meta
                and "hidden_by" not in measure.meta
            ):
                measure.value_normalized = measure.value
                # Update meta to track this normalization
                measure.meta = {**measure.meta, "normalized_by": type(self).__name__}
