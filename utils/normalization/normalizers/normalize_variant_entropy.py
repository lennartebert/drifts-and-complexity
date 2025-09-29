from __future__ import annotations

import math

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizeVariantEntropy(Normalizer):
    """
    Normalize 'Variant Entropy" by "Number of Partitions".

    - Does nothing if the base 'Variant Entropy" measure is absent.
    - Raises if "Number of Partitions" is missing (required dependency).
    - Stores the normalized value in Measure.value_normalized (does not overwrite .value).
    """

    KEY = "Variant Entropy"

    def apply(self, measures: MeasureStore) -> None:
        # If the base measure isn't present, do nothing
        if not measures.has(self.KEY):
            return

        # Retrieve the underlying measure to set .value_normalized
        var_ent_measure = measures.get(self.KEY)
        if var_ent_measure is None:
            return

        var_ent = var_ent_measure.value

        number_partitions_key = "Number of Partitions"

        # Require Number of Traces explicitly
        if not measures.has(number_partitions_key):
            raise Exception(
                "Number of Partitions required to normalize Variant Entropy"
            )

        number_partitions = measures.get_value(number_partitions_key)

        # Guard rails: if values unusable, skip normalization
        if var_ent is None or number_partitions is None or number_partitions <= 0:
            return

        # Compute normalized value (using Miller-Madow constant)
        K = number_partitions
        b = math.e  # this base aligns with Vidgof's implementation
        gamma = (K - 1) / (2 * math.log(b))

        norm_val = float(var_ent) / float(gamma)

        # Store normalized value without overwriting the raw value
        var_ent_measure.value_normalized = norm_val

        # Update meta
        prev_meta = var_ent_measure.meta or {}
        var_ent_measure.meta = {**prev_meta, "normalized_by": type(self).__name__}
