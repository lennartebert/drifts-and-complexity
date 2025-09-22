from __future__ import annotations
import math

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizeDeviationFromRandom(Normalizer):
    """
    Replication-invariant normalization of 'Deviation from Random':
        D' = 1 - (1 - D) / sqrt(1 - 1 / V^2)
    where D = 'Deviation from Random' and V = 'Number of Distinct Activities'.
    Clips to [0,1].
    """

    KEY = "Deviation from Random"

    def apply(self, measures: MeasureStore) -> None:
        # get deviation from random if available - if not, do nothing
        if not measures.has(self.KEY):
            return
        deviation_from_random_measure = measures.get(self.KEY)
        deviation_from_random_value = deviation_from_random_measure.value

        # get deviation from random if available - if not, raise exception
        nda_key = "Number of Distinct Activities"
        if not measures.has(nda_key):
            raise Exception('Number of distinct activities required to normalize devaiation from random')
        number_distinct_activities_value = measures.get_value(nda_key)

        denom_inner = 1.0 - 1.0 / (float(number_distinct_activities_value) ** 2)
        if denom_inner <= 0:
            return
        denom = math.sqrt(denom_inner)
        if denom <= 0:
            return
        
        norm_val = 1.0 - (1.0 - float(deviation_from_random_value)) / denom
        norm_val = max(0.0, min(1.0, float(norm_val)))

        # add norm value to measure
        deviation_from_random_measure.value_normalized = norm_val
        
        # update meta information
        prev_meta = deviation_from_random_measure.meta
        meta = {**prev_meta, "normalized_by": type(self).__name__}
        deviation_from_random_measure.meta = meta
