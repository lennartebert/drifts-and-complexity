import math

from utils.complexity.measures.measure_store import MeasureStore
from utils.normalization.normalizers.normalizer import Normalizer


class NormalizeSequenceEntropy(Normalizer):
    """
    Sequence Entropy (Es) — Miller–Madow correction (cap optional)

    Requires measures:
      - "Sequence Entropy" (Es; computed with natural logs)
      - "Number of Partitions" (K)
      - "Number of Events" (M = |seq(S)|)

    Sets:
      - measure.value_normalized = Es_MM_cap
    """

    KEY = "Sequence Entropy"
    PARTITIONS_KEY = "Number of Partitions"
    EVENTS_KEY = "Number of Events"

    # Toggle feasibility cap at M * ln(K)
    CAP_AT_FEASIBLE_MAX = True

    def apply(self, measures: MeasureStore) -> None:
        if not measures.has(self.KEY):
            return

        m = measures.get(self.KEY)
        if m is None or m.value is None:
            return

        if not measures.has(self.PARTITIONS_KEY):
            raise Exception(
                "Number of Partitions required to normalize Sequence Entropy"
            )
        if not measures.has(self.EVENTS_KEY):
            raise Exception("Number of Events required to normalize Sequence Entropy")

        E = float(m.value)  # raw Es
        K = int(measures.get_value(self.PARTITIONS_KEY))  # partitions
        M = int(measures.get_value(self.EVENTS_KEY))  # events

        # Guardrails
        if K <= 1 or M <= 1:
            m.value_normalized = 0.0
            m.has_normalized = True
            m.meta = {**(m.meta or {}), "normalized_by": type(self).__name__}
            return

        # Miller–Madow additive correction (natural logs)
        gamma = (K - 1) / 2.0
        E_mm = E + gamma

        # Optional feasibility cap: Es <= M * ln(K)
        if self.CAP_AT_FEASIBLE_MAX and K > 1:
            E_mm = min(E_mm, M * math.log(K))

        m.value_normalized = E_mm
        m.has_normalized = True
        m.meta = {
            **(m.meta or {}),
            "normalized_by": type(self).__name__,
            "mm_gamma": gamma,
            "cap_applied": self.CAP_AT_FEASIBLE_MAX,
        }
