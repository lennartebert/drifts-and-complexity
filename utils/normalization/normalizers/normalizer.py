from __future__ import annotations
from abc import ABC, abstractmethod

from utils.complexity.measures.measure_store import MeasureStore


class Normalizer(ABC):
    """
    Base interface for metric normalizers that operate on a MeasureStore.

    Contract:
    - Mutate the provided MeasureStore IN PLACE.
    - Do not add new measures; only modify/hide existing ones.
    - If required inputs are missing or invalid, do nothing.
    """

    @abstractmethod
    def apply(self, measures: MeasureStore) -> None:
        """
        Apply normalization to the provided MeasureStore.

        Implementations:
        - MUST check measures.has(<target_key>) before modifying.
        - SHOULD preserve/augment meta (e.g., {'normalized_by': <ClassName>}).
        - MUST NOT raise on missing/invalid inputs; just no-op.
        """
        ...
