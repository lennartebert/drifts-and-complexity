"""Abstract base class for metric normalizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.complexity.measures.measure_store import MeasureStore


class Normalizer(ABC):
    """Base interface for metric normalizers that operate on a MeasureStore.

    Normalizers transform raw metric values into normalized versions, typically
    for comparison across different scales or to remove size effects. They
    operate in-place on MeasureStore objects and should be robust to missing
    or invalid inputs.

    Contract:
        - Mutate the provided MeasureStore IN PLACE.
        - Do not add new measures; only modify/hide existing ones.
        - If required inputs are missing or invalid, do nothing.

    Examples:
        >>> normalizer = NormalizeNumberOfTraces()
        >>> normalizer.apply(measure_store)
        >>> # measure_store now contains normalized values
    """

    @abstractmethod
    def apply(self, measures: MeasureStore) -> None:
        """Apply normalization to the provided MeasureStore.

        Args:
            measures: MeasureStore to normalize in-place.

        Note:
            Implementations:
            - MUST check measures.has(<target_key>) before modifying.
            - SHOULD preserve/augment meta (e.g., {'normalized_by': <ClassName>}).
            - MUST NOT raise on missing/invalid inputs; just no-op.
        """
        ...
