from typing import Protocol, Any

from utils.complexity.measures.measure_store import MeasureStore


class Metric(Protocol):
    """
    Protocol for an atomic metric definition.

    Each metric:
      - receives input data and an existing MeasureStore (re-usable cache)
      - writes any measures (and hidden by-products) into that store
      - returns None (store is mutated in place)
    """
    name: str
    requires: list[str] = []
    
    def compute(self, input_data: Any, measures: MeasureStore) -> None: ...