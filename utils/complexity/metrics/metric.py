from typing import Protocol

from utils.complexity.measures.measure_store import MeasureStore
from utils.windowing.window import Window


class Metric(Protocol):
    """
    Protocol for an atomic metric definition.

    Each metric:
      - receives the window and an existing MeasureStore (re-usable cache)
      - writes any measures (and hidden by-products) into that store
      - returns None (store is mutated in place)
    """
    name: str
    def compute(self, window: Window, measures: MeasureStore) -> None: ...