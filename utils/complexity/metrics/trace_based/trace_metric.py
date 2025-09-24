from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterable
from utils.complexity.measures.measure_store import MeasureStore  # type: ignore
from utils.complexity.metrics.metric import Metric  # type: ignore

class TraceMetric(Metric, ABC):
    input_kind = "trace"
    
    @abstractmethod
    def compute(self, traces: Iterable[Iterable[Any]], measures: MeasureStore) -> None: ...
