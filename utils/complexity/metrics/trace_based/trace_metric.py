"""Base class for trace-based metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pm4py.objects.log.obj import Trace

from utils.complexity.measures.measure_store import MeasureStore  # type: ignore
from utils.complexity.metrics.metric import Metric  # type: ignore


class TraceMetric(Metric, ABC):
    """Abstract base class for metrics that operate on traces.

    Trace-based metrics receive a list of PM4Py Trace objects and compute
    complexity measures based on the trace structure and content.
    """

    input_kind = "trace"

    @abstractmethod
    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the metric for the given traces.

        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        ...
