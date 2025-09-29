"""Estimated number of acyclic paths metric implementation."""

from __future__ import annotations

from typing import List

from pm4py.objects.log.obj import Trace

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


@register_metric("Estimated Number of Acyclic Paths")
class EstimatedNumberOfAcyclicPaths(TraceMetric):
    """Estimated number of acyclic paths using Pentland formula."""

    name = "Estimated Number of Acyclic Paths"
    requires: list[str] = [
        "Number of Distinct Activities",
        "Number of Distinct Activity Transitions",
    ]

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute estimated number of acyclic paths using Pentland formula.

        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return

        number_of_distinct_activities = measures.get("Number of Distinct Activities")
        number_of_activity_transitions = measures.get(
            "Number of Distinct Activity Transitions"
        )
        if (
            number_of_distinct_activities is None
            or number_of_activity_transitions is None
        ):
            raise ValueError(
                "Required measures missing: "
                f"Number of Distinct Activities={number_of_distinct_activities}, "
                f"Number of Distinct Activity Transitions={number_of_activity_transitions}"
            )

        e = number_of_activity_transitions.value  # i.e., number of edges in DFG
        v = number_of_distinct_activities.value  # i.e., number of vertices in DFG
        value = 10 ** (0.08 * (1 + e - v))
        measures.set(
            self.name,
            float(value),
            hidden=False,
            meta={"formula": "10**(0.08*(1+e-v))", "basis": "derived"},
        )
