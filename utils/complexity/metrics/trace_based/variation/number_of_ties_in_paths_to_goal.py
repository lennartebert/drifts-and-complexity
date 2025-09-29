"""Number of ties in paths to goal metric implementation."""

from __future__ import annotations

from collections import Counter
from typing import Any, List

from pm4py.objects.log.obj import Trace

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.registry import register_metric
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric


def _acts(ev: Any) -> str:
    """Extract activity name from event.

    Args:
        ev: Event object.

    Returns:
        Activity name or None.
    """
    return ev.get("concept:name", ev.get("activity", ev.get("Activity", None)))


def _seq(trace: Trace) -> List[str]:
    """Extract activity sequence from trace.

    Args:
        trace: PM4Py Trace object.

    Returns:
        List of activity names.
    """
    return [_acts(ev) for ev in trace]


@register_metric("Number of Ties in Paths to Goal")
class NumberOfTiesInPathsToGoal(TraceMetric):
    """Number of ties in paths to goal based on Pentland-Haerem task complexity.

    Pentland-Haerem task complexity is defined as "... all possible paths to each goal
    of the task and by summing up the number of ties making up these paths..." (Haerem et al. 2015).

    Ideally, this would be implemented as all paths to the goal through the observed adjacency matrix.
    However, this would lead to too long run times. The previous implementation by Vidgof counted
    the number of child nodes of all leaf nodes in an EPA. However, this does not correctly count
    shorter variants that are contained in the prefix of larger variants. Hence, we resort to
    counting the activity transitions in all unique variants.
    """

    name = "Number of Ties in Paths to Goal"
    requires: list[str] = []

    def compute(self, traces: List[Trace], measures: MeasureStore) -> None:
        """Compute the number of ties in paths to goal.

        Args:
            traces: List of PM4Py Trace objects.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return
        variants = Counter(tuple(_seq(tr)) for tr in traces)
        total_ties = 0
        for var_seq in variants.keys():
            total_ties += max(len(var_seq) - 1, 0)
        measures.set(
            self.name,
            int(total_ties),
            hidden=False,
            meta={
                "note": "sum of transitions across unique variants",
                "basis": "traces",
            },
        )
