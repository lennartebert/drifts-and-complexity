"""Number of distinct directly-follows relations metric implementation (distribution-based)."""

from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.distribution_based.distribution_metric import (
    PopulationDistributionsMetric,
)
from utils.complexity.metrics.registry import register_metric
from utils.population.population_distributions import PopulationDistributions


@register_metric("Number of Distinct Directly-Follows Relations")
class NumberOfDistinctActivityTransitions(PopulationDistributionsMetric):
    """Distribution-based count of distinct directly-follows relations."""

    name = "Number of Distinct Directly-Follows Relations"
    requires: list[str] = []

    def compute(
        self, population_distribution: PopulationDistributions, measures: MeasureStore
    ) -> None:
        """Compute the number of distinct directly-follows relations from population distribution.

        Args:
            population_distribution: PopulationDistributions object.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return

        number_of_activity_transitions = (
            population_distribution.dfg_edges.population_count
        )
        measures.set(
            self.name,
            number_of_activity_transitions,
            hidden=False,
            meta={"bases": "population_distribution"},
        )
