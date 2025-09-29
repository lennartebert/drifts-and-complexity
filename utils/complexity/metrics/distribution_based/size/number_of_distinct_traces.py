"""Number of distinct traces metric implementation (distribution-based)."""

from __future__ import annotations

from utils.complexity.measures.measure_store import MeasureStore  # type: ignore
from utils.complexity.metrics.distribution_based.distribution_metric import (
    PopulationDistributionsMetric,
)
from utils.complexity.metrics.registry import register_metric
from utils.population.population_distributions import PopulationDistributions


@register_metric("Number of Distinct Traces")
class NumberOfDistinctTraces(PopulationDistributionsMetric):
    """Distribution-based count of distinct traces."""

    name = "Number of Distinct Traces"
    requires: list[str] = []

    def compute(
        self, population_distribution: PopulationDistributions, measures: MeasureStore
    ) -> None:
        """Compute the number of distinct traces from population distribution.

        Args:
            population_distribution: PopulationDistributions object.
            measures: MeasureStore to store the computed metric.
        """
        if measures.has(self.name):
            return

        number_of_distinct_traces = population_distribution.trace_variants.count
        measures.set(
            self.name,
            number_of_distinct_traces,
            hidden=False,
            meta={"bases": "population_distribution"},
        )
