from __future__ import annotations
from typing import Any
from utils.complexity.measures.measure_store import MeasureStore  # type: ignore
from utils.complexity.metrics.distribution_based.distribution_metric import PopulationDistributionsMetric
from utils.complexity.metrics.registry import register_metric
from utils.population.population_distributions import PopulationDistributions


@register_metric("Number of Distinct Activity Transitions")
class NumberOfDistinctActivities(PopulationDistributionsMetric):
    name = "Number of Distinct Activity Transitions"
    requires: list[str] = []

    def compute(self, population_distribution: PopulationDistributions, measures: MeasureStore) -> None:
        if measures.has(self.name): return
        
        number_of_activity_transitions = population_distribution.dfg_edges.count
        measures.set(self.name, number_of_activity_transitions, hidden=False, meta={"bases": "population_distribution"})
