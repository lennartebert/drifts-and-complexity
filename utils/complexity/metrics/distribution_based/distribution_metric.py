from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterable
from utils.complexity.measures.measure_store import MeasureStore  # type: ignore
from utils.complexity.metrics.metric import Metric
from utils.population.population_distributions import PopulationDistributions  # type: ignore

class PopulationDistributionsMetric(Metric, ABC):
    input_kind = "distribution"
    @abstractmethod
    def compute(self, population_distribution: PopulationDistributions, measures: MeasureStore) -> None: ...
