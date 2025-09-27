"""Base class for distribution-based metrics."""

from __future__ import annotations
from abc import ABC, abstractmethod
from utils.complexity.measures.measure_store import MeasureStore  # type: ignore
from utils.complexity.metrics.metric import Metric
from utils.population.population_distributions import PopulationDistributions  # type: ignore

class PopulationDistributionsMetric(Metric, ABC):
    """Abstract base class for metrics that operate on population distributions.
    
    Distribution-based metrics receive PopulationDistributions objects and compute
    complexity measures based on population-level estimates rather than observed traces.
    """
    input_kind = "distribution"
    
    @abstractmethod
    def compute(self, population_distribution: PopulationDistributions, measures: MeasureStore) -> None:
        """Compute the metric for the given population distribution.
        
        Args:
            population_distribution: PopulationDistributions object.
            measures: MeasureStore to store the computed metric.
        """
        ...
