from dataclasses import dataclass

from utils.population.population_distribution import PopulationDistribution


@dataclass
class PopulationDistributions:
    """
    Keeps all population distributions in one class. Each field is an iNEXT-style population distribution model.
    """

    activities: PopulationDistribution
    dfg_edges: PopulationDistribution
    trace_variants: PopulationDistribution
