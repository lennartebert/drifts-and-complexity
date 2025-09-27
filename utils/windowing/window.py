"""Window data structure for process mining analysis."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

from pm4py.objects.log.obj import Trace
from utils.population.population_distributions import PopulationDistributions

@dataclass
class Window:
    """A window containing traces and optional population distributions.
    
    Attributes:
        id: Unique identifier for the window.
        size: Number of traces in the window.
        traces: List of PM4Py Trace objects.
        population_distributions: Optional population distribution data.
        first_index: Index of first trace in original log.
        last_index: Index of last trace in original log.
        start_moment: Start timestamp of the window.
        end_moment: End timestamp of the window.
        start_change_point: Index of start change point.
        start_change_point_type: Type of start change point.
        end_change_point: Index of end change point.
        end_change_point_type: Type of end change point.
    """
    id: str
    size: int
    traces: List[Trace]
    population_distributions: Optional[PopulationDistributions] = None
    first_index: Optional[int] = None
    last_index: Optional[int] = None
    start_moment: Optional[datetime] = None
    end_moment: Optional[datetime] = None
    start_change_point: Optional[int] = None
    start_change_point_type: Optional[str] = None
    end_change_point: Optional[int] = None
    end_change_point_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert window to dictionary, excluding traces for serialization.
        
        Returns:
            Dictionary representation of the window.
        """
        d = asdict(self)
        d.pop("traces", None)
        return d