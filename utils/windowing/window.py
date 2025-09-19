from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from datetime import datetime

from utils.population.population_distributions import PopulationDistributions

@dataclass
class Window:
    id: str
    size: int
    traces: list
    population_distributions: Optional[PopulationDistributions] = None
    first_index: Optional[int] = None
    last_index: Optional[int] = None
    start_moment: Optional[datetime] = None
    end_moment: Optional[datetime] = None
    start_change_point: Optional[int] = None
    start_change_point_type: Optional[str] = None
    end_change_point: Optional[int] = None
    end_change_point_type: Optional[str] = None

    def to_dict(self): d = asdict(self); d.pop("traces", None); return d