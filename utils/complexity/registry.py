from __future__ import annotations
from typing import Dict, Iterable, List
from .adapters.base import ComplexityAdapter
from .adapters.vidgof import VidgofSampleAdapter
from .adapters.population_inext import PopulationInextAdapter
from .adapters.population_simple import PopulationSimpleAdapter

_BUILTIN: Dict[str, ComplexityAdapter] = {
    "vidgof_sample": VidgofSampleAdapter(),
    "population_inext": PopulationInextAdapter(),   # supports nboot/conf/auto-coverage
    "population_simple": PopulationSimpleAdapter(), # coverage=1.0 only
}

def available_adapters() -> List[str]:
    return list(_BUILTIN.keys())

def get_adapters(names: Iterable[str]) -> List[ComplexityAdapter]:
    return [_BUILTIN[n] for n in names]
