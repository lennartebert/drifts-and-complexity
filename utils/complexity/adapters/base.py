from __future__ import annotations
from typing import Dict, List, Tuple, Protocol, Any
from utils.windowing.windowing import Window

Metrics = Dict[str, float | int | float]
Info    = Dict[str, Any]

class ComplexityAdapter(Protocol):
    """Return (metrics, info) for each window."""
    name: str

    def compute_one(self, window: Window) -> Tuple[Metrics, Info]:
        ...

    def compute_many(self, windows: List[Window]) -> Dict[str, Tuple[Metrics, Info]]:
        return {w.id: self.compute_one(w) for w in windows}
