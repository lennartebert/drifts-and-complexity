from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Measure:
    """
    A single measured value with visibility + metadata.
    - value: numeric result
    - hidden: if True, it's stored but not shown by default
    - meta: free-form metadata (e.g., {"source": "observed"})
    """
    value: float
    hidden: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)