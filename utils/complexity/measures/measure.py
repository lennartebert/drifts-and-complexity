from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Measure:
    """
    A single measured value with visibility + metadata.
    - name: identifier for the measure
    - value: numeric result
    - hidden: if True, it's stored but not shown by default
    - meta: free-form metadata (e.g., {"source": "observed"})
    - has_normalized: if True, the value_normalized is available (may still be None)
    - value_normalized: normalized version of the value
    """

    name: str
    value: float
    hidden: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    has_normalized: bool = False
    value_normalized: Optional[float] = (
        None  # store normalized value separately from non-normalized value
    )
