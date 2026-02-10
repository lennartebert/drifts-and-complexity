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

    Note: assigning to ``value_normalized`` automatically sets
    ``has_normalized = True`` so callers never need to manage both fields.
    """

    name: str
    value: float
    hidden: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    has_normalized: bool = False
    value_normalized: Optional[float] = (
        None  # store normalized value separately from non-normalized value
    )

    def __post_init__(self) -> None:
        # During __init__, __setattr__ is deliberately inert so that the
        # caller-supplied has_normalized is preserved.  After __init__ we
        # fix up the one edge-case (value_normalized explicitly non-None
        # at construction time) and then enable the auto-set behaviour.
        if self.value_normalized is not None:
            object.__setattr__(self, "has_normalized", True)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "value_normalized" and getattr(self, "_initialized", False):
            object.__setattr__(self, "has_normalized", True)
