from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union

from utils.complexity.measures.measure import Measure


class MeasureStore:
    """
    Mutable store for measures produced while computing a window's metrics.

    Methods
    -------
    set(name, value, hidden=False, meta=None)
        Create/overwrite a measure.
    has(name) -> bool
        Whether a measure exists.
    get(name) -> Optional[Measure]
        Return the full Measure object or None.
    get_value(name) -> Optional[float]
        Convenience: return only the numeric value (or None).
    reveal(name) / reveal_many(names)
        Flip hidden=False for selected measures.
    to_visible_dict() -> Dict[str, float]
        Export only visible measures (name -> value).
    to_dict() -> Dict[str, Measure]
        Export all measures (including hidden and metadata).
    update_from(other)
        Merge/overwrite from another MeasureStore or dict[str, Measure].
    """
    def __init__(self, initial: Optional[Union["MeasureStore", Dict[str, Measure]]] = None):
        self._measures: Dict[str, Measure] = {}
        if isinstance(initial, MeasureStore):
            self._measures.update(initial._measures)
        elif isinstance(initial, dict):
            self._measures.update(initial)

    def set(self, name: str, value: float, *, hidden: bool = False, meta: Optional[Dict[str, Any]] = None) -> None:
        self._measures[name] = Measure(value=float(value), hidden=hidden, meta=meta or {})

    def has(self, name: str) -> bool:
        return name in self._measures

    def get(self, name: str) -> Optional[Measure]:
        return self._measures.get(name)

    def get_value(self, name: str, get_normalized_if_available: bool = False) -> Optional[float]:
        m = self._measures.get(name)
        if m is None: return None
        elif (get_normalized_if_available) and (m.value_normalized is not None):
            return m.value_normalized
        else:
            return m.value

    def reveal(self, name: str) -> None:
        if name in self._measures:
            self._measures[name].hidden = False

    def reveal_many(self, names: Iterable[str]) -> None:
        for n in names:
            self.reveal(n)

    def to_visible_dict(self, get_normalized_if_available: bool = False) -> Dict[str, float]:
        result_dict: Dict[str, float] = {}
        for k, m in self._measures.items():
            if m.hidden: continue
            if (get_normalized_if_available) and (m.value_normalized is not None):
                result_dict[k] = m.value_normalized
            else:
                result_dict[k] = m.value

        return result_dict

    def to_dict(self) -> Dict[str, Measure]:
        return dict(self._measures)

    def update_from(self, other: Union["MeasureStore", Dict[str, Measure]]) -> None:
        if isinstance(other, MeasureStore):
            self._measures.update(other._measures)
        else:
            self._measures.update(other)
    
    def __len__(self) -> int:
        """Return the number of measures in the store."""
        return len(self._measures)
    
    def keys(self):
        """Return an iterator over the measure names."""
        return self._measures.keys()