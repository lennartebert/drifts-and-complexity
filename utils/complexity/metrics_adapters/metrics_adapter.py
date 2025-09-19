from __future__ import annotations
from typing import Protocol, Iterable, Dict, Tuple, Optional, Union, List
from utils.complexity.measures.measure_store import MeasureStore, Measure
from utils.windowing.window import Window

class MetricsAdapter(Protocol):
    """
    Adapter for a source of measures (your local metrics, or a 3rd-party lib).

    Conventions:
    - Implementations should only write into the provided MeasureStore.
    - They must not erase values written by previous adapters.
    - They may set meta (e.g., {"source": "local"} or {"source": "sklearn"}) and hidden flags.
    """
    name: str

    def available_metrics(self) -> Iterable[str]:
        """Return canonical metric names that this adapter can produce."""
        ...

    def compute_measures_for_window(
        self,
        window: "Window",
        measures: Optional[Union[MeasureStore, Dict[str, Measure]]] = None,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Tuple[MeasureStore, Dict]:
        """
        Compute measures for a single window. If include is None -> compute all.
        Exclude takes precedence over include.
        Returns (MeasureStore, info).
        """
        ...

    def compute_measures_for_windows(
        self,
        windows: List["Window"],
        measures_by_id: Optional[Dict[Union[str, int], Union[MeasureStore, Dict[str, Measure]]]] = None,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Dict[Union[str, int], Tuple[MeasureStore, Dict[str, Any]]]:
        """
        Default batch implementation: calls `compute_measures_for_window` for each window.

        Keys the result by `window.id` if present; otherwise by the list index.
        Respects `measures_by_id` (pre-filled MeasureStores) using the same keying.
        """
        measures_by_id = measures_by_id or {}
        out: Dict[Union[str, int], Tuple[MeasureStore, Dict[str, Any]]] = {}

        for idx, w in enumerate(windows):
            key: Union[str, int] = getattr(w, "id", idx)
            existing = measures_by_id.get(key)
            store, info = self.compute_measures_for_window(
                w,
                measures=existing,
                include=include,
                exclude=exclude,
            )
            out[key] = (store, info)

        return out