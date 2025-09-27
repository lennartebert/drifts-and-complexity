from __future__ import annotations
from typing import Any, Protocol, Iterable, Dict, Tuple, Optional, Union, List
from utils.complexity.measures.measure_store import MeasureStore, Measure
from utils.windowing.window import Window

class MetricsAdapter(Protocol):
    """
    Adapter for a source of measures (your local metrics, or a 3rd-party lib).

    Conventions:
    - Implementations should only write into the provided MeasureStore.
    - They must not erase values written by previous adapters.
    - They may set meta (e.g., {"source": "local"} or {"source": "sklearn"}).
    """
    name: str

    def available_metrics(self) -> List[str]:
        """Return canonical metric names that this adapter can produce."""
        ...

    def compute_measures_for_window(
        self,
        window: "Window",
        measures: Optional[Union[MeasureStore, Dict[str, Measure]]] = None,
    ) -> Tuple[MeasureStore, Dict]:
        """
        Compute measures for a single window. 
        Returns (MeasureStore, info).
        """
        ...

    def compute_measures_for_windows(
        self,
        windows: List["Window"],
        measures_by_id: Optional[Dict[Union[str, int], Union[MeasureStore, Dict[str, Measure]]]] = None,
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
            )
            out[key] = (store, info)

        return out


def get_adapters(adapter_names: List[str]) -> List[MetricsAdapter]:
    """Get adapter instances by name.
    
    Args:
        adapter_names: List of adapter names to instantiate.
        
    Returns:
        List of adapter instances.
    """
    from .local_metrics_adapter import LocalMetricsAdapter
    from .vidgof_metrics_adapter import VidgofMetricsAdapter
    
    adapters = []
    for name in adapter_names:
        if name == "local":
            adapters.append(LocalMetricsAdapter())
        elif name == "vidgof_sample":
            adapters.append(VidgofMetricsAdapter())
        else:
            raise ValueError(f"Unknown adapter name: {name}")
    
    return adapters