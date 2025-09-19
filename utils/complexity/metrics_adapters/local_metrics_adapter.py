from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

from utils.complexity.measures.measure_store import Measure, MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import (available_metric_names,
                                               get_metric_class)
from utils.complexity.metrics_adapters.metrics_adapter import MetricsAdapter
from utils.windowing.window import Window


class LocalMetricsAdapter(MetricsAdapter):
    """
    Adapter that computes measures using locally implemented Metric classes
    registered in utils.complexity.metrics.registry.

    Parameters
    ----------
    metrics : Optional[Iterable[str]]
        Names to include by default. If None, all registered metrics are used.
    strict : bool
        If True, unknown metric names raise KeyError in ._resolve_metrics().
        If False, unknown names are silently ignored.
    """
    name: str = "local"

    def __init__(self, metrics: Optional[Iterable[str]] = None, *, strict: bool = True):
        self._selected_metric_names = tuple(metrics) if metrics else tuple(available_metric_names())
        self._strict = strict

    def available_metrics(self) -> Iterable[str]:
        return tuple(available_metric_names())

    # --- API ---

    def compute_measures_for_window(
        self,
        window: Window,
        measures: Optional[Union[MeasureStore, Dict[str, Measure]]] = None,
    ) -> Tuple[MeasureStore, Dict]:
        store = measures if isinstance(measures, MeasureStore) else MeasureStore(measures)

        # get metrics instances from names
        metrics = []

        for name in self._selected_metric_names:
            try:
                metric_cls = get_metric_class(name)
                metrics.append(metric_cls())  # instantiate
            except KeyError:
                if self._strict:
                    raise
                # else: ignore unknown 

        for m in metrics:
            m.compute(window, store)

        # Adapter-level info payload
        info = {
            "adapter": self.name,
            "computed": [m.name for m in metrics],
            "hidden": [k for k, v in store.to_dict().items() if v.hidden],
        }
        return store, info

    def compute_measures_for_windows(
        self,
        windows: List[Window],
        measures_by_id: Optional[Dict[str, Union[MeasureStore, Dict[str, Measure]]]] = None,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Dict[str, Tuple[MeasureStore, Dict]]:
        measures_by_id = measures_by_id or {}
        out: Dict[str, Tuple[MeasureStore, Dict]] = {}
        for w in windows:
            store, info = self.compute_measures_for_window(
                w,
                measures=measures_by_id.get(w.id),
                include=include,
                exclude=exclude,
            )
            out[w.id] = (store, info)
        return out
