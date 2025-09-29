from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

from utils.complexity.measures.measure_store import Measure, MeasureStore
from utils.complexity.metrics import registry
from utils.complexity.metrics.metric_orchestrator import MetricOrchestrator
from utils.complexity.metrics_adapters.metrics_adapter import MetricsAdapter
from utils.windowing.window import Window


class LocalMetricsAdapter(MetricsAdapter):
    """
    Adapter that computes measures using locally implemented Metric classes
    registered in `utils.complexity.metrics.registry`, delegating the actual
    computation, dependency handling, and variant selection to the
    `MetricOrchestrator`.

    Parameters
    ----------
    strict : bool, default True
        Passed through to the orchestrator. If True, missing metrics,
        failed dependencies, or failing variants raise; if False, they are
        skipped when possible.
    prefer : {"auto","trace","distribution"}, default "auto"
        Variant preference policy passed to the orchestrator. With "auto",
        distribution-based variants are preferred when a population
        distribution is present on the window; otherwise trace-based
        variants are preferred.
    """

    name: str = "local"

    def __init__(
        self,
        *,
        strict: bool = True,
        prefer: str = "auto",
    ):
        self._strict = strict
        self._prefer = prefer
        self._orch = MetricOrchestrator(strict=strict, prefer=prefer)

    def available_metrics(self) -> List[str]:
        """
        Return the set of metric names this adapter can compute by default.
        """
        return list(registry.available_metric_names())

    # --- API ---

    def compute_measures_for_window(
        self,
        window: Window,
        measures: Optional[Union[MeasureStore, Dict[str, Measure]]] = None,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Tuple[MeasureStore, Dict]:
        """
        Compute the configured metrics for a single `window`.

        This method delegates to the `MetricOrchestrator` which resolves
        dependencies, selects appropriate variants (trace vs. distribution),
        and writes results into the provided or newly created `MeasureStore`.

        Parameters
        ----------
        window : Window
            Input window providing `traces` and optionally
            `population_distribution`.
        measures : Optional[Union[MeasureStore, Dict[str, Measure]]], optional
            Existing store (or dict-like) to populate/update. If None, a new
            `MeasureStore` is created.
        include : Optional[Iterable[str]], optional
            If provided, only include **available metrics** from this set.
        exclude : Optional[Iterable[str]], optional
            If provided, remove these names from the target set.

        Returns
        -------
        (MeasureStore, Dict)
            The populated store and an info dict that includes:
            - "computed": list of metric names successfully present,
            - "hidden":   list of measure keys currently hidden,
            - "skipped":  list of metric names requested but not produced.
        """
        store = (
            measures if isinstance(measures, MeasureStore) else MeasureStore(measures)
        )

        selected_metrics = self.available_metrics()
        if include is not None:
            selected_metrics = [name for name in include if name in selected_metrics]
        if exclude is not None:
            selected_metrics = [
                name for name in selected_metrics if name not in exclude
            ]

        store, info = self._orch.compute_many_metrics(
            selected_metrics,
            window,
            store=store,
        )

        adapter_info = {
            "adapter": self.name,
            "strict": self._strict,
            "prefer": self._prefer,
            **info,
        }
        return store, adapter_info

    def compute_measures_for_windows(
        self,
        windows: List[Window],
        measures_by_id: Optional[
            Dict[str, Union[MeasureStore, Dict[str, Measure]]]
        ] = None,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Dict[str, Tuple[MeasureStore, Dict]]:
        """
        Batch-compute metrics for multiple windows.

        Parameters
        ----------
        windows : List[Window]
            Windows to process.
        measures_by_id : Optional[Dict[str, Union[MeasureStore, Dict[str, Measure]]]], optional
            Optional per-window pre-filled stores keyed by `window.id`.
        include, exclude : Optional[Iterable[str]]
            Include/exclude filters applied per window.

        Returns
        -------
        Dict[str, Tuple[MeasureStore, Dict]]
            Mapping from `window.id` to `(store, info)` as returned by
            `compute_measures_for_window`.
        """
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
