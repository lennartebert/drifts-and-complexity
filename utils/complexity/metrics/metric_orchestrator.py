from __future__ import annotations
from typing import Iterable, List, Optional, Set, Tuple, Dict, Type

from utils.complexity.measures.measure_store import MeasureStore  # type: ignore
from utils.complexity.metrics.metric import Metric  # type: ignore
from utils.complexity.metrics.trace_based.trace_metric import TraceMetric
from utils.complexity.metrics.distribution_based.distribution_metric import PopulationDistributionsMetric
from utils.complexity.metrics.registry import get_metric_classes
from utils.windowing.window import Window  # type: ignore


class MetricOrchestrator:
    """
    Variant-aware metric orchestrator that dispatches metric computation to the
    appropriate implementation (trace-based or distribution-based) and manages
    inter-metric dependencies.

    Core behavior
    -------------
    - **Variant selection**: For each metric name, multiple classes (variants)
      may be registered under the same public name. The orchestrator chooses an
      implementation based on availability of `window.population_distribution`
      and the `prefer` policy.
    - **Input dispatch**:
        * `TraceMetric` implementations receive `window.traces`.
        * `PopulationDistributionMetric` implementations receive
          `window.population_distribution`.
        * Classes not subclassing either base are dispatched via their
          `input_kind` attribute (`"trace"` or `"distribution"`), defaulting to
          `"trace"`.
    - **Dependencies**: If a metric class declares `requires = ["Other Metric", ...]`,
      those dependencies are computed first (recursively). Measures created
      *only* to satisfy dependencies are marked as hidden.
    - **Robustness**: With `strict=True`, errors bubble up; with `strict=False`,
      the orchestrator skips failing variants and continues.

    Parameters
    ----------
    strict : bool, optional
        If True, raise on missing metrics, failed dependencies, or failed
        variant computations. If False, attempt fallbacks and continue where
        possible. Default is True.
    prefer : {"auto","trace","distribution"}, optional
        Variant preference strategy. "auto" picks `"distribution"` when a
        population distribution is present on the window, otherwise `"trace"`.
        Use `"trace"` or `"distribution"` to force a side. Default is "auto".
    """

    def __init__(self, *, strict: bool = True, prefer: str = "auto") -> None:
        self.strict = strict
        self.prefer = prefer

    def compute_many_metrics(
        self,
        metric_names: Iterable[str],
        window: Window,
        store: Optional[MeasureStore] = None,
    ) -> Tuple[MeasureStore, Dict]:
        """
        Compute a collection of metrics on a single window.

        Parameters
        ----------
        metric_names : Iterable[str]
            Public metric names to compute (as registered in the registry).
        window : Window
            Window object providing `traces` and optionally a
            `population_distribution`.
        store : Optional[MeasureStore], optional
            Existing store to populate/update. If None, a new `MeasureStore`
            is created. Default is None.

        Returns
        -------
        (MeasureStore, Dict)
            The (possibly newly created) `MeasureStore` and an info dict:
            - "computed": List[str] of metric names successfully present in the store.
            - "hidden":   List[str] of measures currently marked hidden.
            - "skipped":  List[str] of metric names that were requested but not
                          produced (e.g., due to failures in non-strict mode).

        Notes
        -----
        - Top-level target metrics (those explicitly requested) are forced
          visible when they are newly created by this call.
        """
        store = store if isinstance(store, MeasureStore) else MeasureStore(store)

        skipped: List[str] = []

        for name in metric_names:
            # check if metric already present
            had = store.has(name)
            # if present -> either unhide or do nothing
            if had:
                measure = store.get(name)
                if measure.hidden:
                    measure.hidden = False
                continue
            else:
                # compute metric
                computation_was_successfull = self._compute_one_metric(name, window, store, recursion_guard=set(), top_level=True)
                
                if not computation_was_successfull:
                    skipped.append(name)

        info = {
            "skipped": skipped,
        }
        return store, info

    def _compute_one_metric(
        self,
        metric_name: str,
        window: Window,
        store: MeasureStore,
        *,
        recursion_guard: Optional[Set[str]] = None,
        top_level: bool = True,
    ) -> bool:
        """
        Compute a single metric by name with variant fallback and dependency resolution.

        Parameters
        ----------
        metric_name : str
            Public metric name to compute.
        window : Window
            Window providing `traces` and optionally `population_distribution`.
        store : MeasureStore
            Measure store to read dependencies from and to write results to.
        recursion_guard : Optional[Set[str]], optional
            Internal set guarding against cycles in dependency graphs. Default None.
        top_level : bool, optional
            True if this computation was requested directly by the caller; False
            if it is being computed as a dependency. Dependency-only measures
            created during non-top-level calls may be marked hidden. Default True.

        Returns
        -------
        bool
            True if the metric is present in the store after this call, else False.

        Raises
        ------
        RuntimeError
            - If a circular dependency is detected (and `strict=True`).
            - If a dependency fails (and `strict=True`).
            - If the selected variant requires unavailable inputs (and `strict=True`).
        KeyError
            If the metric name is unknown in the registry (and `strict=True`).

        Notes
        -----
        The orchestrator iterates over all variants registered for `metric_name`
        in an order determined by `_order_variants_for_window`. It attempts each
        variant until one succeeds or all fail (subject to `strict`).
        """
        if store.has(metric_name):
            return True

        recursion_guard = recursion_guard or set()
        if metric_name in recursion_guard:
            if self.strict:
                raise RuntimeError(f"Circular dependency for '{metric_name}'")
            return False
        recursion_guard.add(metric_name)

        try:
            classes = self._order_variants_for_window(window, get_metric_classes(metric_name))
        except KeyError:
            if self.strict:
                raise
            return False

        last_err: Optional[Exception] = None
        for cls in classes:
            try:
                # 1) dependencies
                deps: List[str] = getattr(cls, "requires", []) or []
                newly: List[str] = []
                for dep in deps:
                    if not store.has(dep):
                        ok_dep = self._compute_one_metric(
                            dep, window, store, recursion_guard=recursion_guard, top_level=False
                        )
                        if ok_dep:
                            newly.append(dep)
                        elif self.strict:
                            raise RuntimeError(f"Failed dependency '{dep}' for '{metric_name}'")

                # 2) dispatch correct input object based on variant
                metric: Metric = cls()  # type: ignore
                if issubclass(cls, PopulationDistributionsMetric):
                    pd = getattr(window, "population_distribution", None)
                    if pd is None:
                        raise RuntimeError("PopulationDistribution required but missing on window")
                    metric.compute(pd, store)  # type: ignore[arg-type]
                elif issubclass(cls, TraceMetric):
                    metric.compute(window.traces, store)  # type: ignore[arg-type]
                else:
                    # Fallback: use 'input_kind' if class doesn't subclass the abstractions
                    kind = getattr(cls, "input_kind", "trace")
                    if kind == "distribution":
                        pd = getattr(window, "population_distribution", None)
                        if pd is None:
                            raise RuntimeError("PopulationDistribution required but missing on window")
                        metric.compute(pd, store)  # type: ignore[arg-type]
                    else:
                        metric.compute(window.traces, store)  # type: ignore[arg-type]

                # 3) finalize visibility for dependency-only measures
                if store.has(metric_name):
                    for dep in newly:
                        self._hide_if_new(store, dep)
                    if not top_level:
                        self._hide_if_new(store, metric_name)
                    return True

            except Exception as e:
                last_err = e
                continue

        if last_err and self.strict:
            raise last_err
        return False

    def _order_variants_for_window(self, window: Window, classes: List[Type[Metric]]) -> List[Type[Metric]]:
        """
        Order metric variant classes for a given window according to preference policy.

        Parameters
        ----------
        window : Window
            The window for which inputs are inspected (presence of
            `population_distribution`).
        classes : List[Type[Metric]]
            All registered implementation classes (variants) for a metric name.

        Returns
        -------
        List[Type[Metric]]
            The same classes ordered so that the most suitable variant is tried first.

        Policy
        ------
        - If `prefer == "distribution"` (or `prefer == "auto"` and the window
          has a population distribution), distribution variants are tried first.
        - Otherwise, trace variants are tried first.
        - A class counts as distribution/trace if it subclasses the respective
          abstraction base or declares `input_kind` accordingly.
        """
        has_pd = getattr(window, "population_distribution", None) is not None
        # check that prefer is valid
        if self.prefer not in ("auto", "trace", "distribution"):
            raise ValueError(f"Invalid prefer policy: {self.prefer}")
        
        prefer = 'distribution' if self.prefer == 'auto' else self.prefer

        def is_dist(cls: Type[Metric]) -> bool:
            return issubclass(cls, PopulationDistributionsMetric) or getattr(cls, "input_kind", "trace") == "distribution"

        def is_trace(cls: Type[Metric]) -> bool:
            return issubclass(cls, TraceMetric) or getattr(cls, "input_kind", "trace") == "trace"

        # get for all classes whether they are dist or trace
        classes_to_type_map: Dict[Type[Metric], str] = {}
        for cls in classes:
            if is_dist(cls):
                classes_to_type_map[cls] = "distribution"
            elif is_trace(cls):
                classes_to_type_map[cls] = "trace"
            else:
                raise ValueError(f"Metric class {cls} is neither identifiable as distribution nor trace based")
            
        # if has_pd is False, remove these distribution classes from map
        if not has_pd:
            classes_to_type_map = {cls: typ for cls, typ in classes_to_type_map.items() if typ != "distribution"}
        
        # sort classes_to_type_map by prefered type (preferred type first)
        sorted_classes = sorted(classes_to_type_map.keys(), key=lambda cls: 0 if classes_to_type_map[cls] == prefer else 1)
        
        return sorted_classes

    @staticmethod
    def _hide_if_new(store: MeasureStore, key: str) -> None:
        """
        Mark a just-created measure as hidden, leaving pre-existing measures untouched.

        Parameters
        ----------
        store : MeasureStore
            Store that may contain the measure under `key`.
        key : str
            Measure key/name to potentially mark as hidden.

        Notes
        -----
        Intended for dependency-only outputs so that top-level results remain visible.
        """
        m = store.get(key)
        if m is None:
            return
        if hasattr(m, "hidden") and not m.hidden:
            m.hidden = True
