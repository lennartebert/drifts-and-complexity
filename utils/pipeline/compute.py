from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

import numpy as np

from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import INextBootstrapSampler
from utils.complexity.measures.measure_store import MeasureStore, Measure
from utils.complexity.metrics_adapters.metrics_adapter import MetricsAdapter
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.normalization.orchestrator import apply_normalizers, DEFAULT_NORMALIZERS
from utils.normalization.normalizers.normalizer import Normalizer
from utils.population.extractors.naive_population_extractor import NaivePopulationExtractor
from utils.population.extractors.population_extractor import PopulationExtractor
from utils.windowing.window import Window


def _merge_measures_and_info(
    adapters: List[MetricsAdapter],
    window: Window,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    base_store: Optional[Union[MeasureStore, Dict[str, Measure]]] = None,
) -> Tuple[MeasureStore, Dict[str, Any]]:
    """
    Run all adapters on a single window and merge their outputs into one MeasureStore.

    Later adapters can overwrite same-named measures depending on their own behavior.
    Adapter info is namespaced under adapter.name.
    """
    store = base_store if isinstance(base_store, MeasureStore) else MeasureStore(base_store)
    all_info: Dict[str, Any] = {}

    for adapter in adapters:
        store, info = adapter.compute_measures_for_window(
            window,
            measures=store,
            include=include,
            exclude=exclude,
        )
        all_info[adapter.name] = info

    return store, all_info


def _compute_cis_from_bootstrap(
    normalized_metrics_per_rep: List[Dict[str, float]],
    keys: List[str],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Percentile CIs for each metric key using bootstrap replicates of *normalized* metrics.
    Skips replicates where a metric is missing or NaN/None.
    """
    cis: Dict[str, Dict[str, Optional[float]]] = {}
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    for k in keys:
        vals: List[float] = []
        for rep_m in normalized_metrics_per_rep:
            v = rep_m.get(k, None)
            if v is None:
                continue
            try:
                x = float(v)
            except Exception:
                continue
            if np.isfinite(x):
                vals.append(x)

        if len(vals) == 0:
            cis[k] = {"low": None, "high": None, "mean": None, "std": None, "n": 0}
            continue

        arr = np.asarray(vals, dtype=float)
        lo = float(np.percentile(arr, lo_q))
        hi = float(np.percentile(arr, hi_q))
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        cis[k] = {"low": lo, "high": hi, "mean": mean, "std": std, "n": int(len(arr))}
    return cis


def compute_metrics_and_CIs(
    window: Window,
    population_extractor: Optional[PopulationExtractor] = None,
    metric_adapters: Optional[List[MetricsAdapter]] = None,
    bootstrap_sampler: Optional[INextBootstrapSampler] = None,
    normalizers: Optional[List[Optional[Normalizer]]] = None,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1) Ensure population info (distributions/counts) on the window via `population_extractor`.
      2) Compute base measures via adapters into a MeasureStore.
      3) Apply normalizers to the visible measures (flat dict of floats).
      4) If a bootstrap sampler is provided, draw replicates, recompute measures+normalization for each,
         and return 95% percentile CIs on the normalized metrics.

    Returns
    -------
    dict with keys:
      - "measures":            visible (pre-normalization) measures {name: value}
      - "measures_all":        all measures as {name: {"value":..., "hidden":..., "meta":...}}
      - "metrics_normalized":  normalized values {name: value}
      - "cis":                 {metric: {"low","high","mean","std","n"}} if bootstrap was run, else {}
      - "info":                merged adapter info + pipeline metadata
    """
    # 1) population extraction
    if population_extractor is None:
        population_extractor = NaivePopulationExtractor()
    population_extractor.apply(window)  # sets window.population_distributions

    # 2) compute base measures via adapters
    if metric_adapters is None:
        metric_adapters = [LocalMetricsAdapter()]  # all registered local metrics by default

    store, adapters_info = _merge_measures_and_info(
        metric_adapters,
        window,
        include=include,
        exclude=exclude,
    )

    # Flat dict of visible values for normalization / reporting
    visible_measures: Dict[str, float] = store.to_visible_dict()

    # 3) apply normalizers
    normalizer_instances = [n for n in (normalizers or DEFAULT_NORMALIZERS) if n is not None]
    normalized_metrics: Dict[str, float] = apply_normalizers(visible_measures, normalizer_instances)

    result: Dict[str, Any] = {
        "measures": visible_measures,
        "measures_all": {
            k: {"value": m.value, "hidden": m.hidden, **({"meta": m.meta} if m.meta else {})}
            for k, m in store.to_dict().items()
        },
        "metrics_normalized": normalized_metrics,
        "cis": {},
        "info": {
            "adapters": adapters_info,
            "pipeline": {
                "population_extractor": population_extractor.__class__.__name__,
                "bootstrap": None if bootstrap_sampler is None else bootstrap_sampler.__class__.__name__,
                "normalizers": [type(n).__name__ for n in normalizer_instances],
                "include": list(include) if include is not None else None,
                "exclude": list(exclude) if exclude is not None else None,
            },
        },
    }

    # 4) bootstrap (optional)
    if bootstrap_sampler is not None:
        reps = bootstrap_sampler.sample(window)

        rep_norms: List[Dict[str, float]] = []
        for rep_w in reps:
            # Ensure population info per replicate if your extractor affects estimates
            population_extractor.apply(rep_w)

            rep_store, _ = _merge_measures_and_info(
                metric_adapters,
                rep_w,
                include=include,
                exclude=exclude,
            )
            rep_visible = rep_store.to_visible_dict()
            rep_norm = apply_normalizers(rep_visible, normalizer_instances)
            rep_norms.append(rep_norm)

        # CI keys aligned to baseline normalized metrics
        ci_keys = list(normalized_metrics.keys())
        cis = _compute_cis_from_bootstrap(rep_norms, ci_keys, alpha=0.05)

        result["cis"] = cis
        result["info"]["pipeline"]["bootstrap_replicates"] = len(reps)
        result["info"]["pipeline"]["ci_method"] = "percentile_95"

    return result
