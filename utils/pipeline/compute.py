# Full compute code with parallel window processing.

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import (
    INextBootstrapSampler,
)
from utils.complexity.measures.measure_store import Measure, MeasureStore
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.complexity.metrics_adapters.metrics_adapter import MetricsAdapter
from utils.normalization.normalizers.normalizer import Normalizer
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS, apply_normalizers

# NEW: centralized parallel helper
from utils.parallel import run_parallel
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
)
from utils.population.extractors.population_extractor import PopulationExtractor
from utils.windowing.window import Window


def _merge_measures_and_info(
    adapters: List[MetricsAdapter],
    window: Window,
    *,
    base_store: Optional[Union[MeasureStore, Dict[str, Measure]]] = None,
) -> Tuple[MeasureStore, Dict[str, Any]]:
    """
    Run all adapters on a single window and merge their outputs into one MeasureStore.

    Later adapters can overwrite same-named measures depending on their own behavior.
    Adapter info is namespaced under adapter.name.
    """
    store = (
        base_store if isinstance(base_store, MeasureStore) else MeasureStore(base_store)
    )
    all_info: Dict[str, Any] = {}

    for adapter in adapters:
        store, info = adapter.compute_measures_for_window(
            window,
            measures=store,
        )
        all_info[adapter.name] = info

    return store, all_info


def _compute_cis_from_bootstrap(
    normalized_measures_per_rep: List[Dict[str, float]],
    keys: List[str],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Percentile CIs for each metric key using bootstrap replicates of *normalized* measures.
    Skips replicates where a metric is missing or NaN/None.
    """
    cis: Dict[str, Dict[str, Optional[float]]] = {}
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    for k in keys:
        vals: List[float] = []
        for rep_m in normalized_measures_per_rep:
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
    normalizers: Optional[List[Normalizer]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1) Ensure population info on the window via `population_extractor`.
      2) Compute measures via adapters into a MeasureStore.
      3) Apply normalizers in place (MeasureStore) and return the normalized, visible measures.
      4) If a bootstrap sampler is provided, draw replicates, recompute + normalize each,
         and return 95% percentile CIs on the normalized measures.

    Returns
    -------
    dict with keys:
      - "measures": normalized, visible measures {name: value}
      - "cis":      {metric: {"low","high","mean","std","n"}} if bootstrap was run, else {}
      - "info":     merged adapter info + pipeline metadata
    """
    # 1) population extraction
    if population_extractor is None:
        population_extractor = NaivePopulationExtractor()
    population_extractor.apply(window)  # sets window.population_distributions

    # 2) compute measures via adapters
    if metric_adapters is None:
        metric_adapters = [
            LocalMetricsAdapter()
        ]  # all registered local metrics by default

    base_store, adapters_info = _merge_measures_and_info(
        metric_adapters,
        window,
    )

    # 3) normalize and get visible normalized measures
    normalized_store: MeasureStore = apply_normalizers(
        base_store, normalizers
    )  # handles None
    normalized_measures: Dict[str, float] = normalized_store.to_visible_dict(
        get_normalized_if_available=True
    )

    result: Dict[str, Any] = {
        "measures": normalized_measures,
        "cis": {},
        "info": {
            "adapters": adapters_info,
            "pipeline": {
                "population_extractor": population_extractor.__class__.__name__,
                "bootstrap": (
                    None
                    if bootstrap_sampler is None
                    else bootstrap_sampler.__class__.__name__
                ),
                "normalizers": (
                    None
                    if normalizers is None
                    else [type(n).__name__ for n in normalizers]
                ),
            },
        },
    }

    # 4) bootstrap (optional). Keep this SEQUENTIAL within the worker for robustness.
    if bootstrap_sampler is not None:
        reps = bootstrap_sampler.sample(window)

        rep_norms_visible: List[Dict[str, float]] = []
        for rep_w in reps:
            # Ensure population info per replicate if the extractor affects estimates
            population_extractor.apply(rep_w)

            rep_store, _ = _merge_measures_and_info(
                metric_adapters,
                rep_w,
            )
            rep_store = apply_normalizers(
                rep_store, normalizers
            )  # returns MeasureStore
            rep_norms_visible.append(
                rep_store.to_visible_dict(get_normalized_if_available=True)
            )

        # CI keys aligned to baseline normalized measures
        ci_keys = list(normalized_measures.keys())
        cis = _compute_cis_from_bootstrap(rep_norms_visible, ci_keys, alpha=0.05)

        result["cis"] = cis
        result["info"]["pipeline"]["bootstrap_replicates"] = len(reps)
        result["info"]["pipeline"]["ci_method"] = "percentile_95"

    return result


# --------------------------------------------------------------------
# Top-level worker used by the parallel window loop
# --------------------------------------------------------------------
def _compute_one_window_task(
    args: Tuple[
        int,
        int,
        "Window",
        Optional["PopulationExtractor"],
        Optional[List["MetricsAdapter"]],
        Optional["INextBootstrapSampler"],
        Optional[List["Normalizer"]],
    ],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    One task = compute metrics + CIs for a single (window_size, sample_id, window).
    Returns:
      - measures_row: base columns + normalized measures
      - ci_low_row:   base columns + CI lows
      - ci_high_row:  base columns + CI highs
      - keys:         set of all metric keys seen (to align columns later)
    """
    (
        window_size,
        sample_id,
        window,
        population_extractor,
        metric_adapters,
        bootstrap_sampler,
        normalizers,
    ) = args

    res = compute_metrics_and_CIs(
        window=window,
        population_extractor=population_extractor,
        metric_adapters=metric_adapters,
        bootstrap_sampler=bootstrap_sampler,
        normalizers=normalizers,
    )

    measures: Dict[str, float] = res.get("measures", {}) or {}
    cis: Dict[str, Dict[str, float]] = res.get("cis", {}) or {}

    keys: Set[str] = set(measures.keys()) | set(cis.keys())
    base = {"sample_size": window_size, "sample_id": sample_id}

    # Measures row
    measures_row = {**base, **measures}

    # CI rows
    low_row: Dict[str, Any] = dict(base)
    high_row: Dict[str, Any] = dict(base)
    for metric, bounds in cis.items():
        if bounds is not None:
            low_row[metric] = bounds.get("low")
            high_row[metric] = bounds.get("high")

    return measures_row, low_row, high_row, keys


def run_metrics_over_samples(
    window_samples: Iterable[Tuple[int, int, "Window"]],
    *,
    population_extractor: Optional["PopulationExtractor"] = None,
    metric_adapters: Optional[List["MetricsAdapter"]] = None,
    bootstrap_sampler: Optional["INextBootstrapSampler"] = None,
    normalizers: Optional[List[Optional["Normalizer"]]] = None,
    sorted_metrics: Optional[Iterable[str]] = None,
    # --- Parallel knobs ---
    parallel_backend: Literal["off", "auto", "processes", "threads"] = "auto",
    n_jobs: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Iterate samples and collect:
      - measures_df:  sample_size, sample_id, [all normalized measures]
      - ci_low_df:    sample_size, sample_id, [CI lows per measure]
      - ci_high_df:   sample_size, sample_id, [CI highs per measure]

    Parallelization strategy:
      * Parallelize over WINDOWS ONLY (best cost/benefit).
      * Bootstraps remain sequential inside each worker to avoid nested pools.
      * Default n_jobs uses SLURM_CPUS_PER_TASK when running on Slurm.

    Returns:
        (measures_df, ci_low_df, ci_high_df)
    """
    rows_measures: List[Dict[str, Any]] = []
    rows_ci_low: List[Dict[str, Any]] = []
    rows_ci_high: List[Dict[str, Any]] = []
    all_measure_keys: Set[str] = set()

    # Materialize the sample iterable (sampling returns a finite set)
    items = list(window_samples)

    # Prepare task args once
    task_args = [
        (
            w_size,
            s_id,
            win,
            population_extractor,
            metric_adapters,
            bootstrap_sampler,
            normalizers,
        )
        for (w_size, s_id, win) in items
    ]

    # Run tasks (sequential if backend="off" or n_jobs==1)
    results = run_parallel(
        task_args,
        _compute_one_window_task,
        backend=parallel_backend,
        n_jobs=n_jobs,  # defaults to SLURM_CPUS_PER_TASK
        chunksize=chunksize,  # auto for processes; 1 for threads
        unordered=True,  # faster; order doesn't matter here
    )

    # Collect rows
    for measures_row, low_row, high_row, keys in results:
        rows_measures.append(measures_row)
        rows_ci_low.append(low_row)
        rows_ci_high.append(high_row)
        all_measure_keys.update(keys)

    # Align columns
    sorted_metrics = list(sorted_metrics or [])
    ordered_metrics = [key for key in sorted_metrics if key in all_measure_keys]
    extra_metrics = sorted(all_measure_keys - set(sorted_metrics))
    all_metrics = ordered_metrics + extra_metrics
    ordered_cols = ["sample_size", "sample_id"] + all_metrics

    measures_df = pd.DataFrame(rows_measures).reindex(columns=ordered_cols)
    ci_low_df = pd.DataFrame(rows_ci_low).reindex(columns=ordered_cols)
    ci_high_df = pd.DataFrame(rows_ci_high).reindex(columns=ordered_cols)

    return measures_df, ci_low_df, ci_high_df
