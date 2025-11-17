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
from utils.SampleConfidenceIntervalExtractor import SampleConfidenceIntervalExtractor
from utils.windowing.window import Window


def _merge_measures_and_info(
    adapters: List[MetricsAdapter],
    window: Window,
    *,
    base_store: Optional[Union[MeasureStore, Dict[str, Measure]]] = None,
    include_metrics: Optional[Iterable[str]] = None,
) -> Tuple[MeasureStore, Dict[str, Any]]:
    """
    Run all adapters on a single window and merge their outputs into one MeasureStore.

    Later adapters can overwrite same-named measures depending on their own behavior.
    Adapter info is namespaced under adapter.name.

    Parameters
    ----------
    adapters : List[MetricsAdapter]
        Adapters to compute measures.
    window : Window
        Input window to process.
    base_store : Optional[Union[MeasureStore, Dict[str, Measure]]]
        Optional pre-filled store to write into.
    include : Optional[Iterable[str]]
        If provided, adapters that support filtering will only compute metrics
        in this list. Others will compute normally and results will be filtered later.
    """
    store = (
        base_store if isinstance(base_store, MeasureStore) else MeasureStore(base_store)
    )
    all_info: Dict[str, Any] = {}

    for adapter in adapters:
        # Some adapters support include/exclude filters. Prefer passing include when available.
        try:
            store, info = adapter.compute_measures_for_window(
                window,
                measure_store=store,
                include_metrics=include_metrics,
            )
        except TypeError:
            store, info = adapter.compute_measures_for_window(
                window,
                measure_store=store,
            )
        all_info[adapter.name] = info

    return store, all_info


def _compute_cis_from_bootstrap(
    normalized_measures_per_rep: List[Dict[str, float]],
    keys: List[str],
    alpha: float = 0.05,
    method: str = "inext",
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    CIs for each metric key using bootstrap replicates of *normalized* measures.
    Skips replicates where a metric is missing or NaN/None.

    Parameters
    ----------
    normalized_measures_per_rep : List[Dict[str, float]]
        Normalized measures for each bootstrap replicate.
    keys : List[str]
        Metric keys to compute CIs for.
    alpha : float, default=0.05
        Significance level (1 - confidence level).
    method : str, default="inext"
        CI method: "inext" for mean ± 1.96*se, "percentile" for percentile-based.
    """
    cis: Dict[str, Dict[str, Optional[float]]] = {}

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
            cis[k] = {
                "low": None,
                "high": None,
                "mean": None,
                "std": None,
                "se": None,
                "n": 0,
            }
            continue

        # Use float64 for intermediate calculations to prevent overflow
        arr = np.asarray(vals, dtype=np.float64)
        mean_64 = np.mean(arr)
        std_64 = np.std(arr, ddof=1) if len(arr) > 1 else np.float64(0.0)
        se_64 = (
            std_64 / np.sqrt(np.float64(len(arr))) if len(arr) > 1 else np.float64(0.0)
        )

        # Cast back to regular float - overflow becomes inf naturally
        mean = float(mean_64)
        std = float(std_64)
        se = float(se_64)

        if method == "inext":
            # iNEXT-style: mean ± 1.96 * se
            z_score = 1.96  # 95% confidence
            lo = mean - z_score * se
            hi = mean + z_score * se
        else:  # percentile
            lo_q = 100.0 * (alpha / 2.0)
            hi_q = 100.0 * (1.0 - alpha / 2.0)
            lo = float(np.percentile(arr, lo_q))
            hi = float(np.percentile(arr, hi_q))

        cis[k] = {
            "low": lo,
            "high": hi,
            "mean": mean,
            "std": std,
            "se": se,
            "n": int(len(arr)),
        }
    return cis


def compute_metrics_and_CIs(
    window: Window,
    population_extractor: Optional[PopulationExtractor] = None,
    metric_adapters: Optional[List[MetricsAdapter]] = None,
    bootstrap_sampler: Optional[INextBootstrapSampler] = None,
    normalizers: Optional[List[Normalizer]] = None,
    include_metrics: Optional[Iterable[str]] = None,
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

    Notes
    -----
    If `include_metrics` is provided, only those metrics are computed when possible
    and returned in the results. Unknown metrics in `include_metrics` are ignored.
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
        include_metrics=include_metrics,
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
            # Bootstrap replicates have population_distributions = None
            # Re-apply population extractor to get proper population distributions
            population_extractor.apply(rep_w)

            rep_store, _ = _merge_measures_and_info(
                metric_adapters,
                rep_w,
                include_metrics=include_metrics,
            )
            rep_store = apply_normalizers(
                rep_store, normalizers
            )  # returns MeasureStore
            rep_norms_visible.append(
                rep_store.to_visible_dict(get_normalized_if_available=True)
            )

        # CI keys aligned to baseline normalized measures
        ci_keys = list(normalized_measures.keys())
        cis = _compute_cis_from_bootstrap(
            rep_norms_visible, ci_keys, alpha=0.05, method="inext"
        )

        result["cis"] = cis
        result["info"]["pipeline"]["bootstrap_replicates"] = len(reps)
        result["info"]["pipeline"]["ci_method"] = "inext_normal_95"

    return result


# --------------------------------------------------------------------
# Top-level worker used by the parallel window loop
# --------------------------------------------------------------------
def _compute_one_window_task(
    args: Tuple[
        int,
        Union[int, str],
        "Window",
        Optional["PopulationExtractor"],
        Optional[List["MetricsAdapter"]],
        Optional["INextBootstrapSampler"],
        Optional[List["Normalizer"]],
        Optional[Iterable[str]],
    ],
) -> List[Dict[str, Any]]:
    """
    One task = compute metrics + CIs for a single (window_size, sample_id, window).
    Returns long format rows: one row per metric with columns:
      - Sample Size, Sample ID, Metric, Value, Bootstrap CI Low, Bootstrap CI High
    """
    (
        window_size,
        sample_id,
        window,
        population_extractor,
        metric_adapters,
        bootstrap_sampler,
        normalizers,
        include_metrics,
    ) = args

    res = compute_metrics_and_CIs(
        window=window,
        population_extractor=population_extractor,
        metric_adapters=metric_adapters,
        bootstrap_sampler=bootstrap_sampler,
        normalizers=normalizers,
        include_metrics=include_metrics,
    )

    measures: Dict[str, float] = res.get("measures", {}) or {}
    cis: Dict[str, Dict[str, float]] = res.get("cis", {}) or {}

    # Build long format rows directly
    rows = []
    for metric, value in measures.items():
        row: Dict[str, Any] = {
            "Sample Size": window_size,
            "Sample ID": sample_id,
            "Metric": metric,
            "Value": value,
        }
        # Add bootstrap CIs if available
        if metric in cis and cis[metric] is not None:
            row["Bootstrap CI Low"] = cis[metric].get("low")
            row["Bootstrap CI High"] = cis[metric].get("high")
        else:
            row["Bootstrap CI Low"] = None
            row["Bootstrap CI High"] = None
        rows.append(row)

    return rows


def compute_metrics_for_samples(
    window_samples: Iterable[Tuple[int, Union[int, str], "Window"]],
    # sample_id may be int or str depending on sampling helper
    *,
    population_extractor: Optional["PopulationExtractor"] = None,
    metric_adapters: Optional[List["MetricsAdapter"]] = None,
    bootstrap_sampler: Optional["INextBootstrapSampler"] = None,
    normalizers: Optional[List[Optional["Normalizer"]]] = None,
    include_metrics: Optional[Iterable[str]] = None,
    # --- Parallel knobs ---
    parallel_backend: Literal["off", "auto", "processes", "threads"] = "auto",
    n_jobs: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute metrics for window samples and return raw metric values in long format.

    The returned DataFrame has columns:
      - Metric: metric name
      - Sample Size: window size
      - Sample ID: sample/replicate identifier
      - Value: the metric value
      - Bootstrap CI Low: lower bootstrap CI bound (if bootstrap_sampler provided)
      - Bootstrap CI High: upper bootstrap CI bound (if bootstrap_sampler provided)

    Note: Bootstrap CIs are included as they are computed during window processing.
    Other analysis (sample CIs, correlations, plateau detection) should be done
    separately using compute_analysis_for_metrics.

    Parallelization strategy:
      * Parallelize over WINDOWS ONLY (best cost/benefit).
      * Bootstraps remain sequential inside each worker to avoid nested pools.
      * Default n_jobs uses SLURM_CPUS_PER_TASK when running on Slurm.

    If `include_metrics` is provided, only those metrics are computed (when
    adapters support filtering) and included in the output DataFrame, in the
    same order as provided. If not provided, all discovered metrics are returned
    in alphabetical order.

    Returns:
        Single DataFrame in long format with Metric and Sample Size as index columns.
    """
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
            list(include_metrics) if include_metrics is not None else None,
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

    # Collect long format rows directly (each result is a list of rows)
    all_rows = []
    all_measure_keys: Set[str] = set()
    for task_rows in results:
        all_rows.extend(task_rows)
        for row in task_rows:
            all_measure_keys.add(row["Metric"])

    # Build DataFrame directly in long format
    long_df = pd.DataFrame(all_rows)

    # Filter metrics if include_metrics specified
    if include_metrics is not None:
        include_list = list(include_metrics)
        long_df = long_df[long_df["Metric"].isin(include_list)]
        # Ensure metrics are in the specified order by creating a categorical
        metric_order = {m: i for i, m in enumerate(include_list)}
        long_df["_metric_order"] = long_df["Metric"].map(metric_order)
        long_df = long_df.sort_values(
            by=["_metric_order", "Sample Size", "Sample ID"]
        ).drop(columns=["_metric_order"])
    else:
        # Sort by metric name, then sample size, then sample ID
        long_df = long_df.sort_values(by=["Metric", "Sample Size", "Sample ID"])

    # Set index to Metric and Sample Size
    long_df = long_df.set_index(["Metric", "Sample Size"]).sort_index()

    return long_df


def compute_analysis_for_metrics(
    metrics_df: pd.DataFrame,
    *,
    sample_confidence_interval_extractor: Optional[
        "SampleConfidenceIntervalExtractor"
    ] = None,
    include_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Compute analysis metrics from raw metric values: mean values, CIs, correlations, and plateau detection.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Long format DataFrame with columns: Sample Size, Sample ID, Metric, Value
        (and optionally Bootstrap CI Low, Bootstrap CI High).
        May have a MultiIndex (will be reset if present).
    sample_confidence_interval_extractor : Optional[SampleConfidenceIntervalExtractor]
        If provided, computes sample confidence intervals across samples.
    include_metrics : Optional[Iterable[str]]
        If provided, only analyze these metrics.

    Returns
    -------
    pd.DataFrame with index (Metric, Sample Size) and columns:
        - Mean Value: mean metric value per (Metric, Sample Size)
        - Sample CI Low: lower sample CI bound (if extractor provided)
        - Sample CI High: upper sample CI bound (if extractor provided)
        - Sample CI Rel Width: relative CI width (if extractor provided)
        - Pearson Rho: Pearson correlation coefficient (constant per Metric)
        - Pearson P: Pearson p-value (constant per Metric)
        - Spearman Rho: Spearman correlation coefficient (constant per Metric)
        - Spearman P: Spearman p-value (constant per Metric)
        - Plateau n: sample size where plateau is reached (constant per Metric)
        - Plateau Found: boolean indicating if plateau was found (constant per Metric)
    """
    from utils import helpers

    # Reset index if present
    if isinstance(metrics_df.index, pd.MultiIndex) or any(
        name is not None for name in metrics_df.index.names
    ):
        metrics_df = metrics_df.reset_index()

    # Filter metrics if specified
    if include_metrics is not None:
        include_list = list(include_metrics)
        metrics_df = metrics_df[metrics_df["Metric"].isin(include_list)]
        metric_list = include_list
    else:
        metric_list = sorted(metrics_df["Metric"].unique().tolist())

    # 1. Compute mean values per metric and sample size (base structure)
    analysis_df = (
        metrics_df.groupby(["Sample Size", "Metric"], as_index=True)["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": "Mean Value"})
    )

    # 2. Compute sample confidence intervals if extractor provided
    if sample_confidence_interval_extractor is not None:
        sample_ci_df = sample_confidence_interval_extractor.compute_sample_ci_long(
            metrics_df
        )
        analysis_df = analysis_df.merge(
            sample_ci_df[
                [
                    "Sample Size",
                    "Metric",
                    "Sample CI Low",
                    "Sample CI High",
                    "Sample CI Rel Width",
                ]
            ],
            on=["Sample Size", "Metric"],
            how="left",
        )
    else:
        analysis_df["Sample CI Low"] = None
        analysis_df["Sample CI High"] = None
        analysis_df["Sample CI Rel Width"] = None

    # 3. Compute correlations (per metric, not per sample size)
    correlations_df = helpers.compute_correlations_from_long_format(metrics_df)
    if include_metrics is not None:
        correlations_df = correlations_df[correlations_df["Metric"].isin(metric_list)]

    # Merge correlations - they're constant per metric, so broadcast across all Sample Size values
    analysis_df = analysis_df.merge(
        correlations_df[
            ["Metric", "Pearson Rho", "Pearson P", "Spearman Rho", "Spearman P"]
        ],
        on="Metric",
        how="left",
    )

    # 4. Compute plateau detection (based on mean values, per metric)
    # Convert to wide format for plateau detection
    mean_values_wide = analysis_df.pivot_table(
        index="Sample Size",
        columns="Metric",
        values="Mean Value",
    ).reset_index()
    mean_values_wide = mean_values_wide.rename(columns={"Sample Size": "sample_size"})
    mean_values_wide["sample_id"] = 0

    plateau_df = helpers.detect_plateau_df(
        measures_per_log={"log": mean_values_wide},
        metric_columns=metric_list if include_metrics else None,
        rel_threshold=0.025,  # 2.5% threshold
        agg="mean",
        min_runs=1,
        report="current",
    )

    # Extract plateau values and merge (constant per metric)
    # plateau_df now has MultiIndex columns: (log_name, 'Plateau n') and (log_name, 'Plateau Found')
    if isinstance(plateau_df.columns, pd.MultiIndex):
        # Extract Plateau n and Plateau Found columns for the "log" log name
        if ("log", "Plateau n") in plateau_df.columns:
            # Extract the Series and create a clean DataFrame
            plateau_n_series = plateau_df[("log", "Plateau n")]
            plateau_n_df = pd.DataFrame(
                {"Metric": plateau_n_series.index, "Plateau n": plateau_n_series.values}
            )
            analysis_df = analysis_df.merge(
                plateau_n_df,
                on="Metric",
                how="left",
            )
        else:
            analysis_df["Plateau n"] = None

        if ("log", "Plateau Found") in plateau_df.columns:
            # Extract the Series and create a clean DataFrame
            plateau_found_series = plateau_df[("log", "Plateau Found")]
            plateau_found_df = pd.DataFrame(
                {
                    "Metric": plateau_found_series.index,
                    "Plateau Found": plateau_found_series.values,
                }
            )
            analysis_df = analysis_df.merge(
                plateau_found_df,
                on="Metric",
                how="left",
            )
        else:
            analysis_df["Plateau Found"] = False
    else:
        # Fallback for old format (shouldn't happen with new code)
        analysis_df["Plateau n"] = None
        analysis_df["Plateau Found"] = False

    # Set index to Metric and Sample Size
    analysis_df = analysis_df.set_index(["Metric", "Sample Size"]).sort_index()

    return analysis_df
