#!/usr/bin/env python
# type: ignore
# pylint: disable=all
# flake8: noqa
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

from utils import constants, helpers
from utils.bootstrapping.bootstrap_samplers.bootstrap_sampler import BootstrapSampler
from utils.comparison_table import build_comparison_table_if_exists
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import (
    VidgofMetricsAdapter,
)
from utils.master_table import build_and_save_master_csv, combine_analysis_with_means
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS
from utils.pipeline.compute import (
    compute_analysis_for_metrics,
    compute_metrics_for_samples,
)
from utils.plotting.plot_cis import plot_bootstrap_cis, plot_empirical_cis
from utils.plotting.plot_correlations import plot_all_correlation_results

# plot_inext_curves module removed - no longer needed
from utils.population.extractors.chao1_population_extractor import (
    Chao1PopulationExtractor,
)
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
)
from utils.sample_confidence_interval_extractor import SampleConfidenceIntervalExtractor
from utils.windowing.window import Window

from .yaml_input_handling import (
    ExperimentSettings,
    build_scenarios_registry,
    experiment_settings_from_profile,
    load_experiment_settings,
    load_scenarios_yaml,
)

# --- defaults ---
SORTED_METRICS = constants.ALL_METRIC_NAMES  # constants.PC_METRICS

PLOT_GRID = [
    # ----- LENGTH -----
    [["Number of Events"], ["Number of Traces"], []],
    # ----- SIZE -----
    [
        ["Number of Distinct Activities"],
        ["Number of Distinct Traces"],
        ["Number of Distinct Directly-Follows Relations"],
    ],
    [["Min. Trace Length"], ["Avg. Trace Length"], ["Max. Trace Length"]],
    # ----- VARIATION -----
    [
        ["Percentage of Distinct Traces"],
        ["Average Distinct Activities per Trace"],
        ["Structure"],
    ],
    [
        ["Estimated Number of Acyclic Paths"],
        ["Number of Ties in Paths to Goal"],
        ["Lempel-Ziv Complexity"],
    ],
    # ----- DISTANCE -----
    [["Average Affinity"], ["Deviation from Random"], ["Average Edit Distance"]],
    # ----- GRAPH ENTROPY -----
    [["Sequence Entropy"], ["Normalized Sequence Entropy"], []],
    [["Variant Entropy"], ["Normalized Variant Entropy"], []],
]

PLOT_ROW_GROUPS = [
    "Length",
    "Size",
    "Size",
    "Variation",
    "Variation",
    "Distance",
    "Graph Entropy",
    "Graph Entropy",
]


_EXPERIMENT_YAML = load_experiment_settings()

default_population_extractor = NaivePopulationExtractor()
chao1_population_extractor = Chao1PopulationExtractor()
default_metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
default_normalizers: Optional[List] = None
default_sample_confidence_interval_extractor = SampleConfidenceIntervalExtractor(
    conf_level=0.95
)

SCENARIOS = build_scenarios_registry(
    load_scenarios_yaml(),
    default_sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
    default_metric_adapters=default_metric_adapters,
    naive_population_extractor=default_population_extractor,
    chao1_population_extractor=chao1_population_extractor,
    default_normalizers=DEFAULT_NORMALIZERS,
)


# Metrics used for the log-level statistics CSV
METRICS_FOR_LOG_STATISTICS = [
    "Number of Events",
    "Number of Traces",
    "Number of Distinct Activities",
    "Number of Distinct Traces",
    "Number of Distinct Directly-Follows Relations",
    "Avg. Trace Length",
]


def _sample_consecutive_windows_by_size_and_sample_id(
    pm4py_log: Any,
    size_to_sample_ids: Dict[int, List[str]],
    random_state: int,
) -> List[tuple[int, str, Window]]:
    """
    Deterministically sample consecutive windows for specific (size, sample_id) pairs.

    Sampling seed is derived from (random_state, size, sample_id), so recomputing only
    missing sample IDs yields the exact same windows as full recomputation.
    """
    traces = list(pm4py_log)
    n_traces = len(traces)
    if n_traces == 0:
        return []

    out: List[tuple[int, str, Window]] = []
    base_seed = int(random_state)
    for size in sorted(size_to_sample_ids.keys()):
        s = int(size)
        if s <= 0:
            continue
        if s > n_traces:
            continue
        for sample_id in sorted(size_to_sample_ids[s], key=lambda x: int(x)):
            sid_i = int(sample_id)
            seed_seq = np.random.SeedSequence([base_seed, s, sid_i])
            derived_seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
            rng = np.random.default_rng(derived_seed)
            start_idx = int(rng.integers(low=0, high=n_traces - s + 1))
            chosen_traces = traces[start_idx : start_idx + s]
            out.append(
                (
                    s,
                    str(sample_id),
                    Window(id=str(sample_id), size=s, traces=chosen_traces),
                )
            )
    return out


# --- Helper functions for event log statistics ---
def compute_metrics_for_log_statistics(
    pm4py_log, metric_adapter: LocalMetricsAdapter, population_extractor
) -> Dict[str, float]:
    """
    Compute basic metrics for a full event log.

    Parameters
    ----------
    pm4py_log
        PM4Py event log.
    metric_adapter
        Metrics adapter to compute metrics.
    population_extractor
        Population extractor to apply to window.

    Returns
    -------
    Dict[str, float]
        Dictionary with metric names as keys and values.
    """
    # Create a window from the full log
    window = Window(
        id="full_log",
        size=len(pm4py_log),
        traces=list(pm4py_log),
        population_distributions=None,
    )

    # Apply population extractor
    population_extractor.apply(window)

    # Compute metrics
    store, _ = metric_adapter.compute_measures_for_window(
        window, include_metrics=METRICS_FOR_LOG_STATISTICS
    )

    # Extract values
    result = {}
    for metric_name in METRICS_FOR_LOG_STATISTICS:
        if store.has(metric_name):
            measure = store.get(metric_name)
            result[metric_name] = measure.value
        else:
            result[metric_name] = None

    return result


def build_log_statistics_dataframe(log_data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a DataFrame from collected log statistics.

    Parameters
    ----------
    log_data_list
        List of dictionaries, each containing log statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Type, Event Log, Description, plus metric columns
        as defined in METRICS_FOR_LOG_STATISTICS.
    """
    df = pd.DataFrame(log_data_list)
    # Capitalize type values
    if "type" in df.columns:
        df["type"] = df["type"].str.capitalize()
    # Rename only non-metric columns for output
    column_mapping = {
        "type": "Type",
        "log_name": "Event Log",
        "description": "Description",
    }
    df = df.rename(columns=column_mapping)

    # Reorder columns: Type, Event Log, Description, then metrics in the global list
    column_order: List[str] = ["Type", "Event Log", "Description"]
    # Only include metrics that are present in the DataFrame
    for metric in METRICS_FOR_LOG_STATISTICS:
        if metric in df.columns:
            column_order.append(metric)

    df = df[column_order]

    return df


def _merge_correlation_reliability_plateau(
    corr_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    plateau_median: Dict[str, float],
    plateau_found_majority: Dict[str, bool],
) -> pd.DataFrame:
    """Combine correlation analysis, reliability CIs, and plateau summary for master/plots."""
    c = corr_df.reset_index()
    r = rel_df.reset_index()
    const = (
        c.drop_duplicates(subset=["Metric"])
        .set_index("Metric")[["Pearson Rho", "Pearson P", "Spearman Rho", "Spearman P"]]
    )
    merged = pd.merge(
        c,
        r,
        on=["Metric", "Sample Size"],
        how="outer",
        suffixes=("_corr", "_rel"),
    )
    merged["Mean Value"] = merged["Mean Value_rel"].fillna(merged["Mean Value_corr"])
    merged["Median Value"] = merged["Median Value_rel"].fillna(merged["Median Value_corr"])
    drop_cols = [
        c
        for c in merged.columns
        if c.endswith("_corr") and c not in ("Metric", "Sample Size")
    ]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    interval_columns = [
        "Bootstrap CI Low",
        "Bootstrap CI High",
        "Sample CI Low",
        "Sample CI High",
        "Sample CI Rel Width",
        "Sample Q05",
        "Sample Q95",
        "Empirical CI Rel Width",
    ]
    for col in interval_columns:
        corr_col = f"{col}_corr"
        rel_col = f"{col}_rel"
        if corr_col in merged.columns or rel_col in merged.columns:
            rel_vals = (
                merged[rel_col]
                if rel_col in merged.columns
                else pd.Series(np.nan, index=merged.index)
            )
            corr_vals = (
                merged[corr_col]
                if corr_col in merged.columns
                else pd.Series(np.nan, index=merged.index)
            )
            merged[col] = rel_vals.fillna(corr_vals)
    drop_interval_suffix_cols = [
        c
        for c in merged.columns
        if c.endswith("_corr") or c.endswith("_rel")
    ]
    merged = merged.drop(columns=drop_interval_suffix_cols, errors="ignore")
    for col in ["Pearson Rho", "Pearson P", "Spearman Rho", "Spearman P"]:
        merged[col] = merged["Metric"].map(const[col])
    merged["Plateau n"] = merged["Metric"].map(plateau_median)
    merged["Plateau Found"] = merged["Metric"].map(plateau_found_majority)
    merged = merged.set_index(["Metric", "Sample Size"]).sort_index()
    return merged


# --- core compute function ---
def compute_results(
    scenario_name: str,
    settings: ExperimentSettings,
    scenario: Dict[str, Any],
    *,
    reuse_raw_metrics_if_available: bool = True,
) -> None:
    list_of_logs = scenario["logs"]
    population_extractor = scenario["population_extractor"]
    metric_adapters = scenario["metric_adapters"]
    bootstrap_sampler = scenario["bootstrap_sampler"]
    normalizers = scenario["normalizers"]
    sample_confidence_interval_extractor = scenario.get(
        "sample_confidence_interval_extractor",
        default_sample_confidence_interval_extractor,
    )
    base_scenario_name = scenario["base_scenario_name"]
    print(f"Generating results for {scenario_name}")
    include_metrics = settings.include_metrics
    if bootstrap_sampler is None:
        bootstrap_sampler = BootstrapSampler(
            B=settings.bootstrap_replica_count, seed=settings.random_state
        )
    data_dictionary = helpers.load_data_dictionary(
        constants.get_data_dictionary_path(), get_real=True, get_synthetic=True
    )
    data_dictionary = {
        log: info for log, info in data_dictionary.items() if log in list_of_logs
    }

    scenario_out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name
    scenario_out_dir.mkdir(parents=True, exist_ok=True)
    existing_log_stats_path = scenario_out_dir / "log_statistics.csv"
    log_statistics_by_log: Dict[str, Dict[str, Any]] = {}
    if existing_log_stats_path.exists():
        existing_stats_df = pd.read_csv(existing_log_stats_path)
        for _, row in existing_stats_df.iterrows():
            existing_log_name = str(row.get("Event Log", "")).strip()
            if not existing_log_name:
                continue
            existing_entry: Dict[str, Any] = {
                "type": row.get("Type", ""),
                "log_name": existing_log_name,
                "description": row.get("Description", ""),
            }
            for metric_name in METRICS_FOR_LOG_STATISTICS:
                if metric_name in existing_stats_df.columns:
                    existing_entry[metric_name] = row.get(metric_name)
            log_statistics_by_log[existing_log_name] = existing_entry

    # Store population sizes (number of traces) for FPC
    log_population_sizes: Dict[str, int] = {}
    # Store analysis results per log
    analysis_per_log: Dict[str, pd.DataFrame] = {}

    # Get metric adapter for computing basic log statistics
    basic_metrics_adapter = LocalMetricsAdapter()

    for log_name, dataset_info in data_dictionary.items():
        print(f"Computing for {log_name}")
        existing_stats_for_log = log_statistics_by_log.get(log_name)
        if existing_stats_for_log is not None:
            n_pop_val = existing_stats_for_log.get("Number of Traces")
            if pd.notna(n_pop_val):
                try:
                    log_population_sizes[log_name] = int(float(n_pop_val))
                except (TypeError, ValueError):
                    pass

        out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name / log_name
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_metrics_path = out_dir / "raw_metrics.csv"
        raw_metrics_df: Optional[pd.DataFrame] = None
        pm4py_log = None
        all_window_sizes_f: List[int] = []

        if reuse_raw_metrics_if_available and raw_metrics_path.exists():
            raw_existing = pd.read_csv(raw_metrics_path)
            if {"Metric", "Sample Size", "Sample ID", "Value"}.issubset(
                set(raw_existing.columns)
            ):
                required_sizes = sorted(int(s) for s in settings.window_sizes)
                expected_sample_ids = [str(i) for i in range(settings.samples_per_size)]
                sample_ids_by_size = (
                    raw_existing[["Sample Size", "Sample ID"]]
                    .dropna(subset=["Sample Size", "Sample ID"])
                    .drop_duplicates()
                    .assign(**{"Sample Size": lambda d: d["Sample Size"].astype(int)})
                    .assign(**{"Sample ID": lambda d: d["Sample ID"].astype(str)})
                    .groupby("Sample Size")["Sample ID"]
                    .apply(list)
                )
                missing_by_size_from_cache: Dict[int, List[str]] = {}
                for s in required_sizes:
                    existing_ids = set(sample_ids_by_size.get(s, []))
                    missing_ids = [sid for sid in expected_sample_ids if sid not in existing_ids]
                    if missing_ids:
                        missing_by_size_from_cache[s] = missing_ids

                if not missing_by_size_from_cache:
                    raw_metrics_df = raw_existing.set_index(["Metric", "Sample Size"])
                    print(
                        f"Reusing existing raw metrics without log parsing: {raw_metrics_path}"
                    )
                    cached_sizes = {
                        int(s)
                        for s in raw_existing["Sample Size"].dropna().astype(int).unique()
                    }
                    all_window_sizes_f = [s for s in settings.window_sizes if s in cached_sizes]
                else:
                    print(
                        "Found existing raw metrics but some required (size, sample_id) "
                        "pairs are missing. Computing missing pairs only."
                    )
            else:
                print(
                    "Existing raw metrics file has unexpected schema. "
                    "Recomputing from scratch."
                )

        if raw_metrics_df is None:
            log_path = Path(dataset_info["path"])
            pm4py_log = xes_importer.apply(str(log_path))
            # Store population size (number of traces) for FPC
            log_n = len(pm4py_log)
            log_population_sizes[log_name] = log_n
            all_window_sizes_f = [s for s in settings.window_sizes if s <= log_n]

            # Compute basic log statistics (only when log is loaded)
            basic_metrics = compute_metrics_for_log_statistics(
                pm4py_log, basic_metrics_adapter, population_extractor
            )
            log_statistics_by_log[log_name] = {
                "type": dataset_info["type"],
                "log_name": log_name,  # Use the key (e.g., BPIC12)
                "description": dataset_info["name"],
                **basic_metrics,
            }

            if reuse_raw_metrics_if_available and raw_metrics_path.exists():
                raw_existing = pd.read_csv(raw_metrics_path)
                if {"Metric", "Sample Size", "Sample ID", "Value"}.issubset(
                    set(raw_existing.columns)
                ):
                    required_sizes = sorted(int(s) for s in all_window_sizes_f)
                    expected_sample_ids = [str(i) for i in range(settings.samples_per_size)]
                    sample_ids_by_size = (
                        raw_existing[["Sample Size", "Sample ID"]]
                        .dropna(subset=["Sample Size", "Sample ID"])
                        .drop_duplicates()
                        .assign(**{"Sample Size": lambda d: d["Sample Size"].astype(int)})
                        .assign(**{"Sample ID": lambda d: d["Sample ID"].astype(str)})
                        .groupby("Sample Size")["Sample ID"]
                        .apply(list)
                    )
                    missing_by_size: Dict[int, List[str]] = {}
                    for s in required_sizes:
                        existing_ids = set(sample_ids_by_size.get(s, []))
                        missing_ids = [sid for sid in expected_sample_ids if sid not in existing_ids]
                        if missing_ids:
                            missing_by_size[s] = missing_ids

                    if missing_by_size:
                        window_samples_missing = _sample_consecutive_windows_by_size_and_sample_id(
                            pm4py_log,
                            missing_by_size,
                            settings.random_state,
                        )
                        raw_missing_df = compute_metrics_for_samples(
                            window_samples_missing,
                            population_extractor=population_extractor,
                            metric_adapters=metric_adapters,
                            bootstrap_sampler=bootstrap_sampler,
                            normalizers=normalizers,
                            include_metrics=include_metrics,
                        ).sort_index()
                        raw_metrics_df = pd.concat(
                            [raw_existing, raw_missing_df.reset_index()],
                            ignore_index=True,
                        )
                        raw_metrics_df = raw_metrics_df.drop_duplicates(
                            subset=["Metric", "Sample Size", "Sample ID"],
                            keep="last",
                        )
                        raw_metrics_df = raw_metrics_df.set_index(
                            ["Metric", "Sample Size"]
                        ).sort_index()
                        raw_metrics_df.to_csv(raw_metrics_path)
                    else:
                        raw_metrics_df = raw_existing.set_index(["Metric", "Sample Size"])
                        print(f"Reusing existing raw metrics: {raw_metrics_path}")
                else:
                    full_pairs = {
                        int(s): [str(i) for i in range(settings.samples_per_size)]
                        for s in all_window_sizes_f
                    }
                    window_samples_all = _sample_consecutive_windows_by_size_and_sample_id(
                        pm4py_log,
                        full_pairs,
                        settings.random_state,
                    )
                    raw_metrics_df = compute_metrics_for_samples(
                        window_samples_all,
                        population_extractor=population_extractor,
                        metric_adapters=metric_adapters,
                        bootstrap_sampler=bootstrap_sampler,
                        normalizers=normalizers,
                        include_metrics=include_metrics,
                    ).sort_index()
                    raw_metrics_df.to_csv(raw_metrics_path)
            else:
                full_pairs = {
                    int(s): [str(i) for i in range(settings.samples_per_size)]
                    for s in all_window_sizes_f
                }
                window_samples_all = _sample_consecutive_windows_by_size_and_sample_id(
                    pm4py_log,
                    full_pairs,
                    settings.random_state,
                )
                raw_metrics_df = compute_metrics_for_samples(
                    window_samples_all,
                    population_extractor=population_extractor,
                    metric_adapters=metric_adapters,
                    bootstrap_sampler=bootstrap_sampler,
                    normalizers=normalizers,
                    include_metrics=include_metrics,
                ).sort_index()
                raw_metrics_df.to_csv(raw_metrics_path)

        correlation_sizes_f = [
            s
            for s in all_window_sizes_f
            if settings.correlation_start <= s <= settings.correlation_stop
        ]
        reliability_sizes_f = [s for s in settings.reliability_sizes if s in set(all_window_sizes_f)]

        if raw_metrics_df is None:
            raise RuntimeError(f"raw_metrics_df was not built for log {log_name}")

        # Correlation analysis (rho / Pearson vs window size)
        corr_metrics = raw_metrics_df.reset_index()
        corr_metrics = corr_metrics[
            corr_metrics["Sample Size"].isin(correlation_sizes_f)
        ].set_index(["Metric", "Sample Size"])

        analysis_correlation = compute_analysis_for_metrics(
            corr_metrics,
            sample_confidence_interval_extractor=None,
            include_metrics=include_metrics,
            include_sample_ci=False,
            include_correlations=True,
            include_plateau=False,
        )
        analysis_correlation.to_csv(out_dir / "analysis_correlation.csv")

        # Reliability: relative CIs at reliability window sizes
        rel_metrics = raw_metrics_df.reset_index()
        rel_metrics = rel_metrics[
            rel_metrics["Sample Size"].isin(reliability_sizes_f)
        ].set_index(["Metric", "Sample Size"])

        analysis_reliability = compute_analysis_for_metrics(
            rel_metrics,
            sample_confidence_interval_extractor=sample_confidence_interval_extractor,
            include_metrics=include_metrics,
            include_sample_ci=True,
            include_correlations=False,
            include_plateau=False,
        )
        analysis_reliability.to_csv(out_dir / "analysis_reliability.csv")

        plateau_analysis = compute_analysis_for_metrics(
            raw_metrics_df,
            sample_confidence_interval_extractor=None,
            include_metrics=include_metrics,
            include_sample_ci=False,
            include_correlations=False,
            include_plateau=True,
            plateau_windows_per_test=settings.plateau_windows_per_test,
            plateau_step_between_tests=settings.plateau_step_between_tests,
            plateau_alpha=settings.plateau_alpha,
            plateau_number_consecutive_non_trending_tests=settings.plateau_number_consecutive_non_trending_tests,
            plateau_start_window_size=min(all_window_sizes_f) if all_window_sizes_f else None,
        )
        plateau_summary = (
            plateau_analysis.reset_index()[
                [
                    "Metric",
                    "Plateau n",
                    "Plateau Found",
                    "MK p-value",
                    "MK alpha",
                    "MK tail window sizes",
                    "MK tail means",
                ]
            ]
            .drop_duplicates(subset=["Metric"])
            .copy()
        )
        plateau_summary["Plateau Classification"] = plateau_summary["Plateau Found"].map(
            lambda x: "plateauing" if bool(x) else "trending"
        )
        plateau_summary = plateau_summary[
            [
                "Metric",
                "Plateau n",
                "Plateau Found",
                "Plateau Classification",
                "MK p-value",
                "MK alpha",
                "MK tail window sizes",
                "MK tail means",
            ]
        ]
        plateau_summary.to_csv(out_dir / "plateau_summary.csv", index=False)
        plateau_med = {
            str(r["Metric"]): (float(r["Plateau n"]) if pd.notna(r["Plateau n"]) else float("nan"))
            for _, r in plateau_summary.iterrows()
        }
        plateau_maj = {
            str(r["Metric"]): bool(r["Plateau Found"])
            for _, r in plateau_summary.iterrows()
        }

        # Merged table for master + correlation plots
        analysis_df = _merge_correlation_reliability_plateau(
            analysis_correlation,
            analysis_reliability,
            plateau_med,
            plateau_maj,
        )
        analysis_df.to_csv(out_dir / "analysis.csv")

        ci_plot_analysis = compute_analysis_for_metrics(
            raw_metrics_df,
            sample_confidence_interval_extractor=sample_confidence_interval_extractor,
            include_metrics=include_metrics,
            include_sample_ci=True,
            include_correlations=False,
            include_plateau=False,
        )
        ci_plot_df = ci_plot_analysis.reset_index()

        # Remove legacy filenames so outputs reflect the current naming scheme.
        for legacy_name in ("sample_cis.png", "sample_cis.pdf"):
            legacy_path = out_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

        plot_bootstrap_cis(
            analysis_df=ci_plot_df,
            plot_grid=PLOT_GRID,
            plot_row_groups=PLOT_ROW_GROUPS,
            out_dir=out_dir,
        )
        plot_empirical_cis(
            analysis_df=ci_plot_df,
            plot_grid=PLOT_GRID,
            plot_row_groups=PLOT_ROW_GROUPS,
            out_dir=out_dir,
        )

        analysis_per_log[log_name] = analysis_df

    out_dir = scenario_out_dir

    # Compute sample size for FPC (correlation interval over configured window grid)
    correlation_window_count = len(
        [
            s
            for s in settings.window_sizes
            if settings.correlation_start <= s <= settings.correlation_stop
        ]
    )
    n_samples = correlation_window_count * settings.samples_per_size
    avg_population_size = (
        int(np.mean(list(log_population_sizes.values())))
        if log_population_sizes
        else None
    )

    # Combine all analysis data and add mean rows
    combined_analysis_df = combine_analysis_with_means(
        analysis_per_log=analysis_per_log,
        ref_sizes=settings.reliability_sizes,
        measure_basis_map=constants.METRIC_BASIS_MAP,
        n=n_samples,
        N_pop=avg_population_size,
        metric_columns=include_metrics,
    )

    # Create correlation plots
    plot_all_correlation_results(
        combined_analysis_df=combined_analysis_df,
        out_dir=out_dir,
    )

    # Build the master table (CSV-first, scenario-agnostic)
    master_csv_path = str(out_dir / "master.csv")
    csv_path = build_and_save_master_csv(
        combined_analysis_df=combined_analysis_df,
        out_csv_path=master_csv_path,
    )
    print(f"Master table saved to: {csv_path}")

    # 3) Build comparison table if base scenario exists
    if base_scenario_name is not None:
        before_master_csv_path = str(
            constants.BIAS_STUDY_RESULTS_DIR / base_scenario_name / "master.csv"
        )
        # Save comparison table in the current scenario folder (same as master.csv)
        comparison_csv_path = str(out_dir / "metrics_comparison.csv")

        build_comparison_table_if_exists(
            before_csv_path=before_master_csv_path,
            after_csv_path=csv_path,
            out_csv_path=comparison_csv_path,
        )

    if log_statistics_by_log:
        log_stats_df = build_log_statistics_dataframe(list(log_statistics_by_log.values()))
        log_stats_csv_path = out_dir / "log_statistics.csv"
        log_stats_df.to_csv(log_stats_csv_path, index=False)
        print(f"Log statistics table saved to: {log_stats_csv_path}")


def main() -> None:
    all_scenario_names = list(SCENARIOS.keys())
    non_test_scenario_names = [n for n in all_scenario_names if n != "test"]
    parser = argparse.ArgumentParser(
        description=(
            "Bias study: run scenarios from scenarios.yaml. "
            "Use --test for the smoke 'test' scenario and experiment_settings profile 'test'. "
            "Otherwise use profile 'full' and run every scenario except 'test', "
            "or pass --scenarios to choose by name."
        )
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only the 'test' scenario with experiment_settings profile 'test'.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Scenario names to run (profile 'full' only), e.g. real_base synthetic_base. "
            "Default: all scenarios except 'test'. Cannot be used with --test."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Override experiment_settings include_metrics. Shorthand: "
            f"{list(constants.METRIC_SHORTHAND.keys())} or full names."
        ),
    )
    parser.add_argument(
        "--recompute-raw-metrics",
        action="store_true",
        help=(
            "Recompute window metrics even if raw_metrics.csv already exists. "
            "By default, existing raw metrics are reused."
        ),
    )
    args = parser.parse_args()

    if args.test and args.scenarios is not None:
        raise SystemExit("Cannot use --scenarios together with --test (use --test alone).")

    cli_include_metrics: Optional[List[str]] = None
    if args.metrics is not None:
        try:
            cli_include_metrics = helpers.resolve_metric_names(args.metrics)
        except ValueError as e:
            raise SystemExit(str(e))

    if args.test:
        experiment_settings = experiment_settings_from_profile(_EXPERIMENT_YAML["test"])
        if cli_include_metrics is not None:
            experiment_settings = replace(
                experiment_settings, include_metrics=cli_include_metrics
            )
        test_scenario = SCENARIOS["test"].copy()
        scenarios_to_run = [("test", test_scenario, experiment_settings)]
    else:
        scenarios_to_run = []
        full_settings = experiment_settings_from_profile(_EXPERIMENT_YAML["full"])
        if cli_include_metrics is not None:
            full_settings = replace(full_settings, include_metrics=cli_include_metrics)

        if args.scenarios is None:
            # run all scenarios except "test"
            for scenario_name in non_test_scenario_names:
                scenario_config = SCENARIOS[scenario_name].copy()
                scenarios_to_run.append((scenario_name, scenario_config, full_settings))
        else:
            if not args.scenarios:
                raise SystemExit(
                    "--scenarios requires at least one scenario name (or omit --scenarios to run all except 'test')."
                )
            for name in args.scenarios:
                if name == "test":
                    raise SystemExit(
                        "Scenario 'test' is only run with --test (smoke settings). "
                        "Omit it from --scenarios or use --test."
                    )
                if name not in SCENARIOS:
                    raise SystemExit(
                        f"Unknown scenario {name!r}. Valid names: {all_scenario_names}"
                    )
                scenario_config = SCENARIOS[name].copy()
                scenarios_to_run.append((name, scenario_config, full_settings))

    for scenario_name, sc, experiment_settings in scenarios_to_run:
        print(f"\n=== Running scenario: {scenario_name} ===")
        compute_results(
            scenario_name=scenario_name,
            settings=experiment_settings,
            scenario=sc,
            reuse_raw_metrics_if_available=not args.recompute_raw_metrics,
        )


if __name__ == "__main__":
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    import multiprocessing as mp

    mp.freeze_support()  # harmless on Linux; required on Windows
    main()
