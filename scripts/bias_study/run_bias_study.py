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
from scipy.stats import kendalltau

from utils import constants, helpers, sampling_helper
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
    plateau_test_start_sizes_for_log,
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


def _metrics_df_long(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.reset_index()
    if "Sample Size" not in df.columns and "Sample Size" in df.index.names:
        df = df.reset_index()
    return df


def _mean_at_window(df: pd.DataFrame, metric: str, window_size: int) -> Optional[float]:
    """Mean ``Value`` across samples for one metric at ``window_size``; None if missing."""
    sub = df[(df["Metric"] == metric) & (df["Sample Size"] == window_size)]
    if sub.empty:
        return None
    v = float(sub["Value"].mean())
    return v if np.isfinite(v) else None


def _plateau_summary_by_metric(
    include_metrics: List[str],
    plateau_n_by_metric: Dict[str, float],
    plateau_p_by_metric: Dict[str, float],
    plateau_alpha: float,
    tail_window_sizes_by_metric: Dict[str, List[int]],
    tail_means_by_metric: Dict[str, List[float]],
) -> tuple[pd.DataFrame, Dict[str, float], Dict[str, bool]]:
    """
    One row per metric summarizing plateau detection on mean metric values across samples
    at each window size (not per-sample trajectories).

    Plateau logic (Pass C in ``compute_results``): for rolling tails of window sizes,
    Kendall's tau is computed between the tail index and the sequence of tail means. A tail
    is non-trending when the p-value is above ``plateau_alpha`` (no significant monotonic
    trend), or when all tail means are constant (Kendall p-value undefined; treated as
    non-trending). After ``plateau_number_consecutive_non_trending_tests`` consecutive
    non-trending tails, ``Plateau n`` is the first window size in that streak and
    ``Plateau Found`` is True. Otherwise ``Plateau Found`` is False and ``Plateau n`` is NaN.
    """
    rows = []
    plateau_med: Dict[str, float] = {}
    plateau_maj: Dict[str, bool] = {}
    for m in include_metrics:
        pn = plateau_n_by_metric.get(m, np.nan)
        if not np.isfinite(pn):
            pn = np.nan
        found = bool(np.isfinite(pn))
        p_val = plateau_p_by_metric.get(m, np.nan)
        rows.append(
            {
                "Metric": m,
                "Plateau n": pn,
                "Plateau Found": found,
                "Plateau Classification": "plateauing" if found else "trending",
                "MK p-value": p_val,
                "MK alpha": plateau_alpha,
                "MK tail window sizes": ",".join(
                    str(x) for x in tail_window_sizes_by_metric.get(m, [])
                ),
                "MK tail means": ",".join(
                    f"{x:.10g}" for x in tail_means_by_metric.get(m, [])
                ),
            },
        )
        plateau_med[m] = float(pn) if found else float("nan")
        plateau_maj[m] = found
    return pd.DataFrame(rows), plateau_med, plateau_maj


# --- core compute function ---
def compute_results(
    scenario_name: str,
    settings: ExperimentSettings,
    scenario: Dict[str, Any],
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

    # Store population sizes (number of traces) for FPC
    log_population_sizes: Dict[str, int] = {}
    # Store analysis results per log
    analysis_per_log: Dict[str, pd.DataFrame] = {}
    # Store log statistics for summary table
    log_statistics: List[Dict[str, any]] = []

    # Get metric adapter for computing basic log statistics
    basic_metrics_adapter = LocalMetricsAdapter()

    for log_name, dataset_info in data_dictionary.items():
        print(f"Computing for {log_name}")
        log_path = Path(dataset_info["path"])
        pm4py_log = xes_importer.apply(str(log_path))
        # Store population size (number of traces) for FPC
        log_n = len(pm4py_log)
        log_population_sizes[log_name] = log_n
        # Correlation / reliability: only cap by log length (feasible window sizes).
        # Plateau test starts and per-test windows are capped by log length.
        correlation_sizes_f = [s for s in settings.correlation_sizes if s <= log_n]
        reliability_sizes_f = [s for s in settings.reliability_sizes if s <= log_n]

        # Compute basic log statistics
        basic_metrics = compute_metrics_for_log_statistics(
            pm4py_log, basic_metrics_adapter, population_extractor
        )
        log_statistics.append(
            {
                "type": dataset_info["type"],
                "log_name": log_name,  # Use the key (e.g., BPIC12)
                "description": dataset_info["name"],
                **basic_metrics,
            }
        )

        out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name / log_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pass A: correlation window sizes (e.g., 50–500, capped by log)
        window_samples_base = (
            sampling_helper.sample_consecutive_trace_windows_with_replacement(
                pm4py_log,
                correlation_sizes_f,
                settings.samples_per_size,
                settings.random_state,
            )
        )
        metrics_base = compute_metrics_for_samples(
            window_samples_base,
            population_extractor=population_extractor,
            metric_adapters=metric_adapters,
            bootstrap_sampler=bootstrap_sampler,
            normalizers=normalizers,
            include_metrics=include_metrics,
        )

        # Pass B: reliability-only sizes not in correlation grid (e.g., 1000)
        extra_rel_sizes = [s for s in reliability_sizes_f if s not in set(correlation_sizes_f)]
        metrics_extra_list: list[pd.DataFrame] = [metrics_base]
        if extra_rel_sizes:
            window_samples_rel = (
                sampling_helper.sample_consecutive_trace_windows_with_replacement(
                    pm4py_log,
                    extra_rel_sizes,
                    settings.samples_per_size,
                    settings.random_state,
                )
            )
            metrics_rel_only = compute_metrics_for_samples(
                window_samples_rel,
                population_extractor=population_extractor,
                metric_adapters=metric_adapters,
                bootstrap_sampler=bootstrap_sampler,
                normalizers=normalizers,
                include_metrics=include_metrics,
            )
            metrics_extra_list.append(metrics_rel_only)

        raw_metrics_df = pd.concat(metrics_extra_list, axis=0)
        raw_metrics_df = raw_metrics_df.sort_index()

        # --- Pass C: plateau (rolling MK test on tail means; no bootstrap) ---
        # Old criterion: relative change vs previous window.
        # New criterion: Mann-Kendall on rolling tails (N mean readings, shift by one step).
        plateau_df = _metrics_df_long(raw_metrics_df)
        test_start_sizes = plateau_test_start_sizes_for_log(settings, log_n)
        mk_tail_n = max(1, int(settings.plateau_windows_per_test_count))
        mk_tail_step = max(1, int(settings.plateau_windows_per_test_step))
        mk_alpha = float(settings.plateau_alpha)
        mk_required_consecutive = max(
            1, int(settings.plateau_number_consecutive_non_trending_tests)
        )
        plateau_by_metric: Dict[str, float] = {}
        plateau_p_by_metric: Dict[str, float] = {}
        tail_window_sizes_by_metric: Dict[str, List[int]] = {}
        tail_means_by_metric: Dict[str, List[float]] = {}
        plateau_extra_parts: list[pd.DataFrame] = []
        pending_metrics = set(include_metrics)
        non_trending_streak: Dict[str, int] = {m: 0 for m in include_metrics}
        non_trending_streak_start: Dict[str, float] = {}
        for test_start in test_start_sizes:
            if not pending_metrics:
                break
            tail_window_sizes = [
                int(test_start + i * mk_tail_step) for i in range(mk_tail_n)
            ]
            tail_window_sizes = [w for w in tail_window_sizes if w <= log_n]
            # Strict mode: evaluate only complete tests with the full configured window count.
            if len(tail_window_sizes) < mk_tail_n:
                continue

            missing_windows = [
                pw
                for pw in tail_window_sizes
                if any(_mean_at_window(plateau_df, m, pw) is None for m in pending_metrics)
            ]
            if missing_windows:
                plateau_window_samples = (
                    sampling_helper.sample_consecutive_trace_windows_with_replacement(
                        pm4py_log,
                        missing_windows,
                        settings.samples_per_size,
                        settings.random_state,
                    )
                )
                # Compute all missing windows for this test in one call
                # so the existing window-level parallelization can fan out.
                plateau_batch_df = compute_metrics_for_samples(
                    plateau_window_samples,
                    population_extractor=population_extractor,
                    metric_adapters=metric_adapters,
                    bootstrap_sampler=None,
                    normalizers=normalizers,
                    include_metrics=list(pending_metrics),
                )
                plateau_extra_parts.append(plateau_batch_df)
                plateau_dfb = plateau_batch_df.reset_index()
                if (
                    "Sample Size" not in plateau_dfb.columns
                    and "Sample Size" in plateau_dfb.index.names
                ):
                    plateau_dfb = plateau_dfb.reset_index()
                plateau_df = pd.concat([plateau_df, plateau_dfb], ignore_index=True)

            for m in list(pending_metrics):
                ys: list[float] = []
                xs: list[int] = []
                for pw in tail_window_sizes:
                    mu = _mean_at_window(plateau_df, m, pw)
                    if mu is None or not np.isfinite(mu):
                        continue
                    xs.append(int(pw))
                    ys.append(float(mu))
                tail_window_sizes_by_metric[m] = xs
                tail_means_by_metric[m] = ys
                if len(ys) < 2:
                    plateau_p_by_metric[m] = float("nan")
                    continue
                # Constant tail means: Kendall tau p-value is undefined (often nan); treat as no trend.
                if float(np.min(ys)) == float(np.max(ys)):
                    p_value_f = 1.0
                else:
                    _, p_value = kendalltau(range(len(ys)), ys)
                    p_value_f = (
                        float(p_value)
                        if p_value is not None and np.isfinite(p_value)
                        else float("nan")
                    )
                plateau_p_by_metric[m] = p_value_f
                if np.isfinite(p_value_f) and p_value_f > mk_alpha:
                    if non_trending_streak[m] == 0:
                        non_trending_streak_start[m] = float(xs[0])
                    non_trending_streak[m] += 1
                    if non_trending_streak[m] >= mk_required_consecutive:
                        plateau_by_metric[m] = non_trending_streak_start[m]
                        pending_metrics.discard(m)
                else:
                    non_trending_streak[m] = 0
                    non_trending_streak_start.pop(m, None)

        plateau_summary, plateau_med, plateau_maj = _plateau_summary_by_metric(
            include_metrics,
            plateau_by_metric,
            plateau_p_by_metric,
            mk_alpha,
            tail_window_sizes_by_metric,
            tail_means_by_metric,
        )
        plateau_extra = (
            pd.concat(plateau_extra_parts, ignore_index=False)
            if plateau_extra_parts
            else pd.DataFrame()
        )
        if not plateau_extra.empty:
            raw_metrics_df = pd.concat([raw_metrics_df, plateau_extra], axis=0)
            raw_metrics_df = raw_metrics_df.sort_index()

        raw_metrics_df.to_csv(out_dir / "raw_metrics.csv")
        plateau_summary.to_csv(out_dir / "plateau_summary.csv", index=False)

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

        # Merged table for master + correlation plots
        analysis_df = _merge_correlation_reliability_plateau(
            analysis_correlation,
            analysis_reliability,
            plateau_med,
            plateau_maj,
        )
        analysis_df.to_csv(out_dir / "analysis.csv")

        ci_plot_df = analysis_reliability.reset_index()
        ci_plot_df = ci_plot_df[ci_plot_df["Sample Size"].isin(correlation_sizes_f)]

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

    out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute sample size for FPC (correlation grid)
    n_samples = len(list(settings.correlation_sizes)) * settings.samples_per_size
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

    if log_statistics:
        log_stats_df = build_log_statistics_dataframe(log_statistics)
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
        )


if __name__ == "__main__":
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    import multiprocessing as mp

    mp.freeze_support()  # harmless on Linux; required on Windows
    main()
