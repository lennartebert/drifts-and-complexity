#!/usr/bin/env python
# type: ignore
# pylint: disable=all
# flake8: noqa
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

from utils import constants, helpers, sampling_helper
from utils.bootstrapping.bootstrap_samplers.bootstrap_sampler import BootstrapSampler
from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import (
    INextBootstrapSampler,
)
from utils.comparison_table import build_comparison_table_if_exists
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import (
    VidgofMetricsAdapter,
)
from utils.latex_table_generation import generate_all_latex_tables
from utils.master_table import build_and_save_master_csv, combine_analysis_with_means
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS
from utils.pipeline.compute import (
    compute_analysis_for_metrics,
    compute_metrics_for_samples,
)
from utils.plotting.plot_cis import plot_sample_cis
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

# --- defaults (same as before) ---
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
# --- Analysis-specific globals (see plan: correlation / plateau / reliability) ---
SAMPLES_PER_SIZE = 100
RANDOM_STATE = 123
# Bootstrap replicate count (B) for window-level bootstrap CIs
# TODO: restore to 200 after validation; temporarily 2 for faster local runs
BOOTSTRAP_REPLICA_COUNT = 2

# 1) Correlation analysis: rho vs window size (50–500)
CORRELATION_SIZES = range(50, 501, 50)

# 2) Plateau analysis: consecutive relative change vs previous size (per sample_id)
PLATEAU_MIN = 50
PLATEAU_MAX_CAP = 10000
PLATEAU_STEP = 50
PLATEAU_THRESHOLD = 0.025

# 3) Reliability: across-sample relative CIs at selected sizes
RELIABILITY_SIZES = [50, 500, 1000]

REF_SIZES = [50, 500, 1000]

# Back-compat alias for bootstrap sampler construction
BOOTSTRAP_SIZE = BOOTSTRAP_REPLICA_COUNT

BREAKDOWN_BY = "dimension"  # None, "basis", or "dimension"

CORRELATION_TYPE = (
    "Spearman"  # Correlation type to use in LaTeX tables ("Pearson" or "Spearman")
)

default_population_extractor = NaivePopulationExtractor()
default_metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
default_bootstrap_sampler = BootstrapSampler(
    B=BOOTSTRAP_REPLICA_COUNT, seed=RANDOM_STATE
)
default_normalizers: Optional[List] = None
default_sample_confidence_interval_extractor = SampleConfidenceIntervalExtractor(
    conf_level=0.95
)


# Metrics used for the log-level statistics (CSV + LaTeX table)
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


def generate_latex_log_statistics_table(
    df: pd.DataFrame,
    caption: str = "Event Logs Used in this Study",
    label: str = "tab:list_event_logs",
) -> str:
    """
    Generate LaTeX table from log statistics DataFrame.

    Parameters
    ----------
    df
        DataFrame with log statistics.
    caption
        Table caption.
    label
        Table label.

    Returns
    -------
    str
        LaTeX table code.
    """
    # Determine LaTeX column specification dynamically:
    #  - 3 left-aligned columns (Type, Event Log, Description)
    #  - one centered column per metric
    num_metrics = len(METRICS_FOR_LOG_STATISTICS)
    colspec = "l@{}ll" + "c" * num_metrics + "@{}"

    # Header row: use METRIC_NAMES_TO_LATEX_MAP for metric display names
    metric_headers = [
        constants.METRIC_NAMES_TO_LATEX_MAP.get(metric, metric)
        for metric in METRICS_FOR_LOG_STATISTICS
    ]
    header_metrics_part = " & ".join(metric_headers)

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt} % reduce column padding",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        f" Type&Event Log   &Description&  {header_metrics_part}\\\\",
        "",
    ]

    # Sort rows by type and event log
    df_sorted = df.sort_values(by=["Type", "Event Log"])

    current_type: Optional[str] = None
    for _, row in df_sorted.iterrows():
        row_type = row["Type"]
        log_name = row["Event Log"]
        description = row["Description"]

        # Escape underscores in log names for LaTeX
        if isinstance(log_name, str):
            escaped_log_name = log_name.replace("_", "\\_")
        else:
            escaped_log_name = str(log_name)

        # Add type label if it changed
        if row_type != current_type:
            if current_type is not None:
                lines.append("")
                lines.append("\\midrule")
            current_type = row_type
            type_label = row_type.capitalize()
        else:
            type_label = ""

        # Build metric value cells in the order of METRICS_FOR_LOG_STATISTICS
        metric_cells: List[str] = []
        for metric in METRICS_FOR_LOG_STATISTICS:
            if metric not in row or pd.isna(row[metric]):
                metric_cells.append("")
                continue

            value = row[metric]
            # Heuristic: average-like metrics as floats, others as integers with thousands separator
            if (
                isinstance(value, (int, np.integer))
                and "Avg." not in metric
                and "Average" not in metric
            ):
                metric_cells.append(f"{int(value):,}")
            elif isinstance(value, (float, np.floating)) and (
                "Avg." in metric or "Average" in metric
            ):
                metric_cells.append(f"{float(value):.2f}")
            else:
                # Fallback: string representation
                metric_cells.append(str(value))

        metrics_part = " & ".join(metric_cells)
        line = f" {type_label}&{escaped_log_name}& {description}&  {metrics_part}\\\\"
        lines.append(line)

    lines.append("")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _long_metrics_to_value_map(
    metrics_df: pd.DataFrame,
) -> Dict[tuple[str, str], Dict[int, float]]:
    """Map (sample_id_str, metric) -> {window_size -> value}."""
    df = metrics_df.reset_index()
    out: Dict[tuple[str, str], Dict[int, float]] = {}
    for _, row in df.iterrows():
        sid = str(row["Sample ID"])
        m = str(row["Metric"])
        sz = int(row["Sample Size"])
        out.setdefault((sid, m), {})[sz] = float(row["Value"])
    return out


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
    rename_ci = {
        f"{k}_rel": k
        for k in ["Sample CI Low", "Sample CI High", "Sample CI Rel Width"]
        if f"{k}_rel" in merged.columns
    }
    merged = merged.rename(columns=rename_ci)
    for col in ["Pearson Rho", "Pearson P", "Spearman Rho", "Spearman P"]:
        merged[col] = merged["Metric"].map(const[col])
    merged["Plateau n"] = merged["Metric"].map(plateau_median)
    merged["Plateau Found"] = merged["Metric"].map(plateau_found_majority)
    merged = merged.set_index(["Metric", "Sample Size"]).sort_index()
    return merged


def _adaptive_plateau_extension(
    pm4py_log,
    base_metrics_df: pd.DataFrame,
    *,
    include_metrics: List[str],
    samples_per_size: int,
    max_win: int,
    step: int,
    rel_threshold: float,
    population_extractor,
    metric_adapters,
    bootstrap_sampler,
    normalizers,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, bool]]:
    """
    Extend beyond 550 only when needed. Returns (extra_metrics_df, plateau_per_sample_df,
    plateau_median_by_metric, plateau_found_majority_by_metric).
    """
    maps = _long_metrics_to_value_map(base_metrics_df)
    pending: set[tuple[str, str]] = set()
    plateau_n: Dict[tuple[str, str], float] = {}

    for sid, m in list(maps.keys()):
        if m not in include_metrics:
            continue
        sm = maps[(sid, m)]
        pn, ok = helpers.consecutive_plateau_first_size(
            sm,
            step=step,
            rel_threshold=rel_threshold,
            max_win=min(500, max_win),
        )
        if ok:
            plateau_n[(sid, m)] = pn
        elif max_win > 500:
            pending.add((sid, m))

    extra_parts: list[pd.DataFrame] = []
    if max_win <= 500 or not pending:
        plateau_per_sample = _plateau_summary_records(
            include_metrics, samples_per_size, plateau_n, maps
        )
        med, maj = _plateau_aggregate(plateau_per_sample, samples_per_size)
        return pd.DataFrame(), plateau_per_sample, med, maj

    for curr_size in range(500 + step, max_win + 1, step):
        need_sids = sorted({sid for (sid, _m) in pending})
        if not need_sids:
            break
        window_samples = (
            sampling_helper.sample_consecutive_trace_windows_with_replacement(
                pm4py_log, [curr_size], samples_per_size, random_state
            )
        )
        batch_df = compute_metrics_for_samples(
            window_samples,
            population_extractor=population_extractor,
            metric_adapters=metric_adapters,
            bootstrap_sampler=bootstrap_sampler,
            normalizers=normalizers,
            include_metrics=include_metrics,
        )
        extra_parts.append(batch_df)
        dfb = batch_df.reset_index()
        for _, row in dfb.iterrows():
            sid = str(row["Sample ID"])
            m = str(row["Metric"])
            sz = int(row["Sample Size"])
            if (sid, m) in maps:
                maps[(sid, m)][sz] = float(row["Value"])
            else:
                maps[(sid, m)] = {sz: float(row["Value"])}

        for sid, m in list(pending):
            sm = maps.get((sid, m), {})
            prev = curr_size - step
            if prev not in sm or curr_size not in sm:
                continue
            vp = sm[prev]
            vc = sm[curr_size]
            if not (np.isfinite(vp) and np.isfinite(vc)):
                continue
            if abs(vp) < 1e-15:
                continue
            if abs(vc - vp) / abs(vp) <= rel_threshold:
                plateau_n[(sid, m)] = float(curr_size)
                pending.discard((sid, m))

    plateau_per_sample = _plateau_summary_records(
        include_metrics, samples_per_size, plateau_n, maps
    )
    med, maj = _plateau_aggregate(plateau_per_sample, samples_per_size)
    extra_df = (
        pd.concat(extra_parts, ignore_index=False)
        if extra_parts
        else pd.DataFrame()
    )
    return extra_df, plateau_per_sample, med, maj


def _plateau_summary_records(
    include_metrics: List[str],
    samples_per_size: int,
    plateau_n: Dict[tuple[str, str], float],
    maps: Dict[tuple[str, str], Dict[int, float]],
) -> pd.DataFrame:
    rows = []
    sample_ids = [str(i) for i in range(samples_per_size)]
    for sid in sample_ids:
        for m in include_metrics:
            key = (sid, m)
            pn = plateau_n.get(key, np.nan)
            if not np.isfinite(pn):
                pn = np.nan
            rows.append(
                {
                    "Sample ID": sid,
                    "Metric": m,
                    "Plateau n": pn,
                    "Plateau Found": bool(np.isfinite(pn)),
                }
            )
    return pd.DataFrame(rows)


def _plateau_aggregate(
    plateau_per_sample: pd.DataFrame, samples_per_size: int
) -> tuple[Dict[str, float], Dict[str, bool]]:
    med: Dict[str, float] = {}
    maj: Dict[str, bool] = {}
    for m, grp in plateau_per_sample.groupby("Metric"):
        vals = grp["Plateau n"].to_numpy(dtype=float)
        if not np.any(np.isfinite(vals)):
            med[m] = float("nan")
        else:
            med[m] = float(np.nanmedian(vals))
        frac = float(np.mean(grp["Plateau Found"].to_numpy(dtype=bool)))
        maj[m] = frac >= 0.5
    return med, maj


# --- core compute function ---
def compute_results(
    list_of_logs: List[str],
    results_name: str,
    scenario_name: str,
    clear_name: str,
    population_extractor=default_population_extractor,
    metric_adapters=default_metric_adapters,
    bootstrap_sampler=None,
    normalizers=default_normalizers,
    include_metrics: Optional[List[str]] = None,
    sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
    base_scenario_name: Optional[str] = None,  # type: ignore
) -> None:
    print(f"Generating results for {results_name}")
    if include_metrics is None:
        include_metrics = SORTED_METRICS
    if bootstrap_sampler is None:
        bootstrap_sampler = BootstrapSampler(
            B=BOOTSTRAP_REPLICA_COUNT, seed=RANDOM_STATE
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
        log_population_sizes[log_name] = len(pm4py_log)
        max_win = min(PLATEAU_MAX_CAP, len(pm4py_log))
        correlation_sizes_f = [s for s in CORRELATION_SIZES if s <= max_win]
        reliability_sizes_f = [s for s in RELIABILITY_SIZES if s <= max_win]

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

        # Pass A: correlation window sizes (50–500, capped by log)
        window_samples_base = (
            sampling_helper.sample_consecutive_trace_windows_with_replacement(
                pm4py_log, correlation_sizes_f, SAMPLES_PER_SIZE, RANDOM_STATE
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

        # Pass B: reliability-only sizes not in correlation grid (typically 1000)
        extra_rel_sizes = [s for s in reliability_sizes_f if s not in set(correlation_sizes_f)]
        metrics_extra_list: list[pd.DataFrame] = [metrics_base]
        if extra_rel_sizes:
            window_samples_rel = (
                sampling_helper.sample_consecutive_trace_windows_with_replacement(
                    pm4py_log, extra_rel_sizes, SAMPLES_PER_SIZE, RANDOM_STATE
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

        # Pass C: adaptive plateau beyond 500
        plateau_extra, plateau_per_sample, plateau_med, plateau_maj = (
            _adaptive_plateau_extension(
                pm4py_log,
                metrics_base,
                include_metrics=include_metrics,
                samples_per_size=SAMPLES_PER_SIZE,
                max_win=max_win,
                step=PLATEAU_STEP,
                rel_threshold=PLATEAU_THRESHOLD,
                population_extractor=population_extractor,
                metric_adapters=metric_adapters,
                bootstrap_sampler=bootstrap_sampler,
                normalizers=normalizers,
                random_state=RANDOM_STATE,
            )
        )
        if not plateau_extra.empty:
            raw_metrics_df = pd.concat([raw_metrics_df, plateau_extra], axis=0)
            raw_metrics_df = raw_metrics_df.sort_index()

        raw_metrics_df.to_csv(out_dir / "raw_metrics.csv")
        plateau_per_sample.to_csv(out_dir / "plateau_per_sample.csv", index=False)

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

        # Reliability: relative CIs at REF sizes
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

        plot_sample_cis(
            analysis_df=analysis_reliability.reset_index(),
            plot_grid=PLOT_GRID,
            plot_row_groups=PLOT_ROW_GROUPS,
            out_dir=out_dir,
        )

        analysis_per_log[log_name] = analysis_df

    out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute sample size for FPC (correlation grid)
    n_samples = len(list(CORRELATION_SIZES)) * SAMPLES_PER_SIZE
    avg_population_size = (
        int(np.mean(list(log_population_sizes.values())))
        if log_population_sizes
        else None
    )

    # Combine all analysis data and add mean rows
    combined_analysis_df = combine_analysis_with_means(
        analysis_per_log=analysis_per_log,
        ref_sizes=REF_SIZES,
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

    # 4) Generate LaTeX tables from CSVs
    latex_out_dir = out_dir / "latex"
    comparison_csv_path = (
        str(out_dir / "metrics_comparison.csv")
        if base_scenario_name is not None
        else None
    )

    generate_all_latex_tables(
        master_csv_path=csv_path,
        out_dir=str(latex_out_dir),
        scenario_key=scenario_name,
        scenario_title=clear_name,
        correlation=CORRELATION_TYPE,
        comparison_csv_path=comparison_csv_path,
        breakdown_by=BREAKDOWN_BY,
    )

    # 5) Generate and save log statistics table
    if log_statistics:
        log_stats_df = build_log_statistics_dataframe(log_statistics)
        # Save CSV
        log_stats_csv_path = out_dir / "log_statistics.csv"
        log_stats_df.to_csv(log_stats_csv_path, index=False)
        print(f"Log statistics table saved to: {log_stats_csv_path}")

        # Generate and save LaTeX
        latex_table = generate_latex_log_statistics_table(log_stats_df)
        log_stats_latex_path = latex_out_dir / "log_statistics.tex"
        log_stats_latex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_stats_latex_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"Log statistics LaTeX table saved to: {log_stats_latex_path}")


# --- scenario registry ---
SCENARIOS = {
    "synthetic_base": dict(
        logs=["O2C_S", "CLAIM_S", "LOAN_S", "CREDIT_S"],
        clear_name="Synthetic (Base)",
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=None,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name=None,
    ),
    "synthetic_normalized": dict(
        logs=["O2C_S", "CLAIM_S", "LOAN_S", "CREDIT_S"],
        clear_name="Synthetic (Normalized)",
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        base_scenario_name="synthetic_base",
    ),
    "synthetic_normalized_and_population": dict(
        logs=["O2C_S", "CLAIM_S", "LOAN_S", "CREDIT_S"],
        clear_name="Synthetic (Normalized + Population)",
        population_extractor=Chao1PopulationExtractor(),
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name="synthetic_base",
    ),
    "real_base": dict(
        logs=["BPIC12", "RTFMP"],
        clear_name="Real (Base)",
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=None,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name=None,
    ),
    "real_normalized": dict(
        logs=["BPIC12", "RTFMP"],
        clear_name="Real (Normalized)",
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name="real_base",
    ),
    "real_normalized_and_population": dict(
        logs=["BPIC12", "RTFMP"],
        clear_name="Real (Normalized + Population)",
        population_extractor=Chao1PopulationExtractor(),
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name="real_base",
    ),
    # Full correlation / plateau / reliability grid on TEST_BPIC12 only (use with BOOTSTRAP_REPLICA_COUNT=2 while testing)
    "test_bpic12": dict(
        logs=["TEST_BPIC12"],
        clear_name="TEST BPIC12 (full grid)",
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=None,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name=None,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenarios by ID or name.")
    parser.add_argument(
        "scenarios",
        nargs="*",
        help=f"Scenario IDs [0..{len(SCENARIOS)-1}] or scenario names: {list(SCENARIOS.keys())}. If none provided, runs all scenarios.",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with reduced parameters"
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=f"Metrics to calculate. Use shorthand names: {list(constants.METRIC_SHORTHAND.keys())} or full names. Default: all metrics",
    )
    args = parser.parse_args()

    # Process metrics parameter
    try:
        if args.metrics is None:
            # Use default sorted_metrics
            sorted_selected_metrics = SORTED_METRICS
        else:
            # Resolve shorthand names to full names
            sorted_selected_metrics = helpers.resolve_metric_names(args.metrics)
    except ValueError as e:
        raise SystemExit(str(e))

    # Modify global parameters for test mode
    global SAMPLES_PER_SIZE, BOOTSTRAP_REPLICA_COUNT, BOOTSTRAP_SIZE
    global CORRELATION_SIZES, RELIABILITY_SIZES, PLATEAU_MAX_CAP
    if args.test:
        SAMPLES_PER_SIZE = 2
        BOOTSTRAP_REPLICA_COUNT = 2
        BOOTSTRAP_SIZE = BOOTSTRAP_REPLICA_COUNT
        CORRELATION_SIZES = range(50, 101, 50)
        RELIABILITY_SIZES = [50, 100, 150]
        PLATEAU_MAX_CAP = 300

        # Create test scenario
        test_scenario = dict(
            logs=["TEST_BPIC12"],
            clear_name="Test",
            population_extractor=default_population_extractor,
            metric_adapters=default_metric_adapters,
            bootstrap_sampler=None,
            normalizers=None,
            include_metrics=sorted_selected_metrics,
            sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
            base_scenario_name=None,
        )

        scenarios_to_run = [("test", test_scenario)]
    else:
        scenarios_to_run = []
        scenario_names = list(SCENARIOS.keys())

        # If no scenarios specified, run all
        if not args.scenarios:
            print("No scenarios specified, running all scenarios...")
            for scenario_name, scenario_config in SCENARIOS.items():
                # Add include_metrics to scenario config
                scenario_config = scenario_config.copy()
                scenario_config["include_metrics"] = sorted_selected_metrics
                scenarios_to_run.append((scenario_name, scenario_config))
        else:
            for scenario_input in args.scenarios:
                # Try to parse as integer (scenario ID)
                try:
                    scenario_id = int(scenario_input)
                    if scenario_id < 0 or scenario_id >= len(SCENARIOS):
                        raise SystemExit(
                            f"Invalid scenario_id {scenario_id}. Valid range: 0-{len(SCENARIOS)-1}"
                        )
                    scenario_name = scenario_names[scenario_id]
                    scenario_config = SCENARIOS[scenario_name].copy()
                    scenario_config["include_metrics"] = sorted_selected_metrics
                    scenarios_to_run.append((scenario_name, scenario_config))
                except ValueError:
                    # Not an integer, treat as scenario name
                    if scenario_input not in SCENARIOS:
                        raise SystemExit(
                            f"Invalid scenario name '{scenario_input}'. Valid names: {scenario_names}"
                        )
                    scenario_config = SCENARIOS[scenario_input].copy()
                    scenario_config["include_metrics"] = sorted_selected_metrics
                    scenarios_to_run.append((scenario_input, scenario_config))

    # Run each scenario
    for scenario_name, sc in scenarios_to_run:
        print(f"\n=== Running scenario: {scenario_name} ===")
        compute_results(
            list_of_logs=sc["logs"],  # type: ignore
            results_name=scenario_name,
            scenario_name=scenario_name,
            clear_name=sc["clear_name"],  # type: ignore
            population_extractor=sc["population_extractor"],  # type: ignore
            metric_adapters=sc["metric_adapters"],  # type: ignore
            bootstrap_sampler=sc["bootstrap_sampler"],  # type: ignore
            normalizers=sc["normalizers"],  # type: ignore
            include_metrics=sc["include_metrics"],  # type: ignore
            sample_confidence_interval_extractor=sc.get(
                "sample_confidence_interval_extractor",
                default_sample_confidence_interval_extractor,
            ),
            base_scenario_name=sc["base_scenario_name"],  # type: ignore
        )


if __name__ == "__main__":
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    import multiprocessing as mp

    mp.freeze_support()  # harmless on Linux; required on Windows
    main()
