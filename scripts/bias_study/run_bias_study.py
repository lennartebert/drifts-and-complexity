#!/usr/bin/env python
# type: ignore
# pylint: disable=all
# flake8: noqa
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

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
from utils.plotting.plot_cis import plot_ci_results
from utils.plotting.plot_correlations import plot_all_correlation_results

# plot_inext_curves module removed - no longer needed
from utils.population.extractors.chao1_population_extractor import (
    Chao1PopulationExtractor,
)
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
)
from utils.SampleConfidenceIntervalExtractor import SampleConfidenceIntervalExtractor

# --- defaults (same as before) ---
SORTED_METRICS = constants.ALL_METRIC_NAMES  # constants.PC_METRICS

SAMPLES_PER_SIZE = 200
RANDOM_STATE = 123
BOOTSTRAP_SIZE = 200
SIZES = range(50, 501, 50)

REF_SIZES = [50, 250, 500]

BREAKDOWN_BY = "dimension"  # None, "basis", or "dimension"

CORRELATION_TYPE = (
    "Spearman"  # Correlation type to use in LaTeX tables ("Pearson" or "Spearman")
)

default_population_extractor = NaivePopulationExtractor()
default_metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
default_bootstrap_sampler = BootstrapSampler(B=BOOTSTRAP_SIZE)
default_normalizers: Optional[List] = None
default_sample_confidence_interval_extractor = SampleConfidenceIntervalExtractor(
    conf_level=0.95
)


# --- core compute function ---
def compute_results(
    list_of_logs: List[str],
    results_name: str,
    scenario_name: str,
    clear_name: str,
    population_extractor=default_population_extractor,
    metric_adapters=default_metric_adapters,
    bootstrap_sampler=default_bootstrap_sampler,
    normalizers=default_normalizers,
    include_metrics: Optional[List[str]] = None,
    sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
    base_scenario_name: Optional[str] = None,  # type: ignore
) -> None:
    print(f"Generating results for {results_name}")
    if include_metrics is None:
        include_metrics = SORTED_METRICS
    data_dictionary = helpers.load_data_dictionary(
        constants.DATA_DICTIONARY_FILE_PATH, get_real=True, get_synthetic=True
    )
    data_dictionary = {
        log: info for log, info in data_dictionary.items() if log in list_of_logs
    }

    # Store population sizes (number of traces) for FPC
    log_population_sizes: Dict[str, int] = {}
    # Store analysis results per log
    analysis_per_log: Dict[str, pd.DataFrame] = {}

    for log_name, dataset_info in data_dictionary.items():
        print(f"Computing for {log_name}")
        log_path = Path(dataset_info["path"])
        pm4py_log = xes_importer.apply(str(log_path))
        # Store population size (number of traces) for FPC
        log_population_sizes[log_name] = len(pm4py_log)

        window_samples = (
            sampling_helper.sample_consecutive_trace_windows_with_replacement(
                pm4py_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE
            )
        )

        out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name / log_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compute raw metrics
        metrics_df = compute_metrics_for_samples(
            window_samples,
            population_extractor=population_extractor,
            metric_adapters=metric_adapters,
            bootstrap_sampler=bootstrap_sampler,
            normalizers=normalizers,
            include_metrics=include_metrics,
        )

        # Save raw metrics
        metrics_df.to_csv(out_dir / "raw_metrics.csv")

        # Compute analysis (mean values, CIs, correlations, plateau)
        analysis_df = compute_analysis_for_metrics(
            metrics_df,
            sample_confidence_interval_extractor=sample_confidence_interval_extractor,
            include_metrics=include_metrics,
        )

        # Save analysis results to CSV
        analysis_df.to_csv(out_dir / "analysis.csv")

        # Create CI plots
        plot_ci_results(
            metrics_df=metrics_df,
            analysis_df=analysis_df,
            out_dir=out_dir,
            plot_breakdown=BREAKDOWN_BY,
            ncols=3,
            metric_order=include_metrics,
        )

        # Store for downstream processing
        analysis_per_log[log_name] = analysis_df

    out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute sample size for FPC
    n_samples = len(SIZES) * SAMPLES_PER_SIZE
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
        logs=["BPIC12", "BPIC15_1", "BPIC15_2", "ITHD"],
        clear_name="Real (Base)",
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=None,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name=None,
    ),
    "real_normalized": dict(
        logs=["BPIC12", "BPIC15_1", "BPIC15_2", "ITHD"],
        clear_name="Real (Normalized)",
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name="real_base",
    ),
    "real_normalized_and_population": dict(
        logs=["BPIC12", "BPIC15_1", "BPIC15_2", "ITHD"],
        clear_name="Real (Normalized + Population)",
        population_extractor=Chao1PopulationExtractor(),
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name="real_base",
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
    global SAMPLES_PER_SIZE, BOOTSTRAP_SIZE, SIZES
    if args.test:
        SAMPLES_PER_SIZE = 2
        BOOTSTRAP_SIZE = 2
        SIZES = range(50, 101, 50)

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
            base_scenario_name="test1",
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
