#!/usr/bin/env python
# type: ignore
# pylint: disable=all
# flake8: noqa
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

from utils import constants, helpers, sampling_helper
from utils.bootstrapping.bootstrap_samplers.bootstrap_sampler import BootstrapSampler
from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import (
    INextBootstrapSampler,
)
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import (
    VidgofMetricsAdapter,
)
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS
from utils.pipeline.compute import run_metrics_over_samples
from utils.plotting.plot_cis import (
    plot_aggregated_measures_bootstrap_cis,
    plot_aggregated_measures_sample_cis,
)
from utils.plotting.plot_correlations import plot_correlation_results

# plot_inext_curves module removed - no longer needed
from utils.population.extractors.chao1_population_extractor import (
    Chao1PopulationExtractor,
)
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
)
from utils.SampleConfidenceIntervalExtractor import SampleConfidenceIntervalExtractor

# --- defaults (same as before) ---
sorted_metrics = constants.ALL_METRIC_NAMES

SAMPLES_PER_SIZE = 200
RANDOM_STATE = 123
BOOTSTRAP_SIZE = 200
SIZES = range(50, 501, 50)

REF_SIZES = [50, 250, 500]

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
        include_metrics = sorted_metrics
    data_dictionary = helpers.load_data_dictionary(
        constants.DATA_DICTIONARY_FILE_PATH, get_real=True, get_synthetic=True
    )
    data_dictionary = {
        log: info for log, info in data_dictionary.items() if log in list_of_logs
    }

    (
        measures_per_log,
        bootstrap_ci_low_per_log,
        bootstrap_ci_high_per_log,
        sample_ci_low_per_log,
        sample_ci_high_per_log,
        sample_ci_rel_width_per_log,
    ) = ({}, {}, {}, {}, {}, {})
    for log_name, dataset_info in data_dictionary.items():
        print(f"Computing for {log_name}")
        log_path = Path(dataset_info["path"])
        pm4py_log = xes_importer.apply(str(log_path))

        window_samples = (
            sampling_helper.sample_random_windows_no_replacement_within_only(
                pm4py_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE
            )
        )

        (
            measures_df,
            bootstrap_ci_low_df,
            bootstrap_ci_high_df,
            sample_ci_low_df,
            sample_ci_high_df,
            sample_ci_rel_width_df,
        ) = run_metrics_over_samples(
            window_samples,
            population_extractor=population_extractor,
            metric_adapters=metric_adapters,
            bootstrap_sampler=bootstrap_sampler,
            normalizers=normalizers,
            include_metrics=include_metrics,
            sample_confidence_interval_extractor=sample_confidence_interval_extractor,
        )

        out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name / log_name
        out_dir.mkdir(parents=True, exist_ok=True)
        measures_df.to_csv(out_dir / "measures.csv")

        if bootstrap_ci_low_df is not None:
            bootstrap_ci_low_df.to_csv(out_dir / "bootstrap_ci_low.csv")

        if bootstrap_ci_high_df is not None:
            bootstrap_ci_high_df.to_csv(out_dir / "bootstrap_ci_high.csv")

        if sample_ci_low_df is not None:
            sample_ci_low_df.to_csv(out_dir / "sample_ci_low.csv")

        if sample_ci_high_df is not None:
            sample_ci_high_df.to_csv(out_dir / "sample_ci_high.csv")

        if sample_ci_rel_width_df is not None:
            sample_ci_rel_width_df.to_csv(out_dir / "sample_ci_rel_width.csv")

        # create plots
        if bootstrap_ci_low_df is not None and bootstrap_ci_high_df is not None:
            plot_aggregated_measures_bootstrap_cis(
                measures_df,
                bootstrap_ci_low_df,
                bootstrap_ci_high_df,
                out_path=str(out_dir / "measures_bootstrap_cis_mean.png"),
                agg="mean",
                title=f"{log_name} - Aggregated measures with bootstrap CIs (mean)",
                ncols=3,
            )

        if sample_ci_low_df is not None and sample_ci_high_df is not None:
            plot_aggregated_measures_sample_cis(
                measures_df,
                sample_ci_low_df,
                sample_ci_high_df,
                out_path=str(out_dir / "measures_sample_cis_mean.png"),
                agg="mean",
                title=f"{log_name} - Aggregated measures with sample CIs (mean)",
                ncols=3,
            )

        measures_per_log[log_name] = measures_df
        bootstrap_ci_low_per_log[log_name] = bootstrap_ci_low_df
        bootstrap_ci_high_per_log[log_name] = bootstrap_ci_high_df
        sample_ci_low_per_log[log_name] = sample_ci_low_df
        sample_ci_high_per_log[log_name] = sample_ci_high_df
        sample_ci_rel_width_per_log[log_name] = sample_ci_rel_width_df

    out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df, pval_df = helpers.get_correlations_for_dictionary(
        sample_metrics_per_log=measures_per_log,
        rename_dictionary_map=None,
        metric_columns=include_metrics,
        base_column="sample_size",
    )
    corr_df.to_csv(out_dir / "correlations_r.csv")
    pval_df.to_csv(out_dir / "correlations_p.csv")
    helpers.corr_p_to_latex_stars(
        corr_df, pval_df, out_dir / "correlations.tex", f"correlations_{results_name}"
    )
    plot_correlation_results(
        corr_df, out_path=out_dir / "correlations_box_plot.png", plot_type="box"
    )
    plot_correlation_results(
        corr_df, out_path=out_dir / "correlations_dot_plot.png", plot_type="dot"
    )

    # 1) Compute plateau_df once (same matrix shape as corr_df/pval_df)
    plateau_df = helpers.detect_plateau_df(
        measures_per_log=measures_per_log,
        metric_columns=include_metrics,
        rel_threshold=0.05,  # your plateau threshold
        agg="mean",
        min_runs=1,  # or 2 if you want consecutive confirmations
        report="current",  # or "next"
    )

    if base_scenario_name is not None:
        # load master_table.csv from base_scenario_name
        base_csv_path = (
            constants.BIAS_STUDY_RESULTS_DIR / base_scenario_name / "master_table.csv"
        )
        base_df = pd.read_csv(base_csv_path)
        # Convert to dict keyed by log (exclude mean rows)
        base_measures_per_log = {}
        for log in base_df["log"].unique():
            if log != "" and pd.notna(log):  # Skip mean rows (empty log)
                base_measures_per_log[log] = base_df[base_df["log"] == log].copy()
        # Convert current corr_df to dict format expected by compute_significant_improvement
        current_measures_per_log = {}
        for log in corr_df.columns:
            # Convert corr_df column to DataFrame with metric and rho columns
            log_corr = corr_df[log].reset_index()
            log_corr.columns = ["metric", "rho"]
            current_measures_per_log[log] = log_corr
        # compute if improvement was significant
        improvement_per_log_df, improvement_summary_df = (
            helpers.compute_significant_improvement(
                measures_per_log=current_measures_per_log,  # AFTER
                base_measures_per_log=base_measures_per_log,  # BEFORE
                include_metrics=include_metrics,
                num_window_sizes=len(SIZES),  # <-- k
                rho_col="rho",
                alpha=0.05,
            )
        )
    else:
        improvement_per_log_df, improvement_summary_df = None, None

    # 2) Build the master table using precomputed corr/pval and plateau_df
    label = f"tab:master_table_{scenario_name}"
    csv_path, tex_path = helpers.create_master_table(
        measures_per_log=measures_per_log,
        sample_ci_rel_width_per_log=sample_ci_rel_width_per_log,
        corr_df=corr_df,
        pval_df=pval_df,
        plateau_df=plateau_df,
        out_csv_path=str(out_dir / "master_table.csv"),
        metric_columns=include_metrics,
        ref_sizes=REF_SIZES,
        measure_basis_map=constants.METRIC_BASIS_MAP,
        pretty_plateau=True,  # prints '---' instead of NaN
        label=label,
        improvement_per_log_df=improvement_per_log_df,
        improvement_summary_df=improvement_summary_df,
    )
    print(csv_path, tex_path)

    # print aggregated master table

    means_label = f"tab:master_table_means_{scenario_name}"
    helpers.write_means_only_table_from_master_csv(
        master_csv_path=csv_path,
        means_csv_path=out_dir / "master_table_means.csv",
        means_tex_path=out_dir / "master_table_means.tex",
        caption="Assessment of Measures (Means Across Logs)",
        label=means_label,
    )


# --- scenario registry ---
SCENARIOS = {
    "synthetic_base": dict(
        logs=["O2C_S", "CLAIM_S", "LOAN_S", "CREDIT_S"],
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=None,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name=None,
    ),
    "synthetic_normalized": dict(
        logs=["O2C_S", "CLAIM_S", "LOAN_S", "CREDIT_S"],
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        base_scenario_name="synthetic_base",
    ),
    "synthetic_normalized_and_population": dict(
        logs=["O2C_S", "CLAIM_S", "LOAN_S", "CREDIT_S"],
        population_extractor=Chao1PopulationExtractor(),
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name="synthetic_base",
    ),
    "real_base": dict(
        logs=["BPIC12", "BPIC15_1", "BPIC15_2", "ITHD"],
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=None,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name=None,
    ),
    "real_normalized": dict(
        logs=["BPIC12", "BPIC15_1", "BPIC15_2", "ITHD"],
        population_extractor=default_population_extractor,
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=None,
        normalizers=DEFAULT_NORMALIZERS,
        sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
        base_scenario_name="real_base",
    ),
    "real_normalized_and_population": dict(
        logs=["BPIC12", "BPIC15_1", "BPIC15_2", "ITHD"],
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
    if args.metrics is None:
        # Use default sorted_metrics
        selected_metrics = sorted_metrics
    else:
        # Resolve shorthand names to full names
        selected_metrics = []
        for metric_input in args.metrics:
            if metric_input in constants.METRIC_SHORTHAND:
                # It's a shorthand, resolve to full name
                full_name = constants.METRIC_SHORTHAND[metric_input]
                selected_metrics.append(full_name)
            elif metric_input in constants.ALL_METRIC_NAMES:
                # It's already a full name
                selected_metrics.append(metric_input)
            else:
                # Invalid metric name
                available_shorthand = list(constants.METRIC_SHORTHAND.keys())
                available_full = constants.ALL_METRIC_NAMES
                raise SystemExit(
                    f"Invalid metric '{metric_input}'. "
                    f"Available shorthand: {available_shorthand} "
                    f"or full names: {available_full}"
                )

        # Remove duplicates while preserving order
        selected_metrics = list(dict.fromkeys(selected_metrics))

    # Modify global parameters for test mode
    global SAMPLES_PER_SIZE, BOOTSTRAP_SIZE, SIZES
    if args.test:
        SAMPLES_PER_SIZE = 10
        BOOTSTRAP_SIZE = 10
        SIZES = range(50, 201, 50)

        # Create test scenario
        test_scenario = dict(
            logs=["TEST_BPIC12"],
            population_extractor=default_population_extractor,
            metric_adapters=default_metric_adapters,
            bootstrap_sampler=None,
            normalizers=None,
            include_metrics=selected_metrics,
            sample_confidence_interval_extractor=default_sample_confidence_interval_extractor,
            base_scenario_name="test",
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
                scenario_config["include_metrics"] = selected_metrics
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
                    scenario_config["include_metrics"] = selected_metrics
                    scenarios_to_run.append((scenario_name, scenario_config))
                except ValueError:
                    # Not an integer, treat as scenario name
                    if scenario_input not in SCENARIOS:
                        raise SystemExit(
                            f"Invalid scenario name '{scenario_input}'. Valid names: {scenario_names}"
                        )
                    scenario_config = SCENARIOS[scenario_input].copy()
                    scenario_config["include_metrics"] = selected_metrics
                    scenarios_to_run.append((scenario_input, scenario_config))

    # Run each scenario
    for scenario_name, sc in scenarios_to_run:
        print(f"\n=== Running scenario: {scenario_name} ===")
        compute_results(
            list_of_logs=sc["logs"],  # type: ignore
            results_name=scenario_name,
            scenario_name=scenario_name,
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
