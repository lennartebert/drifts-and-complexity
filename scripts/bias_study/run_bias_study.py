#!/usr/bin/env python
from __future__ import annotations

import os
import math
import argparse
from pathlib import Path

from pm4py.objects.log.importer.xes import importer as xes_importer

from utils import constants, helpers, sampling_helper
from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import INextBootstrapSampler
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import VidgofMetricsAdapter
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS
from utils.pipeline.compute import run_metrics_over_samples
from utils.plotting.plot_cis import plot_aggregated_measures_cis
from utils.plotting.plot_correlations import plot_correlation_results
from utils.population.extractors.chao1_population_extractor import Chao1PopulationExtractor
from utils.population.extractors.naive_population_extractor import NaivePopulationExtractor

# --- defaults (same as before) ---
sorted_metrics = constants.ALL_METRIC_NAMES

SAMPLES_PER_SIZE = 200
RANDOM_STATE = 123
BOOTSTRAP_SIZE = 50
SIZES = range(50, 501, 50)

default_population_extractor = NaivePopulationExtractor()
default_metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
default_bootstrap_sampler = INextBootstrapSampler(B=BOOTSTRAP_SIZE)
default_normalizers = None

# --- core compute function ---
def compute_results(
    list_of_logs, results_name, out_path,
    population_extractor=default_population_extractor,
    metric_adapters=default_metric_adapters,
    bootstrap_sampler=default_bootstrap_sampler,
    normalizers=default_normalizers,
):
    print(f'Generating results for {results_name}')
    data_dictionary = helpers.load_data_dictionary(
        constants.DATA_DICTIONARY_FILE_PATH, get_real=True, get_synthetic=True
    )
    data_dictionary = {log: info for log, info in data_dictionary.items() if log in list_of_logs}

    measures_per_log, ci_low_per_log, ci_high_per_log = {}, {}, {}
    for log_name, dataset_info in data_dictionary.items():
        print(f'Computing for {log_name}')
        log_path = Path(dataset_info["path"])
        pm4py_log = xes_importer.apply(str(log_path))

        window_samples = sampling_helper.sample_random_windows_no_replacement_within_only(
            pm4py_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE
        )

        measures_df, ci_low_df, ci_high_df = run_metrics_over_samples(
            window_samples,
            population_extractor=population_extractor,
            metric_adapters=metric_adapters,
            bootstrap_sampler=bootstrap_sampler,
            normalizers=normalizers,
            sorted_metrics=sorted_metrics,
        )

        out_dir = constants.BIAS_STUDY_RESULTS_DIR / log_name
        out_dir.mkdir(parents=True, exist_ok=True)
        measures_df.to_csv(out_dir / "measures.csv")
        ci_low_df.to_csv(out_dir / "ci_low.csv")
        ci_high_df.to_csv(out_dir / "ci_high.csv")

        if not ci_low_df.empty:
            plot_aggregated_measures_cis(
                measures_df, ci_low_df, ci_high_df,
                out_path=out_dir / "measures_cis_mean.png",
                agg="mean", title=f"{log_name} - Aggregated measures with CIs (mean)",
                ncols=3,
            )

        measures_per_log[log_name] = measures_df
        ci_low_per_log[log_name] = ci_low_df
        ci_high_per_log[log_name] = ci_high_df

    out_dir = constants.BIAS_STUDY_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df, pval_df = helpers.get_correlations_for_dictionary(
        sample_metrics_per_log=measures_per_log,
        rename_dictionary_map=None,
        metric_columns=sorted_metrics,
        base_column='sample_size',
    )
    corr_df.to_csv(out_dir / "correlations_r.csv")
    pval_df.to_csv(out_dir / "correlations_p.csv")
    helpers.corr_p_to_latex_stars(corr_df, pval_df, out_dir / "correlations.tex", f"correlations_{results_name}")
    plot_correlation_results(corr_df, out_path=out_dir / "correlations_box_plot.png", plot_type="box")
    plot_correlation_results(corr_df, out_path=out_dir / "correlations_dot_plot.png", plot_type="dot")

# --- scenario registry ---
SCENARIOS = [
    dict(  # 0
        logs=['O2C_S','CLAIM_S','LOAN_S','CREDIT_S'],
        name="synthetic_base",
        out="results/correlations/synthetic/base",
        population_extractor=default_population_extractor,
        normalizers=default_normalizers,
    ),
    dict(  # 1
        logs=['O2C_S','CLAIM_S','LOAN_S','CREDIT_S'],
        name="synthetic_normalized",
        out="results/correlations/synthetic/normalized",
        population_extractor=default_population_extractor,
        normalizers=DEFAULT_NORMALIZERS,
    ),
    dict(  # 2
        logs=['O2C_S','CLAIM_S','LOAN_S','CREDIT_S'],
        name="synthetic_normalized_and_population",
        out="results/correlations/synthetic/normalized_and_population",
        population_extractor=Chao1PopulationExtractor(),
        normalizers=DEFAULT_NORMALIZERS,
    ),
    dict(  # 3
        logs=['RTFMP'],
        name="real_base",
        out="results/correlations/real/base",
        population_extractor=default_population_extractor,
        normalizers=default_normalizers,
    ),
    dict(  # 4
        logs=['RTFMP'],
        name="real_normalized",
        out="results/correlations/real/normalized",
        population_extractor=default_population_extractor,
        normalizers=DEFAULT_NORMALIZERS,
    ),
    dict(  # 5
        logs=['RTFMP'],
        name="real_normalized_and_population",
        out="results/correlations/real/normalized_and_population",
        population_extractor=Chao1PopulationExtractor(),
        normalizers=DEFAULT_NORMALIZERS,
    ),
]

def main():
    parser = argparse.ArgumentParser(description="Run one scenario by integer ID.")
    parser.add_argument("scenario_id", type=int, help=f"Scenario ID [0..{len(SCENARIOS)-1}]")
    parser.add_argument("--test", action="store_true", help="Run in test mode with reduced parameters")
    args = parser.parse_args()

    # Modify global parameters for test mode
    global SAMPLES_PER_SIZE, BOOTSTRAP_SIZE, SIZES
    if args.test:
        SAMPLES_PER_SIZE = 1
        BOOTSTRAP_SIZE = 2
        SIZES = range(50, 101, 50)  # Just 2 sizes: 50 and 100
        
        # Create test scenario
        test_scenario = dict(
            logs=['RTFMP'],
            name="test",
            out="results/tests/test",
            population_extractor=Chao1PopulationExtractor(),
            normalizers=default_normalizers,
        )
        
        sc = test_scenario
    else:
        if args.scenario_id < 0 or args.scenario_id >= len(SCENARIOS):
            raise SystemExit(f"Invalid scenario_id {args.scenario_id}")
        sc = SCENARIOS[args.scenario_id]

    compute_results(
        list_of_logs=sc["logs"],
        results_name=sc["name"],
        out_path=sc["out"],
        population_extractor=sc["population_extractor"],
        metric_adapters=default_metric_adapters,
        bootstrap_sampler=default_bootstrap_sampler,
        normalizers=sc["normalizers"],
    )

if __name__ == "__main__":
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    import multiprocessing as mp
    mp.freeze_support()          # harmless on Linux; required on Windows
    main()
