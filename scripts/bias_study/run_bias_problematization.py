#### Complexity Metric Problematization - Examples

import datetime
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
from pm4py.objects.log.obj import Event, EventLog, Trace

from utils import constants, helpers, sampling_helper
from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import \
    INextBootstrapSampler
from utils.complexity.metrics_adapters.local_metrics_adapter import \
    LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import \
    VidgofMetricsAdapter
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS
from utils.pipeline.compute import run_metrics_over_samples
from utils.plotting.plot_cis import plot_aggregated_measures_cis
from utils.population.extractors.chao1_population_extractor import \
    Chao1PopulationExtractor
from utils.population.extractors.naive_population_extractor import \
    NaivePopulationExtractor

sorted_metrics = constants.ALL_METRIC_NAMES

## helper functions
def make_trace(variant, trace_id: int):
    # give each trace a unique name/id
    trace = Trace(attributes={"concept:name": f"Case{trace_id}"})
    # add events with both concept:name and timestamp
    base_time = datetime.datetime(2020, 1, 1, 0, 0, 0)  # arbitrary starting point
    for i, act in enumerate(variant):
        event = Event({
            "concept:name": act,
            "time:timestamp": base_time + datetime.timedelta(minutes=i)
        })
        trace.append(event)
    return trace

def apply_fixes(sample_collection, problems_to_fix=['p1', 'p2', 'p3']):
    # get the right configuration for our computation pipeline

    population_extractor = default_population_extractor
    metric_adapters = default_metric_adapters
    bootstrap_sampler = default_bootstrap_sampler
    normalizers = default_normalizers

    if 'p1' in problems_to_fix:
        normalizers = DEFAULT_NORMALIZERS
    
    if 'p2' in problems_to_fix:
        population_extractor = Chao1PopulationExtractor()
    
    if 'p3' in problems_to_fix:
        bootstrap_sampler = INextBootstrapSampler(B=50)

    results_per_log = {}
    for log_name, window_samples in sample_collection.items():
        measures_df, ci_low_df, ci_high_df = run_metrics_over_samples(window_samples,
            population_extractor = population_extractor,
            metric_adapters = metric_adapters,
            bootstrap_sampler = bootstrap_sampler,
            normalizers = normalizers,
            sorted_metrics=sorted_metrics)
        
        results_per_log[log_name] = (measures_df, ci_low_df, ci_high_df)
    return results_per_log

SAMPLES_PER_SIZE = 200 # do 200 samples in final computation
RANDOM_STATE = 1
NUM_TRACES_PER_LOG = 10000 # do 10000 in final computation
SIZES = range(50, 501, 50)

# set defaults
default_population_extractor = NaivePopulationExtractor()
default_metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
default_bootstrap_sampler = None
default_normalizers = None

rename_map = {
        "strict_monotone_growth": "Log 1",
        "rare_occurrences": "Log 2",
        "variance": "Log 3",
    }

def main():
    ## P1 Strict Monotone growth
    # The complexity metric grows with increasing window size, although there is no variance, rare species or loops.

    sample_collection = {}
    results_per_log = {}

    log_name = 'strict_monotone_growth'

    # Create a trace variant A-B-C-D-E
    variant = ["A", "B", "C", "D", "E"]

    # Build EventLog with 10,000 identical traces
    event_log = EventLog([make_trace(variant, id) for id in range(NUM_TRACES_PER_LOG)])

    # get window samples
    window_samples = sampling_helper.sample_random_windows_no_replacement_within_only(event_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE) # returns (sample_size, sample_id, window)
    sample_collection[log_name] = window_samples

    # compute measures
    measures_df, ci_low_df, ci_high_df = run_metrics_over_samples(window_samples,
        population_extractor = default_population_extractor,
        metric_adapters = default_metric_adapters,
        bootstrap_sampler = default_bootstrap_sampler,
        normalizers = default_normalizers,
        sorted_metrics=sorted_metrics)

    results_per_log[log_name] = (measures_df, ci_low_df, ci_high_df)

    print(measures_df)

    # plot Number of events
    fig, ax = plt.subplots(figsize=(6, 4))
    measures_df.boxplot(
        column="Number of Events",
        by="sample_size",
        ax=ax,
        grid=False
    )
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Number of Events")
    ax.set_title("Number of Events vs Window Size (Single Variant Log)")
    plt.suptitle("")  # remove automatic title from pandas

    # Ensure output directory exists
    out_path = Path(f"results/correlations/problematization/{log_name}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


    # ## Infinite support
    # # Looped models cause unbounded number of distinct variants

    # from pm4py.objects.bpmn.importer import importer as bpmn_importer
    # from pm4py.objects.conversion.bpmn import converter as bpmn_converter
    # from pm4py.algo.simulation.playout.petri_net import algorithm as simulator

    # # Path to the BPMN file exported from BPMN.io (A->B->C with self-loop on B)
    # bpmn_path = r"data\synthetic\simple_loop\simple_loop.bpmn"
    # # Import BPMN
    # bpmn_graph = bpmn_importer.apply(bpmn_path)

    # # Convert BPMN -> (Petri net, initial marking, final marking)
    # net, im, fm = bpmn_converter.apply(bpmn_graph)

    # event_log = simulator.apply(
    #     net,
    #     im,
    #     fm,
    #     parameters = {
    #         "no_traces": NUM_TRACES_PER_LOG,
    #         "random_seed": 1
    #     },
    #     variant=simulator.Variants.BASIC_PLAYOUT,
    # )

    # # quick peek
    # for i, trace in enumerate(event_log[:5], 1):
    #     print(f"Trace {i}:", [ev["concept:name"] for ev in trace])

    # sizes = range(50, 501, 50)
    # samples = sample_random_trace_sets_no_replacement_within_only(event_log, sizes, SAMPLES_PER_SIZE, RANDOM_STATE)
    # sample_collection['infinite_support'] = samples
    # adapters = ["vidgof_sample"]
    # df_metrics = compute_metrics_for_samples(samples, adapters)
    # sample_metrics_per_log['infinite_support'] = df_metrics
    # print(df_metrics)

    # # Boxplot Distinct traces vs Window Size
    # fig, ax = plt.subplots(figsize=(6, 4))
    # df_metrics.boxplot(
    #     column="Number of Distinct Traces",
    #     by="sample_size",
    #     ax=ax,
    #     grid=False
    # )

    # ax.set_xlabel("Window Size")
    # ax.set_ylabel("Number of Distinct Traces")
    # ax.set_title("Number of Distinct Traces vs Window Size (Looping Log)")
    # plt.suptitle("")  # remove automatic title from pandas

    # # Ensure output directory exists
    # out_path = Path("results/correlations/problematization/p2.png")
    # out_path.parent.mkdir(parents=True, exist_ok=True)

    # # Save figure
    # plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # plt.close(fig)

    ## P3 Rare occurrences
    # The complexity metric under-estimates rarely occurring behavior due to skewed occurrence distributions.

    log_name = 'rare_occurrences'

    # Create trace variants
    frequent_variant = ["A", "B"]
    once_occuring_variants = [["A", f"B_{i}"] for i in range (0, 10)] # 10 variants that occur only once

    # Build EventLog with 10,000 traces of two variants
    rare_trace_variant_count = 10

    traces = []
    for id in range(NUM_TRACES_PER_LOG):
        indeces_of_common_variants = NUM_TRACES_PER_LOG - rare_trace_variant_count
        if id < indeces_of_common_variants:
            traces.append(make_trace(frequent_variant, id))
        else:
            traces.append(make_trace(once_occuring_variants[id - indeces_of_common_variants], id))

    # shuffle traces (should not matter due to random sampling but just to be sure)
    random.seed(1)
    random.shuffle(traces)

    event_log = EventLog(traces)

    # get window samples
    window_samples = sampling_helper.sample_random_windows_no_replacement_within_only(event_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE) # returns (sample_size, sample_id, window)
    sample_collection[log_name] = window_samples

    # compute measures
    measures_df, ci_low_df, ci_high_df = run_metrics_over_samples(window_samples,
        population_extractor = default_population_extractor,
        metric_adapters = default_metric_adapters,
        bootstrap_sampler = default_bootstrap_sampler,
        normalizers = default_normalizers,
        sorted_metrics=sorted_metrics)

    results_per_log[log_name] = (measures_df, ci_low_df, ci_high_df)

    print(measures_df)

    # Boxplot Variety vs Window Size
    fig, ax = plt.subplots(figsize=(6, 4))
    measures_df.boxplot(
        column="Number of Distinct Activities",
        by="sample_size",
        ax=ax,
        grid=False
    )

    ax.set_xlabel("Window Size")
    ax.set_ylabel("Number of Distinct Activities")
    ax.set_title("Number of Distinct Activities vs Window Size (Skewed Variant Log)")
    plt.suptitle("")  # remove automatic title from pandas

    # Ensure output directory exists
    out_path = Path(f"results/correlations/problematization/{log_name}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


    ## P4 Variance
    # The complexity metric fluctuates at small sample sizes due to the law of small numbers. Only at higher sample window sizes, the metric becomes asymptotic.

    log_name = 'variance'

    # Create a trace variant A-B-C-D-E
    variant_a = ["A", "B", "C", "D", "E"] # length: 5
    variant_b = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] # length: 10]


    # Build EventLog with 10,000 traces of two variants
    traces = []
    for id in range(NUM_TRACES_PER_LOG):
        if id % 2 == 0:
            traces.append(make_trace(variant_a, id))
        else:
            traces.append(make_trace(variant_b, id))

    event_log = EventLog(traces)

    # get window samples
    window_samples = sampling_helper.sample_random_windows_no_replacement_within_only(event_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE) # returns (sample_size, sample_id, window)
    sample_collection[log_name] = window_samples

    # compute measures
    measures_df, ci_low_df, ci_high_df = run_metrics_over_samples(window_samples,
        population_extractor = default_population_extractor,
        metric_adapters = default_metric_adapters,
        bootstrap_sampler = default_bootstrap_sampler,
        normalizers = default_normalizers,
        sorted_metrics=sorted_metrics)

    results_per_log[log_name] = (measures_df, ci_low_df, ci_high_df)

    print(measures_df)

    # plot Trace length
    fig, ax = plt.subplots(figsize=(6, 4))
    measures_df.boxplot(
        column="Avg. Trace Length",
        by="sample_size",
        ax=ax,
        grid=False
    )
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Avg. Trace Length")
    ax.set_title("Avg. Trace Length vs Window Size (Two Variant Log)")
    plt.suptitle("")  # remove automatic title from pandas

    # Ensure output directory exists
    out_path = Path(f"results/correlations/problematization/{log_name}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    measures_per_log = {log_name: measures_df for log_name, (measures_df, _, _) in results_per_log.items()}

    corr_df, pval_df = helpers.get_correlations_for_dictionary(sample_metrics_per_log=measures_per_log, rename_dictionary_map=rename_map, metric_columns=sorted_metrics, base_column='sample_size')

    # Save
    out_dir = constants.BIAS_STUDY_RESULTS_DIR / "problematization"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_dir / "correlations_r.csv")
    pval_df.to_csv(out_dir / "correlations_p.csv")


    helpers.corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table.tex"), 'correlations_not_fixed')

    ##### Fixing of problems

    # apply the fix to p1
    results_per_log_p1_fixed = apply_fixes(sample_collection, problems_to_fix=['p1'])

    measures_per_log = {log_name: measures_df for log_name, (measures_df, _, _) in results_per_log_p1_fixed.items()}

    # get new correlations
    corr_df, pval_df = helpers.get_correlations_for_dictionary(sample_metrics_per_log=measures_per_log, rename_dictionary_map=rename_map, metric_columns=sorted_metrics, base_column='sample_size')

    # Save
    out_dir = constants.BIAS_STUDY_RESULTS_DIR / "problematization"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_dir / "correlations_r_p1_fixed.csv")
    pval_df.to_csv(out_dir / "correlations_p_p1_fixed.csv")

    helpers.corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table_p1_fixed.tex"), 'correlations_p1_fixed')

    # apply the fix to p1 and p2 and p3
    results_per_log_p1_p2_p3_fixed = apply_fixes(sample_collection, problems_to_fix=['p1', 'p2', 'p3'])

    measures_per_log = {log_name: measures_df for log_name, (measures_df, _, _) in results_per_log_p1_p2_p3_fixed.items()}

    # get new correlations

    corr_df, pval_df = helpers.get_correlations_for_dictionary(sample_metrics_per_log=measures_per_log, rename_dictionary_map=rename_map, metric_columns=sorted_metrics, base_column='sample_size')

    # Save
    out_dir = constants.BIAS_STUDY_RESULTS_DIR / "problematization"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_dir / "correlations_r_p1_p2_fixed.csv")
    pval_df.to_csv(out_dir / "correlations_p_p1_p2_fixed.csv")

    helpers.corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table_p1_p2_fixed.tex"), 'correlations_p1_p2_fixed')

    # get the variance after applying fixes for p1 and p2 (can use the same results as before)

    # One plot per log. Create both mean and median versions (two files).
    plots_dir = "results/plots_measures_cis"
    os.makedirs(plots_dir, exist_ok=True)

    for log, (measures_df, ci_low_df, ci_high_df) in results_per_log_p1_p2_p3_fixed.items():
        # Mean
        out_mean = os.path.join(plots_dir, f"{log}_measures_cis_mean.png")
        plot_aggregated_measures_cis(
            measures_df, ci_low_df, ci_high_df,
            out_path=out_mean,
            agg="mean",
            title=f"{log} — Aggregated measures with CIs (mean)",
            ncols=3,
        )

        # Median
        out_median = os.path.join(plots_dir, f"{log}_measures_cis_median.png")
        plot_aggregated_measures_cis(
            measures_df, ci_low_df, ci_high_df,
            out_path=out_median,
            agg="median",
            title=f"{log} — Aggregated measures with CIs (median)",
            ncols=3,
        )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()