from pathlib import Path

from pm4py.objects.log.importer.xes import importer as xes_importer

from utils import constants, helpers, sampling_helper
from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import INextBootstrapSampler
from utils.complexity.metrics_adapters.local_metrics_adapter import \
    LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import \
    VidgofMetricsAdapter
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS
from utils.pipeline.compute import run_metrics_over_samples
from utils.plotting.plot_cis import plot_aggregated_measures_cis
from utils.plotting.plot_correlations import plot_correlation_results
from utils.population.extractors.chao1_population_extractor import Chao1PopulationExtractor
from utils.population.extractors.naive_population_extractor import \
    NaivePopulationExtractor

sorted_metrics = [
	'Number of Events',
	'Number of Distinct Activities',
	'Number of Traces',
	'Number of Distinct Traces',
	'Min. Trace Length',
    'Avg. Trace Length',
    'Max. Trace Length',
	'Percentage of Distinct Traces',
	'Average Distinct Activities per Trace',
	'Structure',
	'Estimated Number of Acyclic Paths',
	'Number of Ties in Paths to Goal',
	'Lempel-Ziv Complexity',
	'Average Affinity',
	'Deviation from Random',
	'Average Edit Distance',
	'Sequence Entropy',
    'Normalized Sequence Entropy',
    'Variant Entropy',
    'Normalized Variant Entropy'
]

# set defaults
SAMPLES_PER_SIZE = 200 # select 200 in final eval
RANDOM_STATE = 123
BOOTSTRAP_SIZE = 50 # select 50 in final eval
SIZES = range(50, 501, 50)

default_population_extractor = NaivePopulationExtractor()
default_metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
default_bootstrap_sampler = INextBootstrapSampler(B=BOOTSTRAP_SIZE)
default_normalizers = None

# get correlations between metrics and window size for selected synthetic and real logs

# analyze the synthetic logs

def compute_results(list_of_logs, results_name, out_path, population_extractor=default_population_extractor, metric_adapters=default_metric_adapters, bootstrap_sampler=default_bootstrap_sampler, normalizers=default_normalizers):
    print(f'Generating results for {results_name}')
    data_dictionary = helpers.load_data_dictionary(constants.DATA_DICTIONARY_FILE_PATH, get_real=True, get_synthetic=True)

    data_dictionary = {log: info for log, info in data_dictionary.items() if log in list_of_logs}

    measures_per_log = {}
    ci_low_per_log = {}
    ci_high_per_log = {}

    for log_name, dataset_info in data_dictionary.items():
        print(f'Performing computations for {log_name}')
        # Load log
        log_path = Path(dataset_info["path"])
        pm4py_log = xes_importer.apply(str(log_path))

        # get windows of different sizes
        window_samples = sampling_helper.sample_random_windows_no_replacement_within_only(pm4py_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE)

        # compute measures
        measures_df, ci_low_df, ci_high_df = run_metrics_over_samples(window_samples,
            population_extractor = population_extractor,
            metric_adapters = metric_adapters,
            bootstrap_sampler = bootstrap_sampler,
            normalizers = normalizers,
            sorted_metrics=sorted_metrics)
        
        measures_per_log[log_name] = measures_df
        ci_low_per_log[log_name] = ci_low_df
        ci_high_per_log[log_name] = ci_high_df

        # save results to file
        out_dir = Path(out_path) / log_name
        out_dir.mkdir(parents=True, exist_ok=True)
        measures_df.to_csv(out_dir / "measures.csv")
        ci_low_df.to_csv(out_dir / "ci_low.csv")
        ci_high_df.to_csv(out_dir / "ci_high.csv")

        # plot the variance curves if available
        
        if not ci_low_df.empty:
            plot_aggregated_measures_cis(
                measures_df, ci_low_df, ci_high_df,
                out_path=out_dir / f"measures_cis_mean.png",
                agg="mean",
                title=f"{log_name} - Aggregated measures with CIs (mean)",
                ncols=3,
            )

    # get the correlations
    corr_df, pval_df = helpers.get_correlations_for_dictionary(sample_metrics_per_log=measures_per_log, rename_dictionary_map=None, metric_columns=sorted_metrics, base_column='sample_size')

    # Save correlations
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_dir / "correlations_r.csv")
    pval_df.to_csv(out_dir / "correlations_p.csv")

    # also save as latex
    helpers.corr_p_to_latex_stars(corr_df, pval_df, out_dir / 'correlations.tex', f'correlations_{results_name}')

    # plot correlations 
    plot_correlation_results(corr_df, out_path=out_dir/'correlations_box_plot.png', plot_type='box')
    plot_correlation_results(corr_df, out_path=out_dir/'correlations_dot_plot.png', plot_type='dot')

print('Result generation started')

list_of_logs = ['O2C_S', 'CLAIM_S', 'LOAN_S', 'CREDIT_S']
results_name = 'synthetic_base'
out_path = "results/correlations/synthetic/base"
compute_results(list_of_logs, results_name, out_path, population_extractor=default_population_extractor, metric_adapters=default_metric_adapters, bootstrap_sampler=default_bootstrap_sampler, normalizers=default_normalizers)

# perform fix of p1
list_of_logs = ['O2C_S', 'CLAIM_S', 'LOAN_S', 'CREDIT_S']
results_name = 'synthetic_normalized'
out_path = "results/correlations/synthetic/normalized"
compute_results(list_of_logs, results_name, out_path, population_extractor=default_population_extractor, metric_adapters=default_metric_adapters, bootstrap_sampler=default_bootstrap_sampler, normalizers=DEFAULT_NORMALIZERS)

# perform fix of p1 and p2
list_of_logs = ['O2C_S', 'CLAIM_S', 'LOAN_S', 'CREDIT_S']
results_name = 'synthetic_normalized_and_population'
out_path = "results/correlations/synthetic/normalized_and_population"
compute_results(list_of_logs, results_name, out_path, population_extractor=Chao1PopulationExtractor(), metric_adapters=default_metric_adapters, bootstrap_sampler=default_bootstrap_sampler, normalizers=DEFAULT_NORMALIZERS)

### real-world evaluation
list_of_logs = ['RTFMP']
results_name = 'real_base'
out_path = "results/correlations/real/base"
compute_results(list_of_logs, results_name, out_path, population_extractor=default_population_extractor, metric_adapters=default_metric_adapters, bootstrap_sampler=default_bootstrap_sampler, normalizers=default_normalizers)

# perform fix of p1
list_of_logs = ['RTFMP']
results_name = 'real_normalized'
out_path = "results/correlations/real/normalized"
compute_results(list_of_logs, results_name, out_path, population_extractor=default_population_extractor, metric_adapters=default_metric_adapters, bootstrap_sampler=default_bootstrap_sampler, normalizers=DEFAULT_NORMALIZERS)

# perform fix of p1 and p2
list_of_logs = ['RTFMP']
results_name = 'real_normalized_and_population'
out_path = "results/correlations/real/normalized_and_population"
compute_results(list_of_logs, results_name, out_path, population_extractor=Chao1PopulationExtractor(), metric_adapters=default_metric_adapters, bootstrap_sampler=default_bootstrap_sampler, normalizers=DEFAULT_NORMALIZERS)
