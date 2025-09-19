#### Complexity Metric Problematization - Examples

import datetime
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pm4py.objects.log.obj import Event, EventLog, Trace
from typing import Callable, Iterable, Optional

from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import INextBootstrapSampler
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import VidgofMetricsAdapter
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS
from utils.pipeline.compute import compute_metrics_and_CIs
from utils.population.extractors.chao1_population_extractor import Chao1PopulationExtractor
from utils.population.extractors.naive_population_extractor import NaivePopulationExtractor
from utils.windowing.window import Window

measures_in_scope = {
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
}

from complexity_sample_size_correlation_analysis import (
    compute_metrics_for_samples, sample_random_trace_sets_no_replacement_within_only)


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


SAMPLES_PER_SIZE = 10 # do 200 samples in final computation
RANDOM_STATE = 1
NUM_TRACES_PER_LOG = 1000 # do 1000 in final computation

## P1 Strict Monotone growth
# The complexity metric grows with increasing window size, although there is no variance, rare species or loops.

sample_collection = {}
sample_metrics_per_log = {}

# Create a trace variant A-B-C-D-E
variant = ["A", "B", "C", "D", "E"]

# Build EventLog with 10,000 identical traces
event_log = EventLog([make_trace(variant, id) for id in range(NUM_TRACES_PER_LOG)])


# bake event log into window

window = Window(id=1, size=len(event_log), traces=event_log)

# apply pipeline

# get local and vidgof metrics
metrics_and_CIs = compute_metrics_and_CIs(window, 
    population_extractor = NaivePopulationExtractor(),
    metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()],
    bootstrap_sampler = None,
    normalizers = [])
print(metrics_and_CIs)

# get normalized metrics
metrics_and_CIs_norm = compute_metrics_and_CIs(window, 
    population_extractor = NaivePopulationExtractor(),
    metric_adapters = [LocalMetricsAdapter()],
    bootstrap_sampler = None,
    normalizers = DEFAULT_NORMALIZERS)
print(metrics_and_CIs_norm)

# get population level metrics
metrics_and_CIs_norm_pop = compute_metrics_and_CIs(window, 
    population_extractor = Chao1PopulationExtractor(),
    metric_adapters = [LocalMetricsAdapter()],
    bootstrap_sampler = None,
    normalizers = DEFAULT_NORMALIZERS)
print(metrics_and_CIs_norm_pop)

# normalized and bootstrapped
metrics_and_CIs_norm_bootstrapped = compute_metrics_and_CIs(window, 
    population_extractor = NaivePopulationExtractor(),
    metric_adapters = [LocalMetricsAdapter()],
    bootstrap_sampler = INextBootstrapSampler(B=200),
    normalizers = DEFAULT_NORMALIZERS)
print(metrics_and_CIs_norm_bootstrapped)

# normalized populalation-level bootstrapped
metrics_and_CIs_norm_pop_bootstrapped = compute_metrics_and_CIs(window, 
    population_extractor = Chao1PopulationExtractor(),
    metric_adapters = [LocalMetricsAdapter()],
    bootstrap_sampler = INextBootstrapSampler(B=200),
    normalizers = DEFAULT_NORMALIZERS)
print(metrics_and_CIs_norm_bootstrapped)


sizes = range(50, 501, 50)
samples = sample_random_trace_sets_no_replacement_within_only(event_log, sizes, SAMPLES_PER_SIZE, RANDOM_STATE)
window_samples = (Window(id=sample_id, size=len(trace_list), traces=trace_list) for (window_size, sample_id, trace_list) in samples)


# sample_collection['strict_monotone_growth'] = samples

# df_metrics = compute_metrics_for_samples(samples)




# adapters = ["vidgof_sample"]
# df_metrics = compute_metrics_for_samples(samples, adapters)
# sample_metrics_per_log['strict_monotone_growth'] = df_metrics
# print(df_metrics)

# # plot Number of events
# fig, ax = plt.subplots(figsize=(6, 4))
# df_metrics.boxplot(
#     column="Number of Events",
#     by="window_size",
#     ax=ax,
#     grid=False
# )
# ax.set_xlabel("Window Size")
# ax.set_ylabel("Number of Events")
# ax.set_title("Number of Events vs Window Size (Single Variant Log)")
# plt.suptitle("")  # remove automatic title from pandas

# # Ensure output directory exists
# out_path = Path("results/correlations/problematization/p1.png")
# out_path.parent.mkdir(parents=True, exist_ok=True)

# # Save figure
# plt.savefig(out_path, dpi=300, bbox_inches="tight")
# plt.close(fig)



# # ## P2 Infinite support
# # # Looped models cause unbounded number of distinct variants

# # from pm4py.objects.bpmn.importer import importer as bpmn_importer
# # from pm4py.objects.conversion.bpmn import converter as bpmn_converter
# # from pm4py.algo.simulation.playout.petri_net import algorithm as simulator

# # # Path to the BPMN file exported from BPMN.io (A->B->C with self-loop on B)
# # bpmn_path = r"data\synthetic\simple_loop\simple_loop.bpmn"
# # # Import BPMN
# # bpmn_graph = bpmn_importer.apply(bpmn_path)

# # # Convert BPMN -> (Petri net, initial marking, final marking)
# # net, im, fm = bpmn_converter.apply(bpmn_graph)

# # event_log = simulator.apply(
# #     net,
# #     im,
# #     fm,
# #     parameters = {
# #         "no_traces": NUM_TRACES_PER_LOG,
# #         "random_seed": 1
# #     },
# #     variant=simulator.Variants.BASIC_PLAYOUT,
# # )

# # # quick peek
# # for i, trace in enumerate(event_log[:5], 1):
# #     print(f"Trace {i}:", [ev["concept:name"] for ev in trace])

# # sizes = range(50, 501, 50)
# # samples = sample_random_trace_sets_no_replacement_within_only(event_log, sizes, SAMPLES_PER_SIZE, RANDOM_STATE)
# # sample_collection['infinite_support'] = samples
# # adapters = ["vidgof_sample"]
# # df_metrics = compute_metrics_for_samples(samples, adapters)
# # sample_metrics_per_log['infinite_support'] = df_metrics
# # print(df_metrics)

# # # Boxplot Distinct traces vs Window Size
# # fig, ax = plt.subplots(figsize=(6, 4))
# # df_metrics.boxplot(
# #     column="Number of Distinct Traces",
# #     by="window_size",
# #     ax=ax,
# #     grid=False
# # )

# # ax.set_xlabel("Window Size")
# # ax.set_ylabel("Number of Distinct Traces")
# # ax.set_title("Number of Distinct Traces vs Window Size (Looping Log)")
# # plt.suptitle("")  # remove automatic title from pandas

# # # Ensure output directory exists
# # out_path = Path("results/correlations/problematization/p2.png")
# # out_path.parent.mkdir(parents=True, exist_ok=True)

# # # Save figure
# # plt.savefig(out_path, dpi=300, bbox_inches="tight")
# # plt.close(fig)

# ## P3 Rare occurrences
# # The complexity metric under-estimates rarely occurring behavior due to skewed occurrence distributions.

# # Create trace variants
# frequent_variant = ["A", "B"]
# once_occuring_variants = [["A", f"B_{i}"] for i in range (0, 10)] # 10 variants that occur only once

# # Build EventLog with 10,000 traces of two variants
# rare_trace_variant_count = 10

# traces = []
# for id in range(NUM_TRACES_PER_LOG):
#     indeces_of_common_variants = NUM_TRACES_PER_LOG - rare_trace_variant_count
#     if id < indeces_of_common_variants:
#         traces.append(make_trace(frequent_variant, id))
#     else:
#         traces.append(make_trace(once_occuring_variants[id - indeces_of_common_variants], id))

# # shuffle traces (should not matter due to random sampling but just to be sure)
# random.seed(1)
# random.shuffle(traces)

# event_log = EventLog(traces)

# sizes = range(50, 501, 50)
# samples = sample_random_trace_sets_no_replacement_within_only(event_log, sizes, SAMPLES_PER_SIZE, RANDOM_STATE)
# sample_collection['rare_occurrences'] = samples
# adapters = ["vidgof_sample"]
# df_metrics = compute_metrics_for_samples(samples, adapters)
# sample_metrics_per_log['rare_occurrences'] = df_metrics
# print(df_metrics)

# # Boxplot Variety vs Window Size
# fig, ax = plt.subplots(figsize=(6, 4))
# df_metrics.boxplot(
#     column="Number of Distinct Activities",
#     by="window_size",
#     ax=ax,
#     grid=False
# )

# ax.set_xlabel("Window Size")
# ax.set_ylabel("Number of Distinct Activities")
# ax.set_title("Number of Distinct Activities vs Window Size (Skewed Variant Log)")
# plt.suptitle("")  # remove automatic title from pandas

# # Ensure output directory exists
# out_path = Path("results/correlations/problematization/p3.png")
# out_path.parent.mkdir(parents=True, exist_ok=True)

# # Save figure
# plt.savefig(out_path, dpi=300, bbox_inches="tight")
# plt.close(fig)


# ## P4 Variance
# # The complexity metric fluctuates at small sample sizes due to the law of small numbers. Only at higher sample window sizes, the metric becomes asymptotic.

# # Create a trace variant A-B-C-D-E
# variant_a = ["A", "B", "C", "D", "E"] # length: 5
# variant_b = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] # length: 10]


# # Build EventLog with 10,000 traces of two variants
# traces = []
# for id in range(NUM_TRACES_PER_LOG):
#     if id % 2 == 0:
#         traces.append(make_trace(variant_a, id))
#     else:
#         traces.append(make_trace(variant_b, id))

# event_log = EventLog(traces)

# sizes = range(50, 501, 50)
# samples = sample_random_trace_sets_no_replacement_within_only(event_log, sizes, SAMPLES_PER_SIZE, RANDOM_STATE)
# sample_collection['variance'] = samples
# adapters = ["vidgof_sample"]
# df_metrics = compute_metrics_for_samples(samples, adapters)
# sample_metrics_per_log['variance'] = df_metrics
# print(df_metrics)

# # plot Trace length
# fig, ax = plt.subplots(figsize=(6, 4))
# df_metrics.boxplot(
#     column="Avg. Trace Length",
#     by="window_size",
#     ax=ax,
#     grid=False
# )
# ax.set_xlabel("Window Size")
# ax.set_ylabel("Avg. Trace Length")
# ax.set_title("Avg. Trace Length vs Window Size (Two Variant Log)")
# plt.suptitle("")  # remove automatic title from pandas

# # Ensure output directory exists
# out_path = Path("results/correlations/problematization/p4.png")
# out_path.parent.mkdir(parents=True, exist_ok=True)

# # Save figure
# plt.savefig(out_path, dpi=300, bbox_inches="tight")
# plt.close(fig)

# out_path

# def get_correlations_for_dictionary(sample_metrics_per_log):
#     # create a correlation analysis for all measures
#     from pathlib import Path
#     import pandas as pd
#     from scipy import stats

#     # mapping of your dict keys to desired column names
#     rename_map = {
#         "strict_monotone_growth": "P1",
#         "infinite_support": "P2",
#         "rare_occurrences": "P3",
#         "variance": "P4",
#     }

#     r_results, p_results = {}, {}

#     for key, df in sample_metrics_per_log.items():
#         col_tag = rename_map[key]
#         r_results[col_tag] = {}
#         p_results[col_tag] = {}

#         for col in df.columns:
#             if col not in measures_in_scope:
#                 continue
#             # drop missing values pairwise
#             tmp = df[["window_size", col]].dropna()
#             if len(tmp) < 2:
#                 r, p = float("nan"), float("nan")
#             else:
#                 r, p = stats.pearsonr(tmp["window_size"], tmp[col])
#             r_results[col_tag][col] = r
#             p_results[col_tag][col] = p

#     # DataFrames: measures as index, P1..P4 as columns
#     corr_df = pd.DataFrame(r_results)
#     pval_df = pd.DataFrame(p_results).reindex(corr_df.index)
#     print("Correlations:")
#     print(corr_df)
#     print()
#     print("P-values:")
#     print(pval_df)

#     return corr_df, pval_df

# corr_df, pval_df = get_correlations_for_dictionary(sample_metrics_per_log)

# # Save
# out_dir = Path("results/correlations/problematization")
# out_dir.mkdir(parents=True, exist_ok=True)
# corr_df.to_csv(out_dir / "correlations_r.csv")
# pval_df.to_csv(out_dir / "correlations_p.csv")


# # Create Latex output
# import pandas as pd
# import numpy as np
# from pathlib import Path

# def _stars(p: float) -> str:
#     if pd.isna(p):
#         return ""
#     if p < 0.001:
#         return "***"
#     if p < 0.01:
#         return "**"
#     if p < 0.05:
#         return "*"
#     return ""

# def corr_p_to_latex_stars(corr_df: pd.DataFrame, pval_df: pd.DataFrame, out_path: Path, label: str) -> None:
#     # keep P1..P4 order if present
#     cols = [c for c in ["P1", "P2", "P3", "P4"] if c in corr_df.columns]
#     corr = corr_df[cols].copy()
#     pval = pval_df[cols].copy()
#     corr, pval = corr.align(pval, join="outer", axis=0)

#     # Build display DataFrame with "+/-r***" format
#     disp = corr.copy().astype(object)
#     for c in cols:
#         out_col = []
#         for r, p in zip(corr[c], pval[c]):
#             if pd.isna(r):
#                 out_col.append("")
#             else:
#                 out_col.append(f"{r:+.2f}{_stars(p)}")
#         disp[c] = out_col

#     latex_body = disp.to_latex(
#         escape=True,
#         na_rep="",
#         index=True,
#         column_format="l" + "c"*len(cols),
#         bold_rows=False
#     )

#     wrapped = rf"""
#     \begin{{table}}[htbp]
#     \label{{label}}
#     \centering
#     \caption{{Pearson correlation ($r$) between window size and each measure.}}
#     \scriptsize
#     \setlength{{\tabcolsep}}{{6pt}}
#     \renewcommand{{\arraystretch}}{{1.15}}
#     {latex_body}
#     \vspace{{2pt}}
#     \begin{{minipage}}{{0.95\linewidth}}\footnotesize
#     Stars denote significance: $^*p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.
#     \end{{minipage}}
#     \end{{table}}
#     """

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(wrapped)

# corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table.tex"), 'correlations_not_fixed')


# ##### Fixing of problems

# def fix_p1(metrics_df):
#     fixed_metrics_df = metrics_df.copy()
    # fixed_metrics_df['Number of Events'] = metrics_df['Number of Events'] / metrics_df['Number of Traces'] # normalize by number of traces to get avg trace length
    # fixed_metrics_df['Number of Traces'] = None # do not report number of traces as this will always correlate with trace count based sample sizes (alternatively, devide by number of traces which would lead to constant 1)
    # fixed_metrics_df['Percentage of Distinct Traces'] = metrics_df['Percentage of Distinct Traces'] * metrics_df['Number of Traces'] # this gives the same as number of distinct traces
    # fixed_metrics_df['Deviation from Random'] = (
    #     1 - (1 - metrics_df['Deviation from Random'])
    #     / np.sqrt(1 - 1/(metrics_df['Number of Distinct Activities']**2))
    # ) # devision by the worst case scenario (single activity transition)
    # fixed_metrics_df['Lempel-Ziv Complexity'] = (
    #     metrics_df['Lempel-Ziv Complexity']
    #     / (
    #         metrics_df['Number of Events']
    #         / (
    #             np.log(metrics_df['Number of Events'])
    #             / np.log(metrics_df['Number of Distinct Activities'])
    #         )
    #     )
    # ) #  see Kaspar and Schuster 1987

#     return fixed_metrics_df

# def fix_p2(metrics_df, log_name):
#     samples = sample_collection[log_name]
#     adapters = ["population_inext"]
#     metrics_df_populations = compute_metrics_for_samples(samples, adapters)
#     # remove " (full coverage)" from end of column strings
#     metrics_df_populations.columns = [
#         col.replace(" (full coverage)", "") for col in metrics_df_populations.columns
#     ]
#     # align overlapping columns (replace values)
#     for col in metrics_df_populations.columns:
#         metrics_df[col] = metrics_df_populations[col]
    
#     return metrics_df


# def apply_fixes_all_metrics_dicts(sample_metrics_per_log, problems_to_fix=['p1', 'p3']):
#     fixed_metrics_per_log = {}
#     for log_name in sample_metrics_per_log:
#         fixed_metrics_df = sample_metrics_per_log[log_name].copy()
#         if 'p3' in problems_to_fix:
#             # run the inext adapter
#             fixed_metrics_df = fix_p2(fixed_metrics_df, log_name)

#         if 'p1' in problems_to_fix:
#             # fix p1
#             fixed_metrics_df = fix_p1(fixed_metrics_df)
#         # TODO fix further problems

#         fixed_metrics_per_log[log_name] = fixed_metrics_df

#     return fixed_metrics_per_log

# ## Expectation 1: convergence

# fixed_metrics_per_log = apply_fixes_all_metrics_dicts(sample_metrics_per_log, problems_to_fix=['p1'])

# # get new correlations
# corr_df, pval_df = get_correlations_for_dictionary(fixed_metrics_per_log)

# # Save
# out_dir = Path("results/correlations/problematization")
# out_dir.mkdir(parents=True, exist_ok=True)
# corr_df.to_csv(out_dir / "correlations_r_p1_fixed.csv")
# pval_df.to_csv(out_dir / "correlations_p_p1_fixed.csv")

# corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table_p1_fixed.tex"), 'correlations_p1_fixed')

# ## Expectation 2: Representation of full population

# fixed_metrics_per_log = apply_fixes_all_metrics_dicts(sample_metrics_per_log, problems_to_fix=['p1', 'p3'])

# # get new correlations
# corr_df, pval_df = get_correlations_for_dictionary(fixed_metrics_per_log)

# # Save
# out_dir = Path("results/correlations/problematization")
# out_dir.mkdir(parents=True, exist_ok=True)
# corr_df.to_csv(out_dir / "correlations_r_p3_fixed.csv")
# pval_df.to_csv(out_dir / "correlations_p_p3_fixed.csv")

# corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table_p3_fixed.tex"), 'correlations_p3_fixed')


# # Expectation 3: Confidence
# def bootstrap_ci_single_sample(
#     sample: tuple,
#     adapters,
#     compute_metrics_for_samples: Callable,
#     sampler: Callable,
#     B: int = 50,
#     alpha: float = 0.05,
#     random_state: Optional[int] = None,
# ):
#     """
#     Bootstrap metrics for a single sample of traces by resampling WITH replacement
#     inside that sample using your sampler.

#     Parameters
#     ----------
#     sample : tuple
#         (window_size, sample_id, trace_list)
#     adapters : any
#         Passed through to compute_metrics_for_samples.
#     compute_metrics_for_samples : callable
#         Function that takes [(window_size, sample_id, trace_list)]
#         and returns a DataFrame with metrics.
#     sampler : callable
#         Sampling helper:
#         sample_random_trace_sets_no_replacement_within_only(event_log, [SAMPLE_SIZE], 1, random_state)
#     B : int
#         Number of bootstrap replicates.
#     alpha : float
#         1 - confidence level (0.05 → 95% CIs).
#     random_state : int or None
#         RNG seed.

#     Returns
#     -------
#     DataFrame with one row per metric:
#         window_size, sample_id, measure, est, ci_low, ci_high, se_boot, B
#     """
#     rng = np.random.default_rng(random_state)
#     window_size, sample_id, traces = sample

#     # observed metrics
#     df_obs = compute_metrics_for_samples([sample], adapters)
#     metric_cols = [c for c in df_obs.columns if c not in ("window_size", "sample_id")]

#     # collect bootstrap values per metric
#     boot_vals = {m: np.empty(B, dtype=float) for m in metric_cols}

#     for b in range(B):
#         # Resample traces with replacement using your sampler
#         boot_samples = sampler(traces, [window_size], 1, rng.integers(0, 1_000_000))
#         boot_sample = boot_samples[0]  # (window_size, new_sample_id, resampled_traces)

#         df_b = compute_metrics_for_samples([boot_sample], adapters)
#         for m in metric_cols:
#             boot_vals[m][b] = df_b[m].iloc[0]

#     # summarize results
#     rows = []
#     for m in metric_cols:
#         est = df_obs[m].iloc[0]
#         se_boot = boot_vals[m].std(ddof=1)
#         lo = np.quantile(boot_vals[m], alpha / 2)
#         hi = np.quantile(boot_vals[m], 1 - alpha / 2)

#         rows.append({
#             "window_size": window_size,
#             "sample_id": sample_id,
#             "measure": m,
#             "est": est,
#             "ci_low": lo,
#             "ci_high": hi,
#             "se_boot": se_boot,
#             "ci_width": hi - lo,
#             "rel_halfwidth": ((hi - lo) / 2) / est if est != 0 else np.nan,
#             "B": B,
#             "ci_method": "percentile",
#         })

#     return pd.DataFrame(rows)


# def bootstrap_metrics_for_many_samples(
#     samples: Iterable[tuple],
#     adapters,
#     compute_metrics_for_samples: Callable,
#     sampler: Callable,
#     B: int = 1000,
#     alpha: float = 0.05,
#     random_state: Optional[int] = None,
# ) -> pd.DataFrame:
#     """
#     Orchestrate bootstrapping over multiple samples.
#     Calls bootstrap_ci_single_sample() for each sample.

#     Parameters
#     ----------
#     samples : iterable of tuples
#         (window_size, sample_id, trace_list)
#     adapters : any
#         Passed to compute_metrics_for_samples.
#     compute_metrics_for_samples : callable
#         Function returning DataFrame of metrics.
#     sampler : callable
#         Your trace sampler (see bootstrap_ci_single_sample).
#     B, alpha, random_state
#         Passed through to bootstrap_ci_single_sample.

#     Returns
#     -------
#     DataFrame concatenating bootstrap CI summaries
#     for all samples x metrics.
#     """
#     dfs = []
#     for sample in samples:
#         df_ci = bootstrap_ci_single_sample(
#             sample=sample,
#             adapters=adapters,
#             compute_metrics_for_samples=compute_metrics_for_samples,
#             sampler=sampler,
#             B=B,
#             alpha=alpha,
#             random_state=random_state,
#         )
#         dfs.append(df_ci)

#     return pd.concat(dfs, ignore_index=True)

# ci_df = bootstrap_metrics_for_many_samples(
#     samples=samples,
#     adapters=adapters,
#     compute_metrics_for_samples=compute_metrics_for_samples,
#     sampler=sample_random_trace_sets_no_replacement_within_only,
#     B=50,
#     alpha=0.05,
#     random_state=RANDOM_STATE,
# )

# print(ci_df.head())
