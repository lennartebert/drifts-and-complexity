#### Complexity Metric Problematization - Examples

import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pm4py.objects.log.obj import Event, EventLog, Trace

from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import \
    INextBootstrapSampler
from utils.complexity.metrics_adapters.local_metrics_adapter import \
    LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import \
    VidgofMetricsAdapter
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS
from utils.pipeline.compute import (compute_metrics_and_CIs,
                                    run_metrics_over_samples)
from utils.population.extractors.chao1_population_extractor import \
    Chao1PopulationExtractor
from utils.population.extractors.naive_population_extractor import \
    NaivePopulationExtractor
from utils.windowing.window import Window

sorted_metrics = {
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

from complexity_sample_size_correlation_analysis import \
    sample_random_windows_no_replacement_within_only


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


SAMPLES_PER_SIZE = 200 # do 200 samples in final computation
RANDOM_STATE = 1
NUM_TRACES_PER_LOG = 10000 # do 10000 in final computation
SIZES = range(50, 501, 50)

# set defaults
default_population_extractor = NaivePopulationExtractor()
default_metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
default_bootstrap_sampler = None
default_normalizers = None


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
window_samples = sample_random_windows_no_replacement_within_only(event_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE) # returns (sample_size, sample_id, window)
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
window_samples = sample_random_windows_no_replacement_within_only(event_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE) # returns (sample_size, sample_id, window)
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
window_samples = sample_random_windows_no_replacement_within_only(event_log, SIZES, SAMPLES_PER_SIZE, RANDOM_STATE) # returns (sample_size, sample_id, window)
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


def get_correlations_for_dictionary(sample_metrics_per_log):
    # create a correlation analysis for all measures
    from pathlib import Path

    import pandas as pd
    from scipy import stats

    # mapping of your dict keys to desired column names
    rename_map = {
        "strict_monotone_growth": "Log 1",
        "rare_occurrences": "Log 2",
        "variance": "Log 3",
    }

    r_results, p_results = {}, {}

    for key, df in sample_metrics_per_log.items():
        col_tag = rename_map[key]
        r_results[col_tag] = {}
        p_results[col_tag] = {}

        for col in df.columns:
            if col not in sorted_metrics:
                continue
            # drop missing values pairwise
            tmp = df[["sample_size", col]].dropna()
            if len(tmp) < 2:
                r, p = float("nan"), float("nan")
            else:
                r, p = stats.pearsonr(tmp["sample_size"], tmp[col])
            r_results[col_tag][col] = r
            p_results[col_tag][col] = p

    # DataFrames: measures as index, P1..P4 as columns
    corr_df = pd.DataFrame(r_results)
    pval_df = pd.DataFrame(p_results).reindex(corr_df.index)
    print("Correlations:")
    print(corr_df)
    print()
    print("P-values:")
    print(pval_df)

    return corr_df, pval_df

measures_per_log = {log_name: measures_df for log_name, (measures_df, _, _) in results_per_log}

corr_df, pval_df = get_correlations_for_dictionary(measures_per_log)

# Save
out_dir = Path("results/correlations/problematization")
out_dir.mkdir(parents=True, exist_ok=True)
corr_df.to_csv(out_dir / "correlations_r.csv")
pval_df.to_csv(out_dir / "correlations_p.csv")


# Create Latex output
def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def corr_p_to_latex_stars(corr_df: pd.DataFrame, pval_df: pd.DataFrame, out_path: Path, label: str) -> None:
    # keep P1..P4 order if present
    cols = [c for c in ["P1", "P2", "P3", "P4"] if c in corr_df.columns]
    corr = corr_df[cols].copy()
    pval = pval_df[cols].copy()
    corr, pval = corr.align(pval, join="outer", axis=0)

    # Build display DataFrame with "+/-r***" format
    disp = corr.copy().astype(object)
    for c in cols:
        out_col = []
        for r, p in zip(corr[c], pval[c]):
            if pd.isna(r):
                out_col.append("")
            else:
                out_col.append(f"{r:+.2f}{_stars(p)}")
        disp[c] = out_col

    latex_body = disp.to_latex(
        escape=True,
        na_rep="",
        index=True,
        column_format="l" + "c"*len(cols),
        bold_rows=False
    )

    wrapped = rf"""
    \begin{{table}}[htbp]
    \label{{label}}
    \centering
    \caption{{Pearson correlation ($r$) between window size and each measure.}}
    \scriptsize
    \setlength{{\tabcolsep}}{{6pt}}
    \renewcommand{{\arraystretch}}{{1.15}}
    {latex_body}
    \vspace{{2pt}}
    \begin{{minipage}}{{0.95\linewidth}}\footnotesize
    Stars denote significance: $^*p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.
    \end{{minipage}}
    \end{{table}}
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(wrapped)

corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table.tex"), 'correlations_not_fixed')


##### Fixing of problems

def apply_fixes(sample_collection, problems_to_fix=['p1', 'p2', 'p3']):
    fixed_metrics_per_log = {}
    for log_name, window_samples in sample_collection.items():
        population_extractor = default_population_extractor,
        metric_adapters = default_metric_adapters,
        bootstrap_sampler = default_bootstrap_sampler,
        normalizers = default_normalizers,

        if 'p1' in problems_to_fix:
            normalizers = DEFAULT_NORMALIZERS
        
        if 'p2' in problems_to_fix:
            population_extractor = Chao1PopulationExtractor()
        
        if 'p3' in problems_to_fix:
            bootstrap_sampler = INextBootstrapSampler(B=200)
        
        measures_df, ci_low_df, ci_high_df = run_metrics_over_samples(window_samples,
            population_extractor = population_extractor,
            metric_adapters = metric_adapters,
            bootstrap_sampler = bootstrap_sampler,
            normalizers = normalizers,
            sorted_metrics=sorted_metrics)
        
        fixed_metrics_per_log[log_name] = (measures_df, ci_low_df, ci_high_df)


# apply the fix to p1
results_per_log_p1_fixed = apply_fixes(sample_collection, problems_to_fix=['p1'])

measures_per_log = {log_name: measures_df for log_name, (measures_df, _, _) in results_per_log_p1_fixed}

# get new correlations
corr_df, pval_df = get_correlations_for_dictionary(measures_per_log)

# Save
out_dir = Path("results/correlations/problematization")
out_dir.mkdir(parents=True, exist_ok=True)
corr_df.to_csv(out_dir / "correlations_r_p1_fixed.csv")
pval_df.to_csv(out_dir / "correlations_p_p1_fixed.csv")

corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table_p1_fixed.tex"), 'correlations_p1_fixed')

# apply the fix to p1 and p2 and p3
results_per_log_p1_p2_p3_fixed = apply_fixes(sample_collection, problems_to_fix=['p1', 'p2', 'p3'])

measures_per_log = {log_name: measures_df for log_name, (measures_df, _, _) in results_per_log_p1_p2_p3_fixed}

# get new correlations
corr_df, pval_df = get_correlations_for_dictionary(measures_per_log)

# Save
out_dir = Path("results/correlations/problematization")
out_dir.mkdir(parents=True, exist_ok=True)
corr_df.to_csv(out_dir / "correlations_r_p1_p2_fixed.csv")
pval_df.to_csv(out_dir / "correlations_p_p1_p2_fixed.csv")

corr_p_to_latex_stars(corr_df, pval_df, Path("results/correlations/problematization/correlations_table_p1_p2_fixed.tex"), 'correlations_p1_p2_fixed')

# get the variance after applying fixes for p1 and p2 (can use the same results as before)

import os
from typing import Iterable, Tuple
import math
import pandas as pd
import matplotlib.pyplot as plt


def _get_measure_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("sample_size", "sample_id")]


def plot_aggregated_measures_cis(
    measures_df: pd.DataFrame,
    ci_low_df: pd.DataFrame,
    ci_high_df: pd.DataFrame,
    out_path: str,
    agg: str = "mean",
    title: str | None = None,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 3.2),
) -> str:
    """
    Create a multi-panel plot (one subplot per measure) that shows, over sample_size:
      - aggregated center line (mean or median) of the measure values
      - aggregated CI low
      - aggregated CI high

    Parameters
    ----------
    measures_df : DataFrame with columns ['sample_size', 'sample_id', <measure columns...>]
    ci_low_df   : DataFrame with same shape/columns as measures_df for CI lows
    ci_high_df  : DataFrame with same shape/columns as measures_df for CI highs
    out_path    : File path to save the figure (PNG, PDF, etc.). Parent dir will be created.
    agg         : 'mean' or 'median' — which center to plot
    title       : Optional figure title
    ncols       : Number of subplot columns
    figsize_per_panel : Size per subplot; overall size is scaled by number of panels.

    Returns
    -------
    str : The path where the figure was saved.
    """
    agg = agg.lower()
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'.")

    # Ensure columns align and pick measure columns
    measure_cols = _get_measure_columns(measures_df)
    # guard: intersect with CI frames in case of drift
    measure_cols = [c for c in measure_cols if c in ci_low_df.columns and c in ci_high_df.columns]

    # Group by sample_size and aggregate
    group = measures_df.groupby("sample_size", as_index=True)
    group_low = ci_low_df.groupby("sample_size", as_index=True)
    group_high = ci_high_df.groupby("sample_size", as_index=True)

    if agg == "mean":
        center = group[measure_cols].mean()
        low = group_low[measure_cols].mean()
        high = group_high[measure_cols].mean()
    else:  # median
        center = group[measure_cols].median()
        low = group_low[measure_cols].median()
        high = group_high[measure_cols].median()

    # Sort by sample_size to ensure monotone x
    center = center.sort_index()
    low = low.reindex(center.index)
    high = high.reindex(center.index)

    # Figure layout
    n_measures = len(measure_cols)
    nrows = math.ceil(n_measures / ncols)
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    x = center.index.values

    for i, m in enumerate(measure_cols):
        ax = axes_flat[i]
        # center line
        ax.plot(x, center[m].values, label=f"{agg.capitalize()}", linewidth=2)
        # CI low/high as lines
        ax.plot(x, low[m].values, label="CI low", linestyle="--")
        ax.plot(x, high[m].values, label="CI high", linestyle="--")

        ax.set_title(m, fontsize=10)
        ax.set_xlabel("Sample size")
        ax.set_xlim(x.min(), x.max())
        ax.grid(True, alpha=0.25)

        # Y label only on left-most column
        if i % ncols == 0:
            ax.set_ylabel("Value")

        # Keep legends tight
        ax.legend(fontsize=8, loc="best")

    # Hide any unused axes
    for j in range(n_measures, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# One plot per log. Create both mean and median versions (two files).
plots_dir = "plots_measures_cis"
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