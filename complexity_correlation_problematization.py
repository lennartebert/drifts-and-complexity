#### Complexity Metric Problematization - Examples

import datetime
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pm4py.objects.log.obj import Event, EventLog, Trace

from complexity_sample_size_correlation_analysis import (
    compute_metrics_for_samples, sample_random_traces_with_replacement)


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


## P1 Monotone growth
# The complexity metric grows with increasing window size, although there is no variance, rare species or loops.

# Create a trace variant A-B-C-D-E
variant = ["A", "B", "C", "D", "E"]

# Build EventLog with 10,000 identical traces
event_log = EventLog([make_trace(variant, id) for id in range(10000)])

sizes = range(50, 501, 50)
samples_per_size = 200
random_state = 1
samples = sample_random_traces_with_replacement(event_log, sizes, samples_per_size, random_state)
adapters = ["vidgof_sample"]
df_metrics = compute_metrics_for_samples(samples, adapters)
print(df_metrics)

# plot Lempel-Ziv complexity
fig, ax = plt.subplots(figsize=(6, 4))
df_metrics.boxplot(
    column="Lempel-Ziv complexity",
    by="window_size",
    ax=ax,
    grid=False
)
ax.set_xlabel("Window Size")
ax.set_ylabel("Lempel-Ziv Complexity")
ax.set_title("Lempel-Ziv Complexity vs Window Size (Single Variant Log)")

# Ensure output directory exists
out_path = Path("results/correlations/problematization/p1.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Save figure
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

out_path

## P2 Variance
# The complexity metric fluctuates at small sample sizes due to the law of small numbers. Only at higher sample window sizes, the metric becomes asymptotic.

# Create a trace variant A-B-C-D-E
variant_a = ["A", "B", "C", "D", "E"] # length: 5
variant_b = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] # length: 10]


# Build EventLog with 10,000 traces of two variants
traces = []
for id in range(10000):
    if id % 2 == 0:
        traces.append(make_trace(variant_a, id))
    else:
        traces.append(make_trace(variant_b, id))

event_log = EventLog(traces)

sizes = range(50, 501, 50)
samples_per_size = 200
random_state = 1
samples = sample_random_traces_with_replacement(event_log, sizes, samples_per_size, random_state)
adapters = ["vidgof_sample"]
df_metrics = compute_metrics_for_samples(samples, adapters)
print(df_metrics)

# plot Trace length
fig, ax = plt.subplots(figsize=(6, 4))
df_metrics.boxplot(
    column="Trace length avg",
    by="window_size",
    ax=ax,
    grid=False
)
ax.set_xlabel("Window Size")
ax.set_ylabel("Avg. Trace Length")
ax.set_title("Avg. Trace Length vs Window Size (Two Variant Log)")

# Ensure output directory exists
out_path = Path("results/correlations/problematization/p2.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Save figure
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

out_path

## P3 Rare occurrences
# The complexity metric under-estimates rarely occurring behavior due to skewed occurrence distributions.

# Create trace variants
frequent_variant = ["A"]
once_occuring_variants = [[f"B_{i}"] for i in range (0, 10)] # 10 variants that occur only once

# Build EventLog with 10,000 traces of two variants
traces = []
for id in range(10000):
    if id < 9990:
        traces.append(make_trace(frequent_variant, id))
    else:
        traces.append(make_trace(once_occuring_variants[id - 9990], id))

# shuffle traces (should not matter due to random sampling but just to be sure)
random.seed(1)
random.shuffle(traces)

event_log = EventLog(traces)

sizes = range(100, 1001, 100)
samples_per_size = 200
random_state = 1
samples = sample_random_traces_with_replacement(event_log, sizes, samples_per_size, random_state)
adapters = ["vidgof_sample"]
df_metrics = compute_metrics_for_samples(samples, adapters)
print(df_metrics)

# Boxplot Variety vs Window Size
fig, ax = plt.subplots(figsize=(6, 4))
df_metrics.boxplot(
    column="Variety",
    by="window_size",
    ax=ax,
    grid=False
)

ax.set_xlabel("Window Size")
ax.set_ylabel("Variety (number of activities)")
ax.set_title("Variety vs Window Size (Skewed Variant Log)")
plt.suptitle("")  # remove automatic 'Variety by window_size' title from pandas

# Ensure output directory exists
out_path = Path("results/correlations/problematization/p3.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Save figure
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

out_path

## P4 Infinite support
# Looped models cause unbounded number of distinct variants

from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator

# Path to the BPMN file exported from BPMN.io (A->B->C with self-loop on B)
bpmn_path = r"data\synthetic\simple_loop\simple_loop.bpmn"
# Import BPMN
bpmn_graph = bpmn_importer.apply(bpmn_path)

# Convert BPMN -> (Petri net, initial marking, final marking)
net, im, fm = bpmn_converter.apply(bpmn_graph)


event_log = simulator.apply(
    net,
    im,
    fm,
    parameters = {
        "no_traces": 10000,
    },
    variant=simulator.Variants.BASIC_PLAYOUT,
)

# quick peek
for i, trace in enumerate(event_log[:5], 1):
    print(f"Trace {i}:", [ev["concept:name"] for ev in trace])

sizes = range(50, 501, 50)
samples_per_size = 200
random_state = 1
samples = sample_random_traces_with_replacement(event_log, sizes, samples_per_size, random_state)
adapters = ["vidgof_sample"]
df_metrics = compute_metrics_for_samples(samples, adapters)
print(df_metrics)

# Boxplot Distinct traces vs Window Size
fig, ax = plt.subplots(figsize=(6, 4))
df_metrics.boxplot(
    column="Distinct traces",
    by="window_size",
    ax=ax,
    grid=False
)

ax.set_xlabel("Window Size")
ax.set_ylabel("Distinct traces")
ax.set_title("Distinct traces vs Window Size (Looping Log)")
plt.suptitle("")

# Ensure output directory exists
out_path = Path("results/correlations/problematization/p4.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Save figure
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

out_path