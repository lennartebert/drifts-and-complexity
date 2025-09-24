from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

# Adapter uses the registry + orchestrator internally
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.population.extractors.naive_population_extractor import NaivePopulationExtractor
from utils.windowing.window import Window
from pm4py.objects.log.obj import Event, EventLog, Trace



# --- Helper: pretty print a subset of measures using store.get_value if available ---

def print_measures(store: Any, names: Iterable[str]) -> None:
    print("  measures:")
    for name in names:
        val = None
        if hasattr(store, "get_value"):
            try:
                val = store.get_value(name)
            except Exception:
                val = None
        # fallback if there's no get_value() (shouldn't happen in your codebase)
        if val is None and hasattr(store, "to_dict"):
            try:
                d = store.to_dict()
                v = d.get(name)
                val = getattr(v, "value", None) if v is not None else None
            except Exception:
                pass
        print(f"    - {name}: {val}")

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

traces = [
        [{"concept:name": "A"}, {"concept:name": "B"}, {"concept:name": "C"}],
        [{"concept:name": "A"}, {"concept:name": "C"}],
        [{"concept:name": "B"}, {"concept:name": "C"}],
        [{"concept:name": "A"}, {"concept:name": "B"}],
    ]

# create event log from traces
event_log = EventLog(traces)

# create windows from event log
window = Window('1', len(event_log), event_log)

adapter = LocalMetricsAdapter(strict=True, prefer="auto")

print("=== TRACE-ONLY (no population distribution) ===")
store1, info1 = adapter.compute_measures_for_window(window, include=sorted_metrics)
print("  skipped :", info1.get("skipped"))
print_measures(store1, sorted_metrics)

# get the population distributions
pop_extractor = NaivePopulationExtractor()
pop_extractor.apply(window)

adapter = LocalMetricsAdapter(strict=True, prefer="auto")

print("=== TRACE-ONLY (no population distribution) ===")
store1, info1 = adapter.compute_measures_for_window(window, include=sorted_metrics)
print("  skipped :", info1.get("skipped"))
print_measures(store1, sorted_metrics)

