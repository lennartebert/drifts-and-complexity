"""Tests ensuring NaivePopulationExtractor keeps distinct trace counts exact."""

from __future__ import annotations

from pathlib import Path

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import Event, EventLog, Trace

from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
)
from utils.windowing.window import Window


def _make_trace(name: str, activities: list[str]) -> Trace:
    trace = Trace()
    trace.attributes["concept:name"] = name
    for idx, act in enumerate(activities):
        trace.append(
            Event(
                {"concept:name": act, "time:timestamp": f"2024-01-01T00:{idx:02d}:00"}
            )
        )
    return trace


def _build_test_log() -> EventLog:
    log = EventLog()
    # Variant 1 repeated twice
    log.append(_make_trace("t1", ["A", "B", "C"]))
    log.append(_make_trace("t2", ["A", "B", "C"]))
    # Variant 2
    log.append(_make_trace("t3", ["A", "C", "D"]))
    # Variant 3
    log.append(_make_trace("t4", ["X", "Y"]))
    return log


def test_naive_population_preserves_distinct_trace_count():
    """Ensure naive extractor does not inflate number of distinct traces."""
    log = _build_test_log()

    window = Window(id="full", size=len(log), traces=list(log))

    extractor = NaivePopulationExtractor()
    extractor.apply(window)

    # Population distribution count should match observed distinct variants (3).
    pd = window.population_distributions
    assert pd is not None
    assert pd.trace_variants.population_count == 3

    adapter = LocalMetricsAdapter()
    store, _ = adapter.compute_measures_for_window(
        window,
        include_metrics=["Number of Distinct Traces"],
    )

    assert store.has("Number of Distinct Traces")
    measure = store.get("Number of Distinct Traces")
    assert measure is not None
    assert measure.value == 3


def test_credit_s_has_expected_distinct_traces():
    """Verify CREDIT_S log contains the expected number of trace variants."""
    log_path = Path(
        "data/synthetic/BPM fundamentals_7.10_credit application process with rework/simulation_logs.xes.gz"
    )
    log = xes_importer.apply(str(log_path))

    # Manual unique trace count using tuples of concept names
    unique_variants = {tuple(event["concept:name"] for event in trace) for trace in log}
    observed_distinct = len(unique_variants)
    assert observed_distinct == 20

    window = Window(id="credit_s", size=len(log), traces=list(log))
    NaivePopulationExtractor().apply(window)
    pd = window.population_distributions
    assert pd is not None
    assert pd.trace_variants.population_count == observed_distinct
