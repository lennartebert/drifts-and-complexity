"""Pytest configuration and shared fixtures for the test suite."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pytest
from pm4py.objects.log.obj import Event, EventLog, Trace
from pm4py.objects.log.util import xes as xes_util

from utils.complexity.measures.measure_store import MeasureStore
from utils.population.population_distributions import PopulationDistributions

# Central tolerance map for numeric comparisons
TOLERANCE_MAP = {
    "default": 1e-10,
    "float32": 1e-6,
    "float64": 1e-10,
    "lempel_ziv": 1e-6,  # LZ complexity can have small floating point differences
    "edit_distance": 1e-10,
    "entropy": 1e-10,
    "statistics": 1e-10,
}


@pytest.fixture(scope="session")
def seeded_rng():
    """Provide a seeded random number generator for deterministic tests."""
    return np.random.RandomState(seed=42)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def measure_store() -> MeasureStore:
    """Provide a fresh MeasureStore instance for each test."""
    return MeasureStore()


# Test data fixtures
@pytest.fixture
def empty_traces() -> List[Trace]:
    """Empty trace list for edge case testing."""
    return []


@pytest.fixture
def single_trace() -> List[Trace]:
    """Single trace for edge case testing."""
    trace = Trace()
    trace.attributes["concept:name"] = "trace_1"
    trace.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T10:00:00"}))
    trace.append(Event({"concept:name": "B", "time:timestamp": "2023-01-01T10:01:00"}))
    return [trace]


@pytest.fixture
def simple_traces() -> List[Trace]:
    """Simple trace set for basic testing."""
    traces = []

    # Trace 1: A -> B -> C
    trace1 = Trace()
    trace1.attributes["concept:name"] = "trace_1"
    trace1.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T10:00:00"}))
    trace1.append(Event({"concept:name": "B", "time:timestamp": "2023-01-01T10:01:00"}))
    trace1.append(Event({"concept:name": "C", "time:timestamp": "2023-01-01T10:02:00"}))
    traces.append(trace1)

    # Trace 2: A -> B -> D
    trace2 = Trace()
    trace2.attributes["concept:name"] = "trace_2"
    trace2.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T11:00:00"}))
    trace2.append(Event({"concept:name": "B", "time:timestamp": "2023-01-01T11:01:00"}))
    trace2.append(Event({"concept:name": "D", "time:timestamp": "2023-01-01T11:02:00"}))
    traces.append(trace2)

    # Trace 3: A -> B -> C (duplicate of trace1)
    trace3 = Trace()
    trace3.attributes["concept:name"] = "trace_3"
    trace3.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T12:00:00"}))
    trace3.append(Event({"concept:name": "B", "time:timestamp": "2023-01-01T12:01:00"}))
    trace3.append(Event({"concept:name": "C", "time:timestamp": "2023-01-01T12:02:00"}))
    traces.append(trace3)

    return traces


@pytest.fixture
def complex_traces() -> List[Trace]:
    """Complex trace set for comprehensive testing."""
    traces = []

    # Various trace patterns
    patterns = [
        ["A", "B", "C", "D"],
        ["A", "B", "C", "D"],  # duplicate
        ["A", "B", "E", "F"],
        ["A", "C", "B", "D"],
        ["A", "A", "B", "B"],  # repeated activities
        ["X", "Y", "Z"],
        ["A"],  # single activity
        ["A", "B"],  # minimal trace
    ]

    for i, pattern in enumerate(patterns):
        trace = Trace()
        trace.attributes["concept:name"] = f"trace_{i+1}"
        for j, activity in enumerate(pattern):
            trace.append(
                Event(
                    {
                        "concept:name": activity,
                        "time:timestamp": f"2023-01-01T{10+i:02d}:{j:02d}:00",
                    }
                )
            )
        traces.append(trace)

    return traces


@pytest.fixture
def standard_test_log() -> List[Trace]:
    """
    Standard test event log for golden tests and expected value validation.

    This log is carefully designed to have predictable metric values that can be
    computed by hand for verification. The log contains:
    - 6 traces total
    - 4 distinct trace variants
    - 6 distinct activities (A, B, C, D, E, F)
    - 8 distinct activity transitions
    - Various trace lengths (1, 2, 3, 4 events)
    """
    traces = []

    # Trace 1: A -> B -> C (3 events)
    trace1 = Trace()
    trace1.attributes["concept:name"] = "trace_1"
    trace1.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T10:00:00"}))
    trace1.append(Event({"concept:name": "B", "time:timestamp": "2023-01-01T10:01:00"}))
    trace1.append(Event({"concept:name": "C", "time:timestamp": "2023-01-01T10:02:00"}))
    traces.append(trace1)

    # Trace 2: A -> B -> D (3 events) - different from trace1
    trace2 = Trace()
    trace2.attributes["concept:name"] = "trace_2"
    trace2.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T11:00:00"}))
    trace2.append(Event({"concept:name": "B", "time:timestamp": "2023-01-01T11:01:00"}))
    trace2.append(Event({"concept:name": "D", "time:timestamp": "2023-01-01T11:02:00"}))
    traces.append(trace2)

    # Trace 3: A -> B -> C (3 events) - duplicate of trace1
    trace3 = Trace()
    trace3.attributes["concept:name"] = "trace_3"
    trace3.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T12:00:00"}))
    trace3.append(Event({"concept:name": "B", "time:timestamp": "2023-01-01T12:01:00"}))
    trace3.append(Event({"concept:name": "C", "time:timestamp": "2023-01-01T12:02:00"}))
    traces.append(trace3)

    # Trace 4: E -> F (2 events)
    trace4 = Trace()
    trace4.attributes["concept:name"] = "trace_4"
    trace4.append(Event({"concept:name": "E", "time:timestamp": "2023-01-01T13:00:00"}))
    trace4.append(Event({"concept:name": "F", "time:timestamp": "2023-01-01T13:01:00"}))
    traces.append(trace4)

    # Trace 5: A -> B -> C -> D (4 events)
    trace5 = Trace()
    trace5.attributes["concept:name"] = "trace_5"
    trace5.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T14:00:00"}))
    trace5.append(Event({"concept:name": "B", "time:timestamp": "2023-01-01T14:01:00"}))
    trace5.append(Event({"concept:name": "C", "time:timestamp": "2023-01-01T14:02:00"}))
    trace5.append(Event({"concept:name": "D", "time:timestamp": "2023-01-01T14:03:00"}))
    traces.append(trace5)

    # Trace 6: A (1 event)
    trace6 = Trace()
    trace6.attributes["concept:name"] = "trace_6"
    trace6.append(Event({"concept:name": "A", "time:timestamp": "2023-01-01T15:00:00"}))
    traces.append(trace6)

    return traces


@pytest.fixture
def population_distribution() -> PopulationDistributions:
    """Mock population distribution for distribution-based metrics."""

    # This is a simplified mock - in practice, PopulationDistributions would be more complex
    class MockPopulationDistributions:
        def __init__(self):
            self.trace_variants = MockCount(5)  # 5 distinct trace variants
            self.activities = MockCount(8)  # 8 distinct activities
            self.activity_transitions = MockCount(12)  # 12 distinct transitions
            self.dfg_edges = MockCount(
                12
            )  # 12 distinct DFG edges (same as transitions)

    class MockCount:
        def __init__(self, count):
            self.count = count

    return MockPopulationDistributions()


@pytest.fixture
def empty_population_distribution() -> PopulationDistributions:
    """Empty population distribution for edge case testing."""

    class MockEmptyPopulationDistributions:
        def __init__(self):
            self.trace_variants = MockCount(0)
            self.activities = MockCount(0)
            self.activity_transitions = MockCount(0)
            self.dfg_edges = MockCount(0)

    class MockCount:
        def __init__(self, count):
            self.count = count

    return MockEmptyPopulationDistributions()


# Utility functions for tests
def assert_close(
    a: float,
    b: float,
    tolerance_type: str = "default",
    rtol: float = None,
    atol: float = None,
) -> None:
    """Assert that two values are close within tolerance."""
    if rtol is None and atol is None:
        atol = TOLERANCE_MAP.get(tolerance_type, TOLERANCE_MAP["default"])
        rtol = 0

    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def create_trace_from_activities(activities: List[str], trace_id: str = None) -> Trace:
    """Create a PM4Py Trace from a list of activity names."""
    trace = Trace()
    trace.attributes["concept:name"] = trace_id or f"trace_{len(activities)}"

    for i, activity in enumerate(activities):
        event = Event(
            {"concept:name": activity, "time:timestamp": f"2023-01-01T10:{i:02d}:00"}
        )
        trace.append(event)

    return trace


def create_traces_from_patterns(patterns: List[List[str]]) -> List[Trace]:
    """Create multiple traces from activity patterns."""
    return [
        create_trace_from_activities(pattern, f"trace_{i}")
        for i, pattern in enumerate(patterns)
    ]
