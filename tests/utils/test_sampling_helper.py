"""Unit tests for sampling_helper module."""

from __future__ import annotations

import pytest
from pm4py.objects.log.obj import Event, Trace

from utils.sampling_helper import (
    sample_consecutive_trace_windows_with_replacement,
    sample_consecutive_traces,
    sample_random_trace_sets_no_replacement_global,
    sample_random_trace_sets_no_replacement_within_only,
    sample_random_traces,
    sample_random_traces_with_replacement,
    sample_random_windows_no_replacement_global,
    sample_random_windows_no_replacement_within_only,
    sample_random_windows_with_replacement,
)


@pytest.fixture
def test_traces():
    """Create a test log with 10 traces."""
    traces = []
    for i in range(10):
        trace = Trace()
        trace.attributes["concept:name"] = f"trace_{i}"
        trace.append(
            Event({"concept:name": "A", "time:timestamp": f"2023-01-01T10:{i:02d}:00"})
        )
        trace.append(
            Event({"concept:name": "B", "time:timestamp": f"2023-01-01T10:{i:02d}:01"})
        )
        traces.append(trace)
    return traces


def test_sample_random_traces_with_replacement(test_traces):
    """Test random sampling with replacement."""
    result = sample_random_traces_with_replacement(
        event_log=test_traces, sizes=[3, 5], samples_per_size=2, random_state=42
    )
    assert len(result) == 4  # 2 sizes * 2 samples
    for window_size, sample_id, trace_list in result:
        assert window_size in [3, 5]
        assert sample_id in ["0", "1"]
        assert len(trace_list) == window_size
        # With replacement, duplicates are allowed
        assert all(isinstance(t, Trace) for t in trace_list)


def test_sample_random_trace_sets_no_replacement_within_only(test_traces):
    """Test random sampling without replacement within samples."""
    result = sample_random_trace_sets_no_replacement_within_only(
        event_log=test_traces, sizes=[3, 5], samples_per_size=2, random_state=42
    )
    assert len(result) == 4  # 2 sizes * 2 samples
    for window_size, sample_id, trace_list in result:
        assert window_size in [3, 5]
        assert sample_id in ["0", "1"]
        assert len(trace_list) == window_size
        # No duplicates within a sample
        trace_ids = [t.attributes["concept:name"] for t in trace_list]
        assert len(trace_ids) == len(set(trace_ids))


def test_sample_random_trace_sets_no_replacement_global(test_traces):
    """Test random sampling without replacement globally."""
    result = sample_random_trace_sets_no_replacement_global(
        event_log=test_traces, sizes=[3, 2], samples_per_size=2, random_state=42
    )
    assert len(result) == 4  # 2 sizes * 2 samples
    # Check that all traces are unique across all samples
    all_trace_ids = []
    for _, _, trace_list in result:
        trace_ids = [t.attributes["concept:name"] for t in trace_list]
        all_trace_ids.extend(trace_ids)
    # Total traces used: 2*(3+2) = 10, which equals log size
    assert len(all_trace_ids) == len(set(all_trace_ids))


def test_sample_random_traces_policy_within_and_across(test_traces):
    """Test sample_random_traces with within_and_across policy."""
    result = sample_random_traces(
        event_log=test_traces,
        sizes=[3],
        samples_per_size=2,
        policy="within_and_across",
        random_state=42,
    )
    assert len(result) == 2
    # With replacement, duplicates are allowed
    for _, _, trace_list in result:
        assert len(trace_list) == 3


def test_sample_random_traces_policy_within_only(test_traces):
    """Test sample_random_traces with within_only policy."""
    result = sample_random_traces(
        event_log=test_traces,
        sizes=[3],
        samples_per_size=2,
        policy="within_only",
        random_state=42,
    )
    assert len(result) == 2
    for _, _, trace_list in result:
        assert len(trace_list) == 3
        # No duplicates within sample
        trace_ids = [t.attributes["concept:name"] for t in trace_list]
        assert len(trace_ids) == len(set(trace_ids))


def test_sample_random_traces_policy_none(test_traces):
    """Test sample_random_traces with none policy."""
    result = sample_random_traces(
        event_log=test_traces,
        sizes=[3, 2],
        samples_per_size=1,
        policy="none",
        random_state=42,
    )
    assert len(result) == 2
    # All traces should be unique
    all_trace_ids = []
    for _, _, trace_list in result:
        trace_ids = [t.attributes["concept:name"] for t in trace_list]
        all_trace_ids.extend(trace_ids)
    assert len(all_trace_ids) == len(set(all_trace_ids))


def test_sample_random_windows_with_replacement(test_traces):
    """Test random window sampling with replacement."""
    result = sample_random_windows_with_replacement(
        event_log=test_traces, sizes=[3], samples_per_size=2, random_state=42
    )
    assert len(result) == 2
    for window_size, sample_id, window in result:
        assert window_size == 3
        assert sample_id in ["0", "1"]
        assert window.size == 3
        assert len(window.traces) == 3
        assert window.id == sample_id


def test_sample_random_windows_no_replacement_within_only(test_traces):
    """Test random window sampling without replacement within windows."""
    result = sample_random_windows_no_replacement_within_only(
        event_log=test_traces, sizes=[3], samples_per_size=2, random_state=42
    )
    assert len(result) == 2
    for window_size, sample_id, window in result:
        assert window_size == 3
        assert window.size == 3
        # No duplicates within window
        trace_ids = [t.attributes["concept:name"] for t in window.traces]
        assert len(trace_ids) == len(set(trace_ids))


def test_sample_random_windows_no_replacement_global(test_traces):
    """Test random window sampling without replacement globally."""
    result = sample_random_windows_no_replacement_global(
        event_log=test_traces, sizes=[3, 2], samples_per_size=1, random_state=42
    )
    assert len(result) == 2
    # All traces should be unique across windows
    all_trace_ids = []
    for _, _, window in result:
        trace_ids = [t.attributes["concept:name"] for t in window.traces]
        all_trace_ids.extend(trace_ids)
    assert len(all_trace_ids) == len(set(all_trace_ids))


def test_sample_consecutive_traces(test_traces):
    """Test consecutive trace sampling."""
    result = sample_consecutive_traces(
        event_log=test_traces, sizes=[3, 4], samples_per_size=2, random_state=42
    )
    assert len(result) == 4  # 2 sizes * 2 samples
    for window_size, sample_id, trace_list in result:
        assert window_size in [3, 4]
        assert sample_id in ["0", "1"]
        assert len(trace_list) == window_size
        # Traces should be consecutive (check indices)
        if len(trace_list) > 1:
            first_idx = test_traces.index(trace_list[0])
            for i, trace in enumerate(trace_list[1:], 1):
                expected_idx = first_idx + i
                assert test_traces[expected_idx] == trace


def test_sample_consecutive_trace_windows_with_replacement(test_traces):
    """Test consecutive window sampling with replacement."""
    result = sample_consecutive_trace_windows_with_replacement(
        event_log=test_traces, sizes=[3], samples_per_size=2, random_state=42
    )
    assert len(result) == 2
    for window_size, sample_id, window in result:
        assert window_size == 3
        assert sample_id in ["0", "1"]
        assert window.size == 3
        assert len(window.traces) == 3
        # Traces should be consecutive
        if len(window.traces) > 1:
            first_idx = test_traces.index(window.traces[0])
            for i, trace in enumerate(window.traces[1:], 1):
                expected_idx = first_idx + i
                assert test_traces[expected_idx] == trace


def test_sample_consecutive_traces_size_too_large(test_traces):
    """Test that consecutive sampling raises error for size > log length."""
    with pytest.raises(ValueError, match="exceeds number of traces"):
        sample_consecutive_traces(
            event_log=test_traces, sizes=[15], samples_per_size=1, random_state=42
        )


def test_sample_consecutive_trace_windows_size_too_large(test_traces):
    """Test that consecutive window sampling raises error for size > log length."""
    with pytest.raises(ValueError, match="exceeds number of traces"):
        sample_consecutive_trace_windows_with_replacement(
            event_log=test_traces, sizes=[15], samples_per_size=1, random_state=42
        )


def test_sample_random_traces_empty_log():
    """Test sampling from empty log."""
    result = sample_random_traces(
        event_log=[], sizes=[3], samples_per_size=2, random_state=42
    )
    assert result == []


def test_sample_consecutive_traces_empty_log():
    """Test consecutive sampling from empty log."""
    result = sample_consecutive_traces(
        event_log=[], sizes=[3], samples_per_size=2, random_state=42
    )
    assert result == []


def test_sample_random_traces_zero_samples_per_size(test_traces):
    """Test sampling with zero samples_per_size."""
    result = sample_random_traces(
        event_log=test_traces, sizes=[3], samples_per_size=0, random_state=42
    )
    assert result == []


def test_sample_consecutive_traces_zero_samples_per_size(test_traces):
    """Test consecutive sampling with zero samples_per_size."""
    result = sample_consecutive_traces(
        event_log=test_traces, sizes=[3], samples_per_size=0, random_state=42
    )
    assert result == []


def test_sample_random_traces_invalid_policy(test_traces):
    """Test that invalid policy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown policy"):
        sample_random_traces(
            event_log=test_traces,
            sizes=[3],
            samples_per_size=1,
            policy="invalid",  # type: ignore[arg-type]
            random_state=42,
        )
