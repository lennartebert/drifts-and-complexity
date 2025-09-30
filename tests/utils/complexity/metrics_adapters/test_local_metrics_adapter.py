"""Tests for the LocalMetricsAdapter."""

from typing import Any, Dict, List

import pytest

from tests.conftest import (
    complex_traces,
    empty_population_distribution,
    empty_traces,
    population_distribution,
    simple_traces,
    single_trace,
    standard_test_log,
)
from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.constants import ALL_METRIC_NAMES
from utils.windowing.window import Window


class TestLocalMetricsAdapter:
    """Test the LocalMetricsAdapter functionality."""

    @pytest.fixture
    def adapter(self) -> LocalMetricsAdapter:
        """Create a LocalMetricsAdapter instance."""
        return LocalMetricsAdapter(strict=True, prefer="trace")

    @pytest.fixture
    def adapter_auto(self) -> LocalMetricsAdapter:
        """Create a LocalMetricsAdapter with auto preference."""
        return LocalMetricsAdapter(strict=True, prefer="auto")

    @pytest.fixture
    def adapter_distribution(self) -> LocalMetricsAdapter:
        """Create a LocalMetricsAdapter with distribution preference."""
        return LocalMetricsAdapter(strict=True, prefer="distribution")

    def test_adapter_initialization(self):
        """Test adapter initialization with different parameters."""
        # Test default parameters
        adapter = LocalMetricsAdapter()
        assert adapter.name == "local"
        assert adapter._strict is True
        assert adapter._prefer == "auto"

        # Test custom parameters
        adapter = LocalMetricsAdapter(strict=False, prefer="trace")
        assert adapter._strict is False
        assert adapter._prefer == "trace"

    def test_available_metrics(self, adapter):
        """Test that available_metrics returns the expected metrics."""
        metrics = adapter.available_metrics()

        # Should return all registered metrics
        assert len(metrics) > 0
        assert isinstance(metrics, list)

        # Should include most metrics from our constants (excluding entropy metrics from vidgof)
        entropy_metrics = [
            "Sequence Entropy",
            "Normalized Sequence Entropy",
            "Variant Entropy",
            "Normalized Variant Entropy",
        ]
        for metric_name in ALL_METRIC_NAMES:
            if (
                metric_name not in entropy_metrics
            ):  # Entropy metrics are from vidgof adapter
                assert (
                    metric_name in metrics
                ), f"Expected metric {metric_name} not found"

    def test_compute_measures_for_window_empty_traces(self, adapter, empty_traces):
        """Test computing measures for empty traces."""
        window = Window(
            id="empty_window",
            size=len(empty_traces),
            traces=empty_traces,
            population_distributions=None,
        )

        # Empty traces may cause division by zero errors in some metrics
        # Use lenient mode to handle this gracefully
        adapter_lenient = LocalMetricsAdapter(strict=False)
        store, info = adapter_lenient.compute_measures_for_window(window)

        # Should return store (may be empty or have some computed metrics)
        assert isinstance(store, MeasureStore)
        assert info["adapter"] == "local"
        assert info["strict"] is False
        assert info["prefer"] == "auto"  # Default is auto

    def test_compute_measures_for_window_single_trace(self, adapter, single_trace):
        """Test computing measures for single trace."""
        window = Window(
            id="single_window",
            size=len(single_trace),
            traces=single_trace,
            population_distributions=None,
        )

        store, info = adapter.compute_measures_for_window(window)

        # Should compute some metrics
        assert isinstance(store, MeasureStore)
        assert len(store.to_dict()) > 0

        # Check that basic metrics are computed
        assert store.has("Number of Traces")
        assert store.has("Number of Events")
        assert store.get("Number of Traces").value == 1
        assert store.get("Number of Events").value == 2

        # Check info
        assert info["adapter"] == "local"
        assert "skipped" in info

    def test_compute_measures_for_window_standard_log(self, adapter, standard_test_log):
        """Test computing measures for standard test log."""
        window = Window(
            id="standard_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        store, info = adapter.compute_measures_for_window(window)

        # Should compute many metrics
        assert isinstance(store, MeasureStore)
        assert len(store.to_dict()) > 0

        # Check specific metrics
        assert store.has("Number of Traces")
        assert store.has("Number of Events")
        assert store.get("Number of Traces").value == 6
        assert store.get("Number of Events").value == 16

        # Check that metrics are not hidden by default
        for name in store.to_dict().keys():
            measure = store.get(name)
            assert not measure.hidden, f"Metric {name} should not be hidden"

    def test_compute_measures_with_include_exclude(self, adapter, standard_test_log):
        """Test computing measures with include/exclude filters."""
        window = Window(
            id="filtered_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        # Test include filter
        include_metrics = ["Number of Traces", "Number of Events", "Avg. Trace Length"]
        store, info = adapter.compute_measures_for_window(
            window, include_metrics=include_metrics
        )

        # Should only compute included metrics as visible
        computed_metrics = store.to_dict()
        visible_metric_names = [
            name for name, measure in computed_metrics.items() if not measure.hidden
        ]
        for metric in visible_metric_names:
            assert metric in include_metrics, f"Unexpected metric {metric} computed"

        # Test exclude filter
        exclude_metrics = ["Number of Traces", "Number of Events"]
        store, info = adapter.compute_measures_for_window(
            window, exclude_metrics=exclude_metrics
        )

        computed_metrics = store.to_dict()
        visible_metric_names = [
            name for name, measure in computed_metrics.items() if not measure.hidden
        ]
        # Should not hold excluded metrics as visible
        for metric in exclude_metrics:
            if store.has(metric):
                measure = store.get(metric)
                assert (
                    measure.hidden
                ), f"Excluded metric {metric} was computed as visible"

    def test_compute_measures_with_existing_store(self, adapter, standard_test_log):
        """Test computing measures with existing MeasureStore."""
        # Create existing store with some values
        existing_store = MeasureStore()
        existing_store.set("Custom Metric", 42.0, hidden=False, meta={"custom": True})

        window = Window(
            id="existing_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        store, info = adapter.compute_measures_for_window(
            window, measure_store=existing_store
        )

        # Should preserve existing values
        assert store.has("Custom Metric")
        assert store.get("Custom Metric").value == 42.0
        assert store.get("Custom Metric").meta["custom"] is True

        # Should add new computed metrics
        assert store.has("Number of Traces")
        assert store.has("Number of Events")

    def test_compute_measures_with_population_distribution(
        self, adapter_distribution, standard_test_log, population_distribution
    ):
        """Test computing measures with population distribution (distribution preference)."""
        window = Window(
            id="distribution_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=population_distribution,
        )

        store, info = adapter_distribution.compute_measures_for_window(window)

        # Should compute metrics using distribution-based variants when available
        assert isinstance(store, MeasureStore)
        assert len(store.to_dict()) > 0

        # Check that distribution-based metrics are computed
        if store.has("Number of Distinct Traces"):
            measure = store.get("Number of Distinct Traces")
            # Should use distribution-based implementation
            assert measure.meta.get("bases") == "population_distribution"

    def test_compute_measures_for_windows_batch(
        self, adapter, simple_traces, complex_traces
    ):
        """Test batch computing measures for multiple windows."""
        windows = [
            Window(
                id="simple",
                size=len(simple_traces),
                traces=simple_traces,
                population_distributions=None,
            ),
            Window(
                id="complex",
                size=len(complex_traces),
                traces=complex_traces,
                population_distributions=None,
            ),
        ]

        results = adapter.compute_measures_for_windows(windows)

        # Should return results for both windows
        assert len(results) == 2
        assert "simple" in results
        assert "complex" in results

        # Each result should be a tuple of (store, info)
        for window_id, (store, info) in results.items():
            assert isinstance(store, MeasureStore)
            assert isinstance(info, dict)
            assert info["adapter"] == "local"
            assert len(store.to_dict()) > 0

    def test_compute_measures_for_windows_with_existing_stores(
        self, adapter, simple_traces, complex_traces
    ):
        """Test batch computing with existing stores."""
        windows = [
            Window(
                id="simple",
                size=len(simple_traces),
                traces=simple_traces,
                population_distributions=None,
            ),
            Window(
                id="complex",
                size=len(complex_traces),
                traces=complex_traces,
                population_distributions=None,
            ),
        ]

        # Create existing stores
        existing_stores = {"simple": MeasureStore(), "complex": MeasureStore()}
        existing_stores["simple"].set("Custom", 1.0)
        existing_stores["complex"].set("Custom", 2.0)

        results = adapter.compute_measures_for_windows(
            windows, measures_by_id=existing_stores
        )

        # Should preserve existing values
        assert results["simple"][0].get("Custom").value == 1.0
        assert results["complex"][0].get("Custom").value == 2.0

        # Should add computed metrics
        assert results["simple"][0].has("Number of Traces")
        assert results["complex"][0].has("Number of Traces")

    def test_strict_mode_behavior(self, standard_test_log):
        """Test strict mode behavior."""
        # Test strict=True (should raise on errors)
        adapter_strict = LocalMetricsAdapter(strict=True)
        window = Window(
            id="strict_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        # Should not raise with valid input
        store, info = adapter_strict.compute_measures_for_window(window)
        assert isinstance(store, MeasureStore)

        # Test strict=False (should skip problematic metrics)
        adapter_lenient = LocalMetricsAdapter(strict=False)
        store, info = adapter_lenient.compute_measures_for_window(window)
        assert isinstance(store, MeasureStore)

    def test_prefer_parameter_behavior(
        self, standard_test_log, population_distribution
    ):
        """Test different prefer parameter behaviors."""
        window = Window(
            id="prefer_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=population_distribution,
        )

        # Test trace preference
        adapter_trace = LocalMetricsAdapter(prefer="trace")
        store_trace, _ = adapter_trace.compute_measures_for_window(window)

        # Test distribution preference
        adapter_dist = LocalMetricsAdapter(prefer="distribution")
        store_dist, _ = adapter_dist.compute_measures_for_window(window)

        # Test auto preference
        adapter_auto = LocalMetricsAdapter(prefer="auto")
        store_auto, _ = adapter_auto.compute_measures_for_window(window)

        # All should compute successfully
        assert isinstance(store_trace, MeasureStore)
        assert isinstance(store_dist, MeasureStore)
        assert isinstance(store_auto, MeasureStore)

    def test_metric_metadata_consistency(self, adapter, standard_test_log):
        """Test that computed metrics have consistent metadata."""
        window = Window(
            id="metadata_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        store, info = adapter.compute_measures_for_window(window)

        for name in store.to_dict().keys():
            measure = store.get(name)

            # Should have metadata
            assert measure.meta is not None

            # Should not be hidden by default
            assert not measure.hidden

            # Should have a value
            assert measure.value is not None

    def test_adapter_info_structure(self, adapter, standard_test_log):
        """Test that adapter info has the expected structure."""
        window = Window(
            id="info_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        store, info = adapter.compute_measures_for_window(window)

        # Check required info fields
        assert "adapter" in info
        assert "strict" in info
        assert "prefer" in info
        assert "visible" in info
        assert "skipped" in info

        # Check values
        assert info["adapter"] == "local"
        assert info["strict"] is True
        assert info["prefer"] == "trace"
        assert isinstance(info["visible"], list)
        assert isinstance(info["skipped"], list)
