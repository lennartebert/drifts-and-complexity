"""Tests for the metric registry system."""

from typing import List, Type

import pytest

from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import (
    available_metric_names,
    discover_metrics,
    get_metric_class,
    get_metric_classes,
    register_metric,
)
from utils.constants import ALL_METRIC_NAMES, DUAL_VARIANT_METRIC_NAMES


class TestMetricRegistry:
    """Test the metric registry functionality."""

    def test_discover_metrics(self):
        """Test that metrics are discovered correctly."""
        # Clear any previous discovery
        discover_metrics()

        # Get available metric names
        metric_names = list(available_metric_names())

        assert len(metric_names) > 0, "No metrics discovered"

        # Check that we have all expected metrics
        # Should include all metrics from our constants (excluding entropy metrics from vidgof)
        entropy_metrics = [
            "Sequence Entropy",
            "Normalized Sequence Entropy",
            "Variant Entropy",
            "Normalized Variant Entropy",
        ]
        for expected_metric in ALL_METRIC_NAMES:
            if (
                expected_metric not in entropy_metrics
            ):  # Entropy metrics are from vidgof adapter
                assert (
                    expected_metric in metric_names
                ), f"Expected metric {expected_metric} not found"

        # Verify we have the expected number of metrics (excluding entropy metrics)
        expected_count = len(ALL_METRIC_NAMES) - len(entropy_metrics)
        assert (
            len(metric_names) == expected_count
        ), f"Expected {expected_count} metrics (excluding entropy), found {len(metric_names)}"

    def test_get_metric_class(self):
        """Test getting a single metric class."""
        discover_metrics()

        # Test getting a metric class
        metric_class = get_metric_class("Number of Traces")
        assert metric_class is not None
        assert hasattr(metric_class, "name")
        assert metric_class.name == "Number of Traces"

        # Test getting a non-existent metric
        with pytest.raises(KeyError):
            get_metric_class("Non-existent Metric")

    def test_get_metric_classes(self):
        """Test getting all metric classes for a given name."""
        discover_metrics()

        # Test getting classes for a metric with multiple implementations
        classes = get_metric_classes("Number of Distinct Traces")
        assert len(classes) >= 1, "Should have at least one implementation"

        # All classes should have the same name
        for cls in classes:
            assert cls.name == "Number of Distinct Traces"

        # Test getting classes for a non-existent metric
        with pytest.raises(KeyError):
            get_metric_classes("Non-existent Metric")

    def test_metric_class_properties(self):
        """Test that metric classes have required properties."""
        discover_metrics()

        metric_names = list(available_metric_names())

        for metric_name in metric_names:
            classes = get_metric_classes(metric_name)

            for cls in classes:
                # Should have a name attribute
                assert hasattr(cls, "name")
                assert cls.name == metric_name

                # Should have a compute method
                assert hasattr(cls, "compute")
                assert callable(getattr(cls, "compute"))

                # Should have requires attribute (may be empty list)
                assert hasattr(cls, "requires")
                assert isinstance(cls.requires, list)

    def test_metric_variants(self):
        """Test that metrics with multiple variants are handled correctly."""
        discover_metrics()

        # Check for metrics that should have multiple variants (both trace-based and distribution-based)
        multi_variant_metrics = DUAL_VARIANT_METRIC_NAMES

        for metric_name in multi_variant_metrics:
            classes = get_metric_classes(metric_name)

            # Should have exactly 2 implementations (trace-based and distribution-based)
            assert (
                len(classes) == 2
            ), f"Metric {metric_name} should have exactly 2 implementations, found {len(classes)}"

            # Check that we have both trace-based and distribution-based variants
            trace_based = [cls for cls in classes if "trace_based" in cls.__module__]
            distribution_based = [
                cls for cls in classes if "distribution_based" in cls.__module__
            ]

            # Should have exactly one of each type
            assert (
                len(trace_based) == 1
            ), f"Metric {metric_name} should have exactly 1 trace-based implementation, found {len(trace_based)}"
            assert (
                len(distribution_based) == 1
            ), f"Metric {metric_name} should have exactly 1 distribution-based implementation, found {len(distribution_based)}"

            # Verify the module paths are correct
            trace_module = trace_based[0].__module__
            distribution_module = distribution_based[0].__module__

            assert (
                "trace_based" in trace_module
            ), f"Trace-based implementation should be in trace_based module, found {trace_module}"
            assert (
                "distribution_based" in distribution_module
            ), f"Distribution-based implementation should be in distribution_based module, found {distribution_module}"

    def test_dual_variant_metrics_retrieval(self):
        """Test that metrics with both trace-based and distribution-based variants can be retrieved correctly."""
        discover_metrics()

        # Metrics that should have both trace-based and distribution-based variants
        dual_variant_metrics = DUAL_VARIANT_METRIC_NAMES

        for metric_name in dual_variant_metrics:
            # Get all classes for this metric
            classes = get_metric_classes(metric_name)
            assert (
                len(classes) == 2
            ), f"Metric {metric_name} should have exactly 2 variants"

            # Separate trace-based and distribution-based classes
            trace_class = next(
                cls for cls in classes if "trace_based" in cls.__module__
            )
            distribution_class = next(
                cls for cls in classes if "distribution_based" in cls.__module__
            )

            # Both should have the same name
            assert trace_class.name == metric_name
            assert distribution_class.name == metric_name

            # Both should have the same requirements
            assert (
                trace_class.requires == distribution_class.requires
            ), f"Trace-based and distribution-based variants of {metric_name} should have same requirements"

            # Both should be callable
            assert callable(trace_class.compute)
            assert callable(distribution_class.compute)

            # Verify they are different classes
            assert (
                trace_class != distribution_class
            ), f"Trace-based and distribution-based variants of {metric_name} should be different classes"

    def test_registry_consistency(self):
        """Test that the registry is consistent across multiple calls."""
        # Clear and rediscover
        discover_metrics()
        names1 = set(available_metric_names())

        # Rediscover
        discover_metrics()
        names2 = set(available_metric_names())

        # Should be the same
        assert (
            names1 == names2
        ), "Registry should be consistent across multiple discovery calls"

        # Test that getting classes is consistent
        for name in names1:
            classes1 = get_metric_classes(name)
            classes2 = get_metric_classes(name)

            assert len(classes1) == len(
                classes2
            ), f"Class count should be consistent for {name}"
            assert classes1 == classes2, f"Classes should be consistent for {name}"


class TestMetricRegistration:
    """Test metric registration functionality."""

    def test_register_metric_decorator(self):
        """Test the @register_metric decorator."""

        # Create a test metric class
        @register_metric("Test Metric")
        class TestMetric:
            name = "Test Metric"
            requires = []

            def compute(self, traces, measures):
                measures.set(self.name, 42, hidden=False, meta={"test": True})

        # Discover metrics to register our test metric
        discover_metrics()

        # Should be able to get the metric class
        metric_class = get_metric_class("Test Metric")
        assert metric_class is not None
        assert metric_class.name == "Test Metric"

        # Should be able to get all classes
        classes = get_metric_classes("Test Metric")
        assert len(classes) >= 1
        assert TestMetric in classes

    def test_register_metric_with_requirements(self):
        """Test registering a metric with requirements."""

        @register_metric("Test Metric with Requirements")
        class TestMetricWithRequirements:
            name = "Test Metric with Requirements"
            requires = ["Number of Traces", "Number of Events"]

            def compute(self, traces, measures):
                measures.set(self.name, 100, hidden=False, meta={"test": True})

        # Discover metrics
        discover_metrics()

        # Should be able to get the metric class
        metric_class = get_metric_class("Test Metric with Requirements")
        assert metric_class is not None
        assert metric_class.requires == ["Number of Traces", "Number of Events"]


class TestMetricOrchestratorIntegration:
    """Test integration between registry and orchestrator."""

    def test_orchestrator_uses_registry(self, standard_test_log):
        """Test that the orchestrator uses the registry correctly."""
        from utils.complexity.metrics.metric_orchestrator import MetricOrchestrator
        from utils.windowing.window import Window

        discover_metrics()
        metric_names = list(available_metric_names())

        # Create a window
        window = Window(
            id="test_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        # Create orchestrator
        orchestrator = MetricOrchestrator(strict=True, prefer="trace")

        # Compute all metrics
        store, info = orchestrator.compute_many_metrics(metric_names, window)

        # All metrics should be computed
        computed_count = len([name for name in metric_names if store.has(name)])
        assert computed_count == len(
            metric_names
        ), f"All metrics should be computed, got {computed_count}/{len(metric_names)}"

        # Check that no metrics were skipped
        assert (
            len(info.get("skipped", [])) == 0
        ), f"Some metrics were skipped: {info.get('skipped', [])}"

    def test_orchestrator_variant_selection(self, standard_test_log):
        """Test that the orchestrator selects the correct variant."""
        from utils.complexity.metrics.metric_orchestrator import MetricOrchestrator
        from utils.windowing.window import Window

        discover_metrics()

        # Create a window
        window = Window(
            id="test_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        # Test trace preference
        orchestrator_trace = MetricOrchestrator(strict=True, prefer="trace")
        store_trace, _ = orchestrator_trace.compute_many_metrics(
            ["Number of Distinct Traces"], window
        )

        # Should use trace-based implementation
        if store_trace.has("Number of Distinct Traces"):
            measure = store_trace.get("Number of Distinct Traces")
            assert (
                measure.meta.get("basis") == "traces"
            ), "Should use trace-based implementation"

    def test_orchestrator_dependency_resolution(self, standard_test_log):
        """Test that the orchestrator resolves dependencies correctly."""
        from utils.complexity.metrics.metric_orchestrator import MetricOrchestrator
        from utils.windowing.window import Window

        discover_metrics()

        # Create a window
        window = Window(
            id="test_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None,
        )

        # Create orchestrator
        orchestrator = MetricOrchestrator(strict=True, prefer="trace")

        # Test computing a metric that might have dependencies
        store, info = orchestrator.compute_many_metrics(["Avg. Trace Length"], window)

        # The metric should be computed
        assert store.has("Avg. Trace Length"), "Avg. Trace Length should be computed"

        # Check that dependencies were resolved (TraceLengthStats should have been computed)
        assert store.has(
            "Min. Trace Length"
        ), "Min. Trace Length dependency should be resolved"
        assert store.has(
            "Max. Trace Length"
        ), "Max. Trace Length dependency should be resolved"
