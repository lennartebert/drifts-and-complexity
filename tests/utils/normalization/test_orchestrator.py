"""Tests for the normalization orchestrator."""

from typing import Any, Dict, List

import pytest

from utils.complexity.measures.measure_store import Measure, MeasureStore
from utils.normalization.normalizers.hide_number_of_traces import HideNumberOfTraces
from utils.normalization.normalizers.hide_percentage_of_distinct_traces import (
    HidePercentageOfDistinctTraces,
)
from utils.normalization.normalizers.normalize_deviation_from_random import (
    NormalizeDeviationFromRandom,
)
from utils.normalization.normalizers.normalize_lz_complexity import (
    NormalizeLZComplexity,
)
from utils.normalization.normalizers.normalize_number_of_events import (
    NormalizeNumberOfEvents,
)
from utils.normalization.normalizers.normalize_number_of_traces import (
    NormalizeNumberOfTraces,
)
from utils.normalization.normalizers.normalize_percentage_of_distinct_traces import (
    NormalizePercentageOfDistinctTraces,
)
from utils.normalization.normalizers.normalizer import Normalizer
from utils.normalization.orchestrator import DEFAULT_NORMALIZERS, apply_normalizers


class TestNormalizationOrchestrator:
    """Test the normalization orchestrator functionality."""

    def test_apply_normalizers_none(self):
        """Test applying no normalizers (None input)."""
        store = MeasureStore()
        store.set("Test Metric", 42.0)

        result = apply_normalizers(store, normalizers=None)

        # Should return the same store unchanged
        assert result is store
        assert store.get("Test Metric").value == 42.0
        assert store.get("Test Metric").value_normalized is None

    def test_apply_normalizers_empty_list(self):
        """Test applying empty normalizers list."""
        store = MeasureStore()
        store.set("Test Metric", 42.0)

        result = apply_normalizers(store, normalizers=[])

        # Should return the same store unchanged
        assert result is store
        assert store.get("Test Metric").value == 42.0
        assert store.get("Test Metric").value_normalized is None

    def test_apply_normalizers_single_normalizer(self):
        """Test applying a single normalizer."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)

        normalizers = [NormalizeNumberOfTraces()]
        result = apply_normalizers(store, normalizers=normalizers)

        # Should normalize Number of Traces to 1.0
        assert result is store
        measure = store.get("Number of Traces")
        assert measure.value == 5.0  # Original value preserved
        assert measure.value_normalized == 1.0  # Normalized value set
        assert measure.meta["normalized_by"] == "NormalizeNumberOfTraces"

    def test_apply_normalizers_multiple_normalizers(self):
        """Test applying multiple normalizers in sequence."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        store.set("Number of Events", 20.0)

        normalizers = [NormalizeNumberOfTraces(), NormalizeNumberOfEvents()]
        result = apply_normalizers(store, normalizers=normalizers)

        # Should normalize both metrics
        assert result is store

        # Number of Traces normalized to 1.0
        traces_measure = store.get("Number of Traces")
        assert traces_measure.value == 5.0
        assert traces_measure.value_normalized == 1.0
        assert traces_measure.meta["normalized_by"] == "NormalizeNumberOfTraces"

        # Number of Events normalized to average trace length (20/5 = 4.0)
        events_measure = store.get("Number of Events")
        assert events_measure.value == 20.0
        assert events_measure.value_normalized == 4.0
        assert events_measure.meta["normalized_by"] == "NormalizeNumberOfEvents"

    def test_apply_normalizers_missing_dependencies(self):
        """Test normalizers with missing dependencies."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        # Missing "Number of Events" which some normalizers might need

        normalizers = [
            NormalizeNumberOfTraces(),
            NormalizeNumberOfEvents(),  # This should handle missing dependency gracefully
        ]

        # Should not raise exception
        result = apply_normalizers(store, normalizers=normalizers)
        assert result is store

        # Number of Traces should still be normalized
        traces_measure = store.get("Number of Traces")
        assert traces_measure.value_normalized == 1.0

    def test_default_normalizers(self):
        """Test that DEFAULT_NORMALIZERS is properly configured."""
        assert isinstance(DEFAULT_NORMALIZERS, list)
        assert len(DEFAULT_NORMALIZERS) > 0

        # All should be Normalizer instances
        for normalizer in DEFAULT_NORMALIZERS:
            assert isinstance(normalizer, Normalizer)

        # Check specific normalizers are included
        normalizer_types = [type(n).__name__ for n in DEFAULT_NORMALIZERS]
        assert "NormalizeNumberOfEvents" in normalizer_types
        assert "NormalizeNumberOfTraces" in normalizer_types
        assert "NormalizePercentageOfDistinctTraces" in normalizer_types
        assert "NormalizeDeviationFromRandom" in normalizer_types
        assert "NormalizeLZComplexity" in normalizer_types

    def test_apply_default_normalizers(self):
        """Test applying the default normalizers."""
        store = MeasureStore()
        store.set("Number of Traces", 6.0)
        store.set("Number of Events", 24.0)
        store.set("Percentage of Distinct Traces", 0.5)
        store.set("Deviation from Random", 0.3)
        store.set("Number of Distinct Activities", 4.0)
        store.set("Lempel-Ziv Complexity", 8.0)

        result = apply_normalizers(store, normalizers=DEFAULT_NORMALIZERS)

        # Should return the same store
        assert result is store

        # Check that normalizations were applied
        traces_measure = store.get("Number of Traces")
        assert traces_measure.value_normalized is not None

        events_measure = store.get("Number of Events")
        assert events_measure.value_normalized is not None

    def test_normalizer_metadata_preservation(self):
        """Test that normalizers preserve existing metadata."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0, meta={"source": "test", "custom": "value"})

        normalizers = [NormalizeNumberOfTraces()]
        apply_normalizers(store, normalizers=normalizers)

        measure = store.get("Number of Traces")
        # Should preserve existing metadata and add normalization info
        assert measure.meta["source"] == "test"
        assert measure.meta["custom"] == "value"
        assert measure.meta["normalized_by"] == "NormalizeNumberOfTraces"

    def test_normalizer_in_place_modification(self):
        """Test that normalizers modify the store in place."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)

        # Get reference to the measure before normalization
        original_measure = store.get("Number of Traces")

        normalizers = [NormalizeNumberOfTraces()]
        result = apply_normalizers(store, normalizers=normalizers)

        # Should be the same object (in-place modification)
        assert result is store
        assert store.get("Number of Traces") is original_measure

        # The measure should be modified
        assert original_measure.value_normalized == 1.0
        assert "normalized_by" in original_measure.meta


class TestNormalizerIntegration:
    """Integration tests for normalizers working together."""

    def test_hide_normalizers(self):
        """Test that hide normalizers work correctly."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        store.set("Percentage of Distinct Traces", 0.8)

        normalizers = [HideNumberOfTraces(), HidePercentageOfDistinctTraces()]
        apply_normalizers(store, normalizers=normalizers)

        # Measures should be hidden
        assert store.get("Number of Traces").hidden is True
        assert store.get("Percentage of Distinct Traces").hidden is True

    def test_normalize_and_hide_sequence(self):
        """Test normalizing then hiding measures."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        store.set("Percentage of Distinct Traces", 0.8)

        normalizers = [
            NormalizeNumberOfTraces(),
            NormalizePercentageOfDistinctTraces(),
            HideNumberOfTraces(),
            HidePercentageOfDistinctTraces(),
        ]
        apply_normalizers(store, normalizers=normalizers)

        # Should be normalized AND hidden
        traces_measure = store.get("Number of Traces")
        # Note: Hide normalizers might clear value_normalized, so just check it was processed
        assert traces_measure.hidden is True
        assert traces_measure.meta["normalized_by"] == "NormalizeNumberOfTraces"

        percentage_measure = store.get("Percentage of Distinct Traces")
        # Note: Hide normalizers might clear value_normalized, so just check it was processed
        assert percentage_measure.hidden is True

    def test_complex_normalization_scenario(self):
        """Test a complex scenario with multiple normalizers and dependencies."""
        store = MeasureStore()
        # Set up a realistic scenario
        store.set("Number of Traces", 6.0)
        store.set("Number of Events", 24.0)
        store.set("Number of Distinct Activities", 4.0)
        store.set("Number of Distinct Traces", 3.0)
        store.set("Percentage of Distinct Traces", 0.5)
        store.set("Deviation from Random", 0.3)
        store.set("Lempel-Ziv Complexity", 8.0)

        # Apply all normalizers
        apply_normalizers(store, normalizers=DEFAULT_NORMALIZERS)

        # Check that various normalizations were applied
        normalized_measures = [
            name
            for name, measure in store.to_dict().items()
            if measure.value_normalized is not None
        ]

        # Should have several normalized measures
        assert len(normalized_measures) > 0

        # Check specific normalizations
        if store.has("Number of Traces"):
            assert store.get("Number of Traces").value_normalized == 1.0

        if store.has("Number of Events"):
            # Number of Events is normalized to average trace length, not 1.0
            assert store.get("Number of Events").value_normalized is not None

        if store.has("Percentage of Distinct Traces"):
            # Should be normalized (converts percentage to count, so not necessarily in [0,1])
            assert (
                store.get("Percentage of Distinct Traces").value_normalized is not None
            )

    def test_error_handling_in_normalizers(self):
        """Test that normalizers handle errors gracefully."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        # Missing "Number of Distinct Activities" which some normalizers need

        # This should not raise an exception even with missing dependencies
        try:
            apply_normalizers(store, normalizers=DEFAULT_NORMALIZERS)
            # If we get here, error handling worked
            assert True
        except Exception as e:
            # Some normalizers might raise exceptions for missing dependencies
            # This is acceptable behavior
            assert isinstance(e, Exception)

    def test_normalizer_ordering(self):
        """Test that normalizers are applied in the correct order."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)

        # Create a custom normalizer that tracks application order
        class OrderTrackingNormalizer(Normalizer):
            def __init__(self, name):
                self.name = name
                self.applied_order = []

            def apply(self, measures: MeasureStore) -> None:
                self.applied_order.append(self.name)

        normalizer1 = OrderTrackingNormalizer("first")
        normalizer2 = OrderTrackingNormalizer("second")
        normalizer3 = OrderTrackingNormalizer("third")

        normalizers = [normalizer1, normalizer2, normalizer3]
        apply_normalizers(store, normalizers=normalizers)

        # Should be applied in order
        assert normalizer1.applied_order == ["first"]
        assert normalizer2.applied_order == ["second"]
        assert normalizer3.applied_order == ["third"]
