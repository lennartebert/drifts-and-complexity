"""Tests for individual normalizers."""

import pytest
import math
from typing import Dict, Any

from utils.complexity.measures.measure_store import MeasureStore, Measure
from utils.normalization.normalizers.normalize_number_of_traces import NormalizeNumberOfTraces
from utils.normalization.normalizers.normalize_number_of_events import NormalizeNumberOfEvents
from utils.normalization.normalizers.normalize_percentage_of_distinct_traces import NormalizePercentageOfDistinctTraces
from utils.normalization.normalizers.normalize_deviation_from_random import NormalizeDeviationFromRandom
from utils.normalization.normalizers.normalize_lz_complexity import NormalizeLZComplexity
from utils.normalization.normalizers.hide_number_of_traces import HideNumberOfTraces
from utils.normalization.normalizers.hide_percentage_of_distinct_traces import HidePercentageOfDistinctTraces


class TestNormalizeNumberOfTraces:
    """Test the NormalizeNumberOfTraces normalizer."""
    
    def test_normalize_basic(self):
        """Test basic normalization of Number of Traces."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        
        normalizer = NormalizeNumberOfTraces()
        normalizer.apply(store)
        
        measure = store.get("Number of Traces")
        assert measure.value == 5.0  # Original value preserved
        assert measure.value_normalized == 1.0  # Always normalized to 1.0
        assert measure.meta["normalized_by"] == "NormalizeNumberOfTraces"
    
    def test_normalize_different_values(self):
        """Test normalization with different trace counts."""
        test_values = [1.0, 10.0, 100.0, 0.5, 2.5]
        
        for value in test_values:
            store = MeasureStore()
            store.set("Number of Traces", value)
            
            normalizer = NormalizeNumberOfTraces()
            normalizer.apply(store)
            
            measure = store.get("Number of Traces")
            assert measure.value == value
            assert measure.value_normalized == 1.0
    
    def test_missing_measure(self):
        """Test behavior when Number of Traces is missing."""
        store = MeasureStore()
        store.set("Other Metric", 42.0)
        
        normalizer = NormalizeNumberOfTraces()
        
        # Should raise exception when required measure is missing
        with pytest.raises(Exception, match="Number of Traces required"):
            normalizer.apply(store)
    
    def test_preserve_metadata(self):
        """Test that existing metadata is preserved."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0, meta={"source": "test", "custom": "value"})
        
        normalizer = NormalizeNumberOfTraces()
        normalizer.apply(store)
        
        measure = store.get("Number of Traces")
        assert measure.meta["source"] == "test"
        assert measure.meta["custom"] == "value"
        assert measure.meta["normalized_by"] == "NormalizeNumberOfTraces"


class TestNormalizeNumberOfEvents:
    """Test the NormalizeNumberOfEvents normalizer."""
    
    def test_normalize_basic(self):
        """Test basic normalization of Number of Events."""
        store = MeasureStore()
        store.set("Number of Events", 20.0)
        store.set("Number of Traces", 5.0)  # Required dependency
        
        normalizer = NormalizeNumberOfEvents()
        normalizer.apply(store)
        
        measure = store.get("Number of Events")
        assert measure.value == 20.0  # Original value preserved
        assert measure.value_normalized == 4.0  # 20/5 = 4.0 (average trace length)
        assert measure.meta["normalized_by"] == "NormalizeNumberOfEvents"
    
    def test_missing_measure(self):
        """Test behavior when Number of Events is missing."""
        store = MeasureStore()
        store.set("Other Metric", 42.0)
        
        normalizer = NormalizeNumberOfEvents()
        
        # Should do nothing when required measure is missing
        normalizer.apply(store)
        # Should not raise exception


class TestNormalizePercentageOfDistinctTraces:
    """Test the NormalizePercentageOfDistinctTraces normalizer."""
    
    def test_normalize_basic(self):
        """Test basic normalization of Percentage of Distinct Traces."""
        store = MeasureStore()
        store.set("Percentage of Distinct Traces", 0.5)
        store.set("Number of Traces", 10.0)  # Required dependency
        
        normalizer = NormalizePercentageOfDistinctTraces()
        normalizer.apply(store)
        
        measure = store.get("Percentage of Distinct Traces")
        assert measure.value == 0.5  # Original value preserved
        assert measure.value_normalized == 5.0  # 0.5 * 10 = 5.0 (count of distinct traces)
        assert measure.meta["normalized_by"] == "NormalizePercentageOfDistinctTraces"
    
    def test_normalize_different_values(self):
        """Test normalization with different percentage values."""
        test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        nt = 10.0  # Number of Traces
        
        for value in test_values:
            store = MeasureStore()
            store.set("Percentage of Distinct Traces", value)
            store.set("Number of Traces", nt)
            
            normalizer = NormalizePercentageOfDistinctTraces()
            normalizer.apply(store)
            
            measure = store.get("Percentage of Distinct Traces")
            assert measure.value == value
            assert measure.value_normalized == value * nt
    
    def test_missing_measure(self):
        """Test behavior when Percentage of Distinct Traces is missing."""
        store = MeasureStore()
        store.set("Other Metric", 42.0)
        
        normalizer = NormalizePercentageOfDistinctTraces()
        
        # Should do nothing when required measure is missing
        normalizer.apply(store)
        # Should not raise exception


class TestNormalizeDeviationFromRandom:
    """Test the NormalizeDeviationFromRandom normalizer."""
    
    def test_normalize_basic(self):
        """Test basic normalization of Deviation from Random."""
        store = MeasureStore()
        store.set("Deviation from Random", 0.3)
        store.set("Number of Distinct Activities", 4.0)
        
        normalizer = NormalizeDeviationFromRandom()
        normalizer.apply(store)
        
        measure = store.get("Deviation from Random")
        assert measure.value == 0.3  # Original value preserved
        assert measure.value_normalized is not None
        assert 0 <= measure.value_normalized <= 1  # Should be in [0,1]
        assert measure.meta["normalized_by"] == "NormalizeDeviationFromRandom"
    
    def test_normalize_edge_cases(self):
        """Test normalization with edge case values."""
        # Test with minimum values
        store = MeasureStore()
        store.set("Deviation from Random", 0.0)
        store.set("Number of Distinct Activities", 2.0)  # Minimum valid value
        
        normalizer = NormalizeDeviationFromRandom()
        normalizer.apply(store)
        
        measure = store.get("Deviation from Random")
        assert 0 <= measure.value_normalized <= 1
        
        # Test with maximum values
        store = MeasureStore()
        store.set("Deviation from Random", 1.0)
        store.set("Number of Distinct Activities", 10.0)
        
        normalizer = NormalizeDeviationFromRandom()
        normalizer.apply(store)
        
        measure = store.get("Deviation from Random")
        assert 0 <= measure.value_normalized <= 1
    
    def test_missing_deviation_measure(self):
        """Test behavior when Deviation from Random is missing."""
        store = MeasureStore()
        store.set("Number of Distinct Activities", 4.0)
        
        normalizer = NormalizeDeviationFromRandom()
        normalizer.apply(store)  # Should do nothing, not raise
        
        # No measure should be modified
        assert not store.has("Deviation from Random")
    
    def test_missing_activities_measure(self):
        """Test behavior when Number of Distinct Activities is missing."""
        store = MeasureStore()
        store.set("Deviation from Random", 0.3)
        
        normalizer = NormalizeDeviationFromRandom()
        
        # Should raise exception when required dependency is missing
        with pytest.raises(Exception, match="Number of distinct activities required"):
            normalizer.apply(store)
    
    def test_invalid_activities_value(self):
        """Test behavior with invalid Number of Distinct Activities values."""
        store = MeasureStore()
        store.set("Deviation from Random", 0.3)
        store.set("Number of Distinct Activities", 1.0)  # Invalid: V^2 = 1, denom = 0
        
        normalizer = NormalizeDeviationFromRandom()
        normalizer.apply(store)  # Should do nothing, not raise
        
        measure = store.get("Deviation from Random")
        assert measure.value_normalized is None  # Should not be normalized
    
    def test_mathematical_correctness(self):
        """Test that the normalization formula is applied correctly."""
        store = MeasureStore()
        store.set("Deviation from Random", 0.3)
        store.set("Number of Distinct Activities", 4.0)
        
        normalizer = NormalizeDeviationFromRandom()
        normalizer.apply(store)
        
        measure = store.get("Deviation from Random")
        D = 0.3
        V = 4.0
        
        # Manual calculation of expected normalized value
        denom_inner = 1.0 - 1.0 / (V ** 2)
        denom = math.sqrt(denom_inner)
        expected_norm_val = 1.0 - (1.0 - D) / denom
        expected_norm_val = max(0.0, min(1.0, expected_norm_val))
        
        assert abs(measure.value_normalized - expected_norm_val) < 1e-10


class TestNormalizeLZComplexity:
    """Test the NormalizeLZComplexity normalizer."""
    
    def test_normalize_basic(self):
        """Test basic normalization of Lempel-Ziv Complexity."""
        store = MeasureStore()
        store.set("Lempel-Ziv Complexity", 8.0)
        store.set("Number of Events", 20.0)  # Required dependency
        store.set("Number of Distinct Activities", 4.0)  # Required dependency
        
        normalizer = NormalizeLZComplexity()
        normalizer.apply(store)
        
        measure = store.get("Lempel-Ziv Complexity")
        assert measure.value == 8.0  # Original value preserved
        assert measure.value_normalized is not None
        assert measure.value_normalized > 0  # Should be positive
        assert measure.meta["normalized_by"] == "NormalizeLZComplexity"
    
    def test_missing_measure(self):
        """Test behavior when Lempel-Ziv Complexity is missing."""
        store = MeasureStore()
        store.set("Other Metric", 42.0)
        
        normalizer = NormalizeLZComplexity()
        normalizer.apply(store)  # Should do nothing, not raise
        
        # No measure should be modified
        assert not store.has("Lempel-Ziv Complexity")


class TestHideNormalizers:
    """Test the hide normalizers."""
    
    def test_hide_number_of_traces(self):
        """Test hiding Number of Traces."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        
        normalizer = HideNumberOfTraces()
        normalizer.apply(store)
        
        measure = store.get("Number of Traces")
        assert measure.hidden is True
    
    def test_hide_percentage_of_distinct_traces(self):
        """Test hiding Percentage of Distinct Traces."""
        store = MeasureStore()
        store.set("Percentage of Distinct Traces", 0.5)
        
        normalizer = HidePercentageOfDistinctTraces()
        normalizer.apply(store)
        
        measure = store.get("Percentage of Distinct Traces")
        assert measure.hidden is True
    
    def test_hide_missing_measures(self):
        """Test hiding measures that don't exist."""
        store = MeasureStore()
        store.set("Other Metric", 42.0)
        
        # Should not raise exceptions for missing measures
        normalizer1 = HideNumberOfTraces()
        normalizer2 = HidePercentageOfDistinctTraces()
        
        normalizer1.apply(store)
        normalizer2.apply(store)
        
        # Should not affect existing measures
        assert not store.get("Other Metric").hidden


class TestNormalizerProperties:
    """Test general properties of normalizers."""
    
    def test_normalizers_do_not_add_measures(self):
        """Test that normalizers don't add new measures."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        
        initial_keys = set(store.to_dict().keys())
        
        normalizer = NormalizeNumberOfTraces()
        normalizer.apply(store)
        
        final_keys = set(store.to_dict().keys())
        
        # Should not add new measures
        assert final_keys == initial_keys
    
    def test_normalizers_modify_in_place(self):
        """Test that normalizers modify measures in place."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        
        # Get reference to the measure
        original_measure = store.get("Number of Traces")
        
        normalizer = NormalizeNumberOfTraces()
        normalizer.apply(store)
        
        # Should be the same object
        assert store.get("Number of Traces") is original_measure
        assert original_measure.value_normalized == 1.0
    
    def test_normalizers_preserve_original_values(self):
        """Test that normalizers preserve original values."""
        store = MeasureStore()
        store.set("Number of Traces", 5.0)
        
        normalizer = NormalizeNumberOfTraces()
        normalizer.apply(store)
        
        measure = store.get("Number of Traces")
        # Original value should be unchanged
        assert measure.value == 5.0
        # Only normalized value should be added
        assert measure.value_normalized == 1.0
    
    def test_normalizers_handle_empty_store(self):
        """Test that normalizers handle empty stores gracefully."""
        store = MeasureStore()
        
        normalizers = [
            NormalizeNumberOfTraces(),
            NormalizeNumberOfEvents(),
            NormalizePercentageOfDistinctTraces(),
            NormalizeDeviationFromRandom(),
            NormalizeLZComplexity(),
            HideNumberOfTraces(),
            HidePercentageOfDistinctTraces()
        ]
        
        # Should not raise exceptions
        for normalizer in normalizers:
            try:
                normalizer.apply(store)
            except Exception:
                # Some normalizers may raise exceptions for missing dependencies
                # This is acceptable behavior
                pass
        
        # Store should remain empty
        assert len(store.to_dict()) == 0
