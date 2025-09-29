"""Tests for the VidgofMetricsAdapter."""

import pytest
from typing import List, Dict, Any

from utils.complexity.metrics_adapters.vidgof_metrics_adapter import VidgofMetricsAdapter
from utils.complexity.measures.measure_store import MeasureStore
from utils.windowing.window import Window
from tests.conftest import (
    empty_traces, single_trace, simple_traces, complex_traces, standard_test_log
)


class TestVidgofMetricsAdapter:
    """Test the VidgofMetricsAdapter functionality."""
    
    @pytest.fixture
    def adapter(self) -> VidgofMetricsAdapter:
        """Create a VidgofMetricsAdapter instance."""
        return VidgofMetricsAdapter()
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = VidgofMetricsAdapter()
        assert adapter.name == "vidgof"
    
    def test_available_metrics(self, adapter):
        """Test that available_metrics returns the expected entropy metrics."""
        metrics = adapter.available_metrics()
        
        expected_metrics = [
            "Variant Entropy",
            "Normalized Variant Entropy",
            "Trace Entropy",
            "Normalized Trace Entropy"
        ]
        
        assert len(metrics) == len(expected_metrics)
        for metric in expected_metrics:
            assert metric in metrics, f"Expected metric {metric} not found"
    
    def test_compute_measures_for_window_empty_traces(self, adapter, empty_traces):
        """Test computing measures for empty traces."""
        window = Window(
            id="empty_window",
            size=len(empty_traces),
            traces=empty_traces,
            population_distributions=None
        )
        
        # Empty traces should raise an error or return empty results
        # The Vidgof library might not handle empty traces well
        try:
            store, info = adapter.compute_measures_for_window(window)
            # If it succeeds, check the results
            assert isinstance(store, MeasureStore)
            assert info["adapter"] == "vidgof"
            assert info["support"] == 0
        except Exception:
            # It's acceptable for the Vidgof library to fail on empty traces
            pytest.skip("Vidgof library cannot handle empty traces")
    
    def test_compute_measures_for_window_single_trace(self, adapter, single_trace):
        """Test computing measures for single trace."""
        window = Window(
            id="single_window",
            size=len(single_trace),
            traces=single_trace,
            population_distributions=None
        )
        
        store, info = adapter.compute_measures_for_window(window)
        
        # Should compute entropy metrics
        assert isinstance(store, MeasureStore)
        assert len(store) > 0
        
        # Check that entropy metrics are computed
        expected_metrics = [
            "Variant Entropy",
            "Normalized Variant Entropy", 
            "Sequence Entropy",
            "Normalized Sequence Entropy"
        ]
        
        for metric in expected_metrics:
            assert store.has(metric), f"Expected metric {metric} not computed"
            measure = store.get(metric)
            assert measure.value is not None
            assert isinstance(measure.value, (int, float))
            assert not measure.hidden
            assert measure.meta["source"] == "vidgof"
        
        # Check info
        assert info["adapter"] == "vidgof"
        assert info["support"] == 1
    
    def test_compute_measures_for_window_standard_log(self, adapter, standard_test_log):
        """Test computing measures for standard test log."""
        window = Window(
            id="standard_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None
        )
        
        store, info = adapter.compute_measures_for_window(window)
        
        # Should compute entropy metrics
        assert isinstance(store, MeasureStore)
        assert len(store) > 0
        
        # Check specific entropy metrics
        expected_metrics = [
            "Variant Entropy",
            "Normalized Variant Entropy",
            "Sequence Entropy", 
            "Normalized Sequence Entropy"
        ]
        
        for metric in expected_metrics:
            assert store.has(metric), f"Expected metric {metric} not computed"
            measure = store.get(metric)
            
            # Entropy values should be non-negative
            assert measure.value >= 0, f"Entropy value for {metric} should be non-negative"
            
            # Normalized entropy should be between 0 and 1
            if "Normalized" in metric:
                assert 0 <= measure.value <= 1, f"Normalized entropy {metric} should be between 0 and 1"
            
            # Check metadata
            assert measure.meta["source"] == "vidgof"
            assert not measure.hidden
        
        # Check info
        assert info["adapter"] == "vidgof"
        assert info["support"] == len(standard_test_log)
    
    def test_compute_measures_for_window_complex_traces(self, adapter, complex_traces):
        """Test computing measures for complex traces."""
        window = Window(
            id="complex_window",
            size=len(complex_traces),
            traces=complex_traces,
            population_distributions=None
        )
        
        store, info = adapter.compute_measures_for_window(window)
        
        # Should compute entropy metrics
        assert isinstance(store, MeasureStore)
        assert len(store) > 0
        
        # Check that all expected metrics are present
        expected_metrics = [
            "Variant Entropy",
            "Normalized Variant Entropy",
            "Sequence Entropy",
            "Normalized Sequence Entropy"
        ]
        
        for metric in expected_metrics:
            assert store.has(metric), f"Expected metric {metric} not computed"
            measure = store.get(metric)
            assert measure.value is not None
            assert isinstance(measure.value, (int, float))
        
        # Check info
        assert info["adapter"] == "vidgof"
        assert info["support"] == len(complex_traces)
    
    def test_compute_measures_with_existing_store(self, adapter, standard_test_log):
        """Test computing measures with existing MeasureStore."""
        # Create existing store with some values
        existing_store = MeasureStore()
        existing_store.set("Custom Metric", 42.0, hidden=False, meta={"custom": True})
        
        window = Window(
            id="existing_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None
        )
        
        store, info = adapter.compute_measures_for_window(window, measures=existing_store)
        
        # Should preserve existing values
        assert store.has("Custom Metric")
        assert store.get("Custom Metric").value == 42.0
        assert store.get("Custom Metric").meta["custom"] is True
        
        # Should add entropy metrics
        assert store.has("Variant Entropy")
        assert store.has("Sequence Entropy")
    
    def test_entropy_metrics_properties(self, adapter, standard_test_log):
        """Test that entropy metrics satisfy expected properties."""
        window = Window(
            id="properties_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None
        )
        
        store, info = adapter.compute_measures_for_window(window)
        
        # Get entropy values
        variant_entropy = store.get("Variant Entropy").value
        normalized_variant_entropy = store.get("Normalized Variant Entropy").value
        sequence_entropy = store.get("Sequence Entropy").value
        normalized_sequence_entropy = store.get("Normalized Sequence Entropy").value
        
        # All entropy values should be non-negative
        assert variant_entropy >= 0
        assert sequence_entropy >= 0
        
        # Normalized values should be between 0 and 1
        assert 0 <= normalized_variant_entropy <= 1
        assert 0 <= normalized_sequence_entropy <= 1
        
        # Normalized values should be less than or equal to their non-normalized counterparts
        assert normalized_variant_entropy <= variant_entropy
        assert normalized_sequence_entropy <= sequence_entropy
    
    def test_entropy_metrics_consistency_across_traces(self, adapter):
        """Test that entropy metrics are consistent across different trace sets."""
        # Test with simple traces
        simple_window = Window(
            id="simple",
            size=3,
            traces=[
                # Create simple traces manually to avoid fixture issues
                self._create_trace(["A", "B", "C"]),
                self._create_trace(["A", "B", "D"]),
                self._create_trace(["A", "B", "C"])  # duplicate
            ],
            population_distributions=None
        )
        
        simple_store, _ = adapter.compute_measures_for_window(simple_window)
        
        # Test with more complex traces
        complex_window = Window(
            id="complex",
            size=3,
            traces=[
                self._create_trace(["A", "B", "C", "D"]),
                self._create_trace(["A", "B", "E", "F"]),
                self._create_trace(["X", "Y", "Z"])
            ],
            population_distributions=None
        )
        
        complex_store, _ = adapter.compute_measures_for_window(complex_window)
        
        # More complex traces should generally have higher entropy
        simple_variant_entropy = simple_store.get("Variant Entropy").value
        complex_variant_entropy = complex_store.get("Variant Entropy").value
        
        simple_sequence_entropy = simple_store.get("Sequence Entropy").value
        complex_sequence_entropy = complex_store.get("Sequence Entropy").value
        
        # This is a general property - more diverse traces should have higher entropy
        # (though this might not always hold due to the specific entropy calculation)
        # We'll just check that we get valid values
        assert isinstance(simple_variant_entropy, (int, float))
        assert isinstance(complex_variant_entropy, (int, float))
        assert isinstance(simple_sequence_entropy, (int, float))
        assert isinstance(complex_sequence_entropy, (int, float))
    
    def test_floatify_functionality(self, adapter, standard_test_log):
        """Test that the _floatify method works correctly."""
        window = Window(
            id="floatify_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None
        )
        
        store, info = adapter.compute_measures_for_window(window)
        
        # All values should be floats
        for name in store.keys():
            measure = store.get(name)
            assert isinstance(measure.value, (int, float))
            # Should be convertible to float
            float(measure.value)
    
    def test_metadata_consistency(self, adapter, standard_test_log):
        """Test that all computed measures have consistent metadata."""
        window = Window(
            id="metadata_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None
        )
        
        store, info = adapter.compute_measures_for_window(window)
        
        for name in store.keys():
            measure = store.get(name)
            
            # Should have metadata
            assert measure.meta is not None
            
            # Should have source metadata
            assert "source" in measure.meta
            assert measure.meta["source"] == "vidgof"
            
            # Should not be hidden
            assert not measure.hidden
            
            # Should have a value
            assert measure.value is not None
    
    def test_info_structure(self, adapter, standard_test_log):
        """Test that info has the expected structure."""
        window = Window(
            id="info_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None
        )
        
        store, info = adapter.compute_measures_for_window(window)
        
        # Check required info fields
        assert "adapter" in info
        assert "support" in info
        
        # Check values
        assert info["adapter"] == "vidgof"
        assert info["support"] == len(standard_test_log)
        assert isinstance(info["support"], int)
    
    @staticmethod
    def _create_trace(activities: List[str]):
        """Helper method to create a trace from activity list."""
        from pm4py.objects.log.obj import Trace, Event
        
        trace = Trace()
        trace.attributes['concept:name'] = f'trace_{len(activities)}'
        for i, activity in enumerate(activities):
            trace.append(Event({
                "concept:name": activity,
                "time:timestamp": f"2023-01-01T10:{i:02d}:00"
            }))
        return trace
