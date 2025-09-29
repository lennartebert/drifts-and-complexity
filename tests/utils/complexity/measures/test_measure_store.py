"""Tests for the MeasureStore class."""

from typing import Any, Dict

import pytest

from utils.complexity.measures.measure import Measure
from utils.complexity.measures.measure_store import MeasureStore


class TestMeasureStore:
    """Test the MeasureStore class."""

    def test_measurestore_creation_empty(self):
        """Test creating an empty MeasureStore."""
        store = MeasureStore()

        assert len(store) == 0
        assert list(store.keys()) == []
        assert store.to_dict() == {}

    def test_measurestore_creation_with_initial_data(self):
        """Test creating MeasureStore with initial data."""
        initial_measures = {
            "metric1": Measure(name="metric1", value=10.0),
            "metric2": Measure(name="metric2", value=20.0, hidden=True),
        }
        store = MeasureStore(initial_measures)

        assert len(store) == 2
        assert store.has("metric1")
        assert store.has("metric2")
        metric1 = store.get("metric1")
        metric2 = store.get("metric2")
        assert metric1 is not None and metric1.value == 10.0
        assert metric2 is not None and metric2.value == 20.0

    def test_measurestore_creation_with_other_store(self):
        """Test creating MeasureStore from another MeasureStore."""
        original = MeasureStore()
        original.set("test_metric", 15.0, meta={"source": "original"})

        new_store = MeasureStore(original)

        assert len(new_store) == 1
        test_metric = new_store.get("test_metric")
        assert test_metric is not None and test_metric.value == 15.0
        assert test_metric.meta["source"] == "original"

    def test_set_and_get_measure(self):
        """Test setting and getting measures."""
        store = MeasureStore()

        store.set("Test Metric", 42.0, hidden=True, meta={"source": "test"})

        assert store.has("Test Metric")
        assert not store.has("Non-existent")

        measure = store.get("Test Metric")
        assert measure is not None
        assert measure.name == "Test Metric"
        assert measure.value == 42.0
        assert measure.hidden is True
        assert measure.meta["source"] == "test"

    def test_set_measure_with_defaults(self):
        """Test setting measure with default parameters."""
        store = MeasureStore()

        store.set("Default Metric", 100.0)

        measure = store.get("Default Metric")
        assert measure is not None
        assert measure.name == "Default Metric"
        assert measure.value == 100.0
        assert measure.hidden is False
        assert measure.meta == {}

    def test_get_nonexistent_measure(self):
        """Test getting a non-existent measure."""
        store = MeasureStore()

        assert store.get("Non-existent") is None

    def test_get_value_basic(self):
        """Test getting just the value of a measure."""
        store = MeasureStore()
        store.set("Value Test", 3.14)

        assert store.get_value("Value Test") == 3.14
        assert store.get_value("Non-existent") is None

    def test_get_value_normalized(self):
        """Test getting normalized value when available."""
        store = MeasureStore()
        store.set("Normalized Test", 10.0, meta={"test": "data"})

        # Set normalized value manually
        measure = store.get("Normalized Test")
        assert measure is not None
        measure.value_normalized = 5.0

        assert store.get_value("Normalized Test") == 10.0
        assert (
            store.get_value("Normalized Test", get_normalized_if_available=True) == 5.0
        )

    def test_get_value_normalized_not_available(self):
        """Test getting normalized value when not available."""
        store = MeasureStore()
        store.set("No Normalized", 7.0)

        # Should return original value when normalized not available
        assert store.get_value("No Normalized", get_normalized_if_available=True) == 7.0

    def test_reveal_measure(self):
        """Test revealing hidden measures."""
        store = MeasureStore()
        store.set("Hidden Metric", 50.0, hidden=True)

        measure = store.get("Hidden Metric")
        assert measure is not None
        assert measure.hidden is True

        store.reveal("Hidden Metric")
        assert measure.hidden is False

    def test_reveal_nonexistent_measure(self):
        """Test revealing non-existent measure (should not raise)."""
        store = MeasureStore()

        # Should not raise exception
        store.reveal("Non-existent")

    def test_reveal_many_measures(self):
        """Test revealing multiple measures."""
        store = MeasureStore()
        store.set("Metric1", 1.0, hidden=True)
        store.set("Metric2", 2.0, hidden=True)
        store.set("Metric3", 3.0, hidden=False)  # Already visible

        store.reveal_many(["Metric1", "Metric2", "Metric3", "Non-existent"])

        metric1 = store.get("Metric1")
        metric2 = store.get("Metric2")
        metric3 = store.get("Metric3")
        assert metric1 is not None and metric1.hidden is False
        assert metric2 is not None and metric2.hidden is False
        assert metric3 is not None and metric3.hidden is False

    def test_to_visible_dict(self):
        """Test getting visible measures as dictionary."""
        store = MeasureStore()
        store.set("Visible1", 10.0, hidden=False)
        store.set("Visible2", 20.0, hidden=False)
        store.set("Hidden1", 30.0, hidden=True)

        visible_dict = store.to_visible_dict()

        assert "Visible1" in visible_dict
        assert "Visible2" in visible_dict
        assert "Hidden1" not in visible_dict
        assert visible_dict["Visible1"] == 10.0
        assert visible_dict["Visible2"] == 20.0

    def test_to_visible_dict_normalized(self):
        """Test getting visible measures with normalized values."""
        store = MeasureStore()
        store.set("Normalized", 10.0, hidden=False)

        # Set normalized value
        measure = store.get("Normalized")
        assert measure is not None
        measure.value_normalized = 5.0

        visible_dict = store.to_visible_dict(get_normalized_if_available=True)
        assert visible_dict["Normalized"] == 5.0

        visible_dict_orig = store.to_visible_dict(get_normalized_if_available=False)
        assert visible_dict_orig["Normalized"] == 10.0

    def test_to_dict(self):
        """Test getting all measures as dictionary."""
        store = MeasureStore()
        store.set("Metric1", 1.0, hidden=False)
        store.set("Metric2", 2.0, hidden=True)

        all_dict = store.to_dict()

        assert len(all_dict) == 2
        assert "Metric1" in all_dict
        assert "Metric2" in all_dict
        assert isinstance(all_dict["Metric1"], Measure)
        assert isinstance(all_dict["Metric2"], Measure)

    def test_update_from_measurestore(self):
        """Test updating from another MeasureStore."""
        store1 = MeasureStore()
        store1.set("Original", 100.0, meta={"source": "store1"})

        store2 = MeasureStore()
        store2.set("New", 200.0, meta={"source": "store2"})
        store2.set("Original", 150.0, meta={"source": "store2"})  # Overwrite

        store1.update_from(store2)

        assert len(store1) == 2
        original = store1.get("Original")
        new = store1.get("New")
        assert original is not None and original.value == 150.0
        assert original.meta["source"] == "store2"
        assert new is not None and new.value == 200.0

    def test_update_from_dict(self):
        """Test updating from a dictionary of measures."""
        store = MeasureStore()
        store.set("Existing", 50.0)

        new_measures = {
            "New1": Measure(name="New1", value=75.0),
            "New2": Measure(name="New2", value=100.0, hidden=True),
        }

        store.update_from(new_measures)

        assert len(store) == 3
        existing = store.get("Existing")
        new1 = store.get("New1")
        new2 = store.get("New2")
        assert existing is not None and existing.value == 50.0
        assert new1 is not None and new1.value == 75.0
        assert new2 is not None and new2.value == 100.0
        assert new2.hidden is True

    def test_len_and_keys(self):
        """Test length and keys methods."""
        store = MeasureStore()
        assert len(store) == 0
        assert list(store.keys()) == []

        store.set("Metric1", 1.0)
        store.set("Metric2", 2.0)

        assert len(store) == 2
        keys = list(store.keys())
        assert "Metric1" in keys
        assert "Metric2" in keys
        assert len(keys) == 2

    def test_overwrite_measure(self):
        """Test overwriting an existing measure."""
        store = MeasureStore()
        store.set("Overwrite", 10.0, hidden=False, meta={"version": "1"})

        # Overwrite with different values
        store.set("Overwrite", 20.0, hidden=True, meta={"version": "2"})

        measure = store.get("Overwrite")
        assert measure is not None
        assert measure.value == 20.0
        assert measure.hidden is True
        assert measure.meta["version"] == "2"

    def test_float_conversion(self):
        """Test that values are converted to float."""
        store = MeasureStore()
        store.set("Int Value", 42)  # int
        store.set("String Value", 3.14)  # float value

        assert store.get_value("Int Value") == 42.0
        assert store.get_value("String Value") == 3.14

    def test_meta_preservation(self):
        """Test that metadata is preserved correctly."""
        store = MeasureStore()
        meta = {"source": "test", "version": "1.0", "nested": {"key": "value"}}
        store.set("Meta Test", 5.0, meta=meta)

        measure = store.get("Meta Test")
        assert measure is not None
        assert measure.meta == meta
        assert measure.meta["source"] == "test"
        assert measure.meta["nested"]["key"] == "value"

    def test_empty_meta_handling(self):
        """Test handling of None and empty meta."""
        store = MeasureStore()

        # None meta should become empty dict
        store.set("None Meta", 1.0, meta=None)
        none_meta_measure = store.get("None Meta")
        assert none_meta_measure is not None
        assert none_meta_measure.meta == {}

        # Empty dict should work
        store.set("Empty Meta", 2.0, meta={})
        empty_meta_measure = store.get("Empty Meta")
        assert empty_meta_measure is not None
        assert empty_meta_measure.meta == {}


class TestMeasureStoreIntegration:
    """Integration tests for Measure and MeasureStore working together."""

    def test_measure_name_consistency(self):
        """Test that measure names are consistent with store keys."""
        store = MeasureStore()
        store.set("Consistent Name", 42.0)

        measure = store.get("Consistent Name")
        assert measure is not None
        assert measure.name == "Consistent Name"

    def test_measure_modification_through_store(self):
        """Test modifying measures through the store."""
        store = MeasureStore()
        store.set("Modifiable", 10.0)

        measure = store.get("Modifiable")
        assert measure is not None
        measure.value_normalized = 5.0
        measure.hidden = True
        measure.meta["modified"] = True

        # Changes should be reflected
        modified_measure = store.get("Modifiable")
        assert modified_measure is not None
        assert modified_measure.value_normalized == 5.0
        assert modified_measure.hidden is True
        assert modified_measure.meta["modified"] is True

    def test_measure_visibility_workflow(self):
        """Test a complete workflow of measure visibility."""
        store = MeasureStore()

        # Add measures with different visibility
        store.set("Public", 1.0, hidden=False)
        store.set("Private", 2.0, hidden=True)
        store.set("Secret", 3.0, hidden=True)

        # Initially only public should be visible
        visible = store.to_visible_dict()
        assert "Public" in visible
        assert "Private" not in visible
        assert "Secret" not in visible

        # Reveal private measures
        store.reveal_many(["Private", "Secret"])

        # Now all should be visible
        visible = store.to_visible_dict()
        assert "Public" in visible
        assert "Private" in visible
        assert "Secret" in visible

    def test_normalization_workflow(self):
        """Test a complete normalization workflow."""
        store = MeasureStore()
        store.set("Raw Metric", 100.0)

        # Simulate normalization
        measure = store.get("Raw Metric")
        assert measure is not None
        measure.value_normalized = 50.0
        measure.meta["normalized_by"] = "TestNormalizer"

        # Test both original and normalized values
        assert store.get_value("Raw Metric") == 100.0
        assert store.get_value("Raw Metric", get_normalized_if_available=True) == 50.0

        # Test visible dict with normalization
        visible_orig = store.to_visible_dict(get_normalized_if_available=False)
        visible_norm = store.to_visible_dict(get_normalized_if_available=True)

        assert visible_orig["Raw Metric"] == 100.0
        assert visible_norm["Raw Metric"] == 50.0

    def test_complex_measure_lifecycle(self):
        """Test a complex lifecycle of measures."""
        store = MeasureStore()

        # Create initial measures
        store.set("Base Metric", 10.0, meta={"phase": "initial"})
        store.set("Derived Metric", 20.0, hidden=True, meta={"phase": "derived"})

        # Process measures
        base_measure = store.get("Base Metric")
        assert base_measure is not None
        base_measure.value_normalized = 1.0
        base_measure.meta["phase"] = "normalized"

        derived_measure = store.get("Derived Metric")
        assert derived_measure is not None
        derived_measure.value_normalized = 2.0
        derived_measure.hidden = False
        derived_measure.meta["phase"] = "processed"

        # Verify final state
        final_base = store.get("Base Metric")
        final_derived = store.get("Derived Metric")
        assert final_base is not None and final_base.value_normalized == 1.0
        assert final_base.meta["phase"] == "normalized"
        assert final_derived is not None and final_derived.hidden is False
        assert final_derived.value_normalized == 2.0
        assert final_derived.meta["phase"] == "processed"

        # All should be visible now
        visible = store.to_visible_dict()
        assert len(visible) == 2
        assert "Base Metric" in visible
        assert "Derived Metric" in visible
