"""Tests for the Measure class."""

from typing import Any, Dict

import pytest

from utils.complexity.measures.measure import Measure


class TestMeasure:
    """Test the Measure dataclass."""

    def test_measure_creation_basic(self):
        """Test basic measure creation with required fields."""
        measure = Measure(name="Test Metric", value=42.0)

        assert measure.name == "Test Metric"
        assert measure.value == 42.0
        assert measure.hidden is False
        assert measure.meta == {}
        assert measure.value_normalized is None

    def test_measure_creation_with_all_fields(self):
        """Test measure creation with all fields specified."""
        meta = {"source": "test", "version": "1.0"}
        measure = Measure(
            name="Complex Metric",
            value=3.14,
            hidden=True,
            meta=meta,
            value_normalized=1.0,
        )

        assert measure.name == "Complex Metric"
        assert measure.value == 3.14
        assert measure.hidden is True
        assert measure.meta == meta
        assert measure.value_normalized == 1.0

    def test_measure_creation_with_defaults(self):
        """Test measure creation with default values."""
        measure = Measure(name="Default Metric", value=0.0)

        assert measure.name == "Default Metric"
        assert measure.value == 0.0
        assert measure.hidden is False
        assert measure.meta == {}
        assert measure.value_normalized is None

    def test_measure_immutability(self):
        """Test that measure fields can be modified (dataclass allows this)."""
        measure = Measure(name="Mutable Metric", value=10.0)

        # Should be able to modify fields
        measure.value = 20.0
        measure.hidden = True
        measure.meta["new_key"] = "new_value"
        measure.value_normalized = 2.0

        assert measure.value == 20.0
        assert measure.hidden is True
        assert measure.meta["new_key"] == "new_value"
        assert measure.value_normalized == 2.0

    def test_measure_string_representation(self):
        """Test string representation of Measure."""
        measure = Measure(name="String Test", value=42.0, hidden=True)
        repr_str = repr(measure)

        assert "String Test" in repr_str
        assert "42.0" in repr_str
        assert "hidden=True" in repr_str

    def test_measure_equality(self):
        """Test measure equality comparison."""
        measure1 = Measure(name="Equal Test", value=5.0, meta={"key": "value"})
        measure2 = Measure(name="Equal Test", value=5.0, meta={"key": "value"})
        measure3 = Measure(name="Different", value=5.0, meta={"key": "value"})

        assert measure1 == measure2
        assert measure1 != measure3

    def test_measure_with_none_values(self):
        """Test measure with None values for optional fields."""
        measure = Measure(
            name="None Test", value=1.0, hidden=False, meta={}, value_normalized=None
        )

        assert measure.value_normalized is None
        assert measure.meta == {}
