"""
Basic test to verify pytest setup is working correctly.
"""

import pytest


def test_basic_math():
    """Test basic mathematical operations."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
    assert 10 / 2 == 5


def test_string_operations():
    """Test basic string operations."""
    assert "hello" + " " + "world" == "hello world"
    assert len("test") == 4
    assert "test".upper() == "TEST"


def test_list_operations():
    """Test basic list operations."""
    test_list = [1, 2, 3, 4, 5]
    assert len(test_list) == 5
    assert sum(test_list) == 15
    assert max(test_list) == 5
    assert min(test_list) == 1


@pytest.mark.unit
def test_unit_marker():
    """Test that pytest markers work."""
    assert True


@pytest.mark.slow
def test_slow_operation():
    """Test marked as slow (can be skipped with -m 'not slow')."""
    # Simulate a slow operation
    result = sum(i for i in range(1000))
    assert result == 499500
