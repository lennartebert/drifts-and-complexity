"""Tests for abstract PopulationExtractor class."""

import pytest
from abc import ABC
from utils.population.extractors.population_extractor import PopulationExtractor
from utils.windowing.window import Window


class TestPopulationExtractor:
    """Test suite for abstract PopulationExtractor class."""
    
    def test_is_abstract_class(self):
        """Test that PopulationExtractor is properly abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            PopulationExtractor()
    
    def test_abstract_methods(self):
        """Test that apply method is properly abstract."""
        # Create a concrete class that doesn't implement apply
        class IncompleteExtractor(PopulationExtractor):
            pass
        
        # Should not be able to instantiate
        with pytest.raises(TypeError):
            IncompleteExtractor()
    
    def test_concrete_implementation(self):
        """Test that a concrete implementation can be created."""
        class ConcreteExtractor(PopulationExtractor):
            def apply(self, window: Window) -> Window:
                # Minimal implementation
                return window
        
        # Should be able to instantiate and call
        extractor = ConcreteExtractor()
        assert isinstance(extractor, PopulationExtractor)
        assert callable(extractor.apply)
    
    def test_inheritance_hierarchy(self):
        """Test inheritance relationships."""
        assert issubclass(PopulationExtractor, ABC)
        
        class TestExtractor(PopulationExtractor):
            def apply(self, window: Window) -> Window:
                return window
        
        extractor = TestExtractor()
        assert isinstance(extractor, PopulationExtractor)
        assert isinstance(extractor, ABC)
    
    def test_method_signature(self):
        """Test that the apply method has the correct signature."""
        class TestExtractor(PopulationExtractor):
            def apply(self, window: Window) -> Window:
                # Check that we can access the window parameter
                assert hasattr(window, 'traces')
                return window
        
        extractor = TestExtractor()
        
        # Create a mock window
        class MockWindow:
            def __init__(self):
                self.traces = []
        
        window = MockWindow()
        result = extractor.apply(window)
        assert result is window
    
    def test_docstring_requirements(self):
        """Test that the class has proper documentation."""
        assert PopulationExtractor.__doc__ is not None
        assert "population data" in PopulationExtractor.__doc__.lower()
        assert "window" in PopulationExtractor.__doc__.lower()
        
        # Test that apply method has docstring
        assert PopulationExtractor.apply.__doc__ is not None
        assert "window" in PopulationExtractor.apply.__doc__.lower()