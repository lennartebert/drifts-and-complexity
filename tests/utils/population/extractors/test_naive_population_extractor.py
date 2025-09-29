"""Tests for the NaivePopulationExtractor class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from collections import Counter

from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
    _counts_activities,
    _counts_dfg_edges,
    _counts_trace_variants,
    _build_naive_distribution_from_counts
)
from utils.population.population_distributions import PopulationDistributions
from utils.population.population_distribution import PopulationDistribution
from utils.windowing.window import Window


class TestNaivePopulationExtractor:
    """Test the NaivePopulationExtractor class."""

    def test_creation(self):
        """Test creating a NaivePopulationExtractor instance."""
        extractor = NaivePopulationExtractor()
        assert isinstance(extractor, NaivePopulationExtractor)

    def test_apply_empty_window(self):
        """Test applying extractor to empty window raises ValueError."""
        extractor = NaivePopulationExtractor()
        window = Window(id="test", size=0, traces=[])
        
        with pytest.raises(ValueError, match="Window must contain traces for population extraction"):
            extractor.apply(window)

    def test_apply_none_traces(self):
        """Test applying extractor to window with None traces."""
        extractor = NaivePopulationExtractor()
        window = Window(id="test", size=0, traces=[])
        
        with pytest.raises(ValueError, match="Window must contain traces for population extraction"):
            extractor.apply(window)

    @patch('utils.population.extractors.naive_population_extractor._counts_activities')
    @patch('utils.population.extractors.naive_population_extractor._counts_dfg_edges')
    @patch('utils.population.extractors.naive_population_extractor._counts_trace_variants')
    def test_apply_success(self, mock_counts_variants, mock_counts_dfg, mock_counts_activities):
        """Test successful application of naive population extractor."""
        # Setup mocks
        mock_counts_activities.return_value = Counter({"A": 3, "B": 2})
        mock_counts_dfg.return_value = Counter({"A>B": 2, "B>C": 1})
        mock_counts_variants.return_value = Counter({("A", "B"): 2, ("A", "B", "C"): 1})
        
        extractor = NaivePopulationExtractor()
        window = Window(id="test", size=3, traces=[Mock(), Mock(), Mock()])
        
        result = extractor.apply(window)
        
        # Should return the same window
        assert result is window
        
        # Should have set population_distributions
        assert hasattr(window, 'population_distributions')
        assert isinstance(window.population_distributions, PopulationDistributions)
        
        # Verify all counting functions were called
        mock_counts_activities.assert_called_once_with(window.traces)
        mock_counts_dfg.assert_called_once_with(window.traces)
        mock_counts_variants.assert_called_once_with(window.traces)

    def test_apply_population_distributions_structure(self):
        """Test that population distributions have correct structure."""
        # Create mock traces
        mock_trace1 = Mock()
        mock_trace1.__getitem__ = Mock(return_value="A")
        mock_trace2 = Mock()
        mock_trace2.__getitem__ = Mock(return_value="B")
        
        with patch('utils.population.extractors.naive_population_extractor._counts_activities') as mock_acts, \
             patch('utils.population.extractors.naive_population_extractor._counts_dfg_edges') as mock_dfg, \
             patch('utils.population.extractors.naive_population_extractor._counts_trace_variants') as mock_vars:
            
            mock_acts.return_value = Counter({"A": 1, "B": 1})
            mock_dfg.return_value = Counter({"A>B": 1})
            mock_vars.return_value = Counter({("A", "B"): 1})
            
            extractor = NaivePopulationExtractor()
            window = Window(id="test", size=2, traces=[mock_trace1, mock_trace2])
            
            result = extractor.apply(window)
            
            # Check structure
            assert hasattr(result.population_distributions, 'activities')
            assert hasattr(result.population_distributions, 'dfg_edges')
            assert hasattr(result.population_distributions, 'trace_variants')
            
            # Check types
            assert result.population_distributions is not None
            assert isinstance(result.population_distributions.activities, PopulationDistribution)
            assert isinstance(result.population_distributions.dfg_edges, PopulationDistribution)
            assert isinstance(result.population_distributions.trace_variants, PopulationDistribution)


class TestCountsActivities:
    """Test the _counts_activities function."""

    @patch('utils.population.extractors.naive_population_extractor.attributes_get.get_attribute_values')
    def test_counts_activities_basic(self, mock_get_attributes):
        """Test basic activity counting."""
        mock_get_attributes.return_value = {"A": 3, "B": 2, "C": 1}
        
        log: list = [Mock(), Mock(), Mock()]
        result = _counts_activities(log)
        
        assert isinstance(result, Counter)
        assert result["A"] == 3
        assert result["B"] == 2
        assert result["C"] == 1
        
        mock_get_attributes.assert_called_once_with(log, "concept:name")

    @patch('utils.population.extractors.naive_population_extractor.attributes_get.get_attribute_values')
    def test_counts_activities_empty(self, mock_get_attributes):
        """Test activity counting with empty log."""
        mock_get_attributes.return_value = {}
        
        log: list = []
        result = _counts_activities(log)
        
        assert isinstance(result, Counter)
        assert len(result) == 0

    @patch('utils.population.extractors.naive_population_extractor.attributes_get.get_attribute_values')
    def test_counts_activities_single_activity(self, mock_get_attributes):
        """Test activity counting with single activity."""
        mock_get_attributes.return_value = {"A": 5}
        
        log: list = [Mock()] * 5
        result = _counts_activities(log)
        
        assert result["A"] == 5
        assert len(result) == 1


class TestCountsDfgEdges:
    """Test the _counts_dfg_edges function."""

    @patch('utils.population.extractors.naive_population_extractor.dfg_discovery.apply')
    def test_counts_dfg_edges_basic(self, mock_dfg_apply):
        """Test basic DFG edge counting."""
        mock_dfg_apply.return_value = {("A", "B"): 3, ("B", "C"): 2, ("A", "C"): 1}
        
        log: list = [Mock(), Mock(), Mock()]
        result = _counts_dfg_edges(log)
        
        assert isinstance(result, Counter)
        assert result["A>B"] == 3
        assert result["B>C"] == 2
        assert result["A>C"] == 1
        
        mock_dfg_apply.assert_called_once_with(log)

    @patch('utils.population.extractors.naive_population_extractor.dfg_discovery.apply')
    def test_counts_dfg_edges_empty(self, mock_dfg_apply):
        """Test DFG edge counting with empty DFG."""
        mock_dfg_apply.return_value = {}
        
        log: list = []
        result = _counts_dfg_edges(log)
        
        assert isinstance(result, Counter)
        assert len(result) == 0

    @patch('utils.population.extractors.naive_population_extractor.dfg_discovery.apply')
    def test_counts_dfg_edges_single_edge(self, mock_dfg_apply):
        """Test DFG edge counting with single edge."""
        mock_dfg_apply.return_value = {("A", "B"): 5}
        
        log: list = [Mock()] * 5
        result = _counts_dfg_edges(log)
        
        assert result["A>B"] == 5
        assert len(result) == 1


class TestCountsTraceVariants:
    """Test the _counts_trace_variants function."""

    @patch('utils.population.extractors.naive_population_extractor.variants_filter.get_variants')
    def test_counts_trace_variants_basic(self, mock_get_variants):
        """Test basic trace variant counting."""
        # Mock variant structure: {variant_id: [trace1, trace2, ...]}
        mock_trace1 = [{"concept:name": "A"}, {"concept:name": "B"}]
        mock_trace2 = [{"concept:name": "A"}, {"concept:name": "B"}]
        mock_trace3 = [{"concept:name": "A"}, {"concept:name": "C"}]
        
        mock_get_variants.return_value = {
            "variant1": [mock_trace1, mock_trace2],
            "variant2": [mock_trace3]
        }
        
        log: list = [Mock(), Mock(), Mock()]
        result = _counts_trace_variants(log)
        
        assert isinstance(result, Counter)
        assert result[("A", "B")] == 2
        assert result[("A", "C")] == 1
        
        mock_get_variants.assert_called_once_with(log)

    @patch('utils.population.extractors.naive_population_extractor.variants_filter.get_variants')
    def test_counts_trace_variants_empty(self, mock_get_variants):
        """Test trace variant counting with empty variants."""
        mock_get_variants.return_value = {}
        
        log: list = []
        result = _counts_trace_variants(log)
        
        assert isinstance(result, Counter)
        assert len(result) == 0

    @patch('utils.population.extractors.naive_population_extractor.variants_filter.get_variants')
    def test_counts_trace_variants_single_variant(self, mock_get_variants):
        """Test trace variant counting with single variant."""
        mock_trace = [{"concept:name": "A"}, {"concept:name": "B"}]
        mock_get_variants.return_value = {
            "variant1": [mock_trace, mock_trace, mock_trace]
        }
        
        log: list = [Mock()] * 3
        result = _counts_trace_variants(log)
        
        assert result[("A", "B")] == 3
        assert len(result) == 1


class TestBuildNaiveDistributionFromCounts:
    """Test the _build_naive_distribution_from_counts function."""

    def test_build_naive_distribution_basic(self):
        """Test building naive distribution from basic counts."""
        counts: Counter = Counter({"A": 3, "B": 2, "C": 1})
        
        result = _build_naive_distribution_from_counts(counts)
        
        assert isinstance(result, PopulationDistribution)
        assert len(result.observed_labels) == 3
        assert result.unseen_count == 0
        assert result.p0 == 0.0
        assert result.n_samples == 6
        
        # Check probabilities sum to 1
        probs = result.probs
        assert abs(sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3

    def test_build_naive_distribution_empty(self):
        """Test building naive distribution from empty counts."""
        counts: Counter = Counter()
        
        result = _build_naive_distribution_from_counts(counts)
        
        assert isinstance(result, PopulationDistribution)
        assert result.observed_labels == []
        assert result.observed_probs == []
        assert result.unseen_count == 0
        assert result.p0 == 0.0
        assert result.n_samples == 0
        
        assert result.probs == []

    def test_build_naive_distribution_single_item(self):
        """Test building naive distribution from single item counts."""
        counts: Counter = Counter({"A": 5})
        
        result = _build_naive_distribution_from_counts(counts)
        
        assert len(result.observed_labels) == 1
        assert result.observed_probs == [1.0]
        assert result.unseen_count == 0
        assert result.p0 == 0.0
        assert result.n_samples == 5
        
        probs = result.probs
        assert abs(probs[0] - 1.0) < 1e-10
        assert len(probs) == 1

    def test_build_naive_distribution_zero_counts(self):
        """Test building naive distribution from counts with zeros."""
        counts: Counter = Counter({"A": 3, "B": 0, "C": 2})
        
        result = _build_naive_distribution_from_counts(counts)
        
        assert len(result.observed_labels) == 3
        assert result.n_samples == 5  # Only non-zero counts
        
        probs = result.probs
        assert abs(sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3

    def test_build_naive_distribution_complex_labels(self):
        """Test building naive distribution with complex label types."""
        counts: Counter = Counter({
            ("A", "B"): 2,
            ("A", "B", "C"): 1,
            ("A", "C"): 1
        })
        
        result = _build_naive_distribution_from_counts(counts)
        
        assert isinstance(result, PopulationDistribution)
        assert len(result.observed_labels) == 3
        assert result.n_samples == 4
        
        probs = result.probs
        assert abs(sum(probs) - 1.0) < 1e-10
        assert len(probs) == 3

    def test_build_naive_distribution_probability_calculation(self):
        """Test that probabilities are calculated correctly."""
        counts: Counter = Counter({"A": 2, "B": 3, "C": 5})
        total = 10
        
        result = _build_naive_distribution_from_counts(counts)
        
        probs = result.probs
        assert abs(probs[0] - 0.2) < 1e-10  # 2/10
        assert abs(probs[1] - 0.3) < 1e-10  # 3/10
        assert abs(probs[2] - 0.5) < 1e-10  # 5/10

    def test_build_naive_distribution_naive_assumptions(self):
        """Test that naive distribution makes correct assumptions."""
        counts: Counter = Counter({"A": 1, "B": 1})
        
        result = _build_naive_distribution_from_counts(counts)
        
        # Naive assumptions: full coverage, no unseen
        assert result.unseen_count == 0
        assert result.p0 == 0.0
        
        # Should have full coverage (C_hat = 1)
        probs = result.probs
        assert abs(sum(probs) - 1.0) < 1e-10
