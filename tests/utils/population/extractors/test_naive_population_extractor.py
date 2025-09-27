"""Tests for NaivePopulationExtractor class."""

import pytest
from collections import Counter
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
    _counts_activities,
    _counts_dfg_edges,
    _counts_trace_variants,
    _build_naive_distribution_from_counts
)
from utils.population.extractors.population_extractor import PopulationExtractor
from utils.population.population_distribution import PopulationDistribution
from utils.windowing.window import Window


class TestNaivePopulationExtractorHelpers:
    """Test suite for helper functions used by NaivePopulationExtractor."""
    
    def test_counts_activities(self, simple_traces):
        """Test activity counting functionality."""
        counts = _counts_activities(simple_traces)
        
        # simple_traces fixture has: A->B->C, A->B->D, A->B->C
        # So A appears 3 times, B appears 3 times, C appears 2 times, D appears 1 time
        assert isinstance(counts, Counter)
        assert counts["A"] == 3
        assert counts["B"] == 3
        assert counts["C"] == 2
        assert counts["D"] == 1
    
    def test_counts_activities_empty(self, empty_traces):
        """Test activity counting with empty traces."""
        counts = _counts_activities(empty_traces)
        assert isinstance(counts, Counter)
        assert len(counts) == 0
    
    def test_counts_activities_single_trace(self, single_trace):
        """Test activity counting with single trace."""
        counts = _counts_activities(single_trace)
        assert isinstance(counts, Counter)
        assert counts["A"] == 1
        assert counts["B"] == 1
        assert len(counts) == 2
    
    def test_counts_dfg_edges(self, simple_traces):
        """Test DFG edge counting functionality."""
        counts = _counts_dfg_edges(simple_traces)
        
        # simple_traces has: A->B->C, A->B->D, A->B->C
        # So A>B appears 3 times, B>C appears 2 times, B>D appears 1 time
        assert isinstance(counts, Counter)
        assert counts["A>B"] == 3
        assert counts["B>C"] == 2
        assert counts["B>D"] == 1
        assert len(counts) == 3
    
    def test_counts_dfg_edges_empty(self, empty_traces):
        """Test DFG edge counting with empty traces."""
        counts = _counts_dfg_edges(empty_traces)
        assert isinstance(counts, Counter)
        assert len(counts) == 0
    
    def test_counts_dfg_edges_single_activity(self):
        """Test DFG edge counting with traces containing single activities."""
        from conftest import create_trace_from_activities
        traces = [create_trace_from_activities(["A"])]
        
        counts = _counts_dfg_edges(traces)
        assert isinstance(counts, Counter)
        assert len(counts) == 0  # No edges in single-activity traces
    
    def test_counts_trace_variants(self, simple_traces):
        """Test trace variant counting functionality."""
        counts = _counts_trace_variants(simple_traces)
        
        # simple_traces has: A->B->C, A->B->D, A->B->C
        # So (A,B,C) appears 2 times, (A,B,D) appears 1 time
        assert isinstance(counts, Counter)
        assert counts[("A", "B", "C")] == 2
        assert counts[("A", "B", "D")] == 1
        assert len(counts) == 2
    
    def test_counts_trace_variants_empty(self, empty_traces):
        """Test trace variant counting with empty traces."""
        counts = _counts_trace_variants(empty_traces)
        assert isinstance(counts, Counter)
        assert len(counts) == 0
    
    def test_build_naive_distribution_from_counts(self):
        """Test building naive distribution from counts."""
        counts = Counter({"A": 10, "B": 5, "C": 3})
        
        dist = _build_naive_distribution_from_counts(counts)
        
        assert isinstance(dist, PopulationDistribution)
        assert len(dist.observed_labels) == 3
        assert len(dist.observed_probs) == 3
        assert dist.unseen_count == 0
        assert dist.p0 == 0.0
        assert dist.n_samples == 18
        
        # Check probabilities are correct (naive assumes full coverage)
        probs = dist.probs
        assert abs(probs[0] - 10/18) < 1e-10  # Depends on order, but should be one of these values
        assert abs(probs[1] - 5/18) < 1e-10 or abs(probs[1] - 10/18) < 1e-10 or abs(probs[1] - 3/18) < 1e-10
        assert abs(probs[2] - 3/18) < 1e-10 or abs(probs[2] - 5/18) < 1e-10 or abs(probs[2] - 10/18) < 1e-10
        assert abs(sum(probs) - 1.0) < 1e-10
    
    def test_build_naive_distribution_empty_counts(self):
        """Test building naive distribution from empty counts."""
        counts = Counter()
        
        dist = _build_naive_distribution_from_counts(counts)
        
        assert len(dist.observed_labels) == 0
        assert len(dist.observed_probs) == 0
        assert dist.unseen_count == 0
        assert dist.p0 == 0.0
        assert dist.n_samples == 0
        assert dist.probs == []
        assert dist.count == 0


class TestNaivePopulationExtractor:
    """Test suite for NaivePopulationExtractor class."""
    
    def test_inheritance(self):
        """Test that NaivePopulationExtractor properly inherits from PopulationExtractor."""
        extractor = NaivePopulationExtractor()
        assert isinstance(extractor, PopulationExtractor)
        assert hasattr(extractor, 'apply')
        assert callable(extractor.apply)
    
    def test_apply_with_simple_traces(self, simple_traces):
        """Test applying extractor to simple traces."""
        window = Window(
            id="test_window",
            size=len(simple_traces),
            traces=simple_traces
        )
        
        extractor = NaivePopulationExtractor()
        result_window = extractor.apply(window)
        
        # Should return the same window object
        assert result_window is window
        
        # Window should now have population_distributions
        assert hasattr(window, 'population_distributions')
        assert window.population_distributions is not None
        
        distributions = window.population_distributions
        
        # Check activities distribution
        assert distributions.activities.count == 4  # A, B, C, D
        assert distributions.activities.p0 == 0.0  # Naive assumes full coverage
        assert distributions.activities.unseen_count == 0
        assert len(distributions.activities.probs) == 4
        assert abs(sum(distributions.activities.probs) - 1.0) < 1e-10
        
        # Check DFG edges distribution
        assert distributions.dfg_edges.count == 3  # A>B, B>C, B>D
        assert distributions.dfg_edges.p0 == 0.0
        assert distributions.dfg_edges.unseen_count == 0
        assert len(distributions.dfg_edges.probs) == 3
        assert abs(sum(distributions.dfg_edges.probs) - 1.0) < 1e-10
        
        # Check trace variants distribution
        assert distributions.trace_variants.count == 2  # (A,B,C), (A,B,D)
        assert distributions.trace_variants.p0 == 0.0
        assert distributions.trace_variants.unseen_count == 0
        assert len(distributions.trace_variants.probs) == 2
        assert abs(sum(distributions.trace_variants.probs) - 1.0) < 1e-10
    
    def test_apply_with_single_trace(self, single_trace):
        """Test applying extractor to single trace."""
        window = Window(
            id="single_window",
            size=len(single_trace),
            traces=single_trace
        )
        
        extractor = NaivePopulationExtractor()
        result_window = extractor.apply(window)
        
        distributions = result_window.population_distributions
        
        # Single trace: A -> B
        assert distributions.activities.count == 2  # A, B
        assert distributions.dfg_edges.count == 1  # A>B
        assert distributions.trace_variants.count == 1  # (A,B)
        
        # All should be naive (full coverage)
        assert distributions.activities.p0 == 0.0
        assert distributions.dfg_edges.p0 == 0.0
        assert distributions.trace_variants.p0 == 0.0
    
    def test_apply_with_empty_traces_raises_error(self, empty_traces):
        """Test that applying extractor to empty traces raises ValueError."""
        window = Window(
            id="empty_window",
            size=len(empty_traces),
            traces=empty_traces
        )
        
        extractor = NaivePopulationExtractor()
        
        with pytest.raises(ValueError, match="Window must contain traces for population extraction"):
            extractor.apply(window)
    
    def test_apply_with_complex_traces(self, complex_traces):
        """Test applying extractor to complex traces."""
        window = Window(
            id="complex_window",
            size=len(complex_traces),
            traces=complex_traces
        )
        
        extractor = NaivePopulationExtractor()
        result_window = extractor.apply(window)
        
        distributions = result_window.population_distributions
        
        # Should have created distributions for all three types
        assert distributions.activities is not None
        assert distributions.dfg_edges is not None
        assert distributions.trace_variants is not None
        
        # All should be naive distributions (p0=0, unseen_count=0)
        assert distributions.activities.p0 == 0.0
        assert distributions.activities.unseen_count == 0
        assert distributions.dfg_edges.p0 == 0.0
        assert distributions.dfg_edges.unseen_count == 0
        assert distributions.trace_variants.p0 == 0.0
        assert distributions.trace_variants.unseen_count == 0
        
        # Probabilities should sum to 1 for each distribution
        assert abs(sum(distributions.activities.probs) - 1.0) < 1e-10
        assert abs(sum(distributions.dfg_edges.probs) - 1.0) < 1e-10
        assert abs(sum(distributions.trace_variants.probs) - 1.0) < 1e-10
    
    def test_multiple_applications_same_window(self, simple_traces):
        """Test that applying extractor multiple times to same window works correctly."""
        window = Window(
            id="test_window",
            size=len(simple_traces),
            traces=simple_traces
        )
        
        extractor = NaivePopulationExtractor()
        
        # Apply twice
        result1 = extractor.apply(window)
        result2 = extractor.apply(window)
        
        # Should be same window
        assert result1 is result2 is window
        
        # Should have distributions (possibly overwritten)
        assert hasattr(window, 'population_distributions')
        distributions = window.population_distributions
        
        # Verify distributions are still valid
        assert distributions.activities.p0 == 0.0
        assert distributions.dfg_edges.p0 == 0.0
        assert distributions.trace_variants.p0 == 0.0