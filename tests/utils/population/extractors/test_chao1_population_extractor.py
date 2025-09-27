"""Tests for Chao1PopulationExtractor class."""

import pytest
from collections import Counter
import math
from utils.population.extractors.chao1_population_extractor import (
    Chao1PopulationExtractor,
    _counts_activities,
    _counts_dfg_edges, 
    _counts_trace_variants,
    _chao1_S_hat_from_counts,
    _coverage_hat,
    _build_chao_distribution_from_counts
)
from utils.population.extractors.population_extractor import PopulationExtractor
from utils.population.population_distribution import PopulationDistribution
from utils.windowing.window import Window
from conftest import assert_close


class TestChao1PopulationExtractorHelpers:
    """Test suite for helper functions used by Chao1PopulationExtractor."""
    
    def test_chao1_S_hat_from_counts_with_doubletons(self):
        """Test Chao1 richness estimator when f2 > 0."""
        # Create counts with known singleton and doubleton frequencies
        counts = Counter({
            "A": 5,  # not singleton or doubleton
            "B": 2,  # doubleton
            "C": 2,  # doubleton  
            "D": 1,  # singleton
            "E": 1,  # singleton
            "F": 3   # not singleton or doubleton
        })
        
        s_obs = 6  # observed species
        f1 = 2     # singletons (D, E)
        f2 = 2     # doubletons (B, C)
        
        expected = s_obs + (f1 * f1) / (2 * f2)  # 6 + 4/4 = 7
        
        result = _chao1_S_hat_from_counts(counts)
        assert_close(result, expected, 'default')
    
    def test_chao1_S_hat_from_counts_no_doubletons(self):
        """Test Chao1 richness estimator when f2 = 0."""
        counts = Counter({
            "A": 5,
            "B": 1,  # singleton
            "C": 1,  # singleton
            "D": 3
        })
        
        s_obs = 4
        f1 = 2
        f2 = 0
        
        # When f2 = 0, formula is: S_obs + f1*(f1-1)/2
        expected = s_obs + (f1 * (f1 - 1)) / 2.0  # 4 + 1 = 5
        
        result = _chao1_S_hat_from_counts(counts)
        assert_close(result, expected, 'default')
    
    def test_chao1_S_hat_from_counts_no_singletons(self):
        """Test Chao1 richness estimator when f1 = 0."""
        counts = Counter({
            "A": 5,
            "B": 2,
            "C": 3
        })
        
        s_obs = 3
        f1 = 0
        
        # When f1 = 0, should return just s_obs
        expected = s_obs
        
        result = _chao1_S_hat_from_counts(counts)
        assert_close(result, expected, 'default')
    
    def test_chao1_S_hat_from_counts_empty(self):
        """Test Chao1 richness estimator with empty counts."""
        counts = Counter()
        result = _chao1_S_hat_from_counts(counts)
        assert result == 0.0
    
    def test_coverage_hat_with_doubletons(self):
        """Test coverage estimator when f2 > 0."""
        N = 100
        f1 = 5
        f2 = 3
        
        # Formula: C = 1 - (f1/N) * A where A = (N-1)*f1 / ((N-1)*f1 + 2*f2)
        A = (N - 1) * f1 / ((N - 1) * f1 + 2 * f2)
        A = 99 * 5 / (99 * 5 + 6)  # 495 / 501
        expected = 1.0 - (f1 / N) * A
        expected = 1.0 - (5 / 100) * (495 / 501)
        
        result = _coverage_hat(N, f1, f2)
        assert_close(result, expected, 'default')
        assert 0.0 <= result <= 1.0  # Should be in valid range
    
    def test_coverage_hat_no_doubletons(self):
        """Test coverage estimator when f2 = 0."""
        N = 50
        f1 = 4
        f2 = 0
        
        # When f2 = 0, A = (N-1)*(f1-1) / ((N-1)*(f1-1) + 2)
        A = (N - 1) * (f1 - 1) / ((N - 1) * (f1 - 1) + 2)
        A = 49 * 3 / (49 * 3 + 2)  # 147 / 149
        expected = 1.0 - (f1 / N) * A
        expected = 1.0 - (4 / 50) * (147 / 149)
        
        result = _coverage_hat(N, f1, f2)
        assert_close(result, expected, 'default')
    
    def test_coverage_hat_no_singletons(self):
        """Test coverage estimator when f1 = 0 (full coverage)."""
        N = 100
        f1 = 0
        f2 = 5
        
        result = _coverage_hat(N, f1, f2)
        assert result == 1.0  # Should be full coverage
    
    def test_coverage_hat_edge_cases(self):
        """Test coverage estimator edge cases."""
        # N = 0
        assert _coverage_hat(0, 0, 0) == 1.0
        
        # f1 = 1, f2 = 0
        result = _coverage_hat(10, 1, 0)
        assert 0.0 <= result <= 1.0
    
    def test_build_chao_distribution_from_counts(self):
        """Test building Chao distribution from counts."""
        counts = Counter({
            "A": 10,
            "B": 5,
            "C": 1,  # singleton
            "D": 1,  # singleton
            "E": 2   # doubleton
        })
        
        dist = _build_chao_distribution_from_counts(counts)
        
        assert isinstance(dist, PopulationDistribution)
        assert len(dist.observed_labels) == 5
        assert len(dist.observed_probs) == 5
        assert dist.n_samples == 19  # sum of counts
        
        # Should have some unseen mass and categories (due to singletons)
        assert dist.p0 > 0.0
        assert dist.unseen_count >= 0
        
        # Coverage should be < 1 due to singletons
        C_hat = 1.0 - dist.p0
        assert C_hat < 1.0
        
        # Total probability should sum to 1
        probs = dist.probs
        assert_close(sum(probs), 1.0, 'default')
    
    def test_build_chao_distribution_empty_counts(self):
        """Test building Chao distribution from empty counts."""
        counts = Counter()
        
        dist = _build_chao_distribution_from_counts(counts)
        
        assert len(dist.observed_labels) == 0
        assert len(dist.observed_probs) == 0
        assert dist.unseen_count == 0
        assert dist.p0 == 0.0
        assert dist.n_samples == 0
        assert dist.count == 0
    
    def test_build_chao_distribution_no_singletons(self):
        """Test building Chao distribution when there are no singletons."""
        counts = Counter({
            "A": 10,
            "B": 5,
            "C": 3,
            "D": 2
        })
        
        dist = _build_chao_distribution_from_counts(counts)
        
        # Should have full coverage (no unseen mass)
        assert dist.p0 == 0.0 or dist.p0 < 1e-10
        assert dist.unseen_count == 0


class TestChao1PopulationExtractor:
    """Test suite for Chao1PopulationExtractor class."""
    
    def test_inheritance(self):
        """Test that Chao1PopulationExtractor properly inherits from PopulationExtractor."""
        extractor = Chao1PopulationExtractor()
        assert isinstance(extractor, PopulationExtractor)
        assert hasattr(extractor, 'apply')
        assert callable(extractor.apply)
    
    def test_apply_with_simple_traces(self, simple_traces):
        """Test applying Chao1 extractor to simple traces."""
        window = Window(
            id="test_window",
            size=len(simple_traces),
            traces=simple_traces
        )
        
        extractor = Chao1PopulationExtractor()
        result_window = extractor.apply(window)
        
        # Should return the same window object
        assert result_window is window
        
        # Window should now have population_distributions
        assert hasattr(window, 'population_distributions')
        assert window.population_distributions is not None
        
        distributions = window.population_distributions
        
        # Check that all distribution types were created
        assert distributions.activities is not None
        assert distributions.dfg_edges is not None
        assert distributions.trace_variants is not None
        
        # Should have valid probability distributions
        for dist in [distributions.activities, distributions.dfg_edges, distributions.trace_variants]:
            probs = dist.probs
            if len(probs) > 0:
                assert_close(sum(probs), 1.0, 'default')
                assert all(p >= 0 for p in probs)
            assert 0.0 <= dist.p0 <= 1.0
            assert dist.unseen_count >= 0
    
    def test_apply_preserves_chao1_properties(self, complex_traces):
        """Test that Chao1 estimator produces statistically reasonable results."""
        window = Window(
            id="complex_window",
            size=len(complex_traces),
            traces=complex_traces
        )
        
        extractor = Chao1PopulationExtractor()
        result_window = extractor.apply(window)
        
        distributions = result_window.population_distributions
        
        # For each distribution type, verify Chao1 properties
        for dist_name, dist in [
            ("activities", distributions.activities),
            ("dfg_edges", distributions.dfg_edges),
            ("trace_variants", distributions.trace_variants)
        ]:
            # Richness estimate should be >= observed richness
            observed_richness = len(dist.observed_labels)
            estimated_richness = dist.count
            assert estimated_richness >= observed_richness, f"Failed for {dist_name}"
            
            # Coverage should be reasonable (not exactly 0 or 1 unless justified)
            coverage = 1.0 - dist.p0
            assert 0.0 <= coverage <= 1.0, f"Invalid coverage for {dist_name}"
            
            # If there are unseen categories, p0 should be > 0
            if dist.unseen_count > 0:
                assert dist.p0 > 0, f"Unseen categories but p0=0 for {dist_name}"
    
    def test_apply_with_high_diversity_data(self):
        """Test Chao1 with data that should have many singletons (high diversity)."""
        from conftest import create_traces_from_patterns
        
        # Create many unique traces (high diversity, many singletons)
        patterns = [
            [f"Activity_{i}"] for i in range(20)  # 20 unique single-activity traces
        ]
        traces = create_traces_from_patterns(patterns)
        
        window = Window(
            id="high_diversity_window",
            size=len(traces),
            traces=traces
        )
        
        extractor = Chao1PopulationExtractor()
        result_window = extractor.apply(window)
        
        distributions = result_window.population_distributions
        
        # Should have high unseen mass due to many singletons
        assert distributions.activities.p0 > 0.1, "Should have significant unseen mass for high diversity data"
        assert distributions.trace_variants.p0 > 0.1, "Should have significant unseen mass for trace variants"
        
        # Should estimate more categories than observed
        assert distributions.activities.count > len(distributions.activities.observed_labels)
        assert distributions.trace_variants.count > len(distributions.trace_variants.observed_labels)
    
    def test_apply_with_low_diversity_data(self):
        """Test Chao1 with data that should have few singletons (low diversity)."""
        from conftest import create_traces_from_patterns
        
        # Create many repetitions of the same few patterns
        patterns = [
            ["A", "B", "C"],  # Repeat this pattern many times
            ["A", "B", "D"],
        ] * 20  # 40 traces total, but only 2 unique patterns
        traces = create_traces_from_patterns(patterns)
        
        window = Window(
            id="low_diversity_window", 
            size=len(traces),
            traces=traces
        )
        
        extractor = Chao1PopulationExtractor()
        result_window = extractor.apply(window)
        
        distributions = result_window.population_distributions
        
        # Should have relatively low unseen mass due to few singletons
        assert distributions.activities.p0 < 0.5, "Should have relatively low unseen mass for low diversity data"
        
        # Estimated richness should be close to observed for well-sampled populations
        activities_ratio = distributions.activities.count / len(distributions.activities.observed_labels)
        assert activities_ratio < 2.0, "Richness estimate shouldn't be too much higher than observed"
    
    def test_apply_multiple_times_same_result(self, simple_traces):
        """Test that applying Chao1 extractor multiple times gives consistent results."""
        window = Window(
            id="test_window",
            size=len(simple_traces),
            traces=simple_traces
        )
        
        extractor = Chao1PopulationExtractor()
        
        # Apply twice
        result1 = extractor.apply(window)
        first_distributions = window.population_distributions
        
        # Store first results
        first_activities_count = first_distributions.activities.count
        first_activities_p0 = first_distributions.activities.p0
        
        result2 = extractor.apply(window)
        second_distributions = window.population_distributions
        
        # Should give same results
        assert second_distributions.activities.count == first_activities_count
        assert_close(second_distributions.activities.p0, first_activities_p0, 'default')
    
    def test_statistical_consistency_with_known_data(self):
        """Test that Chao1 gives statistically reasonable results for known cases."""
        from conftest import create_traces_from_patterns
        
        # Create a known case: many singletons should lead to high richness estimate
        patterns = (
            [["Common", "Activity"]] * 50 +  # 50 copies of common pattern
            [[f"Rare_{i}"]] for i in range(10)  # 10 singleton patterns
        )
        traces = create_traces_from_patterns(patterns)
        
        window = Window(
            id="known_case_window",
            size=len(traces),
            traces=traces
        )
        
        extractor = Chao1PopulationExtractor()
        result_window = extractor.apply(window)
        
        distributions = result_window.population_distributions
        
        # Should estimate more trace variants than observed due to singletons
        observed_variants = len(distributions.trace_variants.observed_labels)  # Should be 11
        estimated_variants = distributions.trace_variants.count
        
        assert estimated_variants > observed_variants, "Chao1 should estimate more variants due to singletons"
        
        # Coverage should be less than 1 due to singletons
        coverage = 1.0 - distributions.trace_variants.p0
        assert coverage < 0.95, "Coverage should be notably less than 1 with many singletons"