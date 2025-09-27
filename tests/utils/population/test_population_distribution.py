"""Tests for PopulationDistribution class."""

import pytest
from utils.population.population_distribution import PopulationDistribution


class TestPopulationDistribution:
    """Test suite for PopulationDistribution class."""
    
    def test_basic_initialization(self):
        """Test basic initialization with valid parameters."""
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",), ("C",)],
            observed_probs=[0.5, 0.3, 0.2],
            unseen_count=2,
            p0=0.1,
            n_samples=100
        )
        
        assert len(dist.observed_labels) == 3
        assert len(dist.observed_probs) == 3
        assert dist.unseen_count == 2
        assert dist.p0 == 0.1
        assert dist.n_samples == 100
    
    def test_initialization_validation_errors(self):
        """Test validation errors during initialization."""
        # Test negative unseen_count
        with pytest.raises(ValueError, match="unseen_count must be >= 0"):
            PopulationDistribution(
                observed_labels=[("A",)],
                observed_probs=[1.0],
                unseen_count=-1,
                p0=0.0,
                n_samples=10
            )
        
        # Test invalid p0 (too high)
        with pytest.raises(ValueError, match="p0 must be within \\[0, 1\\]"):
            PopulationDistribution(
                observed_labels=[("A",)],
                observed_probs=[1.0],
                unseen_count=0,
                p0=1.5,
                n_samples=10
            )
        
        # Test invalid p0 (negative)
        with pytest.raises(ValueError, match="p0 must be within \\[0, 1\\]"):
            PopulationDistribution(
                observed_labels=[("A",)],
                observed_probs=[1.0],
                unseen_count=0,
                p0=-0.1,
                n_samples=10
            )
        
        # Test mismatched labels and probs lengths
        with pytest.raises(ValueError, match="observed_labels and observed_probs must have the same length"):
            PopulationDistribution(
                observed_labels=[("A",), ("B",)],
                observed_probs=[1.0],
                unseen_count=0,
                p0=0.0,
                n_samples=10
            )
    
    def test_empty_distribution(self):
        """Test empty distribution with no observed categories."""
        dist = PopulationDistribution(
            observed_labels=[],
            observed_probs=[],
            unseen_count=0,
            p0=0.0,
            n_samples=0
        )
        
        assert dist.probs == []
        assert dist.count == 0
    
    def test_pure_observed_distribution(self):
        """Test distribution with only observed categories (p0=0)."""
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",), ("C",)],
            observed_probs=[0.5, 0.3, 0.2],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        probs = dist.probs
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-10
        assert dist.count == 3
        
        # Check that probabilities are normalized
        expected = [0.5, 0.3, 0.2]
        for i in range(3):
            assert abs(probs[i] - expected[i]) < 1e-10
    
    def test_pure_unseen_distribution(self):
        """Test distribution with only unseen categories (observed mass = 0)."""
        dist = PopulationDistribution(
            observed_labels=[],
            observed_probs=[],
            unseen_count=5,
            p0=1.0,
            n_samples=0
        )
        
        probs = dist.probs
        assert len(probs) == 5
        assert dist.count == 5
        
        # All unseen categories should have equal probability
        expected_prob = 1.0 / 5
        for prob in probs:
            assert abs(prob - expected_prob) < 1e-10
    
    def test_mixed_observed_unseen_distribution(self):
        """Test distribution with both observed and unseen categories."""
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.6, 0.3],  # will be rescaled to sum to 0.8
            unseen_count=3,
            p0=0.2,
            n_samples=100
        )
        
        probs = dist.probs
        assert len(probs) == 5  # 2 observed + 3 unseen
        assert dist.count == 5
        assert abs(sum(probs) - 1.0) < 1e-10
        
        # First 2 are observed (should sum to 0.8)
        observed_sum = sum(probs[:2])
        assert abs(observed_sum - 0.8) < 1e-10
        
        # Last 3 are unseen (each should be p0/unseen_count = 0.2/3)
        unseen_prob = 0.2 / 3
        for i in range(2, 5):
            assert abs(probs[i] - unseen_prob) < 1e-10
    
    def test_probability_rescaling(self):
        """Test that observed probabilities are rescaled to sum to (1-p0)."""
        # Original probabilities sum to 1.5, should be rescaled to sum to 0.7
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",), ("C",)],
            observed_probs=[0.6, 0.6, 0.3],  # sum = 1.5
            unseen_count=2,
            p0=0.3,  # so observed should sum to 0.7
            n_samples=100
        )
        
        # Check that observed_probs was modified in-place
        expected_scale = 0.7 / 1.5
        assert abs(dist.observed_probs[0] - 0.6 * expected_scale) < 1e-10
        assert abs(dist.observed_probs[1] - 0.6 * expected_scale) < 1e-10
        assert abs(dist.observed_probs[2] - 0.3 * expected_scale) < 1e-10
        
        # Check full probability vector
        probs = dist.probs
        assert abs(sum(probs[:3]) - 0.7) < 1e-10  # observed part
        assert abs(sum(probs[3:]) - 0.3) < 1e-10  # unseen part
    
    def test_zero_observed_probs_with_unseen(self):
        """Test distribution with zero observed mass but unseen categories."""
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.0, 0.0],
            unseen_count=4,
            p0=1.0,
            n_samples=0
        )
        
        probs = dist.probs
        assert len(probs) == 6  # 2 observed (with 0 prob) + 4 unseen
        assert probs[0] == 0.0
        assert probs[1] == 0.0
        
        # Unseen categories get equal share
        unseen_prob = 1.0 / 4
        for i in range(2, 6):
            assert abs(probs[i] - unseen_prob) < 1e-10
    
    def test_cache_invalidation_on_mutation(self):
        """Test that cache is invalidated when observed_probs is mutated."""
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.4, 0.6],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        # Access probs to populate cache
        initial_probs = dist.probs
        assert abs(initial_probs[0] - 0.4) < 1e-10
        assert abs(initial_probs[1] - 0.6) < 1e-10
        
        # Mutate observed_probs in-place
        dist.observed_probs[0] = 0.8
        dist.observed_probs[1] = 0.2
        
        # Cache should be invalidated, new probs should reflect change
        new_probs = dist.probs
        assert abs(new_probs[0] - 0.8) < 1e-10
        assert abs(new_probs[1] - 0.2) < 1e-10
    
    def test_cache_persistence(self):
        """Test that cache persists when no mutations occur."""
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.4, 0.6],
            unseen_count=1,
            p0=0.2,
            n_samples=100
        )
        
        # Multiple accesses should use cached values
        probs1 = dist.probs
        probs2 = dist.probs
        count1 = dist.count
        count2 = dist.count
        
        # Should be the same list objects (not just equal values)
        assert probs1 is not probs2  # returns copy to prevent mutation
        assert count1 == count2
        
        # Values should be identical
        assert probs1 == probs2
    
    def test_edge_case_single_unseen_category(self):
        """Test distribution with exactly one unseen category."""
        dist = PopulationDistribution(
            observed_labels=[("A",)],
            observed_probs=[0.8],
            unseen_count=1,
            p0=0.2,
            n_samples=50
        )
        
        probs = dist.probs
        assert len(probs) == 2
        assert abs(probs[0] - 0.8) < 1e-10
        assert abs(probs[1] - 0.2) < 1e-10
        assert dist.count == 2
    
    def test_numerical_stability_tiny_p0(self):
        """Test numerical stability with very small p0."""
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.5, 0.5],
            unseen_count=1,
            p0=1e-15,
            n_samples=1000
        )
        
        probs = dist.probs
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-10
        
        # Observed should be very close to original
        assert abs(probs[0] - 0.5) < 1e-10
        assert abs(probs[1] - 0.5) < 1e-10
        assert probs[2] < 1e-10  # unseen should be tiny
    
    def test_properties_return_copies(self):
        """Test that probs property returns copies to prevent external mutation."""
        dist = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.4, 0.6],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        probs1 = dist.probs
        probs2 = dist.probs
        
        # Should be different objects
        assert probs1 is not probs2
        
        # But with same values
        assert probs1 == probs2
        
        # Mutating returned copy shouldn't affect the distribution
        probs1[0] = 999.0
        probs3 = dist.probs
        assert abs(probs3[0] - 0.4) < 1e-10
    
    def test_complex_labels(self):
        """Test with complex label structures like trace variants."""
        trace_variants = [
            ("A", "B", "C"),
            ("A", "D", "E", "F"),
            ("G",)
        ]
        
        dist = PopulationDistribution(
            observed_labels=trace_variants,
            observed_probs=[0.5, 0.3, 0.2],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        assert dist.observed_labels == trace_variants
        assert len(dist.probs) == 3
        assert dist.count == 3