"""Tests for the PopulationDistribution class."""

from typing import List, Tuple

import pytest

from utils.population.population_distribution import PopulationDistribution


class TestPopulationDistribution:
    """Test the PopulationDistribution dataclass."""

    def test_creation_basic(self):
        """Test basic population distribution creation."""
        labels = [("A",), ("B",), ("C",)]
        probs = [0.4, 0.3, 0.3]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        assert dist.observed_labels == labels
        assert dist.observed_probs == probs
        assert dist.unseen_count == 0
        assert dist.p0 == 0.0
        assert dist.n_samples == 100

    def test_creation_with_unseen(self):
        """Test population distribution with unseen categories."""
        labels = [("A",), ("B",)]
        probs = [0.6, 0.4]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=2,
            p0=0.2,
            n_samples=50,
        )

        assert dist.observed_labels == labels
        # Probabilities are rescaled in __post_init__ to sum to (1 - p0)
        assert len(dist.observed_probs) == 2
        assert dist.unseen_count == 2
        assert dist.p0 == 0.2
        assert dist.n_samples == 50

    def test_post_init_validation_negative_unseen_count(self):
        """Test validation of negative unseen_count."""
        with pytest.raises(ValueError, match="unseen_count must be >= 0"):
            PopulationDistribution(
                observed_labels=[("A",)],
                observed_probs=[1.0],
                unseen_count=-1,
                p0=0.0,
                n_samples=10,
            )

    def test_post_init_validation_p0_out_of_range(self):
        """Test validation of p0 outside [0, 1] range."""
        # p0 too low
        with pytest.raises(ValueError, match="p0 must be within \\[0, 1\\]"):
            PopulationDistribution(
                observed_labels=[("A",)],
                observed_probs=[1.0],
                unseen_count=0,
                p0=-0.1,
                n_samples=10,
            )

        # p0 too high
        with pytest.raises(ValueError, match="p0 must be within \\[0, 1\\]"):
            PopulationDistribution(
                observed_labels=[("A",)],
                observed_probs=[1.0],
                unseen_count=0,
                p0=1.1,
                n_samples=10,
            )

    def test_post_init_validation_mismatched_lengths(self):
        """Test validation of mismatched label and probability lengths."""
        with pytest.raises(
            ValueError,
            match="observed_labels and observed_probs must have the same length",
        ):
            PopulationDistribution(
                observed_labels=[("A",), ("B",)],
                observed_probs=[1.0],  # Only one probability for two labels
                unseen_count=0,
                p0=0.0,
                n_samples=10,
            )

    def test_post_init_rescaling_observed_probs(self):
        """Test that observed probabilities are rescaled to sum to (1 - p0)."""
        labels = [("A",), ("B",), ("C",)]
        probs = [0.2, 0.3, 0.5]  # Sum to 1.0

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=1,
            p0=0.2,  # So observed should sum to 0.8
            n_samples=100,
        )

        # Should be rescaled to sum to 0.8
        expected_sum = 0.8
        actual_sum = sum(dist.observed_probs)
        assert abs(actual_sum - expected_sum) < 1e-10

    def test_post_init_no_rescaling_when_zero_mass(self):
        """Test that zero mass is not rescaled."""
        labels = [("A",), ("B",)]
        probs = [0.0, 0.0]  # Zero mass

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        # Should remain zeros
        assert dist.observed_probs == [0.0, 0.0]

    def test_probs_property_no_unseen(self):
        """Test probs property when there are no unseen categories."""
        labels = [("A",), ("B",), ("C",)]
        probs = [0.4, 0.3, 0.3]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        result_probs = dist.probs
        # Should be normalized to sum to 1.0
        assert abs(sum(result_probs) - 1.0) < 1e-10
        assert len(result_probs) == 3

    def test_probs_property_with_unseen(self):
        """Test probs property when there are unseen categories."""
        labels = [("A",), ("B",)]
        probs = [0.6, 0.4]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=2,
            p0=0.2,
            n_samples=100,
        )

        result_probs = dist.probs
        # Should have 4 total probabilities (2 observed + 2 unseen)
        assert len(result_probs) == 4
        # First two should be the rescaled observed probabilities
        # Last two should be p0/unseen_count = 0.2/2 = 0.1 each
        assert abs(result_probs[2] - 0.1) < 1e-10
        assert abs(result_probs[3] - 0.1) < 1e-10

    def test_probs_property_zero_mass(self):
        """Test probs property when there is zero total mass."""
        labels = [("A",), ("B",)]
        probs = [0.0, 0.0]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        result_probs = dist.probs
        # Should return empty list when no mass
        assert result_probs == []

    def test_count_property(self):
        """Test count property returns correct number of categories."""
        labels = [("A",), ("B",)]
        probs = [0.6, 0.4]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=3,
            p0=0.2,
            n_samples=100,
        )

        # Should be 2 observed + 3 unseen = 5 total
        assert dist.count == 5

    def test_count_property_zero_mass(self):
        """Test count property when there is zero mass."""
        labels = [("A",), ("B",)]
        probs = [0.0, 0.0]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        assert dist.count == 0

    def test_cache_invalidation(self):
        """Test that cache is invalidated when observed_probs changes."""
        labels = [("A",), ("B",)]
        probs = [0.6, 0.4]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        # Access to populate cache
        original_probs = dist.probs

        # Modify observed_probs in place
        dist.observed_probs[0] = 0.8
        dist.observed_probs[1] = 0.2

        # Should get new values (cache should be invalidated)
        new_probs = dist.probs
        assert new_probs != original_probs
        assert abs(new_probs[0] - 0.8) < 1e-10
        assert abs(new_probs[1] - 0.2) < 1e-10

    def test_cache_key_consistency(self):
        """Test that cache key is consistent for same inputs."""
        labels = [("A",), ("B",)]
        probs = [0.6, 0.4]

        dist1 = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=1,
            p0=0.2,
            n_samples=100,
        )

        dist2 = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=1,
            p0=0.2,
            n_samples=100,
        )

        # Should have same cache key
        assert dist1._current_key() == dist2._current_key()

    def test_edge_case_p0_one(self):
        """Test edge case where p0 = 1.0 (no observed mass)."""
        labels = [("A",), ("B",)]
        probs = [0.0, 0.0]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=2,
            p0=1.0,
            n_samples=100,
        )

        result_probs = dist.probs
        # Should have 2 observed + 2 unseen = 4 total categories
        assert len(result_probs) == 4
        # Last two should be unseen categories with probability 0.5 each
        assert abs(result_probs[2] - 0.5) < 1e-10
        assert abs(result_probs[3] - 0.5) < 1e-10

    def test_edge_case_unseen_count_zero(self):
        """Test edge case where unseen_count = 0."""
        labels = [("A",), ("B",)]
        probs = [0.6, 0.4]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        result_probs = dist.probs
        # Should only have observed probabilities, normalized to sum to 1
        assert len(result_probs) == 2
        assert abs(sum(result_probs) - 1.0) < 1e-10

    def test_probs_returns_copy(self):
        """Test that probs property returns a copy to prevent external mutation."""
        labels = [("A",), ("B",)]
        probs = [0.6, 0.4]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        result_probs = dist.probs
        result_probs[0] = 999.0  # Try to mutate

        # Should not affect the cached value
        new_probs = dist.probs
        assert new_probs[0] != 999.0

    def test_complex_labels(self):
        """Test with complex label types (tuples, strings, etc.)."""
        labels = [
            ("activity1", "activity2"),  # DFG edge
            ("activity2", "activity3"),
            ("activity1", "activity3"),
        ]
        probs = [0.5, 0.3, 0.2]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=1,
            p0=0.1,
            n_samples=50,
        )

        assert dist.observed_labels == labels
        assert dist.count == 4  # 3 observed + 1 unseen

    def test_numerical_stability(self):
        """Test numerical stability with very small probabilities."""
        labels = [("A",), ("B",)]
        probs = [1e-10, 1e-10]

        dist = PopulationDistribution(
            observed_labels=labels,
            observed_probs=probs,
            unseen_count=0,
            p0=0.0,
            n_samples=100,
        )

        # Should handle very small numbers without issues
        result_probs = dist.probs
        assert len(result_probs) == 2
        assert all(p >= 0 for p in result_probs)
        assert abs(sum(result_probs) - 1.0) < 1e-10
