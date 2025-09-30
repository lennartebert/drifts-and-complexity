"""Tests for the PopulationDistribution class."""

from collections import Counter

import pytest

from utils.population.population_distribution import PopulationDistribution


class TestPopulationDistribution:
    """Test the PopulationDistribution dataclass."""

    def test_creation_basic(self):
        """Test basic population distribution creation."""
        observed = Counter({"A": 10, "B": 15, "C": 5})

        dist = PopulationDistribution(
            observed=observed,
            population=observed,
            count=len(observed),
            unseen_count=None,
            p0=None,
            n_samples=30,
        )

        assert dist.observed == observed
        assert dist.population == observed
        assert dist.count == 3
        assert dist.unseen_count is None
        assert dist.p0 is None
        assert dist.n_samples == 30

    def test_creation_with_unseen(self):
        """Test population distribution with unseen categories."""
        observed = Counter({"A": 20, "B": 10})
        population = Counter({"A": 20, "B": 10, "unseen_1": 5, "unseen_2": 5})

        dist = PopulationDistribution(
            observed=observed,
            population=population,
            count=4,
            unseen_count=2,
            p0=0.2,
            n_samples=50,
        )

        assert dist.observed == observed
        assert dist.population == population
        assert dist.count == 4
        assert dist.unseen_count == 2
        assert dist.p0 == 0.2
        assert dist.n_samples == 50

    def test_post_init_validation_negative_unseen_count(self):
        """Test validation of negative unseen_count."""
        with pytest.raises(
            ValueError, match="unseen_count must be non-negative if provided"
        ):
            PopulationDistribution(
                observed=Counter({"A": 10}),
                population=Counter({"A": 10}),
                count=1,
                unseen_count=-1,
                p0=0.0,
                n_samples=10,
            )

    def test_post_init_validation_p0_out_of_range(self):
        """Test validation of p0 outside [0, 1] range."""
        # p0 too low
        with pytest.raises(ValueError, match="p0 must be in \\[0, 1\\] if provided"):
            PopulationDistribution(
                observed=Counter({"A": 10}),
                population=Counter({"A": 10}),
                count=1,
                unseen_count=0,
                p0=-0.1,
                n_samples=10,
            )

        # p0 too high
        with pytest.raises(ValueError, match="p0 must be in \\[0, 1\\] if provided"):
            PopulationDistribution(
                observed=Counter({"A": 10}),
                population=Counter({"A": 10}),
                count=1,
                unseen_count=0,
                p0=1.1,
                n_samples=10,
            )

    def test_post_init_validation_negative_n_samples(self):
        """Test validation of negative n_samples."""
        with pytest.raises(ValueError, match="n_samples must be non-negative"):
            PopulationDistribution(
                observed=Counter({"A": 10}),
                population=Counter({"A": 10}),
                count=1,
                unseen_count=0,
                p0=0.0,
                n_samples=-1,
            )

    def test_immutability(self):
        """Test that the dataclass is immutable."""
        observed = Counter({"A": 10, "B": 5})

        dist = PopulationDistribution(
            observed=observed,
            population=observed,
            count=2,
            unseen_count=None,
            p0=None,
            n_samples=15,
        )

        # Should not be able to modify fields
        with pytest.raises(Exception):  # FrozenInstanceError
            dist.observed = Counter({"C": 1})

        with pytest.raises(Exception):  # FrozenInstanceError
            dist.unseen_count = 1

    def test_complex_labels(self):
        """Test with complex label types (tuples, strings, etc.)."""
        observed = Counter(
            {
                ("activity1", "activity2"): 5,  # DFG edge
                ("activity2", "activity3"): 3,
                ("activity1", "activity3"): 2,
            }
        )

        dist = PopulationDistribution(
            observed=observed,
            population=observed,
            count=len(observed),
            unseen_count=None,
            p0=None,
            n_samples=20,
        )

        assert dist.observed == observed
        assert dist.count == 3
        assert dist.n_samples == 20

    def test_edge_case_p0_one(self):
        """Test edge case where p0 = 1.0 (no observed mass)."""
        observed = Counter()
        population = Counter({"unseen_1": 5, "unseen_2": 5})

        dist = PopulationDistribution(
            observed=observed,
            population=population,
            count=2,
            unseen_count=2,
            p0=1.0,
            n_samples=10,
        )

        assert dist.observed == observed
        assert dist.population == population
        assert dist.count == 2
        assert dist.unseen_count == 2
        assert dist.p0 == 1.0

    def test_numerical_stability(self):
        """Test numerical stability with very small values."""
        observed = Counter({"A": 1, "B": 1})

        dist = PopulationDistribution(
            observed=observed,
            population=observed,
            count=2,
            unseen_count=None,
            p0=None,
            n_samples=2,
        )

        assert dist.observed == observed
        assert dist.count == 2
        assert dist.n_samples == 2
