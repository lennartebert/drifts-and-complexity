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
            n_reference=30,
            n_population=len(observed),
            observed_count=len(observed),
            population_count=len(observed),
            unseen_count=None,
            p0=None,
        )

        assert dist.observed == observed
        assert dist.population == observed
        assert dist.population_count == 3
        assert dist.unseen_count is None
        assert dist.p0 is None
        assert dist.n_reference == 30

    def test_creation_with_unseen(self):
        """Test population distribution with unseen categories."""
        observed = Counter({"A": 20, "B": 10})
        population = Counter({"A": 20, "B": 10, "unseen_1": 5, "unseen_2": 5})

        dist = PopulationDistribution(
            observed=observed,
            population=population,
            n_reference=50,
            n_population=4,
            observed_count=2,
            population_count=4,
            unseen_count=2,
            p0=0.2,
        )

        assert dist.observed == observed
        assert dist.population == population
        assert dist.population_count == 4
        assert dist.unseen_count == 2
        assert dist.p0 == 0.2
        assert dist.n_reference == 50

    def test_immutability(self):
        """Test that the dataclass is immutable."""
        observed = Counter({"A": 10, "B": 5})

        dist = PopulationDistribution(
            observed=observed,
            population=observed,
            n_reference=15,
            n_population=2,
            observed_count=2,
            population_count=2,
            unseen_count=None,
            p0=None,
        )

        # Should not be able to modify fields
        with pytest.raises(Exception):  # FrozenInstanceError
            object.__setattr__(dist, "observed", Counter({"C": 1}))

        with pytest.raises(Exception):  # FrozenInstanceError
            object.__setattr__(dist, "unseen_count", 1)

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
            n_reference=20,
            n_population=len(observed),
            observed_count=len(observed),
            population_count=len(observed),
            unseen_count=None,
            p0=None,
        )

        assert dist.observed == observed
        assert dist.population_count == 3
        assert dist.n_reference == 20

    def test_edge_case_p0_one(self):
        """Test edge case where p0 = 1.0 (no observed mass)."""
        observed: Counter = Counter()
        population = Counter({"unseen_1": 5, "unseen_2": 5})

        dist = PopulationDistribution(
            observed=observed,
            population=population,
            n_reference=10,
            n_population=2,
            observed_count=0,
            population_count=2,
            unseen_count=2,
            p0=1.0,
        )

        assert dist.observed == observed
        assert dist.population == population
        assert dist.population_count == 2
        assert dist.unseen_count == 2
        assert dist.p0 == 1.0

    def test_numerical_stability(self):
        """Test numerical stability with very small values."""
        observed = Counter({"A": 1, "B": 1})

        dist = PopulationDistribution(
            observed=observed,
            population=observed,
            n_reference=2,
            n_population=2,
            observed_count=2,
            population_count=2,
            unseen_count=None,
            p0=None,
        )

        assert dist.observed == observed
        assert dist.population_count == 2
        assert dist.n_reference == 2
