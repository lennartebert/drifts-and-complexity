"""Tests for the PopulationDistributions class."""

import pytest
from typing import List, Tuple

from utils.population.population_distributions import PopulationDistributions
from utils.population.population_distribution import PopulationDistribution


class TestPopulationDistributions:
    """Test the PopulationDistributions dataclass."""

    def test_creation_basic(self):
        """Test basic PopulationDistributions creation."""
        # Create individual distributions
        activities = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.6, 0.4],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        dfg_edges = PopulationDistribution(
            observed_labels=[("A>B",), ("B>C",)],
            observed_probs=[0.7, 0.3],
            unseen_count=1,
            p0=0.1,
            n_samples=100
        )
        
        trace_variants = PopulationDistribution(
            observed_labels=[(("A", "B"),), (("A", "B", "C"),)],
            observed_probs=[0.8, 0.2],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        # Create PopulationDistributions
        dists = PopulationDistributions(
            activities=activities,
            dfg_edges=dfg_edges,
            trace_variants=trace_variants
        )
        
        assert dists.activities == activities
        assert dists.dfg_edges == dfg_edges
        assert dists.trace_variants == trace_variants

    def test_creation_with_different_parameters(self):
        """Test PopulationDistributions with different parameter combinations."""
        # Activities with unseen categories
        activities = PopulationDistribution(
            observed_labels=[("start",), ("process",), ("end",)],
            observed_probs=[0.4, 0.4, 0.2],
            unseen_count=2,
            p0=0.3,
            n_samples=200
        )
        
        # DFG edges with no unseen
        dfg_edges = PopulationDistribution(
            observed_labels=[("start>process",), ("process>end",)],
            observed_probs=[0.6, 0.4],
            unseen_count=0,
            p0=0.0,
            n_samples=200
        )
        
        # Trace variants with high unseen mass
        trace_variants = PopulationDistribution(
            observed_labels=[(("start", "process", "end"),)],
            observed_probs=[1.0],
            unseen_count=5,
            p0=0.8,
            n_samples=200
        )
        
        dists = PopulationDistributions(
            activities=activities,
            dfg_edges=dfg_edges,
            trace_variants=trace_variants
        )
        
        # Verify all distributions are correctly assigned
        assert dists.activities.count == 5  # 3 observed + 2 unseen
        assert dists.dfg_edges.count == 2  # 2 observed + 0 unseen
        assert dists.trace_variants.count == 6  # 1 observed + 5 unseen

    def test_equality_comparison(self):
        """Test equality comparison between PopulationDistributions."""
        # Create identical distributions
        activities1 = PopulationDistribution(
            observed_labels=[("A",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dfg_edges1 = PopulationDistribution(
            observed_labels=[("A>B",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        trace_variants1 = PopulationDistribution(
            observed_labels=[(("A", "B"),)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dists1 = PopulationDistributions(
            activities=activities1,
            dfg_edges=dfg_edges1,
            trace_variants=trace_variants1
        )
        
        # Create identical distributions
        activities2 = PopulationDistribution(
            observed_labels=[("A",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dfg_edges2 = PopulationDistribution(
            observed_labels=[("A>B",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        trace_variants2 = PopulationDistribution(
            observed_labels=[(("A", "B"),)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dists2 = PopulationDistributions(
            activities=activities2,
            dfg_edges=dfg_edges2,
            trace_variants=trace_variants2
        )
        
        assert dists1 == dists2

    def test_inequality_comparison(self):
        """Test inequality comparison between different PopulationDistributions."""
        # Create first set of distributions
        activities1 = PopulationDistribution(
            observed_labels=[("A",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dfg_edges1 = PopulationDistribution(
            observed_labels=[("A>B",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        trace_variants1 = PopulationDistribution(
            observed_labels=[(("A", "B"),)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dists1 = PopulationDistributions(
            activities=activities1,
            dfg_edges=dfg_edges1,
            trace_variants=trace_variants1
        )
        
        # Create different distributions
        activities2 = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.5, 0.5],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dfg_edges2 = PopulationDistribution(
            observed_labels=[("A>B",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        trace_variants2 = PopulationDistribution(
            observed_labels=[(("A", "B"),)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dists2 = PopulationDistributions(
            activities=activities2,
            dfg_edges=dfg_edges2,
            trace_variants=trace_variants2
        )
        
        assert dists1 != dists2

    def test_string_representation(self):
        """Test string representation of PopulationDistributions."""
        activities = PopulationDistribution(
            observed_labels=[("A",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dfg_edges = PopulationDistribution(
            observed_labels=[("A>B",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        trace_variants = PopulationDistribution(
            observed_labels=[(("A", "B"),)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dists = PopulationDistributions(
            activities=activities,
            dfg_edges=dfg_edges,
            trace_variants=trace_variants
        )
        
        repr_str = repr(dists)
        assert "PopulationDistributions" in repr_str
        assert "activities=" in repr_str
        assert "dfg_edges=" in repr_str
        assert "trace_variants=" in repr_str

    def test_accessing_individual_distributions(self):
        """Test accessing individual distribution properties."""
        activities = PopulationDistribution(
            observed_labels=[("start",), ("process",), ("end",)],
            observed_probs=[0.3, 0.4, 0.3],
            unseen_count=1,
            p0=0.2,
            n_samples=100
        )
        
        dfg_edges = PopulationDistribution(
            observed_labels=[("start>process",), ("process>end",)],
            observed_probs=[0.6, 0.4],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        trace_variants = PopulationDistribution(
            observed_labels=[(("start", "process", "end"),)],
            observed_probs=[1.0],
            unseen_count=2,
            p0=0.3,
            n_samples=100
        )
        
        dists = PopulationDistributions(
            activities=activities,
            dfg_edges=dfg_edges,
            trace_variants=trace_variants
        )
        
        # Test accessing properties through the distributions
        assert dists.activities.count == 4  # 3 observed + 1 unseen
        assert dists.dfg_edges.count == 2  # 2 observed + 0 unseen
        assert dists.trace_variants.count == 3  # 1 observed + 2 unseen
        
        # Test accessing probabilities
        activities_probs = dists.activities.probs
        dfg_edges_probs = dists.dfg_edges.probs
        trace_variants_probs = dists.trace_variants.probs
        
        assert len(activities_probs) == 4
        assert len(dfg_edges_probs) == 2
        assert len(trace_variants_probs) == 3

    def test_empty_distributions(self):
        """Test PopulationDistributions with empty distributions."""
        empty_dist = PopulationDistribution(
            observed_labels=[],
            observed_probs=[],
            unseen_count=0,
            p0=0.0,
            n_samples=0
        )
        
        dists = PopulationDistributions(
            activities=empty_dist,
            dfg_edges=empty_dist,
            trace_variants=empty_dist
        )
        
        assert dists.activities.count == 0
        assert dists.dfg_edges.count == 0
        assert dists.trace_variants.count == 0
        
        assert dists.activities.probs == []
        assert dists.dfg_edges.probs == []
        assert dists.trace_variants.probs == []

    def test_mixed_empty_and_non_empty(self):
        """Test PopulationDistributions with mix of empty and non-empty distributions."""
        empty_dist = PopulationDistribution(
            observed_labels=[],
            observed_probs=[],
            unseen_count=0,
            p0=0.0,
            n_samples=0
        )
        
        non_empty_dist = PopulationDistribution(
            observed_labels=[("A",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dists = PopulationDistributions(
            activities=non_empty_dist,
            dfg_edges=empty_dist,
            trace_variants=non_empty_dist
        )
        
        assert dists.activities.count == 1
        assert dists.dfg_edges.count == 0
        assert dists.trace_variants.count == 1

    def test_immutability_of_distributions(self):
        """Test that individual distributions can be modified independently."""
        activities = PopulationDistribution(
            observed_labels=[("A",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dfg_edges = PopulationDistribution(
            observed_labels=[("A>B",)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        trace_variants = PopulationDistribution(
            observed_labels=[(("A", "B"),)],
            observed_probs=[1.0],
            unseen_count=0,
            p0=0.0,
            n_samples=10
        )
        
        dists = PopulationDistributions(
            activities=activities,
            dfg_edges=dfg_edges,
            trace_variants=trace_variants
        )
        
        # Modify one distribution
        dists.activities.observed_probs[0] = 0.5
        dists.activities.observed_probs.append(0.5)
        dists.activities.observed_labels.append(("B",))
        
        # Other distributions should be unchanged
        assert dists.dfg_edges.observed_probs == [1.0]
        assert dists.trace_variants.observed_probs == [1.0]
