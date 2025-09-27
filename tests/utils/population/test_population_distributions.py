"""Tests for PopulationDistributions class."""

import pytest
from utils.population.population_distribution import PopulationDistribution
from utils.population.population_distributions import PopulationDistributions


class TestPopulationDistributions:
    """Test suite for PopulationDistributions dataclass."""
    
    def test_basic_initialization(self):
        """Test basic initialization with PopulationDistribution objects."""
        activities = PopulationDistribution(
            observed_labels=[("A",), ("B",), ("C",)],
            observed_probs=[0.5, 0.3, 0.2],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        dfg_edges = PopulationDistribution(
            observed_labels=[("A>B",), ("B>C",)],
            observed_probs=[0.6, 0.4],
            unseen_count=1,
            p0=0.1,
            n_samples=80
        )
        
        trace_variants = PopulationDistribution(
            observed_labels=[("A", "B", "C"), ("A", "B", "D")],
            observed_probs=[0.7, 0.3],
            unseen_count=2,
            p0=0.2,
            n_samples=50
        )
        
        distributions = PopulationDistributions(
            activities=activities,
            dfg_edges=dfg_edges,
            trace_variants=trace_variants
        )
        
        assert distributions.activities is activities
        assert distributions.dfg_edges is dfg_edges
        assert distributions.trace_variants is trace_variants
    
    def test_dataclass_fields(self):
        """Test that PopulationDistributions is properly configured as a dataclass."""
        # Create minimal distributions
        minimal_dist = PopulationDistribution(
            observed_labels=[],
            observed_probs=[],
            unseen_count=0,
            p0=0.0,
            n_samples=0
        )
        
        distributions = PopulationDistributions(
            activities=minimal_dist,
            dfg_edges=minimal_dist,
            trace_variants=minimal_dist
        )
        
        # Check that fields are accessible
        assert hasattr(distributions, 'activities')
        assert hasattr(distributions, 'dfg_edges')
        assert hasattr(distributions, 'trace_variants')
        
        # Check dataclass behavior (should have __eq__, __repr__, etc.)
        distributions2 = PopulationDistributions(
            activities=minimal_dist,
            dfg_edges=minimal_dist,
            trace_variants=minimal_dist
        )
        
        assert distributions == distributions2
        assert str(distributions)  # Should have string representation
    
    def test_different_distribution_types(self):
        """Test with different types of distributions for each field."""
        # Activities: pure observed
        activities = PopulationDistribution(
            observed_labels=[("Start",), ("Process",), ("End",)],
            observed_probs=[0.2, 0.6, 0.2],
            unseen_count=0,
            p0=0.0,
            n_samples=1000
        )
        
        # DFG edges: mixed observed/unseen
        dfg_edges = PopulationDistribution(
            observed_labels=[("Start>Process",), ("Process>End",)],
            observed_probs=[0.4, 0.4],  # will be rescaled to sum to 0.8
            unseen_count=3,
            p0=0.2,
            n_samples=800
        )
        
        # Trace variants: mostly unseen
        trace_variants = PopulationDistribution(
            observed_labels=[("Start", "Process", "End")],
            observed_probs=[0.1],
            unseen_count=10,
            p0=0.9,
            n_samples=100
        )
        
        distributions = PopulationDistributions(
            activities=activities,
            dfg_edges=dfg_edges,
            trace_variants=trace_variants
        )
        
        # Verify each distribution maintains its properties
        assert distributions.activities.count == 3
        assert distributions.activities.p0 == 0.0
        
        assert distributions.dfg_edges.count == 5  # 2 observed + 3 unseen
        assert distributions.dfg_edges.p0 == 0.2
        
        assert distributions.trace_variants.count == 11  # 1 observed + 10 unseen
        assert distributions.trace_variants.p0 == 0.9
    
    def test_modification_after_creation(self):
        """Test that individual distributions can be accessed and their properties queried."""
        activities = PopulationDistribution(
            observed_labels=[("A",), ("B",)],
            observed_probs=[0.6, 0.4],
            unseen_count=0,
            p0=0.0,
            n_samples=100
        )
        
        dfg_edges = PopulationDistribution(
            observed_labels=[("A>B",)],
            observed_probs=[0.8],
            unseen_count=2,
            p0=0.2,
            n_samples=50
        )
        
        trace_variants = PopulationDistribution(
            observed_labels=[("A", "B")],
            observed_probs=[0.9],
            unseen_count=1,
            p0=0.1,
            n_samples=90
        )
        
        distributions = PopulationDistributions(
            activities=activities,
            dfg_edges=dfg_edges,
            trace_variants=trace_variants
        )
        
        # Access and verify properties through the container
        assert len(distributions.activities.probs) == 2
        assert len(distributions.dfg_edges.probs) == 3  # 1 observed + 2 unseen
        assert len(distributions.trace_variants.probs) == 2  # 1 observed + 1 unseen
        
        # Verify that modifying observed_probs still works
        distributions.activities.observed_probs[0] = 0.8
        distributions.activities.observed_probs[1] = 0.2
        
        new_probs = distributions.activities.probs
        assert abs(new_probs[0] - 0.8) < 1e-10
        assert abs(new_probs[1] - 0.2) < 1e-10
    
    def test_empty_distributions(self):
        """Test with empty distributions."""
        empty_dist = PopulationDistribution(
            observed_labels=[],
            observed_probs=[],
            unseen_count=0,
            p0=0.0,
            n_samples=0
        )
        
        distributions = PopulationDistributions(
            activities=empty_dist,
            dfg_edges=empty_dist,
            trace_variants=empty_dist
        )
        
        assert distributions.activities.count == 0
        assert distributions.dfg_edges.count == 0
        assert distributions.trace_variants.count == 0
        
        assert distributions.activities.probs == []
        assert distributions.dfg_edges.probs == []
        assert distributions.trace_variants.probs == []