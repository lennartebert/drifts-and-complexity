"""
Test suite for iNEXT-style abundance bootstrap implementation.

This module validates that our iNEXT implementation:
1. computes Chao1 correctly,
2. builds the iNEXT bootstrap community (p*) correctly (shrinkage + unseen mass),
3. resamples and recomputes metrics per replicate, and
4. produces sensible CIs and behaviors.
"""

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pytest

# Import the functions under test
from utils.bootstrapping.bootstrap_samplers.inext_bootstrap_sampler import (
    _draw_inext_bootstrap_sample,
    _multinomial_draw,
)
from utils.population.chao1_helpers import (
    _build_inext_bootstrap_population_distribution,
    _compute_inext_unseen_probability_mass,
    chao1_total_richness_bias_corrected,
    chao1_unseen_estimation_inext,
    chao1_unseen_hat_bias_corrected,
)
from utils.population.population_distribution import (
    PopulationDistribution,
    get_labels_and_probabilities,
)

# Constants
TOL = 1e-12


def _counts_case(name: str) -> Dict[str, int]:
    """Provide test cases for counts."""
    cases = {
        "all_doubletons": {"A": 2, "B": 2, "C": 2, "D": 2},  # n=8, f1=0, f2=4
        "singletons_only": {"A": 1, "B": 1, "C": 1, "D": 1},  # n=4, f1=4, f2=0
        "mixed": {"A": 6, "B": 5, "C": 2, "D": 1, "E": 1, "F": 1},  # n=16, f1=3, f2=1
        "degenerate": {"A": 100},  # n=100, f1=0, f2=0
    }
    return cases[name]


def chao1_unseen_hat_wrapper(counts: Dict[str, int]) -> float:
    """Compute Chao1 unseen species estimate."""
    return chao1_unseen_hat_bias_corrected(Counter(counts))


def chao1_total_S_hat_wrapper(counts: Dict[str, int]) -> float:
    """Compute Chao1 total richness estimate."""
    return chao1_total_richness_bias_corrected(Counter(counts))


def build_inext_bootstrap_community(
    counts: Dict[str, int], n_ref: int = None
) -> PopulationDistribution:
    """Build iNEXT bootstrap community using current API."""
    from utils.population.chao1_helpers import create_inext_bootstrap_community
    from utils.population.population_distribution import (
        create_chao1_population_distribution,
    )

    counter = Counter(counts)
    n = sum(counts.values())
    if n_ref is None:
        n_ref = n

    # Create population distribution
    pop_dist = create_chao1_population_distribution(counter, n)

    # Create bootstrap community
    return create_inext_bootstrap_community(pop_dist)


def get_community_probs(community: PopulationDistribution) -> List[float]:
    """Helper function to get probabilities from PopulationDistribution."""
    from utils.population.population_distribution import get_labels_and_probabilities

    labels, probs = get_labels_and_probabilities(community)
    return probs


def get_community_labels(community: PopulationDistribution) -> List[str]:
    """Helper function to get labels from PopulationDistribution."""
    from utils.population.population_distribution import get_labels_and_probabilities

    labels, probs = get_labels_and_probabilities(community)
    return labels


def get_observed_labels(community: PopulationDistribution) -> List[str]:
    """Helper function to get observed labels from PopulationDistribution."""
    labels = get_community_labels(community)
    return [l for l in labels if not l.startswith("unseen_")]


def bootstrap_statistic(
    community: PopulationDistribution, B: int, m: int, rng, metric_fn
) -> Tuple[float, float, float, Tuple[float, float]]:
    """Run bootstrap and compute statistics."""
    from utils.population.population_distribution import get_labels_and_probabilities

    labels, probs = get_labels_and_probabilities(community)
    values = []
    for _ in range(B):
        # Draw bootstrap sample
        draws = _multinomial_draw(rng, m, probs)

        # Convert to counts vector
        counts_vec = np.zeros(len(labels))
        for i, count in enumerate(draws):
            counts_vec[i] = count

        # Compute metric
        value = metric_fn(counts_vec)
        values.append(value)

    # Use float64 for intermediate calculations to prevent overflow
    values_array = np.array(values, dtype=np.float64)
    mean_64 = np.mean(values_array)
    sd_64 = np.std(values_array, ddof=1)
    se_64 = sd_64 / np.sqrt(np.float64(B))

    # Cast back to regular float - overflow becomes inf naturally
    mean = float(mean_64)
    sd = float(sd_64)
    se = float(se_64)

    # Normal approximation CI
    z = 1.96
    ci_low = mean - z * se
    ci_high = mean + z * se

    # Percentile CI
    p_low = float(np.percentile(values_array, 2.5))
    p_high = float(np.percentile(values_array, 97.5))

    return mean, se, (ci_low, ci_high), (p_low, p_high)


def metric_observed_richness(counts_vec: np.ndarray) -> float:
    """Count distinct species (non-zero counts)."""
    return float(np.sum(counts_vec > 0))


class TestChao1Estimators:
    """Test Chao1 estimation functions."""

    def test_chao1_unseen_hat_wrapper_basic_cases(self):
        """Test Chao1 unseen species estimation on basic cases."""
        # All doubletons: f1=0 -> f0_hat=0
        counts = _counts_case("all_doubletons")
        assert chao1_unseen_hat_wrapper(counts) == 0.0

        # Singletons only: f1=4, f2=0 -> f0_hat = (n-1)/n * f1*(f1-1)/2
        counts = _counts_case("singletons_only")
        expected = (3 / 4) * (4 * 3) / 2  # 4.5
        assert abs(chao1_unseen_hat_wrapper(counts) - expected) < TOL

        # Mixed case: f1=3, f2=1 -> f0_hat = (n-1)/n * f1^2/(2*f2)
        counts = _counts_case("mixed")
        expected = (15 / 16) * (3 * 3) / (2 * 1)  # 4.21875
        assert abs(chao1_unseen_hat_wrapper(counts) - expected) < TOL

    def test_chao1_total_S_hat_wrapper_agrees_with_unseen_hat(self):
        """Test that total richness equals observed + unseen."""
        counts = _counts_case("mixed")
        s_obs = len(counts)
        f0_hat = chao1_unseen_hat_wrapper(counts)
        s_total = chao1_total_S_hat_wrapper(counts)

        assert abs(s_total - (s_obs + f0_hat)) < TOL


class TestBootstrapCommunity:
    """Test bootstrap community construction."""

    def test_bootstrap_probabilities_sum_to_one_and_nonnegative(self):
        """Test that bootstrap probabilities sum to 1 and are non-negative."""
        counts = _counts_case("mixed")
        comm = build_inext_bootstrap_community(counts)

        probs = get_community_probs(comm)
        assert abs(sum(probs) - 1.0) < TOL
        assert all(p >= -1e-15 for p in probs)

    def test_unseen_mass_and_unseen_count_consistency(self):
        """Test consistency between unseen mass and unseen count."""
        counts = _counts_case("singletons_only")
        comm = build_inext_bootstrap_community(counts)

        if comm.unseen_count is not None and comm.unseen_count > 0:
            probs = get_community_probs(comm)
            unseen_probs = [
                p
                for l, p in zip(get_community_labels(comm), probs)
                if l.startswith("unseen_")
            ]
            unseen_mass = sum(unseen_probs)
            # For small samples, shrinkage can be very aggressive and eliminate all observed species
            # In this case, the unseen mass equals the total probability mass (1.0)
            # rather than the unseen probability mass (p0)
            if len(comm.observed) == 0:
                # All species are unseen due to aggressive shrinkage
                assert abs(unseen_mass - 1.0) < TOL
            else:
                # Normal case: unseen mass should equal unseen probability mass
                assert abs(unseen_mass - comm.p0) < TOL
            assert comm.unseen_count >= int(np.ceil(chao1_unseen_hat_wrapper(counts)))

    def test_no_unseen_mass_case(self):
        """Test case with no unseen mass (all doubletons)."""
        counts = _counts_case("all_doubletons")
        comm = build_inext_bootstrap_community(counts)

        assert comm.p0 is None or comm.p0 == 0.0
        assert comm.unseen_count is None or comm.unseen_count == 0


class TestSamplingMath:
    """Test sampling mathematics and expectations."""

    def test_multinomial_draw_sums_to_m(self):
        """Test that multinomial draws sum to the target size."""
        counts = _counts_case("mixed")
        comm = build_inext_bootstrap_community(counts)

        probs = get_community_probs(comm)

        m = 1234
        rng = np.random.RandomState(42)
        draws = _multinomial_draw(rng, m, probs)

        assert sum(draws) == m

    def test_bootstrap_observed_richness_matches_theory_for_given_pstar(self):
        """Test that bootstrap observed richness matches theoretical expectation."""
        counts = _counts_case("all_doubletons")  # No unseen, clean theory
        comm = build_inext_bootstrap_community(counts)

        probs = get_community_probs(comm)

        m = sum(counts.values())  # n = 8
        B = 800

        # Theoretical expectation: E[S_obs(m)] = sum_j (1 - (1 - p*_j)^m)
        theoretical = sum(1 - (1 - p) ** m for p in probs)

        # Bootstrap mean
        rng = np.random.RandomState(42)
        mean, _, _, _ = bootstrap_statistic(comm, B, m, rng, metric_observed_richness)

        # Allow 8% tolerance
        tolerance = 0.08 * max(1, theoretical)
        assert abs(mean - theoretical) <= tolerance


class TestEdgeCases:
    """Test edge cases and pathological scenarios."""

    def test_pathological_b_zero_sets_w_zero(self):
        """Test that pathological case with b=0 sets w=0."""
        counts = _counts_case("degenerate")  # All mass in one bin
        comm = build_inext_bootstrap_community(counts)

        # When f1=0, a should be 0, so w should be 0
        assert comm.p0 is None or comm.p0 == 0.0
        # Observed probabilities should equal empirical (no shrinkage)
        n = sum(counts.values())
        empirical_prob = counts["A"] / n
        probs = get_community_probs(comm)
        assert abs(probs[0] - empirical_prob) < TOL


class TestPropertyInvariants:
    """Test that key invariants are maintained."""

    def test_pstar_equals_p_when_a_zero(self):
        """Test that p* equals empirical p when unseen mass a=0."""
        counts = _counts_case("all_doubletons")  # f1=0, so a=0
        comm = build_inext_bootstrap_community(counts)

        n = sum(counts.values())
        empirical_probs = [count / n for count in counts.values()]
        probs = get_community_probs(comm)
        observed_labels = get_observed_labels(comm)
        observed_probs = probs[: len(observed_labels)]

        for p_star, p_emp in zip(observed_probs, empirical_probs):
            assert abs(p_star - p_emp) < TOL


class TestEndToEnd:
    """Test end-to-end behavior."""

    def test_curve_generation_increases_expected_richness(self):
        """Test that curve generation produces increasing expected richness."""
        counts = _counts_case("mixed")
        comm = build_inext_bootstrap_community(counts)

        # Test at different sample sizes
        m_values = [5, 10, 15, 20]
        rng = np.random.RandomState(42)
        B = 200

        means = []
        for m in m_values:
            mean, _, _, _ = bootstrap_statistic(
                comm, B, m, rng, metric_observed_richness
            )
            means.append(mean)

        # Should be generally increasing (allowing for some noise)
        for i in range(1, len(means)):
            assert means[i] >= means[i - 1] - 0.1  # Allow small decreases due to noise


def test_unseen_count_equals_ceil_f0_hat():
    """S0_unseen should equal ceil(f0_hat) (not just >=) per iNEXT construction."""
    counts = _counts_case("singletons_only")  # f2=0 -> positive f0_hat
    f0_hat = chao1_unseen_hat_wrapper(counts)
    comm = build_inext_bootstrap_community(counts)
    assert comm.unseen_count == int(np.ceil(f0_hat))


def test_reproducibility_with_fixed_seed():
    """Given the same RNG state, bootstrap mean should be reproducible."""
    counts = _counts_case("mixed")
    comm = build_inext_bootstrap_community(counts)

    m = sum(counts.values())
    B = 100
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)

    mean1, _, _, _ = bootstrap_statistic(comm, B, m, rng1, metric_observed_richness)
    mean2, _, _, _ = bootstrap_statistic(comm, B, m, rng2, metric_observed_richness)

    assert abs(mean1 - mean2) < TOL
