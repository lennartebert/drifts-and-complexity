"""
Chao1 family of estimators for species richness and unseen species estimation.

This module provides both basic and bias-corrected Chao1 estimators following
the methodology described in Chao (1984, 1987) and Hsieh, Ma & Chao (2016).

References:
- Chao, A. (1984). Nonparametric estimation of the number of classes in a population.
  Scandinavian Journal of Statistics, 11(4), 265-270.
- Chao, A. (1987). Estimating the population size for capture-recapture data with
  unequal catchability. Biometrics, 43(4), 783-791.
- Hsieh, T. C., Ma, K. H., & Chao, A. (2016). iNEXT: an R package for rarefaction and
  extrapolation of species diversity (Hill numbers). Methods in Ecology and Evolution, 7(12), 1451-1456.
"""

from __future__ import annotations

from collections import Counter
from math import ceil, comb
from typing import TYPE_CHECKING, Iterable, List, Tuple

if TYPE_CHECKING:
    from utils.population.population_distribution import PopulationDistribution


def chao1_unseen_hat_basic(counts: Counter) -> float:
    """
    Basic Chao1 estimator for unseen species count.

    This is the original Chao1 formula without bias correction:
        f0_hat = f1^2 / (2 * f2), if f2 > 0
        f0_hat = f1 * (f1 - 1) / 2, if f2 = 0

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for species.

    Returns
    -------
    float
        Estimated number of unseen species (f0_hat).
    """
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)

    if f1 == 0:
        return 0.0

    if f2 > 0:
        return (f1 * f1) / (2.0 * f2)
    else:
        return (f1 * (f1 - 1)) / 2.0


def chao1_unseen_hat_bias_corrected(counts: Counter) -> float:
    """
    Bias-corrected Chao1 estimator for unseen species count.

    This is the iNEXT-style bias-corrected formula:
        f0_hat = ((n-1)/n) * f1^2 / (2 * f2), if f2 > 0
        f0_hat = ((n-1)/n) * f1 * (f1 - 1) / 2, if f2 = 0

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for species.

    Returns
    -------
    float
        Bias-corrected estimate of unseen species count (f0_hat).
    """
    xs = list(counts.values())
    n = sum(xs)
    f1 = sum(1 for x in xs if x == 1)
    f2 = sum(1 for x in xs if x == 2)
    if f1 == 0:
        return 0.0
    if f2 > 0:
        return ((n - 1) / n) * (f1 * f1) / (2.0 * f2)
    return ((n - 1) / n) * (f1 * (f1 - 1)) / 2.0


def chao1_total_richness_basic(counts: Counter) -> float:
    """
    Basic Chao1 estimator for total species richness.

    S_hat = S_obs + f0_hat (basic formula)

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for species.

    Returns
    -------
    float
        Estimated total species richness.
    """
    s_obs = len(counts)
    f0_hat = chao1_unseen_hat_basic(counts)
    return float(s_obs + f0_hat)


def chao1_total_richness_bias_corrected(counts: Counter) -> float:
    """
    Bias-corrected Chao1 estimator for total species richness.

    S_hat = S_obs + f0_hat (bias-corrected formula)

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for species.

    Returns
    -------
    float
        Bias-corrected estimate of total species richness.
    """
    s_obs = len(counts)
    f0_hat = chao1_unseen_hat_bias_corrected(counts)
    return float(s_obs + f0_hat)


def chao1_unseen_estimation_inext(counts: Counter) -> Tuple[float, int]:
    """
    iNEXT-style bias-corrected Chao1 unseen species estimation.

    This function provides both the continuous estimate (f0_hat) and the
    discrete count of pseudo-species to create (S0 = ceil(f0_hat)).

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for species.

    Returns
    -------
    f0_hat : float
        Expected number of unseen species.
    S0 : int
        Ceiling of f0_hat, number of pseudo-species to create.
    """
    f0_hat = chao1_unseen_hat_bias_corrected(counts)
    S0 = max(0, int(ceil(f0_hat)))
    return f0_hat, S0


def chao1_coverage_estimate(counts: Counter) -> float:
    """
    Estimate sample coverage using Chao1 methodology.

    Coverage Ĉ = 1 - f1/n, where f1 is the number of singletons.

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for species.

    Returns
    -------
    float
        Estimated sample coverage (0 ≤ Ĉ ≤ 1).
    """
    n = sum(counts.values())
    if n == 0:
        return 0.0

    f1 = sum(1 for c in counts.values() if c == 1)
    return max(0.0, 1.0 - (f1 / n))


def chao1_unseen_probability_mass(counts: Counter) -> float:
    """
    Estimate unseen probability mass using Chao1 methodology.

    p0 = 1 - Ĉ, where Ĉ is the estimated coverage.

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for species.

    Returns
    -------
    float
        Estimated unseen probability mass (0 ≤ p0 ≤ 1).
    """
    coverage = chao1_coverage_estimate(counts)
    return 1.0 - coverage


def inext_center_line_from_counts(
    counts_dict: Counter, m_grid: List[int]
) -> List[float]:
    """
    Compute the iNEXT center line from original counts.

    This computes Ŝ(m) once from the original counts for each m in m_grid.
    This is the blue line in iNEXT plots.

    Parameters
    ----------
    counts_dict : Counter
        Original abundance counts.
    m_grid : List[int]
        Grid of sample sizes to evaluate.

    Returns
    -------
    List[float]
        Estimated species richness at each m in m_grid.
    """
    xs0 = [int(c) for c in counts_dict.values() if c > 0]
    return [inext_richness_at_size_m(xs0, int(m)) for m in m_grid]


def inext_richness_at_size_m(xs: Iterable[int], m: int) -> float:
    """
    iNEXT size-based estimator for abundance data.

    - m <= n: rarefaction WITHOUT replacement
    - m >  n: Chao extrapolation using A and f0_hat

    Parameters
    ----------
    xs : Iterable[int]
        Abundance counts for species.
    m : int
        Target sample size for richness estimation.

    Returns
    -------
    float
        Estimated species richness at sample size m.
    """
    xs = [int(x) for x in xs if x > 0]
    n = sum(xs)
    if n <= 0:
        return 0.0
    S_obs = len(xs)

    if m <= n:
        denom = comb(n, m)
        # sum_i [1 - C(n - x_i, m) / C(n, m)]
        return sum(1.0 - (comb(n - x, m) / denom) for x in xs)

    # m > n: extrapolation
    f1 = sum(1 for x in xs if x == 1)
    f2 = sum(1 for x in xs if x == 2)
    if f1 == 0:
        f0_hat = 0.0
    elif f2 > 0:
        f0_hat = ((n - 1) / n) * (f1 * f1) / (2.0 * f2)
    else:
        f0_hat = ((n - 1) / n) * (f1 * (f1 - 1)) / 2.0

    if f0_hat <= 0.0 or f1 == 0:
        return float(S_obs)
    A = (n * f0_hat) / (n * f0_hat + f1)
    return S_obs + f0_hat * (1.0 - (1.0 - A) ** (m - n))


# iNEXT Bootstrap Community Functions


def create_inext_bootstrap_community(
    pop_dist: "PopulationDistribution",
) -> "PopulationDistribution":
    """
    Create iNEXT bootstrap community as a PopulationDistribution.

    This implements the "EstiBootComm.Ind" approach from iNEXT:
    1. Shrink observed species probabilities based on rarity
    2. Allocate unseen probability mass equally across pseudo-species
    3. Return a PopulationDistribution for bootstrap sampling

    Parameters
    ----------
    pop_dist : PopulationDistribution
        Population distribution object.

    Returns
    -------
    PopulationDistribution
        Bootstrap community with shrunk observed probabilities and unseen species.

    Raises
    ------
    ValueError
        If no observed species exist.
    """
    if not pop_dist.observed:
        raise ValueError("Cannot build bootstrap community with no observed species")

    # Step 1: Bias-corrected unseen richness estimation
    f0_hat, S0 = chao1_unseen_estimation_inext(pop_dist.observed)

    # Step 2: iNEXT unseen probability mass calculation
    a, b, w = _compute_inext_unseen_probability_mass(pop_dist.observed, f0_hat)

    # Step 3: Build bootstrap community as PopulationDistribution
    return _build_inext_bootstrap_population_distribution(
        pop_dist.observed, f0_hat, S0, a, w, pop_dist.n_samples
    )


def _compute_inext_unseen_probability_mass(
    observed: Counter, f0_hat: float
) -> Tuple[float, float, float]:
    """
    Compute iNEXT unseen probability mass and shrinkage parameters.

    Parameters
    ----------
    observed : Counter
        Observed abundance counts.
    f0_hat : float
        Expected number of unseen species.

    Returns
    -------
    a : float
        Total probability mass for unseen species.
    b : float
        Shrinkage denominator (sum of p_i * (1 - p_i)^n).
    w : float
        Shrinkage weight for observed species.
    """
    n = sum(observed.values())
    f1 = sum(1 for c in observed.values() if c == 1)

    if f0_hat == 0 or f1 == 0:
        return 0.0, 0.0, 0.0

    # iNEXT coverage adjustment
    A = (n * f0_hat) / (n * f0_hat + f1)
    a = (f1 / n) * A  # Total unseen probability mass

    # Shrinkage denominator
    b = sum((count / n) * ((1 - count / n) ** n) for count in observed.values())

    # Shrinkage weight
    w = 0.0 if b == 0.0 else a / b

    return a, b, w


def _build_inext_bootstrap_population_distribution(
    observed: Counter, f0_hat: float, S0: int, a: float, w: float, n_samples: int
) -> PopulationDistribution:
    """
    Build the iNEXT bootstrap community as a PopulationDistribution.

    This function constructs the bootstrap community p* with explicit renormalization
    and invariant verification as per iNEXT specification.

    Parameters
    ----------
    observed : Counter
        Observed abundance counts.
    f0_hat : float
        Expected number of unseen species.
    S0 : int
        Number of pseudo-species to create.
    a : float
        Total probability mass for unseen species.
    w : float
        Shrinkage weight for observed species.
    n_samples : int
        Reference sample size.

    Returns
    -------
    PopulationDistribution
        Bootstrap community with shrunk observed probabilities and unseen species.
    """
    from utils.population.population_distribution import PopulationDistribution

    n = sum(observed.values())

    # Build shrunk observed species probabilities with rarity-based shrinkage
    probs_obs = []
    labels_obs = []
    shrunk_observed = Counter()

    for label, count in observed.items():
        p_i = count / n
        p_i_star = p_i * (1 - w * ((1 - p_i) ** n))
        probs_obs.append(p_i_star)
        labels_obs.append(label)
        # Convert back to counts by scaling by n_samples
        shrunk_count = int(p_i_star * n_samples)
        if shrunk_count > 0:
            shrunk_observed[label] = shrunk_count

    # Build unseen species probabilities with equal allocation
    probs_unseen = []
    labels_unseen = []
    if S0 > 0 and a > 0:
        unseen_mass_per_species = a / S0
        for i in range(1, S0 + 1):
            probs_unseen.append(unseen_mass_per_species)
            labels_unseen.append(f"unseen_{i}")

    # Combine observed first, unseen appended
    probs = probs_obs + probs_unseen
    labels = labels_obs + labels_unseen

    # Explicit renormalization
    s = float(sum(probs))
    if s <= 0:
        raise ValueError("Degenerate p* (sum=0).")
    # FP-safe renorm
    probs = [q / s for q in probs]

    # Invariants verification
    assert abs(sum(probs) - 1.0) < 1e-12, f"p* does not sum to 1: {sum(probs)}"
    if S0 > 0:
        unseen_sum = sum(probs[-S0:])
        assert abs(unseen_sum - a) < 1e-12, f"Unseen mass mismatch: {unseen_sum} vs {a}"

    # Build full population counter from normalized probabilities
    full_population = Counter()
    for label, prob in zip(labels, probs):
        count = int(prob * n_samples)
        if count > 0:
            full_population[label] = count

    # Build unseen species metadata
    unseen_count = 0
    unseen_prob_mass = 0.0
    if S0 > 0 and a > 0:
        unseen_count = S0
        unseen_prob_mass = a

    return PopulationDistribution(
        observed=shrunk_observed,
        population=full_population,
        count=len(full_population),
        unseen_count=unseen_count,
        p0=unseen_prob_mass,
        n_samples=n_samples,
    )
