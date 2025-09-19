"""
Population extraction helpers using Chao1 / sample coverage adjustments.

This module turns *observed* abundances (activities, directly-follows edges, and
trace variants) in a PM4Py event log into *estimated population models* with
(i) an estimated number of unseen species/items and (ii) estimated sample
coverage, following the Chao1 family of estimators. It then exposes a
`Chao1PopulationExtractor` that augments a `Window` with:

- `population_distributions`: an iNEXT-like mixture consisting of
  observed labels with reweighted probabilities that sum to the estimated
  coverage Ĉ, plus an aggregate unseen mass p0 = 1 - Ĉ and a count of
  unseen categories M_unseen.
- `population_counts`: deterministic richness estimates derived from the
  fitted distributions (Ŝ = S_obs + M_unseen for Chao1).

References (for context):
- Chao, A. (1984, 1987). Estimating the number of classes in a population.
- Chao & Jost (2012–2015). Coverage-based rarefaction and extrapolation.
"""

from __future__ import annotations

from collections import Counter
from typing import Tuple

from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.util import xes_constants as xes

from utils.population.extractors.population_extractor import \
    PopulationExtractor
from utils.population.population_distributions import (PopulationDistribution,
                                                       PopulationDistributions)
from utils.windowing.helpers import Window


# ---- observed abundance counters (PM4Py) ----
def _counts_activities(log) -> Counter[str]:
    """
    Count activity occurrences in the given PM4Py event log.

    Parameters
    ----------
    log : pm4py.objects.log.log.EventLog
        Event log of traces and events.

    Returns
    -------
    Counter[str]
        A Counter mapping activity label -> observed frequency (abundance).
    """
    # pm4py utility returns a dict of {activity: count}; wrap in Counter for convenience
    vals = attributes_get.get_attribute_values(log, xes.DEFAULT_NAME_KEY)
    return Counter(vals)


def _counts_dfg_edges(log) -> Counter[str]:
    """
    Count directly-follows (DFG) edge occurrences in the log.

    The DFG is discovered from the log and edge keys are encoded as "A>B".

    Parameters
    ----------
    log : pm4py.objects.log.log.EventLog

    Returns
    -------
    Counter[str]
        A Counter mapping "src>tgt" -> frequency.
    """
    dfg = dfg_discovery.apply(log)
    return Counter({f"{a}>{b}": c for (a, b), c in dfg.items()})


def _counts_trace_variants(log) -> Counter[Tuple[str, ...]]:
    """
    Count trace variants by their activity sequences.

    Variant keys are represented as tuples of activity labels.

    Parameters
    ----------
    log : pm4py.objects.log.log.EventLog

    Returns
    -------
    Counter[Tuple[str, ...]]
        A Counter mapping variant tuple -> frequency (number of traces with that variant).
    """
    varmap = variants_filter.get_variants(log)
    out = Counter()
    # `varmap` is {variant_id: [trace1, trace2, ...]}; we reconstruct a canonical tuple key
    for trs in varmap.values():
        # use the first representative trace to obtain the activity sequence
        key = tuple(ev[xes.DEFAULT_NAME_KEY] for ev in trs[0])
        out[key] = len(trs)
    return out


# ---- Chao / coverage helpers ----
def _chao1_S_hat_from_counts(counts: Counter) -> float:
    """
    Compute the Chao1 richness estimator Ŝ from observed abundances.

    Chao1 estimates the (asymptotic) number of categories as:
        Ŝ = S_obs + f1^2 / (2 f2), if f2 > 0
        Ŝ = S_obs + f1 (f1 - 1) / 2, if f2 = 0
    where f1 is the number of singletons and f2 the number of doubletons.

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for categories.

    Returns
    -------
    float
        Estimated total richness Ŝ.
    """
    s_obs = len(counts)
    if s_obs == 0:
        return 0.0
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)
    return float(s_obs + (f1 * f1) / (2.0 * f2)) if f2 > 0 else float(s_obs + (f1 * (f1 - 1)) / 2.0)


def _coverage_hat(N: int, f1: int, f2: int) -> float:
    """
    Estimate sample coverage Ĉ (Good–Turing style) using Chao's adjustment.

    We compute an estimate of the probability mass of observed species
    (coverage) as:
        Ĉ = 1 - (f1 / N) * A
    where A is a bias-correction term that depends on f2. For details, see
    Chao & Jost (2012–2015) on coverage-based rarefaction/extrapolation.

    Parameters
    ----------
    N : int
        Total number of observations (sum of abundances).
    f1 : int
        Number of singletons.
    f2 : int
        Number of doubletons.

    Returns
    -------
    float
        Estimated coverage Ĉ in [0, 1].
    """
    if N == 0 or f1 == 0:
        # No observations or no singletons -> treat as full coverage
        return 1.0
    if f2 > 0:
        A = (N - 1) * f1 / ((N - 1) * f1 + 2.0 * f2)
    else:
        # When no doubletons, use recommended correction
        A = (N - 1) * (f1 - 1) / ((N - 1) * (f1 - 1) + 2.0) if f1 > 1 else 1.0
    C = 1.0 - (f1 / N) * A
    # Clip to [0, 1] to avoid tiny numerical excursions
    return max(0.0, min(1.0, C))


def _build_chao_distribution_from_counts(counts: Counter) -> PopulationDistribution:
    """
    Fit an iNEXT-like population distribution from observed counts.

    The fitted model includes:
    - labels: observed category labels
    - probs: reweighted observed probabilities that sum to Ĉ
    - unseen_count: M_unseen = round(Ŝ - S_obs) (non-negative)
    - p0: unseen mass = 1 - Ĉ
    - n_ref: reference sample size N

    Parameters
    ----------
    counts : Counter
        Observed abundance counts for categories.

    Returns
    -------
    PopulationDistribution
        A parametric representation of the observed+unseen composition.
    """
    labels = list(counts.keys())
    obs = list(counts.values())
    N = int(sum(obs))
    s_obs = len(obs)
    f1 = sum(1 for c in obs if c == 1)
    f2 = sum(1 for c in obs if c == 2)

    # Richness estimate and implied count of unseen categories
    S_hat = _chao1_S_hat_from_counts(counts)
    M_unseen = max(0, int(round(S_hat - s_obs)))

    # Coverage (observed mass) and implied unseen mass p0
    C_hat = _coverage_hat(N, f1, f2)
    p0 = max(0.0, 1.0 - C_hat)

    # Reweight observed categories to sum to Ĉ; if N==0, distribute uniformly
    if N > 0:
        probs_obs = [C_hat * (c / N) for c in obs]
    elif s_obs > 0:
        probs_obs = [C_hat / s_obs] * s_obs
    else:
        probs_obs = []

    return PopulationDistribution(
        observed_labels=labels, observed_probs=probs_obs, unseen_count=M_unseen, p0=p0, n_samples=N
    )



class Chao1PopulationExtractor(PopulationExtractor):
    """
    Population extractor that augments a `Window` with Chao1-based estimates.

    Behavior
    --------
    - If `window.population_distributions` is already present, it is reused.
    - Otherwise, observed abundances are computed for:
        * Activities (event labels)
        * DFG edges ("A>B")
        * Trace variants (tuples of activity labels)
      and each is converted into a `PopulationDistribution` using Chao1 and
      coverage estimates.
    - `window.population_counts` is then set using the deterministic counts
      obtained from the fitted distributions (Ŝ = S_obs + M_unseen).

    Notes
    -----
    The distribution encodes both observed mass (Ĉ) and unseen mass (p0),
    enabling downstream procedures (e.g., coverage-standardized computations
    or extrapolations).
    """

    def apply(self, window: Window) -> Window:
        """
        Apply the Chao1 population modeling to the given window.

        Parameters
        ----------
        window : Window
            A time or trace window containing a PM4Py event log (in `window.traces`).

        Returns
        -------
        Window
            The same window, mutated to include:
            - `population_distributions` (activities, dfg_edges, trace_variants)
            - `population_counts` (richness estimates for the same three domains)
        """
        # Reuse existing fitted distributions if present (idempotent behavior on repeated calls)
        if window.population_distributions is not None:
            PD = window.population_distributions
        else:
            # Build from traces once (observed abundances -> iNEXT-like model)
            log = window.traces
            pd_acts = _build_chao_distribution_from_counts(_counts_activities(log))
            pd_dfg  = _build_chao_distribution_from_counts(_counts_dfg_edges(log))
            pd_vars = _build_chao_distribution_from_counts(_counts_trace_variants(log))
            PD = PopulationDistributions(
                activities=pd_acts, dfg_edges=pd_dfg, trace_variants=pd_vars
            )
            window.population_distributions = PD

        return window
