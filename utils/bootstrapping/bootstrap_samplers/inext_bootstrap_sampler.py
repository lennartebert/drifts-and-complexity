from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from typing import List, Optional

from utils.bootstrapping.bootstrap_samplers.bootstrap_sampler import BootstrapSampler
from utils.population.extractors.chao1_population_extractor import (
    Chao1PopulationExtractor,
)
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
)
from utils.population.population_distributions import (
    PopulationDistribution,
    PopulationDistributions,
)
from utils.windowing.window import Window


# -----------------
# small helpers
# -----------------
def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed)


def _multinomial_draw(rng: random.Random, n: int, probs: List[float]) -> List[int]:
    """
    Simple cumulative-CDF multinomial sampler.
    Returns a list of category counts that sum to n.
    """
    if n <= 0:
        return [0] * len(probs)
    s = sum(probs)
    if not probs:
        return []
    p = [x / s for x in probs] if s > 0 else [1.0 / len(probs)] * len(probs)

    # cumulative distribution
    cum, acc = [], 0.0
    for x in p:
        acc += x
        cum.append(acc)
    cum[-1] = 1.0  # absorb fp error

    out = [0] * len(p)
    for _ in range(n):
        u = rng.random()
        j = 0
        while j < len(cum) and u > cum[j]:
            j += 1
        out[j] += 1
    return out


def _draw_counts_for_species(
    rng: random.Random, pdist: PopulationDistribution
) -> Counter:
    """
    Draw abundance vector ~ Multinomial(n_ref, probs_obs + unseen bins).
    Observed bins map back to their labels; unseen bins get dummy keys.
    """
    probs = pdist.probs  # observed probs + equal unseen bins (if any)
    draws = _multinomial_draw(rng, pdist.n_samples, probs)
    L = len(pdist.observed_labels)

    counts: Counter = Counter()
    for j, c in enumerate(draws):
        if c == 0:
            continue
        if j < L:
            counts[pdist.observed_labels[j]] = c
        else:
            counts[("UNSEEN", j - L)] = c
    return counts


# -----------------
# iNEXT-style bootstrap (no Chao on the bootstrap)
# -----------------
class INextBootstrapSampler(BootstrapSampler):
    """
    Performs:
      - Multinomial resampling from window.population_distributions (activities/dfg_edges/trace_variants)
      - Optional non-parametric bootstrap of traces with replacement

    It does NOT compute Chao (or any estimator) on the bootstrap result.
    The resampled abundance vectors are attached on each replicate window as
    `window._bootstrap_abundances` (dict[str, Counter]) for downstream use.

    Parameters
    ----------
    B : int
        Number of bootstrap replicates.
    ensure_with : {"chao1", "naive"}
        If the window lacks population_distributions, build them using the
        chosen extractor. (No estimator is applied on bootstrap outputs.)
    random_state : Optional[int]
        Random seed.
    resample_traces : bool
        If True, also bootstrap the traces (with replacement).
    trace_sample_size : Optional[int]
        If None, uses window.size.
    store_abundances_on_window : bool
        If True, store the resampled abundance vectors on each replicate window
        in `_bootstrap_abundances` for later processing.
    """

    def __init__(
        self,
        B: int = 200,
        ensure_with: str = "chao1",
        random_state: Optional[int] = None,
        resample_traces: bool = True,
        trace_sample_size: Optional[int] = None,
        store_abundances_on_window: bool = True,
    ):
        if ensure_with not in ("chao1", "naive"):
            raise ValueError("ensure_with must be 'chao1' or 'naive'")
        self.B = int(B)
        self.ensure_with = ensure_with
        self.random_state = random_state
        self.resample_traces = bool(resample_traces)
        self.trace_sample_size = trace_sample_size
        self.store_abundances_on_window = store_abundances_on_window

    # ---- internal ----
    def _ensure_distributions(self, window: Window) -> None:
        """
        Populate window.population_distributions if missing.
        Uses Chao1PopulationExtractor or NaivePopulationExtractor to build
        the fitted sampling model (probabilities + unseen mass).
        """
        if window.population_distributions is not None:
            return
        if self.ensure_with == "chao1":
            Chao1PopulationExtractor().apply(
                window
            )  # sets distributions (and may set counts; counts are ignored here)
        else:
            NaivePopulationExtractor().apply(window)

    def _resample_traces(self, rng: random.Random, window: Window) -> list:
        """
        Non-parametric bootstrap of traces with replacement.
        Returns a new list of deep-copied traces.
        """
        base = list(window.traces)
        n = len(base)
        if n == 0:
            return []
        m = self.trace_sample_size or n
        idxs = [rng.randrange(n) for _ in range(m)]
        return [deepcopy(base[i]) for i in idxs]

    # ---- public ----
    def sample(self, window: Window) -> List[Window]:
        # make sure we have a fitted probability model to draw from
        self._ensure_distributions(window)
        PD: PopulationDistributions = window.population_distributions  # type: ignore

        rng = _rng(self.random_state)
        reps: List[Window] = []

        for b in range(self.B):
            # 1) abundance-level multinomial draws (iNEXT-style)
            acts_counts = _draw_counts_for_species(rng, PD.activities)
            dfg_counts = _draw_counts_for_species(rng, PD.dfg_edges)
            vars_counts = _draw_counts_for_species(rng, PD.trace_variants)

            # 2) optional non-parametric trace bootstrap (for non-pop metrics)
            if self.resample_traces:
                boot_traces = self._resample_traces(rng, window)
                boot_size = len(boot_traces)
            else:
                boot_traces = (
                    window.traces
                )  # share original traces (population-only bootstrap)
                boot_size = window.size

            # 3) assemble replicate window (DO NOT compute Chao here)
            boot_win = Window(
                id=f"{window.id}::boot{b+1}",
                size=boot_size,
                traces=boot_traces,
                population_distributions=PD,  # reuse fitted model
            )

            # Optionally attach abundance draws so you can compute estimators/metrics later
            if self.store_abundances_on_window:
                # Attach as a private field to avoid API clashes
                boot_win._bootstrap_abundances = {  # type: ignore[attr-defined]
                    "activities": acts_counts,
                    "dfg_edges": dfg_counts,
                    "trace_variants": vars_counts,
                }

            reps.append(boot_win)

        return reps
