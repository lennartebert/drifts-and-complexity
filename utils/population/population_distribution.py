from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PopulationDistribution:
    """
    Population distribution with coverage-aware observed mass and imputed unseen mass.

    Stored fields
    -------------
    observed_labels : List[Tuple]
        Identifiers for observed categories (e.g., activities, DFG edges, variants).
        Each label may be a tuple (e.g., a variant sequence) or any hashable tuple-like key.
    observed_probs : List[float]
        Probabilities assigned to `observed_labels`. Intended to sum to Ĉ = 1 - p0.
        If they do not, they are rescaled in `__post_init__` to sum to (1 - p0), unless
        the vector is all zeros (then it remains zeros and is normalized on access).
    unseen_count : int
        Number of *unseen* categories M (e.g., round(Ŝ - S_obs)) to be represented
        as equally likely mass within p0.
    p0 : float
        Unseen probability mass (1 - Ĉ), i.e., mass not covered by observed categories.
        Must be in [0, 1].
    n_samples : int
        Reference sample size (e.g., the size of the sample from which the estimates were
        derived). Kept for traceability/meta-data; not used in computations here.

    Cached derived attributes (properties)
    --------------------------------------
    probs : List[float]
        Full probability vector: observed probabilities (rescaled to sum to 1 - p0)
        followed by `unseen_count` entries, each equal to p0 / unseen_count.
        If `unseen_count == 0`, this is just the normalized observed probabilities
        (summing to 1.0). If both `p0 == 0` and observed mass sums to 0, returns [].
        This value is cached and recomputed only when (`observed_probs`, `unseen_count`, `p0`) change.
    count : int
        Total number of categories represented in `probs`, i.e., len(probs).
        This value is cached alongside `probs`.

    Notes
    -----
    - The cache is *content-aware*: mutations to `observed_probs` are detected because
      the cache key uses `tuple(observed_probs)`. If you mutate the list in-place, the
      next access will see the change and recompute.
    """

    observed_labels: List[Tuple]
    observed_probs: List[float]
    unseen_count: int
    p0: float
    n_samples: int

    # ---- Internal cache fields (not part of the public API) ----
    _cache_key: Tuple[Tuple[float, ...], int, float] | None = field(
        default=None, init=False, repr=False
    )
    _cached_probs: List[float] | None = field(default=None, init=False, repr=False)
    _cached_count: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Basic validation / clipping
        if self.unseen_count < 0:
            raise ValueError("unseen_count must be >= 0")
        if not (0.0 <= self.p0 <= 1.0):
            raise ValueError("p0 must be within [0, 1]")
        if len(self.observed_labels) != len(self.observed_probs):
            raise ValueError(
                "observed_labels and observed_probs must have the same length"
            )

        # Rescale observed_probs to sum to (1 - p0) when they have positive mass.
        target = 1.0 - float(self.p0)
        s = float(sum(self.observed_probs))
        if s > 0.0:
            if target < 0.0:
                # Shouldn't happen due to p0 ∈ [0,1], but guard anyway
                raise ValueError("Computed target observed mass < 0. Check p0.")
            scale = target / s
            self.observed_probs = [p * scale for p in self.observed_probs]

        # Initialize cache as empty; the first access will compute and store values.
        self._invalidate_cache()

    # ------- Public cached properties -------

    @property
    def probs(self) -> List[float]:
        """
        Full probability vector (observed + unseen), normalized and cached.

        Returns
        -------
        List[float]
            - If unseen_count > 0 and p0 > 0: observed probs (sum to 1 - p0)
              followed by `unseen_count` entries of p0 / unseen_count.
            - If unseen_count == 0: normalized observed probs that sum to 1.0.
            - If there is zero total mass (no observed and p0 == 0): [].
        """
        self._ensure_cache()
        # Return a copy to prevent accidental external mutation of the cached list
        return list(self._cached_probs or [])

    @property
    def count(self) -> int:
        """
        Total number of categories represented by the distribution.

        Returns
        -------
        int
            len(probs), i.e., observed categories plus (potential) unseen categories.
        """
        self._ensure_cache()
        return int(self._cached_count or 0)

    # ------- Internal cache helpers -------

    def _current_key(self) -> Tuple[Tuple[float, ...], int, float]:
        """
        Build a hashable cache key from inputs that affect `probs`/`count`.
        We round p0 slightly to avoid pathological float jitter in equality checks.
        """
        # tuple(observed_probs) captures in-place list mutations
        probs_tuple = tuple(self.observed_probs)
        p0_key = round(float(self.p0), 15)
        return (probs_tuple, int(self.unseen_count), p0_key)

    def _ensure_cache(self) -> None:
        """Compute and cache `probs`/`count` if the inputs changed."""
        key = self._current_key()
        if (
            key == self._cache_key
            and self._cached_probs is not None
            and self._cached_count is not None
        ):
            return  # Cache is valid

        # Recompute
        obs = list(self.observed_probs)
        obs_sum = float(sum(obs))

        if self.unseen_count > 0 and self.p0 > 0.0:
            # Ensure observed sums to (1 - p0); if zero, keep zeros (mass all in unseen)
            if obs_sum > 0.0:
                target = 1.0 - self.p0
                if abs(obs_sum - target) > 1e-12:
                    scale = target / obs_sum if obs_sum > 0 else 0.0
                    obs = [p * scale for p in obs]
            p_unseen = self.p0 / self.unseen_count
            full = obs + [p_unseen] * self.unseen_count
        else:
            # No unseen categories: normalize observed to sum to 1
            if obs_sum > 0.0:
                full = [p / obs_sum for p in obs]
            else:
                full = []

        self._cached_probs = full
        self._cached_count = len(full)
        self._cache_key = key

    def _invalidate_cache(self) -> None:
        """Mark cached values as stale (next access will recompute)."""
        self._cache_key = None
        self._cached_probs = None
        self._cached_count = None
