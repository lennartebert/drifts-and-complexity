from __future__ import annotations

import math
import random
from collections import Counter
from math import ceil, floor
from typing import Dict, List, Optional, Tuple

from utils.population.population_distributions import PopulationDistribution

# ==========================================================
# Coercion helper (observed may be weights; iNEXT needs abundances)
# ==========================================================


def _coerce_to_abundance(observed: Counter, n_reference: int) -> Counter:
    """
    Coerce ``observed`` into an *abundance* Counter that sums to ``n_reference``.

    - If ``observed`` already contains non-negative integers summing to ``n_reference``,
      it is returned unchanged.
    - Otherwise, values are scaled proportionally and converted to counts via the
      largest remainder method so the result sums exactly to ``n_reference``.
    """
    if (
        all(float(v).is_integer() and v >= 0 for v in observed.values())
        and sum(observed.values()) == n_reference
    ):
        return Counter(observed)

    total = float(sum(observed.values()))
    if total <= 0 or n_reference <= 0:
        return Counter()

    names = list(observed.keys())
    probs = [float(observed[k]) / total for k in names]

    floats = [p * n_reference for p in probs]
    floors = [int(math.floor(x)) for x in floats]
    leftover = int(n_reference - sum(floors))

    fracs = sorted(
        ((i, floats[i] - floors[i]) for i in range(len(names))),
        key=lambda t: t[1],
        reverse=True,
    )
    for j in range(leftover):
        floors[fracs[j][0]] += 1

    return Counter({names[i]: floors[i] for i in range(len(names))})


# ==========================================================
# Frequency and Chao1 estimation helpers (iNEXT-aligned)
# ==========================================================


def _freq_of_freqs_basic(x: Counter) -> Tuple[int, int, int]:
    """Return (n, f1, f2) for an *abundance* Counter ``x``."""
    n = sum(x.values())
    f1 = sum(1 for c in x.values() if c == 1)
    f2 = sum(1 for c in x.values() if c == 2)
    return n, f1, f2


def _chao1_f0_hat_inext(n: int, f1: int, f2: int) -> float:
    """
    Bias-corrected Chao1 unseen species estimate (iNEXT style):
      - if f2>0:  ((n-1)/n) * f1^2/(2 f2)
      - if f2=0:  ((n-1)/n) * f1(f1-1)/2
    """
    if f1 == 0:
        return 0.0
    if f2 > 0:
        return ((n - 1) / n) * (f1 * f1) / (2.0 * f2)
    return ((n - 1) / n) * (f1 * (f1 - 1)) / 2.0


def _A_factor_inext(n: int, f1: int, f2: int) -> float:
    """
    iNEXT A factor for coverage/extrapolation using bias-corrected f0_hat.
      A = n*f0_hat / (n*f0_hat + f1)  (A=1 if f1==0; A=0 if f0_hat<=0)
    """
    if f1 == 0:
        return 1.0
    f0_hat = _chao1_f0_hat_inext(n, f1, f2)
    if f0_hat <= 0:
        return 0.0
    return (n * f0_hat) / (n * f0_hat + f1)


# ==========================================================
# Coverage estimation (Chat.Ind and inverse) — kept for completeness
# ==========================================================


def chat_ind(observed: Counter, m: float) -> float:
    """
    iNEXT abundance-based sample coverage Chat.Ind(x, m).
    Rarefaction (m<n) uses a hypergeometric form; Reference (m=n): 1-(f1/n)A;
    Extrapolation (m>n): 1-(f1/n) A^(m-n+1).
    """
    x = Counter({k: v for k, v in observed.items() if v > 0})
    n, f1, f2 = _freq_of_freqs_basic(x)
    if n <= 0:
        return 0.0

    A = _A_factor_inext(n, f1, f2)

    def _C_at_integer_m(k: int) -> float:
        if k < n:
            s = 0.0
            for c in x.values():
                if (n - c) >= k:
                    s += (c / n) * math.exp(
                        math.lgamma(n - c + 1)
                        - math.lgamma(n - c - k + 1)
                        - math.lgamma(n)
                        + math.lgamma(n - k)
                    )
            return 1.0 - s
        elif k == n:
            return 1.0 - (f1 / n) * A
        else:
            return 1.0 - (f1 / n) * (A ** (k - n + 1))

    if float(m).is_integer():
        return max(0.0, min(1.0, _C_at_integer_m(int(m))))

    m_floor, m_ceil = floor(m), ceil(m)
    c_floor = _C_at_integer_m(m_floor)
    c_ceil = _C_at_integer_m(m_ceil)
    c = (m_ceil - m) * c_floor + (m - m_floor) * c_ceil
    return max(0.0, min(1.0, c))


def inv_chat_ind(observed: Counter, target_C: float) -> float:
    """
    iNEXT invChat.Ind(x, C): find m s.t. Chat.Ind(x, m) ≈ target_C.
    Rarefaction uses 1-D search on [1,n]; extrapolation uses closed form with A.
    """
    x = Counter({k: v for k, v in observed.items() if v > 0})
    n, f1, f2 = _freq_of_freqs_basic(x)
    if n <= 0:
        return 1.0

    refC = chat_ind(x, n)
    if abs(refC - target_C) < 1e-12:
        return float(n)

    if refC > target_C:
        # Golden-section search
        def f(mval: float) -> float:
            return abs(chat_ind(x, mval) - target_C)

        a, b = 1.0, float(n)
        phi = (1 + 5**0.5) / 2.0
        invphi = 1 / phi
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(60):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - invphi * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a)
                fd = f(d)
        return max(1.0, (a + b) / 2.0)

    # Extrapolation
    if f1 == 0:
        return float(n)
    if f2 > 0:
        A = (n - 1) * f1 / ((n - 1) * f1 + 2.0 * f2)
    elif f1 > 1 and f2 == 0:
        A = (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2.0)
    else:
        A = 0.0
    if A <= 0.0:
        return float(n)

    mm = (math.log(n / f1) + math.log(1.0 - target_C)) / math.log(A) - 1.0
    return float(n) + mm


# ==========================================================
# Estimated bootstrap community (iNEXT EstiBootComm.Ind)
# ==========================================================


def _esti_boot_comm_ind_probs(
    observed: Counter,
    n: int,
) -> Tuple[Dict[str, float], float, int]:
    """
    iNEXT "estimated bootstrap community" for abundance data.

    Returns (prob_obs, p0, S0) where:
      - prob_obs: adjusted probabilities for observed species (dict name -> p_i)
      - p0: total unseen mass a = (f1/n) * A
      - S0: number of unseen species (ceil(f0_hat))
    """
    items = list(observed.items())
    counts = [c for _, c in items]
    if sum(counts) != n:
        raise ValueError("sum(observed.values()) must equal n after coercion.")

    f1 = sum(1 for c in counts if c == 1)
    f2 = sum(1 for c in counts if c == 2)

    f0_hat = _chao1_f0_hat_inext(n, f1, f2)
    if f1 == 0:
        A = 1.0
    elif f0_hat <= 0:
        A = 0.0
    else:
        A = (n * f0_hat) / (n * f0_hat + f1)

    a = (f1 / n) * A  # p0 (unseen mass at reference)

    rel = [c / n for c in counts]
    b = sum(r * ((1.0 - r) ** n) for r in rel)
    w = 0.0 if f0_hat == 0 else (a / b if b > 0 else 0.0)

    prob_obs: Dict[str, float] = {}
    for (name, c), r in zip(items, rel):
        prob_obs[name] = r * (1.0 - w * ((1.0 - r) ** n))

    S0 = int(ceil(f0_hat)) if f0_hat > 0 else 0
    p0 = a if f0_hat > 0 else 0.0
    return prob_obs, p0, S0


# ==========================================================
# Main constructors (asymptotic S_hat focus)
# ==========================================================


def create_chao1_population_distribution(
    observed: Counter,
    rng: Optional[random.Random] = None,
) -> "PopulationDistribution":
    """
    Build a PopulationDistribution that targets **asymptotic richness** S_hat (Chao1, abundance).

    - Assumes abundance data: integer non-negative counts.
    - n_reference is derived as sum(observed.values()).
    - Computes S_obs, f0_hat (iNEXT bias-corrected), and S_hat = S_obs + f0_hat.
    - Constructs a **species-inventory population** (presence counts):
        1 per observed species + 1 per unseen placeholder (S0 = ceil(f0_hat)).
      Thus sum(population.values()) = S_obs + S0 ≈ S_hat (integerized).
    - Also returns p0 (unseen mass at reference) for completeness.
    """
    _ = rng  # unused

    # --- abundance-only validation
    if not all(float(v).is_integer() and v >= 0 for v in observed.values()):
        raise ValueError("Abundance mode requires integer, non-negative counts.")
    n_reference = int(sum(observed.values()))
    if n_reference <= 0:
        raise ValueError("n_reference (sum of observed counts) must be positive.")

    # --- use observed directly as abundance vector
    obs_abund = Counter(observed)
    n = n_reference
    S_obs = sum(1 for v in obs_abund.values() if v > 0)
    f1 = sum(v == 1 for v in obs_abund.values())
    f2 = sum(v == 2 for v in obs_abund.values())

    f0_hat = _chao1_f0_hat_inext(n, f1, f2)  # iNEXT bias-corrected
    S0 = int(ceil(f0_hat)) if f0_hat > 0 else 0

    # unseen mass p0 via estimated bootstrap community (for reporting)
    prob_obs, p0, _ = _esti_boot_comm_ind_probs(obs_abund, n)

    # --- species-inventory population (presence counts)
    population = Counter({k: 1 for k, c in obs_abund.items() if c > 0})
    for i in range(1, S0 + 1):
        population[f"unseen_{i}"] = 1

    return PopulationDistribution(
        observed=Counter(observed),
        population=population,
        n_reference=n_reference,
        n_population=S_obs + S0,  # interpret as total species represented
        observed_count=S_obs,
        population_count=S_obs + S0,
        unseen_count=S0 if S0 > 0 else None,
        p0=p0 if f0_hat > 0 else None,
        coverage_observed=None,
        coverage_population=None,
        f0_hat=f0_hat,
        s_asymptotic=S_obs + f0_hat,
    )


def create_chao1_bootstrapped_population_distribution(
    base_pd: "PopulationDistribution",
    B: int = 200,
    rng: Optional[random.Random] = None,
) -> List["PopulationDistribution"]:
    """
    Create B bootstrap replicate PopulationDistributions (Chao1, abundance) from a base PD.

    This uses the iNEXT bootstrap scheme:
      1) Build the estimated bootstrap community (observed + unseen) *probabilities*
         from the reference abundance histogram (base_pd.observed, n_reference).
      2) Draw Multinomial(n_reference, prob_full) replicates.
      3) For each replicate, recompute S_obs^b and f0_hat^b and construct a
         species-inventory population: 1 per observed species + S0^b unseen placeholders.

    Parameters
    ----------
    base_pd : PopulationDistribution
        A base Ŝ-focused PD created by `create_chao1_population_distribution`.
        Must contain abundance counts in `base_pd.observed`; `base_pd.population`
        is *not* used to build the generator, since it's a presence inventory.
    B : int, default 200
        Number of bootstrap replicates.
    rng : random.Random, optional
        RNG for reproducibility.

    Returns
    -------
    List[PopulationDistribution]
        A list of B replicate PDs.
    """
    if B <= 0:
        raise ValueError("B must be positive.")
    rng = rng or random.Random()

    # Validate abundance inputs
    if not all(float(v).is_integer() and v >= 0 for v in base_pd.observed.values()):
        raise ValueError(
            "Abundance mode requires integer, non-negative counts in base_pd.observed."
        )
    n_reference = int(sum(base_pd.observed.values()))
    if n_reference <= 0:
        raise ValueError("n_reference (sum of base_pd.observed) must be positive.")

    # 1) Estimated bootstrap community (probabilities) from reference abundance
    obs_abund = Counter(base_pd.observed)
    prob_obs, p0, S0 = _esti_boot_comm_ind_probs(obs_abund, n_reference)  # iNEXT-style
    prob_full = dict(prob_obs)
    if S0 > 0 and p0 > 0:
        p_each = p0 / S0
        for i in range(1, S0 + 1):
            prob_full[f"unseen_{i}"] = p_each

    names = list(prob_full.keys())
    probs = [prob_full[k] for k in names]
    s = sum(probs)
    probs = (
        [p / s for p in probs]
        if s > 0
        else [1.0 / max(1, len(names))] * max(1, len(names))
    )

    # 2) Draw replicates & 3) rebuild PDs as species-inventory with S0^b
    replicate_pds: List[PopulationDistribution] = []
    for _ in range(B):
        # Multinomial draw at n_reference (categorical loop)
        draws = [0] * len(names)
        for _j in range(n_reference):
            r = rng.random()
            acc = 0.0
            for idx, p in enumerate(probs):
                acc += p
                if r <= acc:
                    draws[idx] += 1
                    break
        rep_observed = Counter({names[i]: c for i, c in enumerate(draws) if c > 0})

        # Recompute replicate S_obs^b and f0_hat^b (bias-corrected Chao1)
        S_obs_b = sum(1 for v in rep_observed.values() if v > 0)
        f1b = sum(v == 1 for v in rep_observed.values())
        f2b = sum(v == 2 for v in rep_observed.values())
        f0b = _chao1_f0_hat_inext(n_reference, f1b, f2b)
        S0b = int(ceil(f0b)) if f0b > 0 else 0

        # Optional: unseen mass on replicate (not needed for CI computation itself)
        try:
            _, p0_b, _ = _esti_boot_comm_ind_probs(rep_observed, n_reference)
        except Exception:
            p0_b = None

        # Species-inventory population for the replicate
        rep_population = Counter({k: 1 for k, c in rep_observed.items() if c > 0})
        for i in range(1, S0b + 1):
            rep_population[f"unseen_{i}"] = 1

        rep_pd = PopulationDistribution(
            observed=rep_observed,
            population=rep_population,
            n_reference=n_reference,
            n_population=S_obs_b + S0b,  # interpreted as total species represented
            observed_count=S_obs_b,
            population_count=S_obs_b + S0b,
            unseen_count=S0b if S0b > 0 else None,
            p0=p0_b,
            coverage_observed=None,
            coverage_population=None,
            f0_hat=f0b,
            s_asymptotic=S_obs_b + f0b,
        )
        replicate_pds.append(rep_pd)

    return replicate_pds
