# Population estimates for “species” richness using Chao1.
# Species implemented:
#   - "activities": distinct activities (vertices)
#   - "dfg_edges": distinct directly-follows relations (edges)
#   - "trace_variants": distinct trace variants
#
# Output is a tidy DataFrame with one row per species, carrying:
#   species, Estimator, s_obs, f1_or_Q1, f2_or_Q2, S_hat_inf, CI_low, CI_high
#
# Notes:
# - We implement the bias-corrected Chao1 estimator.
# - We include a conservative fallback for f2 == 0.
# - CI fields are left as None for now (you can wire analytic/bootstrap CIs later).

from __future__ import annotations
from collections import Counter
from typing import Iterable, List, Dict, Tuple, Literal
import pandas as pd
from collections import Counter
from typing import Mapping, Sequence, Any, Tuple
from pm4py.util import xes_constants as xes
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter

SpeciesID = Literal["activities", "dfg_edges", "trace_variants"]


# ---------- Species extractors (abundance-based) ----------

def _extract_activities(traces) -> Counter[str]:
    """Count abundance of each activity in the pm4py EventLog."""
    # Uses the XES "concept:name" key by default
    vals: Mapping[str, int] = attributes_get.get_attribute_values(
        traces, xes.DEFAULT_NAME_KEY
    )
    return Counter(vals)

def _extract_dfg_edges(traces) -> Counter[str]:
    """Count abundance of each directly-follows edge 'A>B' in the pm4py EventLog."""
    # Frequency DFG: {(A, B): count, ...}
    dfg: Mapping[Tuple[str, str], int] = dfg_discovery.apply(traces)
    return Counter({f"{a}>{b}": c for (a, b), c in dfg.items()})

def _extract_trace_variants(traces) -> Counter[Tuple[str, ...]]:
    """Count abundance of each trace variant (sequence of activities) using pm4py."""
    # Group traces into variants; keys are variant ids, values are lists of traces
    variants = variants_filter.get_variants(traces)
    out: Counter[Tuple[str, ...]] = Counter()
    for trs in variants.values():
        # Derive the activity sequence from one representative trace
        seq = tuple(ev[xes.DEFAULT_NAME_KEY] for ev in trs[0])
        out[seq] = len(trs)
    return out


_EXTRACTORS: Dict[SpeciesID, callable] = {
    "activities": _extract_activities,
    "dfg_edges": _extract_dfg_edges,
    "trace_variants": _extract_trace_variants,
}


# ---------- Estimators ----------

def _chao1_from_abundances(counts: Counter) -> Tuple[int, int, int, float, Tuple[float, float] | None, str]:
    """
    Bias-corrected Chao1 richness estimator from species abundances.
    Returns: (s_obs, f1, f2, s_hat, ci, note)
    """
    s_obs = len(counts)
    f1 = sum(1 for x in counts.values() if x == 1)
    f2 = sum(1 for x in counts.values() if x == 2)

    if s_obs == 0:  # empty window
        return 0, 0, 0, 0.0, None, "empty sample"

    if f2 > 0:
        s_hat = s_obs + (f1 * f1) / (2.0 * f2)
        ci = None  # plug analytic or bootstrap CI later
        note = ""
    else:
        # Common fallback when f2 == 0; conservative upward adjustment.
        # (You can switch to iChao1 here later if desired.)
        s_hat = s_obs + (f1 * (f1 - 1)) / 2.0
        ci = None
        note = "f2=0 adjustment"

    return s_obs, f1, f2, float(s_hat), ci, note


# ---------- Public API ----------

def estimate_populations(
    traces: List[List[str]],
    species: Iterable[SpeciesID] = ("activities", "dfg_edges", "trace_variants"),
    estimator: Literal["Chao1"] = "Chao1",
) -> pd.DataFrame:
    """
    Run richness estimation per species using abundance-based Chao1.

    Parameters
    ----------
    traces : list[list[str]]
        Window/sample: list of traces, each a list of activity labels.
    species : iterable of SpeciesID
        Which species to estimate (“activities”, “dfg_edges”, “variants”).
    estimator : "Chao1"
        Currently only Chao1 is implemented.

    Returns
    -------
    pd.DataFrame
        Columns: species, Estimator, s_obs, f1_or_Q1, f2_or_Q2, S_hat_inf, CI_low, CI_high, notes
    """
    rows = []
    for sp in species:
        extractor = _EXTRACTORS[sp]
        abund = extractor(traces)

        if estimator == "Chao1":
            s_obs, f1, f2, s_hat, ci, note = _chao1_from_abundances(abund)
            rows.append(
                {
                    "species": sp,
                    "Estimator": "Chao1",
                    "s_obs": s_obs,
                    "f1_or_Q1": f1,
                    "f2_or_Q2": f2,
                    "S_hat_inf": s_hat,
                    "CI_low": None if not ci else ci[0],
                    "CI_high": None if not ci else ci[1],
                    "notes": note,
                }
            )
        else:
            raise NotImplementedError(f"Estimator {estimator} not implemented.")
    return pd.DataFrame(rows)
