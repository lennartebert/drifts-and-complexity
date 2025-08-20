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
from typing import Iterable, List, Literal, Tuple, Optional, Dict, Any
import pandas as pd
from collections import Counter
from typing import Mapping, Sequence, Any, Tuple
from pm4py.util import xes_constants as xes
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter
from utils.inext_adapter import (
    to_r_abundance_list,
    estimateD_common_coverage,
)

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

def estimate_populations_same_coverage(
    windows: List[List[List[str]]],
    species: Iterable[SpeciesID] = ("activities", "dfg_edges", "trace_variants"),
    estimator: Literal["Chao1"] = "Chao1",     # kept for API continuity
    q_orders: Tuple[int, int, int] = (0, 1, 2),
    nboot: int = 200,
    conf: float = 0.95,
    window_ids: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """
    Standardize all windows to a *shared sample coverage* for each species using iNEXT.
    Returns one DataFrame per window, each with one row per species, containing q in {0,1,2}
    diversity estimates and CIs, plus coverage and sampling-size metadata.

    Parameters
    ----------
    windows : list of samples; each sample is a list of traces (each trace a list of activities)
    species : iterable of species IDs (keys in _EXTRACTORS)
    estimator : kept for API continuity (iNEXT handles q-diversity)
    q_orders : Hill numbers to report (typ. 0,1,2)
    nboot : bootstrap replicates for CIs
    conf : confidence level
    window_ids : optional explicit names for windows; default "W1", "W2", ...

    Returns
    -------
    List[pd.DataFrame]
        For each window i: a DataFrame with rows = species, columns:
        ['coverage','m','method','n_ref','extrapolation_factor',
         'q0','q0_LCL','q0_UCL','q1','q1_LCL','q1_UCL','q2','q2_LCL','q2_UCL']
    """
    if window_ids is None:
        window_ids = [f"W{i+1}" for i in range(len(windows))]

    # prepare per-window accumulator: window_id -> list[row dict]
    per_window_rows: Dict[str, List[Dict[str, Any]]] = {wid: [] for wid in window_ids}

    for sp in species:
        # 1) build abundance dicts per window
        extractor = _EXTRACTORS[sp]
        abund_all = [extractor(traces) for traces in windows]
        n_ref_map = {wid: int(sum(c.values())) for wid, c in zip(window_ids, abund_all)}

        # 2) to R list, run iNEXT at common coverage for this species
        r_list = to_r_abundance_list(abund_all, assemblage_names=window_ids)
        est = estimateD_common_coverage(r_list, q_orders=q_orders, nboot=nboot, conf=conf)

        # Expect columns like: Assemblage, SC, m, Method, Order.q, qD, qD.LCL, qD.UCL
        base_cols = ["Assemblage", "SC", "m", "Method"]
        q_cols = ["Order.q", "qD", "qD.LCL", "qD.UCL"]
        est = est[base_cols + q_cols].copy()

        # meta per window (coverage, m, method)
        meta = (
            est[base_cols]
            .drop_duplicates(subset=["Assemblage"])
            .set_index("Assemblage")
        )

        # wide pivot q results
        wide = (
            est.pivot_table(
                index="Assemblage",
                columns="Order.q",
                values=["qD", "qD.LCL", "qD.UCL"],
            ).sort_index(axis=1)
        )
        # flatten multiindex columns -> qD_q0, qD.LCL_q0, ...
        wide.columns = [f"{a}_q{int(b)}" for a, b in wide.columns.to_flat_index()]

        tmp = meta.join(wide)
        # add reference sample size and extrapolation factor
        tmp["n_ref"] = [n_ref_map[idx] for idx in tmp.index]
        tmp["extrapolation_factor"] = tmp["m"] / tmp["n_ref"].replace(0, pd.NA)

        # rename to clean, prefix-free analytic names
        rename_map = {
            "SC": "coverage",
            "m": "m",
            "Method": "method",
            "qD_q0": "q0",
            "qD_q1": "q1",
            "qD_q2": "q2",
            "qD.LCL_q0": "q0_LCL",
            "qD.LCL_q1": "q1_LCL",
            "qD.LCL_q2": "q2_LCL",
            "qD.UCL_q0": "q0_UCL",
            "qD.UCL_q1": "q1_UCL",
            "qD.UCL_q2": "q2_UCL",
        }
        tmp = tmp.rename(columns=rename_map)

        # For each window, append a row for this species
        for wid in window_ids:
            row = {
                "species": sp,
                "coverage": tmp.at[wid, "coverage"],
                "m": tmp.at[wid, "m"],
                "method": tmp.at[wid, "method"],
                "n_ref": tmp.at[wid, "n_ref"],
                "extrapolation_factor": tmp.at[wid, "extrapolation_factor"],
            }
            # add q orders present in q_orders
            if 0 in q_orders:
                row.update(
                    q0=tmp.at[wid, "q0"],
                    q0_LCL=tmp.at[wid, "q0_LCL"],
                    q0_UCL=tmp.at[wid, "q0_UCL"],
                )
            if 1 in q_orders:
                row.update(
                    q1=tmp.at[wid, "q1"],
                    q1_LCL=tmp.at[wid, "q1_LCL"],
                    q1_UCL=tmp.at[wid, "q1_UCL"],
                )
            if 2 in q_orders:
                row.update(
                    q2=tmp.at[wid, "q2"],
                    q2_LCL=tmp.at[wid, "q2_LCL"],
                    q2_UCL=tmp.at[wid, "q2_UCL"],
                )

            per_window_rows[wid].append(row)

    # materialize one DataFrame per window (species as rows)
    ordered_cols = [
        "species", "coverage", "m", "method", "n_ref", "extrapolation_factor",
        # q0
        *(["q0", "q0_LCL", "q0_UCL"] if 0 in q_orders else []),
        # q1
        *(["q1", "q1_LCL", "q1_UCL"] if 1 in q_orders else []),
        # q2
        *(["q2", "q2_LCL", "q2_UCL"] if 2 in q_orders else []),
    ]

    per_window_dfs: List[pd.DataFrame] = []
    for wid in window_ids:
        df = pd.DataFrame(per_window_rows[wid])[ordered_cols]
        df.insert(0, "window_id", wid)  # optional, handy when concatenating later
        per_window_dfs.append(df)

    return per_window_dfs
