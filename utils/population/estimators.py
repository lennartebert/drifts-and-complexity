from __future__ import annotations
from typing import Iterable, Dict, List, Any
import numpy as np
import pandas as pd
from collections import Counter
from typing import Mapping, Tuple as TTuple
from utils.windowing.windowing import Window
from pm4py.util import xes_constants as xes
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter

# ---------- extractors ----------
def _extract_activities(traces) -> Counter[str]:
    vals: Mapping[str, int] = attributes_get.get_attribute_values(traces, xes.DEFAULT_NAME_KEY)
    return Counter(vals)

def _extract_dfg_edges(traces) -> Counter[str]:
    dfg: Mapping[TTuple[str, str], int] = dfg_discovery.apply(traces)
    return Counter({f"{a}>{b}": c for (a, b), c in dfg.items()})

def _extract_trace_variants(traces) -> Counter[TTuple[str, ...]]:
    variants = variants_filter.get_variants(traces)
    out: Counter[TTuple[str, ...]] = Counter()
    for trs in variants.values():
        seq = tuple(ev[xes.DEFAULT_NAME_KEY] for ev in trs[0])
        out[seq] = len(trs)
    return out

EXTRACTORS = {
    "activities": _extract_activities,
    "dfg_edges": _extract_dfg_edges,
    "trace_variants": _extract_trace_variants,
}

# ---------- SIMPLE (own impl) ----------
def _chao1_from_abundances(counts: Counter) -> Dict[str, float]:
    s_obs = len(counts)
    f1 = sum(1 for x in counts.values() if x == 1)
    f2 = sum(1 for x in counts.values() if x == 2)
    if s_obs == 0:
        return dict(s_obs=0, f1_or_Q1=0, f2_or_Q2=0, S_hat_inf=0.0,
                    q0=0.0, q0_LCL=None, q0_UCL=None, coverage=1.0, m=None, method=None)
    s_hat = s_obs + (f1 * f1) / (2.0 * f2) if f2 > 0 else s_obs + (f1 * (f1 - 1)) / 2.0
    return dict(s_obs=s_obs, f1_or_Q1=f1, f2_or_Q2=f2, S_hat_inf=float(s_hat),
                q0=float(s_hat), q0_LCL=None, q0_UCL=None, coverage=1.0, m=None, method="extrapolation")

def estimate_populations_simple(
    windows: List[Window],
    species: Iterable[str] = ("activities","dfg_edges","trace_variants"),
    coverage_level: float = 1.0,  # must be 1.0 here
) -> Dict[str, pd.DataFrame]:
    if coverage_level != 1.0:
        raise ValueError("population_simple only supports coverage_level=1.0")
    out: Dict[str, List[Dict[str, Any]]] = {w.id: [] for w in windows}
    for sp in species:
        ext = EXTRACTORS[sp]
        for w in windows:
            counts = ext(w.traces)
            row = _chao1_from_abundances(counts)
            row.update(species=sp, n_ref=w.size, extrapolation_factor=None)
            out[w.id].append(row)
    keep = ["species","coverage","m","method","n_ref","extrapolation_factor","q0","q0_LCL","q0_UCL"]
    return {wid: pd.DataFrame(rows)[keep] for wid, rows in out.items()}

# ---------- INEXT ----------
from utils.population.inext_adapter import to_r_abundance_list, iNEXT_estimateD  # type: ignore

def estimate_populations_inext(
    windows: List[Window],
    species: Iterable[str] = ("activities","dfg_edges","trace_variants"),
    coverage_level: float | None = None,
    q_orders=(0,1,2),
    nboot=200,
    conf=0.95,
) -> Dict[str, pd.DataFrame]:
    window_ids = [w.id for w in windows]
    id_map = {w.id: w for w in windows}
    per: Dict[str, List[Dict[str, Any]]] = {wid: [] for wid in window_ids}

    for sp in species:
        ext = EXTRACTORS[sp]
        abund_all = [{w.id: ext(w.traces)} for w in windows]
        r_list = to_r_abundance_list(abund_all)
        est = iNEXT_estimateD(r_list, coverage_level=coverage_level, q_orders=q_orders, nboot=nboot, conf=conf)

        base_cols = ["Assemblage", "SC", "m", "Method"]
        q_cols = ["Order.q", "qD", "qD.LCL", "qD.UCL"]
        est = est[base_cols + q_cols].copy()

        meta = est[base_cols].drop_duplicates(subset=["Assemblage"]).set_index("Assemblage")
        wide = est.pivot_table(index="Assemblage", columns="Order.q", values=["qD","qD.LCL","qD.UCL"]).sort_index(axis=1)
        wide.columns = [f"{a}_q{int(b)}" for a, b in wide.columns.to_flat_index()]
        tmp = meta.join(wide)

        tmp["n_ref"] = [id_map[idx].size for idx in tmp.index]
        tmp["extrapolation_factor"] = tmp["m"] / tmp["n_ref"].replace(0, pd.NA)

        rename = {
            "SC":"coverage","m":"m","Method":"method",
            "qD_q0":"q0","qD.LCL_q0":"q0_LCL","qD.UCL_q0":"q0_UCL",
            "qD_q1":"q1","qD.LCL_q1":"q1_LCL","qD.UCL_q1":"q1_UCL",
            "qD_q2":"q2","qD.LCL_q2":"q2_LCL","qD.UCL_q2":"q2_UCL",
        }
        tmp = tmp.rename(columns=rename)

        for wid in window_ids:
            row = dict(
                species=sp,
                coverage=tmp.at[wid, "coverage"],
                m=tmp.at[wid, "m"],
                method=tmp.at[wid, "method"],
                n_ref=tmp.at[wid, "n_ref"],
                extrapolation_factor=tmp.at[wid, "extrapolation_factor"],
                q0     = _safe_at(tmp, wid, "q0"),
                q0_LCL = _safe_at(tmp, wid, "q0_LCL"),
                q0_UCL = _safe_at(tmp, wid, "q0_UCL"),
                q1     = _safe_at(tmp, wid, "q1"),
                q1_LCL = _safe_at(tmp,  wid, "q1_LCL"),
                q1_UCL = _safe_at(tmp, wid, "q1_UCL"),
                q2     = _safe_at(tmp, wid, "q2"),
                q2_LCL = _safe_at(tmp, wid, "q2_LCL"),
                q2_UCL = _safe_at(tmp, wid, "q2_UCL"),
            )
            per[wid].append(row)

    ordered = ["species","coverage","m","method","n_ref","extrapolation_factor",
               "q0","q0_LCL","q0_UCL","q1","q1_LCL","q1_UCL","q2","q2_LCL","q2_UCL"]
    return {wid: pd.DataFrame(per[wid])[ordered] for wid in window_ids}

def _safe_at(df: pd.DataFrame, row, col, default=np.nan):
    """Safely get df.at[row, col], return default if column or row missing."""
    if col not in df.columns:
        return default
    try:
        return df.at[row, col]
    except KeyError:
        return default
