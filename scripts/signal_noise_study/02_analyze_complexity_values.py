"""Simplified signal/noise analysis with seed-first aggregation.

Outputs only:
- stability_*
- settingChange_*
- processChange_*

For each generated aggregate CSV, a companion "*_counts.csv" is written
containing the number of finite values used in each aggregated cell.
"""

from __future__ import annotations

import argparse
import importlib
import re
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from utils.constants import DIMENSIONS_ORDER, METRIC_DIMENSION_MAP, METRIC_NAMES_TO_LATEX_MAP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATH_AGG = "results/signal_noise_study/aggregate_analysis.csv"
PATH_GEN_INFO = "data/synthetic/sudden_drifts/generation_info.csv"
DIR_CSV = "results/signal_noise_study/csvs"
DIR_LATEX = "results/signal_noise_study/latex"

DEFAULT_TEST_SEEDS = [43, 44] # only used if in test mode (flag --test is set)

MODEL_COMPLEXITY_COL = "Model Complexity"
NOISE_LEVEL_COL = "Noise Level"
CHANGE_MAGNITUDE_COL = "Change Magnitude"
EDIT_OPERATIONS_COL = "Edit Operations"
WINDOW_SIZE_COL = "Window Size"
SEED_COL = "Seed"
LOG_ID_COL = "Log ID"
SPLIT_NAME_COL = "Split Name"
LOG_NUMBER_COL = "Log Number"

NOISE_PROB_TO_LEVEL = {0.0: "None", 0.01: "Low", 0.02: "High"}
COMPLEXITY_MAPPING = {"simple": "Simple", "middle": "Middle", "complex": "Complex"}
CHANGE_OPERATION_MAPPING = {
    "deletion": "deletion",
    "insertion": "insertion",
    "resequentialization": "resequentialization",
    "operator_replacement": "operator_replacement",
    "activity_replacement": "activity_replacement",
    "deletion, insertion, resequentialization, operator_replacement, activity_replacement": "mixed",
}
CHANGE_OPERATION_ORDER = [
    "deletion",
    "insertion",
    "activity_replacement",
    "resequentialization",
    "operator_replacement",
    "mixed",
]

AGG_COL_MAP = {
    "Metric": "Metric",
    "Mean Value": "Mean Value",
    "Median Value": "Median Value",
    "Sample Std": "Sample Std",
    "log_id": LOG_ID_COL,
    "split_name": SPLIT_NAME_COL,
    "window_size": WINDOW_SIZE_COL,
}

# Metric order: by configured dimension order, then by insertion order in
# METRIC_DIMENSION_MAP (same strategy as the previous analysis script).
_METRIC_ORDER = [
    metric
    for dim in DIMENSIONS_ORDER
    for metric, metric_dim in METRIC_DIMENSION_MAP.items()
    if metric_dim == dim
]


# ---------------------------------------------------------------------------
# Input + preparation
# ---------------------------------------------------------------------------
def _finite(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _non_nan(x: pd.Series) -> pd.Series:
    """Numeric coercion + drop NaN, but keep +/-inf values."""
    return pd.to_numeric(x, errors="coerce").dropna()


def _robust_mean(x: pd.Series) -> float:
    vals = _non_nan(x)
    if len(vals) == 0:
        return np.nan
    return float(vals.mean())


def _robust_median(x: pd.Series) -> float:
    vals = _non_nan(x)
    if len(vals) == 0:
        return np.nan
    return float(vals.median())


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float).to_numpy()
    den = pd.to_numeric(denominator, errors="coerce").astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    zero_den = den == 0
    zero_num = num == 0
    out = np.where(zero_den & zero_num, np.nan, out)
    out = np.where(zero_den & (num > 0), np.inf, out)
    return pd.Series(out, index=numerator.index)


def load_and_validate_input(path_agg: str) -> pd.DataFrame:
    """Load required columns only and validate schema."""
    required = set(AGG_COL_MAP.keys())
    optional = {"Sample Size"}
    df = pd.read_csv(
        path_agg,
        low_memory=False,
        usecols=lambda c: c in required or c in optional,
    )
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path_agg}: {missing}")
    if "Sample Size" in df.columns and not (df["Sample Size"] == df["window_size"]).all():
        raise ValueError("If present, 'Sample Size' must equal 'window_size'.")
    if "Sample Size" in df.columns:
        df = df.drop(columns=["Sample Size"])
    return df.rename(columns=AGG_COL_MAP)


def normalize_split_name(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    norm = d[SPLIT_NAME_COL].astype(str).str.lower().str.replace("_", "-", regex=False)
    pre = {"pre", "pre-drift", "pre_drift"}
    post = {"post", "post-drift", "post_drift"}
    d[SPLIT_NAME_COL] = norm.map(
        lambda s: "pre_drift" if s in pre else ("post_drift" if s in post else s)
    )
    return d


def extract_log_number(log_id: str) -> int:
    m = re.match(r"log_(\d+)_", str(log_id))
    if not m:
        raise ValueError(f"Cannot extract log number from log_id: {log_id}")
    return int(m.group(1))


def load_generation_info(path_gen_info: str) -> pd.DataFrame:
    raw = pd.read_csv(path_gen_info, sep=";")
    req = [
        "log_id",
        "Noisy_trace_prob",
        "Process_tree_complexity",
        "Process_tree_evolution_proportion",
        "Allowed_edit_operations",
        "Log_seed",
    ]
    miss = [c for c in req if c not in raw.columns]
    if miss:
        raise ValueError(f"Missing required columns in {path_gen_info}: {miss}")

    out = pd.DataFrame()
    out[LOG_NUMBER_COL] = raw["log_id"].map(extract_log_number)
    out[NOISE_LEVEL_COL] = raw["Noisy_trace_prob"].map(NOISE_PROB_TO_LEVEL)
    out[MODEL_COMPLEXITY_COL] = raw["Process_tree_complexity"].map(COMPLEXITY_MAPPING)
    out[CHANGE_MAGNITUDE_COL] = raw["Process_tree_evolution_proportion"].astype(float)
    out[EDIT_OPERATIONS_COL] = raw["Allowed_edit_operations"].map(CHANGE_OPERATION_MAPPING)
    out[SEED_COL] = raw["Log_seed"]

    if out[NOISE_LEVEL_COL].isna().any():
        raise ValueError("Unmapped noise probabilities in generation_info.")
    if out[MODEL_COMPLEXITY_COL].isna().any():
        raise ValueError("Unmapped complexity values in generation_info.")
    if out[EDIT_OPERATIONS_COL].isna().any():
        raise ValueError("Unmapped edit operation values in generation_info.")
    return out


def enrich_with_experimental_factors(df: pd.DataFrame, gen_info: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d[LOG_NUMBER_COL] = d[LOG_ID_COL].map(extract_log_number)
    out = d.merge(gen_info, on=LOG_NUMBER_COL, how="left").drop(columns=[LOG_NUMBER_COL])
    out = out[
        out[MODEL_COMPLEXITY_COL].notna()
        & out[NOISE_LEVEL_COL].notna()
        & out[EDIT_OPERATIONS_COL].notna()
        & out[CHANGE_MAGNITUDE_COL].notna()
    ].copy()
    if len(out) == 0:
        raise ValueError("No rows left after enrichment with generation info.")
    return out


def select_one_metric_per_dimension(df: pd.DataFrame) -> pd.DataFrame:
    metrics = sorted(df["Metric"].dropna().unique().tolist())
    selected: list[str] = []
    seen_dims: set[str] = set()

    for dim in DIMENSIONS_ORDER:
        candidates = sorted([m for m in metrics if METRIC_DIMENSION_MAP.get(m, "Other") == dim])
        if candidates:
            selected.append(candidates[0])
            seen_dims.add(dim)

    other_dims = sorted(
        {METRIC_DIMENSION_MAP.get(m, "Other") for m in metrics if METRIC_DIMENSION_MAP.get(m, "Other") not in seen_dims}
    )
    for dim in other_dims:
        candidates = sorted([m for m in metrics if METRIC_DIMENSION_MAP.get(m, "Other") == dim])
        if candidates:
            selected.append(candidates[0])

    if not selected:
        return df
    print(f"Test mode metric filter: keeping {len(selected)} metrics ({selected})")
    return df[df["Metric"].isin(selected)].copy()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _add_dimension_and_sort_long(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Dimension"] = d["Metric"].map(METRIC_DIMENSION_MAP).fillna("Other")
    dim_order = list(DIMENSIONS_ORDER) + ["Other"]
    d["_dim_ord"] = d["Dimension"].map(lambda x: dim_order.index(x) if x in dim_order else len(dim_order))
    metric_order = {m: i for i, m in enumerate(_METRIC_ORDER)}
    d["_met_ord"] = d["Metric"].map(lambda m: metric_order.get(m, len(metric_order)))
    d = d.sort_values(["_dim_ord", "_met_ord"]).drop(columns=["_dim_ord", "_met_ord"])
    cols = ["Dimension", "Metric"] + [c for c in d.columns if c not in {"Dimension", "Metric"}]
    return d[cols]


def _add_dimension_and_sort_wide(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if isinstance(d.index, pd.MultiIndex):
        if d.index.nlevels < 2:
            raise ValueError("Expected at least 2 index levels for MultiIndex row sorting.")
        # Normalize to the expected two-level row index.
        d.index = pd.MultiIndex.from_arrays(
            [d.index.get_level_values(0), d.index.get_level_values(1)],
            names=["Dimension", "Metric"],
        )
    else:
        dimension = d.index.map(lambda m: METRIC_DIMENSION_MAP.get(m, "Other"))
        d.index = pd.MultiIndex.from_arrays([dimension, d.index], names=["Dimension", "Metric"])
    dim_order = list(DIMENSIONS_ORDER) + ["Other"]
    desired = []
    for dim in dim_order:
        for metric in _METRIC_ORDER:
            idx = (dim, metric)
            if idx in d.index:
                desired.append(idx)
    for idx in d.index:
        if idx not in desired:
            desired.append(idx)
    return d.reindex(desired)


def _collapse_within_seed(
    df: pd.DataFrame,
    *,
    group_keys_with_seed: list[str],
    value_cols: list[str],
    agg_mode: Literal["mean", "median"] = "median",
) -> pd.DataFrame:
    if SEED_COL not in group_keys_with_seed:
        raise ValueError(f"group_keys_with_seed must include {SEED_COL!r}")
    agg_fn = _robust_mean if agg_mode == "mean" else _robust_median
    grouped = df.groupby(group_keys_with_seed, dropna=False, sort=False)
    return grouped.agg({c: agg_fn for c in value_cols}).reset_index()


def _aggregate_across_seed_with_counts(
    df: pd.DataFrame,
    *,
    group_keys_without_seed: list[str],
    value_col: str,
    across_seed_aggregation: Literal["mean", "median"],
) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        # Keep infinities; exclude only NaNs.
        vals = _non_nan(g[value_col])
        value = _robust_mean(vals) if across_seed_aggregation == "mean" else _robust_median(vals)
        return pd.Series({"Value": value, "N Used": int(len(vals))})

    return df.groupby(group_keys_without_seed, dropna=False, sort=False).apply(_agg).reset_index()


def _reorder_columns(df: pd.DataFrame, order_map: dict[str, list[Any]]) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        name = df.columns.name
        if name in order_map:
            preferred = [c for c in order_map[name] if c in df.columns]
            preferred.extend([c for c in df.columns if c not in preferred])
            return df[preferred]
        return df

    levels = []
    for i, name in enumerate(df.columns.names):
        existing = list(df.columns.get_level_values(i).unique())
        preferred = [v for v in order_map.get(name, []) if v in existing]
        preferred.extend([v for v in existing if v not in preferred])
        levels.append(preferred)
    candidate = pd.MultiIndex.from_product(levels, names=df.columns.names)
    keep = [c for c in candidate if c in df.columns]
    return df[keep]


def _pivot_value_and_counts(
    df_agg: pd.DataFrame,
    *,
    column_levels: list[str],
    order_map: dict[str, list[Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = _add_dimension_and_sort_long(df_agg.copy())
    idx = ["Dimension", "Metric"]
    value_wide = long.pivot_table(
        index=idx, columns=column_levels, values="Value", aggfunc="first", sort=False
    )
    count_wide = long.pivot_table(
        index=idx, columns=column_levels, values="N Used", aggfunc="first", sort=False
    ).astype("Int64")
    # Enforce configured dimension/metric row order after pivoting.
    value_wide = _add_dimension_and_sort_wide(value_wide)
    count_wide = _add_dimension_and_sort_wide(count_wide)
    if order_map:
        value_wide = _reorder_columns(value_wide, order_map)
        count_wide = _reorder_columns(count_wide, order_map)
    return value_wide, count_wide


# ---------------------------------------------------------------------------
# Minimal LaTeX writer (standalone)
# ---------------------------------------------------------------------------
def _headers_underscores_to_spaces(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _fix(x: object) -> str:
        if x is None:
            return ""
        s = str(x)
        if s == LOG_ID_COL:
            return "log\\_id"
        return s.replace("_", " ")

    if isinstance(out.index, pd.MultiIndex):
        out.index = out.index.set_names([_fix(n) for n in out.index.names])
        for lev in range(out.index.nlevels):
            vals = out.index.levels[lev]
            out.index = out.index.set_levels([_fix(v) for v in vals], level=lev)
    else:
        out.index = pd.Index([_fix(v) for v in out.index], name=_fix(out.index.name))

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.set_names([_fix(n) for n in out.columns.names])
        for lev in range(out.columns.nlevels):
            vals = out.columns.levels[lev]
            out.columns = out.columns.set_levels([_fix(v) for v in vals], level=lev)
    else:
        out.columns = pd.Index([_fix(v) for v in out.columns], name=_fix(out.columns.name))

    return out


def _apply_latex_metric_names(df: pd.DataFrame, index: bool) -> pd.DataFrame:
    out = df.copy()
    if index and isinstance(out.index, pd.MultiIndex) and "Metric" in out.index.names:
        p = out.index.names.index("Metric")
        vals = out.index.levels[p]
        out.index = out.index.set_levels([METRIC_NAMES_TO_LATEX_MAP.get(v, str(v)) for v in vals], level=p)
    elif index and out.index.name == "Metric":
        out.index = pd.Index([METRIC_NAMES_TO_LATEX_MAP.get(v, str(v)) for v in out.index], name=out.index.name)
    elif "Metric" in out.columns:
        out["Metric"] = out["Metric"].map(lambda v: METRIC_NAMES_TO_LATEX_MAP.get(v, str(v)))
    return out


def _inject_rank_highlight(latex: str, df: pd.DataFrame, n_index_cols: int) -> str:
    numeric_cols = [j for j in range(len(df.columns)) if pd.api.types.is_numeric_dtype(df.iloc[:, j])]
    if not numeric_cols:
        return latex

    # Determine highest and second-highest per numeric column (ties included).
    best_rows_by_col: dict[int, set[int]] = {}
    second_rows_by_col: dict[int, set[int]] = {}
    for j in numeric_cols:
        col_vals: dict[int, float] = {}
        for i in range(len(df)):
            try:
                v = float(df.iloc[i, j])
            except (ValueError, TypeError):
                continue
            if np.isfinite(v):
                col_vals[i] = v
        if not col_vals:
            best_rows_by_col[j] = set()
            second_rows_by_col[j] = set()
            continue
        uniq = sorted(set(col_vals.values()), reverse=True)
        best = uniq[0]
        second = uniq[1] if len(uniq) > 1 else None
        best_rows_by_col[j] = {i for i, v in col_vals.items() if v == best}
        second_rows_by_col[j] = (
            {i for i, v in col_vals.items() if second is not None and v == second}
            if second is not None
            else set()
        )

    lines = latex.split("\n")
    out_lines: list[str] = []
    body = False
    row_i = 0
    for line in lines:
        if "\\midrule" in line:
            body = True
            out_lines.append(line)
            continue
        if body and ("\\bottomrule" in line or "\\end{tabular}" in line):
            body = False
            out_lines.append(line)
            continue
        if body and row_i < len(df):
            parts = line.split(" & ")
            new_parts = list(parts[:n_index_cols])
            for j in range(len(df.columns)):
                idx = n_index_cols + j
                cell = parts[idx].rstrip().rstrip("\\\\").strip() if idx < len(parts) else ""
                if row_i in best_rows_by_col.get(j, set()):
                    cell = f"\\textbf{{{cell}}}"
                elif row_i in second_rows_by_col.get(j, set()):
                    cell = f"\\underline{{{cell}}}"
                new_parts.append(cell)
            line = " & ".join(new_parts) + " \\\\"
            row_i += 1
        out_lines.append(line)
    return "\n".join(out_lines)


def _fix_flat_table_midrule_position(latex: str, *, index: bool) -> str:
    """
    Pandas may emit \\midrule after the first data row for some flat tables.
    Move it right below the header row when index=False.
    """
    if index:
        return latex
    lines = latex.split("\n")
    top_i = next((i for i, ln in enumerate(lines) if "\\toprule" in ln), -1)
    mid_i = next((i for i, ln in enumerate(lines) if "\\midrule" in ln), -1)
    if top_i == -1 or mid_i == -1:
        return latex
    # Expected header row directly after \toprule.
    header_i = top_i + 1
    desired_mid_i = header_i + 1
    if mid_i == desired_mid_i:
        return latex
    if mid_i > desired_mid_i:
        mid_line = lines.pop(mid_i)
        lines.insert(desired_mid_i, mid_line)
    return "\n".join(lines)


def _write_latex_table(df: pd.DataFrame, stem: str, caption: str, label: str, *, index: bool = True, decimals: int = 2) -> None:
    out_path = Path(DIR_LATEX) / f"{stem}.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = _apply_latex_metric_names(df, index=index)
    table = _headers_underscores_to_spaces(table)

    n_cols = len(table.columns)
    n_idx = len(table.index.names) if index and isinstance(table.index, pd.MultiIndex) else (1 if index else 0)
    if n_idx == 0 and index:
        n_idx = 1
    col_fmt = "l" * n_idx + "c" * n_cols

    latex = table.to_latex(
        caption=caption,
        label=label,
        position="H",
        column_format=col_fmt,
        escape=False,
        index=index,
        float_format=f"%.{decimals}f",
        na_rep="",
        multicolumn=isinstance(table.columns, pd.MultiIndex),
        multicolumn_format="c",
        sparsify=True,
    )
    latex = _fix_flat_table_midrule_position(latex, index=index)
    if len(table) > 0 and len(table.columns) > 0:
        latex = _inject_rank_highlight(latex, table, n_idx)
        legend = (
            "\n\\vspace{2pt}\n"
            "\\begin{minipage}{\\linewidth}\n"
            "\\centering\n"
            "\\tiny\n"
            "\\textit{Formatting:} \\textbf{Bold} = highest per column, "
            "\\underline{underlined} = second highest per column.\n"
            "\\end{minipage}"
        )
        latex = latex.replace("\\end{tabular}", "\\end{tabular}" + legend)
    latex = latex.replace("\\centering\n", "\\centering\n\\scriptsize\n")
    latex = re.sub(
        r"(\\label\{[^}]*\})\n",
        r"\1\n\\setlength{\\tabcolsep}{2pt}  % default is 6pt\n",
        latex,
    )
    out_path.write_text(latex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Domain calculations
# ---------------------------------------------------------------------------
def _build_pre_seed_table(df_enriched: pd.DataFrame) -> pd.DataFrame:
    pre = df_enriched[df_enriched[SPLIT_NAME_COL] == "pre_drift"].copy()
    if len(pre) == 0:
        raise ValueError("No pre_drift rows found.")
    pre["Variance Pre"] = pre["Sample Std"] ** 2
    keys = ["Metric", SEED_COL, MODEL_COMPLEXITY_COL, NOISE_LEVEL_COL, WINDOW_SIZE_COL]
    values = ["Median Value", "Mean Value", "Sample Std", "Variance Pre"]
    return _collapse_within_seed(pre, group_keys_with_seed=keys, value_cols=values, agg_mode="median")


def _build_setting_change_seed(pre_seed: pd.DataFrame) -> pd.DataFrame:
    base_keys = ["Metric", SEED_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL]
    baseline = (
        pre_seed[pre_seed[NOISE_LEVEL_COL] == "None"]
        .groupby(base_keys, dropna=False, sort=False)[["Median Value", "Mean Value"]]
        .agg(_robust_median)
        .reset_index()
        .rename(columns={"Median Value": "Baseline Median", "Mean Value": "Baseline Mean"})
    )
    d = pre_seed.merge(baseline, on=base_keys, how="left")
    d["settingChange_median_absChange"] = (d["Median Value"] - d["Baseline Median"]).abs()
    d["settingChange_mean_absChange"] = (d["Mean Value"] - d["Baseline Mean"]).abs()
    d["settingChange_median_relChange"] = _safe_ratio(d["settingChange_median_absChange"], d["Baseline Median"].abs())
    d["settingChange_mean_relChange"] = _safe_ratio(d["settingChange_mean_absChange"], d["Baseline Mean"].abs())
    d["settingChange_median_invRelChange"] = _safe_ratio(d["Baseline Median"].abs(), d["settingChange_median_absChange"])
    d["settingChange_mean_invRelChange"] = _safe_ratio(d["Baseline Mean"].abs(), d["settingChange_mean_absChange"])
    return d


def _build_process_change_seed(df_enriched: pd.DataFrame) -> pd.DataFrame:
    split_df = df_enriched[df_enriched[SPLIT_NAME_COL].isin(["pre_drift", "post_drift"])].copy()
    needed = [
        "Metric",
        LOG_ID_COL,
        WINDOW_SIZE_COL,
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        CHANGE_MAGNITUDE_COL,
        EDIT_OPERATIONS_COL,
        SEED_COL,
        "Median Value",
        "Mean Value",
        "Sample Std",
        SPLIT_NAME_COL,
    ]
    split_df = split_df[needed]
    pre = split_df[split_df[SPLIT_NAME_COL] == "pre_drift"].rename(
        columns={"Median Value": "Median Pre", "Mean Value": "Mean Pre", "Sample Std": "Std Pre"}
    )
    post = split_df[split_df[SPLIT_NAME_COL] == "post_drift"].rename(
        columns={"Median Value": "Median Post", "Mean Value": "Mean Post", "Sample Std": "Std Post"}
    )
    join_cols = ["Metric", LOG_ID_COL, WINDOW_SIZE_COL]
    pre_cols = join_cols + [
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        CHANGE_MAGNITUDE_COL,
        EDIT_OPERATIONS_COL,
        SEED_COL,
        "Median Pre",
        "Mean Pre",
        "Std Pre",
    ]
    post_cols = join_cols + ["Median Post", "Mean Post", "Std Post"]
    joined = pre[pre_cols].merge(post[post_cols], on=join_cols, how="inner")
    if len(joined) == 0:
        raise ValueError("No matching pre/post rows found.")

    joined["Var Pre"] = joined["Std Pre"] ** 2
    joined["Var Post"] = joined["Std Post"] ** 2
    keys = [
        "Metric",
        SEED_COL,
        WINDOW_SIZE_COL,
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        CHANGE_MAGNITUDE_COL,
        EDIT_OPERATIONS_COL,
    ]
    values = ["Median Pre", "Mean Pre", "Median Post", "Mean Post", "Var Pre", "Var Post"]
    seed = _collapse_within_seed(joined, group_keys_with_seed=keys, value_cols=values, agg_mode="median")

    dm = seed["Median Post"] - seed["Median Pre"]
    dmean = seed["Mean Post"] - seed["Mean Pre"]
    am = dm.abs()
    amean = dmean.abs()
    seed["processChange_median_absChange"] = am
    seed["processChange_mean_absChange"] = amean
    seed["processChange_median_relChange"] = _safe_ratio(am, seed["Median Pre"].abs())
    seed["processChange_mean_relChange"] = _safe_ratio(amean, seed["Mean Pre"].abs())
    seed["processChange_median_invRelChange"] = _safe_ratio(seed["Median Pre"].abs(), am)
    seed["processChange_mean_invRelChange"] = _safe_ratio(seed["Mean Pre"].abs(), amean)

    pooled_sd = np.sqrt((seed["Var Pre"] + seed["Var Post"]) / 2)
    # Consistent with other within-seed median/mean comparisons: use absolute change.
    seed["processChange_median_cohensD"] = _safe_ratio(am, pooled_sd)
    seed["processChange_mean_cohensD"] = _safe_ratio(amean, pooled_sd)
    return seed


def _aggregate_measure(
    *,
    df_seed: pd.DataFrame,
    measure_col: str,
    group_keys_without_seed: list[str],
    column_levels: list[str],
    order_map: dict[str, list[Any]] | None,
    across_seed_aggregation: Literal["mean", "median"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df_seed[group_keys_without_seed + [SEED_COL, measure_col]].copy()
    d = d.rename(columns={measure_col: "Measure Value"})
    agg = _aggregate_across_seed_with_counts(
        d,
        group_keys_without_seed=group_keys_without_seed,
        value_col="Measure Value",
        across_seed_aggregation=across_seed_aggregation,
    )
    return _pivot_value_and_counts(agg, column_levels=column_levels, order_map=order_map)


def _aggregate_grouped_single_level(
    *,
    df_seed: pd.DataFrame,
    measure_col: str,
    group_col: str,
    column_order: list[Any] | None,
    across_seed_aggregation: Literal["mean", "median"],
    filter_col: str | None = None,
    filter_val: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df_seed.copy()
    if filter_col is not None:
        d = d[d[filter_col] == filter_val].copy()
    if len(d) == 0:
        raise ValueError(f"No rows left after filtering {filter_col}={filter_val!r}")
    agg = _aggregate_across_seed_with_counts(
        d[["Metric", group_col, SEED_COL, measure_col]].rename(columns={measure_col: "Measure Value"}),
        group_keys_without_seed=["Metric", group_col],
        value_col="Measure Value",
        across_seed_aggregation=across_seed_aggregation,
    )
    value_wide = agg.pivot_table(index="Metric", columns=group_col, values="Value", aggfunc="first")
    count_wide = agg.pivot_table(index="Metric", columns=group_col, values="N Used", aggfunc="first").astype("Int64")
    if column_order is not None:
        existing = [c for c in column_order if c in value_wide.columns]
        existing.extend([c for c in value_wide.columns if c not in existing])
        value_wide = value_wide[existing]
        count_wide = count_wide[existing]
    return _add_dimension_and_sort_wide(value_wide), _add_dimension_and_sort_wide(count_wide)


def _save_bundle(*, stem: str, value_wide: pd.DataFrame, count_wide: pd.DataFrame, caption: str, label: str) -> None:
    Path(DIR_CSV).mkdir(parents=True, exist_ok=True)
    Path(DIR_LATEX).mkdir(parents=True, exist_ok=True)
    value_csv = Path(DIR_CSV) / f"{stem}.csv"
    count_csv = Path(DIR_CSV) / f"{stem}_counts.csv"
    value_wide.to_csv(value_csv)
    count_wide.to_csv(count_csv)
    _write_latex_table(value_wide, stem, caption=caption, label=label, index=True, decimals=2)
    print(f"Saved {value_csv}")
    print(f"Saved {count_csv}")
    print(f"Saved {Path(DIR_LATEX) / (stem + '.tex')}")


def _robust_median_inf_aware(x: pd.Series) -> float:
    """
    Median where finite values dominate infinities.
    If finite values exist, +/-inf are ignored for the median.
    """
    arr = _non_nan(x).to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        return float(np.median(finite))
    pos_inf_count = np.isposinf(arr).sum()
    neg_inf_count = np.isneginf(arr).sum()
    if pos_inf_count and not neg_inf_count:
        return float(np.inf)
    if neg_inf_count and not pos_inf_count:
        return float(-np.inf)
    return np.nan


def _metric_median_from_observations(
    *,
    df: pd.DataFrame,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    d = df[df["Metric"] != "Number of Traces"].copy()
    agg = (
        d.groupby("Metric", dropna=False, sort=False)[value_col]
        .apply(_robust_median_inf_aware)
        .reset_index()
        .rename(columns={value_col: out_col})
    )
    agg["Dimension"] = agg["Metric"].map(METRIC_DIMENSION_MAP).fillna("Other")
    # Keep legacy compatibility in case older data still carries "Length".
    agg["Dimension"] = agg["Dimension"].replace({"Length": "Size"})
    return agg[["Dimension", "Metric", out_col]]


def _metric_count_from_observations(
    *,
    df: pd.DataFrame,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    d = df[df["Metric"] != "Number of Traces"].copy()
    counts = (
        d.groupby("Metric", dropna=False, sort=False)[value_col]
        .apply(lambda s: int(len(_non_nan(s))))
        .reset_index()
        .rename(columns={value_col: out_col})
    )
    counts["Dimension"] = counts["Metric"].map(METRIC_DIMENSION_MAP).fillna("Other")
    counts["Dimension"] = counts["Dimension"].replace({"Length": "Size"})
    return counts[["Dimension", "Metric", out_col]]


def _build_median_across_all_obs(
    *,
    stability_seed: pd.DataFrame,
    setting_seed: pd.DataFrame,
    process_seed: pd.DataFrame,
) -> pd.DataFrame:
    reliability = _metric_median_from_observations(
        df=stability_seed,
        value_col="stability_inverse_CV",
        out_col="Reliability",
    )
    robustness = _metric_median_from_observations(
        df=setting_seed[setting_seed[NOISE_LEVEL_COL] != "None"].copy(),
        value_col="settingChange_median_invRelChange",
        out_col="Robustness",
    )
    responsiveness = _metric_median_from_observations(
        df=process_seed[process_seed[EDIT_OPERATIONS_COL] == "mixed"].copy(),
        value_col="processChange_median_cohensD",
        out_col="Responsiveness",
    )

    merged = (
        reliability.merge(robustness, on=["Dimension", "Metric"], how="outer")
        .merge(responsiveness, on=["Dimension", "Metric"], how="outer")
        .copy()
    )
    merged = _add_dimension_and_sort_long(merged)

    label_prefix_by_dimension = {
        "Size": "A",
        "Variation": "B",
        "Distance": "C",
        "Graph Entropy": "D",
    }
    label_counts = {k: 0 for k in label_prefix_by_dimension}
    labels: list[str] = []
    for dim in merged["Dimension"].astype(str):
        prefix = label_prefix_by_dimension.get(dim)
        if prefix is None:
            labels.append("")
            continue
        label_counts[dim] += 1
        labels.append(f"{prefix}{label_counts[dim]}")
    merged.insert(2, "Label", labels)
    return merged[["Dimension", "Metric", "Label", "Reliability", "Robustness", "Responsiveness"]]


def _build_median_across_all_obs_counts(
    *,
    stability_seed: pd.DataFrame,
    setting_seed: pd.DataFrame,
    process_seed: pd.DataFrame,
) -> pd.DataFrame:
    reliability = _metric_count_from_observations(
        df=stability_seed,
        value_col="stability_inverse_CV",
        out_col="Reliability",
    )
    robustness = _metric_count_from_observations(
        df=setting_seed[setting_seed[NOISE_LEVEL_COL] != "None"].copy(),
        value_col="settingChange_median_invRelChange",
        out_col="Robustness",
    )
    responsiveness = _metric_count_from_observations(
        df=process_seed[process_seed[EDIT_OPERATIONS_COL] == "mixed"].copy(),
        value_col="processChange_median_cohensD",
        out_col="Responsiveness",
    )

    merged = (
        reliability.merge(robustness, on=["Dimension", "Metric"], how="outer")
        .merge(responsiveness, on=["Dimension", "Metric"], how="outer")
        .copy()
    )
    merged = _add_dimension_and_sort_long(merged)

    label_prefix_by_dimension = {
        "Size": "A",
        "Variation": "B",
        "Distance": "C",
        "Graph Entropy": "D",
    }
    label_counts = {k: 0 for k in label_prefix_by_dimension}
    labels: list[str] = []
    for dim in merged["Dimension"].astype(str):
        prefix = label_prefix_by_dimension.get(dim)
        if prefix is None:
            labels.append("")
            continue
        label_counts[dim] += 1
        labels.append(f"{prefix}{label_counts[dim]}")
    merged.insert(2, "Label", labels)
    merged["Reliability"] = pd.to_numeric(merged["Reliability"], errors="coerce").astype("Int64")
    merged["Robustness"] = pd.to_numeric(merged["Robustness"], errors="coerce").astype("Int64")
    merged["Responsiveness"] = pd.to_numeric(merged["Responsiveness"], errors="coerce").astype("Int64")
    return merged[["Dimension", "Metric", "Label", "Reliability", "Robustness", "Responsiveness"]]


def _save_median_across_all_obs_bundle(median_df: pd.DataFrame, count_df: pd.DataFrame) -> None:
    Path(DIR_CSV).mkdir(parents=True, exist_ok=True)
    value_csv = Path(DIR_CSV) / "median_across_all_obs.csv"
    count_csv = Path(DIR_CSV) / "median_across_all_obs_counts.csv"
    median_df.to_csv(value_csv, index=False)
    count_df.to_csv(count_csv, index=False)
    _write_latex_table(
        median_df,
        "median_across_all_obs",
        caption="Observation-level medians of reliability, robustness, and responsiveness.",
        label="tab:median-across-all-obs",
        index=False,
        decimals=2,
    )
    _write_latex_table(
        count_df,
        "median_across_all_obs_counts",
        caption="Non-NaN observation counts used for median_across_all_obs.",
        label="tab:median-across-all-obs-counts",
        index=False,
        decimals=0,
    )
    print(f"Saved {value_csv}")
    print(f"Saved {count_csv}")
    print(f"Saved {Path(DIR_LATEX) / 'median_across_all_obs.tex'}")
    print(f"Saved {Path(DIR_LATEX) / 'median_across_all_obs_counts.tex'}")


def _plot_median_scatter(median_df: pd.DataFrame) -> None:
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        lines_mod = importlib.import_module("matplotlib.lines")
        Line2D = lines_mod.Line2D
    except ModuleNotFoundError:
        print("Skipping median scatter plot: matplotlib is not installed.")
        return

    plots_dir = Path("results/signal_noise_study/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "median_scatter.png"
    plot_pdf_path = plots_dir / "median_scatter.pdf"

    plot_df = median_df.copy()
    x_col = "Robustness"
    y_col = "Responsiveness"
    for col in ["Reliability", x_col, y_col]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col]).copy()
    if len(plot_df) == 0:
        print("Skipping median scatter plot: no finite Robustness/Responsiveness rows.")
        return

    x = pd.to_numeric(plot_df[x_col], errors="coerce")
    y = pd.to_numeric(plot_df[y_col], errors="coerce")
    x_finite = x[np.isfinite(x)]
    y_finite = y[np.isfinite(y)]

    if x_finite.empty:
        x_min, x_max = 0.0, 1.0
    else:
        x_min, x_max = float(x_finite.min()), float(x_finite.max())
    if y_finite.empty:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = float(y_finite.min()), float(y_finite.max())

    x_pad = max((x_max - x_min) * 0.05, 1e-9)
    y_pad = max((y_max - y_min) * 0.05, 1e-9)
    x_left, x_right = x_min - x_pad, x_max + x_pad
    y_bottom, y_top = y_min - y_pad, y_max + y_pad

    plot_df["x_plot"] = x.replace(np.inf, x_right).replace(-np.inf, x_left)
    plot_df["y_plot"] = y.replace(np.inf, y_top).replace(-np.inf, y_bottom)
    plot_df["x_inf_dir"] = np.where(np.isposinf(x), 1, np.where(np.isneginf(x), -1, 0))
    plot_df["y_inf_dir"] = np.where(np.isposinf(y), 1, np.where(np.isneginf(y), -1, 0))

    rel = pd.to_numeric(plot_df["Reliability"], errors="coerce")
    rel_finite = rel[np.isfinite(rel)]
    min_marker_size = 40.0
    max_marker_size = 240.0
    if rel_finite.empty or float(rel_finite.max()) == float(rel_finite.min()):
        plot_df["marker_size"] = (min_marker_size + max_marker_size) / 2
    else:
        lo = float(rel_finite.min())
        hi = float(rel_finite.max())
        rel_for_scale = rel.copy().replace(np.inf, hi).replace(-np.inf, lo)
        rel_norm = (rel_for_scale - lo) / (hi - lo)
        plot_df["marker_size"] = min_marker_size + rel_norm * (max_marker_size - min_marker_size)
        plot_df["marker_size"] = plot_df["marker_size"].fillna(min_marker_size)

    overlap_decimals = 6
    plot_df["x_group"] = plot_df["x_plot"].round(overlap_decimals)
    plot_df["y_group"] = plot_df["y_plot"].round(overlap_decimals)

    dimensions = list(dict.fromkeys(plot_df["Dimension"].astype(str).tolist()))
    cmap = plt.get_cmap("tab10")
    color_map = {d: cmap(i % 10) for i, d in enumerate(dimensions)}

    FONT_AXIS_LABEL = 13
    FONT_TICKS = 11
    FONT_LEGEND = 10
    FONT_LEGEND_TITLE = 11
    FONT_ANNOT = 9
    FONT_INF_NOTE = 10

    inf_notes = []
    for _, row in plot_df.iterrows():
        flags = []
        if np.isposinf(row[x_col]):
            flags.append("x=+inf")
        elif np.isneginf(row[x_col]):
            flags.append("x=-inf")
        if np.isposinf(row[y_col]):
            flags.append("y=+inf")
        elif np.isneginf(row[y_col]):
            flags.append("y=-inf")
        if flags:
            inf_notes.append(f"{row['Metric']} ({', '.join(flags)})")

    def _add_inf_note_box(ax: Any) -> None:
        if not inf_notes:
            return
        max_notes = 16
        shown = inf_notes[:max_notes]
        more = len(inf_notes) - len(shown)
        note_text = "Infs shown with border arrows:\n" + "\n".join(f"- {n}" for n in shown)
        if more > 0:
            note_text += f"\n- ... and {more} more"
        ax.text(
            1.02,
            0.02,
            note_text,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=FONT_INF_NOTE,
            bbox={"boxstyle": "round", "alpha": 0.15},
        )

    def _add_dimension_legend(ax: Any) -> Any:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color_map[d],
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=7,
                label=d,
            )
            for d in dimensions
        ]
        return ax.legend(
            handles=handles,
            title="Complexity dimension",
            bbox_to_anchor=(1.02, 0.74, 0.28, 0.22),
            loc="upper left",
            mode="expand",
            borderaxespad=0.0,
            fontsize=FONT_LEGEND,
            title_fontsize=FONT_LEGEND_TITLE,
        )

    def _size_from_reliability(v: float) -> float:
        if rel_finite.empty:
            return (min_marker_size + max_marker_size) / 2
        lo = float(rel_finite.min())
        hi = float(rel_finite.max())
        if hi == lo:
            return (min_marker_size + max_marker_size) / 2
        if pd.isna(v):
            vv = lo
        elif np.isposinf(v):
            vv = hi
        elif np.isneginf(v):
            vv = lo
        else:
            vv = float(v)
        vv = min(max(vv, lo), hi)
        t = (vv - lo) / (hi - lo)
        return min_marker_size + t * (max_marker_size - min_marker_size)

    def _add_size_legend(ax: Any, dim_legend: Any = None) -> Any:
        if rel_finite.empty:
            if dim_legend is not None:
                ax.add_artist(dim_legend)
            return None
        raw_vals = [float(rel_finite.min()), float(rel_finite.median()), float(rel_finite.max())]
        rel_vals: list[float] = []
        for v in raw_vals:
            if not any(np.isclose(v, u, rtol=0.0, atol=1e-12) for u in rel_vals):
                rel_vals.append(v)

        size_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=np.sqrt(_size_from_reliability(v)),
                label=f"{v:.2f}",
            )
            for v in rel_vals
        ]
        size_legend = ax.legend(
            handles=size_handles,
            title="Reliability",
            bbox_to_anchor=(1.02, 0.50, 0.28, 0.2),
            loc="upper left",
            mode="expand",
            borderaxespad=0.0,
            frameon=True,
            fontsize=FONT_LEGEND,
            title_fontsize=FONT_LEGEND_TITLE,
        )
        if dim_legend is not None:
            ax.add_artist(dim_legend)
        return size_legend

    def _add_inf_boundary_arrows(ax: Any) -> None:
        per_key_count: dict[tuple[Any, Any], int] = {}
        for r in plot_df.itertuples(index=False):
            if int(r.x_inf_dir) == 0 and int(r.y_inf_dir) == 0:
                continue
            key = (r.x_group, r.y_group)
            k = per_key_count.get(key, 0)
            per_key_count[key] = k + 1
            spread = ((k % 5) - 2) * 4

            if int(r.x_inf_dir) == 1:
                ax.annotate(
                    "",
                    xy=(r.x_plot, r.y_plot),
                    xytext=(20, 0),
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "<-", "lw": 0.8, "alpha": 0.8},
                    annotation_clip=False,
                )
                ax.annotate(
                    "inf",
                    xy=(r.x_plot, r.y_plot),
                    xytext=(10, 12 + spread),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_ANNOT,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8},
                    annotation_clip=False,
                )
            elif int(r.x_inf_dir) == -1:
                ax.annotate(
                    "",
                    xy=(r.x_plot, r.y_plot),
                    xytext=(-20, 0),
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "<-", "lw": 0.8, "alpha": 0.8},
                    annotation_clip=False,
                )
                ax.annotate(
                    "-inf",
                    xy=(r.x_plot, r.y_plot),
                    xytext=(-10, 12 + spread),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_ANNOT,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8},
                    annotation_clip=False,
                )
            if int(r.y_inf_dir) == 1:
                ax.annotate(
                    "inf",
                    xy=(r.x_plot, r.y_plot),
                    xytext=(spread, 14),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_ANNOT,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.75},
                    arrowprops={"arrowstyle": "->", "lw": 0.65, "alpha": 0.7},
                    annotation_clip=False,
                )
            elif int(r.y_inf_dir) == -1:
                ax.annotate(
                    "-inf",
                    xy=(r.x_plot, r.y_plot),
                    xytext=(spread, -14),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=FONT_ANNOT,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.75},
                    arrowprops={"arrowstyle": "->", "lw": 0.65, "alpha": 0.7},
                    annotation_clip=False,
                )

    def _draw_points_with_overlap_styles(ax: Any) -> None:
        xy_groups = plot_df.groupby(["x_group", "y_group"], sort=False, dropna=False)
        xr = max(x_right - x_left, 1e-9)
        yr = max(y_top - y_bottom, 1e-9)
        for _, g in xy_groups:
            rows = list(g.itertuples(index=False))
            xp = float(np.mean([r.x_plot for r in rows]))
            yp = float(np.mean([r.y_plot for r in rows]))
            if len(rows) == 1:
                r = rows[0]
                ax.scatter(
                    [xp],
                    [yp],
                    s=float(r.marker_size),
                    color=color_map[str(r.Dimension)],
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=3,
                )
            elif len(rows) == 2:
                r1, r2 = rows
                mean_size = (float(r1.marker_size) + float(r2.marker_size)) / 2.0
                ax.plot(
                    [xp],
                    [yp],
                    marker="o",
                    linestyle="None",
                    markersize=np.sqrt(mean_size) * 1.05,
                    markerfacecolor=color_map[str(r1.Dimension)],
                    markerfacecoloralt=color_map[str(r2.Dimension)],
                    fillstyle="left",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    alpha=1.0,
                    zorder=5,
                )
            else:
                for i, r in enumerate(rows):
                    theta = 2 * np.pi * i / len(rows)
                    jx = xp + 0.008 * xr * np.cos(theta)
                    jy = yp + 0.012 * yr * np.sin(theta)
                    ax.scatter(
                        [jx],
                        [jy],
                        s=max(float(r.marker_size) * 0.55, 18),
                        color=color_map[str(r.Dimension)],
                        alpha=0.95,
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=5,
                    )

    label_position_map = {
        "A1": "left",
        "A2": "right",
        "A3": "top",
        "A4": "left",
        "A5": "top",
        "A6": "right",
        "A7": "left",
        "B1": "right",
        "B2": "bottom",
        "B3": "left",
        "B4": "right",
        "B5": "right",
        "B6": "top",
        "D1": "bottom",
        "D2": "bottom",
        "D4": "left",
    }
    position_to_style = {
        "top": (0, 10, "center", "bottom"),
        "bottom": (0, -10, "center", "top"),
        "left": (-10, 0, "right", "center"),
        "right": (10, 0, "left", "center"),
    }

    def _annotate_overlap_labels(
        ax: Any,
        text_col: str,
        *,
        position_map: dict[str, str] | None = None,
        default_position: str = "top",
        fontsize: int = FONT_ANNOT,
    ) -> None:
        grouped = plot_df.groupby(["x_group", "y_group"], dropna=False, sort=False)
        dup_idx = grouped.cumcount()
        group_sizes = grouped[text_col].transform("size")
        for i, row in plot_df.iterrows():
            k = int(dup_idx.loc[i])
            n = int(group_sizes.loc[i])
            label_txt = str(row[text_col])
            pos = (position_map or {}).get(label_txt, default_position)
            if pos not in position_to_style:
                pos = default_position
            ox, oy, ha, va = position_to_style[pos]

            if n > 1:
                spread_step = 4
                centered_k = k - (n - 1) / 2.0
                if pos in {"top", "bottom"}:
                    ox += int(round(centered_k * spread_step))
                else:
                    oy += int(round(centered_k * spread_step))

            arrow = {"arrowstyle": "-", "lw": 1.0, "alpha": 0.9, "color": "gray"}
            ax.annotate(
                label_txt,
                (row["x_plot"], row["y_plot"]),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=fontsize,
                alpha=0.97,
                ha=ha,
                va=va,
                bbox={"boxstyle": "round,pad=0.15", "fc": "none", "ec": "none", "alpha": 0.0},
                arrowprops=arrow,
            )

    fig, ax = plt.subplots(figsize=(12, 6.5))
    _draw_points_with_overlap_styles(ax)
    _annotate_overlap_labels(
        ax,
        "Label",
        position_map=label_position_map,
        default_position="top",
        fontsize=FONT_ANNOT,
    )
    ax.set_xlabel("Robustness to log noise change", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Responsiveness", fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICKS)
    ax.grid(True, alpha=0.25)
    dim_leg = _add_dimension_legend(ax)
    _add_size_legend(ax, dim_leg)
    _add_inf_boundary_arrows(ax)
    _add_inf_note_box(ax)
    fig.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(plot_pdf_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved scatter plot to: {plot_path}")
    print(f"Saved scatter plot PDF to: {plot_pdf_path}")


def run(
    *,
    across_seed_aggregation: Literal["mean", "median"] = "median",
    test_mode: bool = False,
    test_seeds: list[int] | None = None,
    only_median_overview: bool = False,
) -> None:
    if test_mode:
        seeds = test_seeds if test_seeds is not None else DEFAULT_TEST_SEEDS
        if len(seeds) != 2:
            raise ValueError("In --test mode you must provide exactly 2 test seeds.")
        print(f"Test mode active with seeds: {seeds}")
    else:
        seeds = None

    print("=" * 60)
    print("SIMPLIFIED STABILITY/SETTING/PROCESS CHANGE ANALYSIS")
    print("=" * 60)
    print(f"Across-seed aggregation: {across_seed_aggregation}")

    print("Loading aggregate analysis input...")
    df = load_and_validate_input(PATH_AGG)
    print(f"  Rows loaded: {len(df):,}")
    print("Normalizing split names...")
    df = normalize_split_name(df)
    print("Loading generation info...")
    gen = load_generation_info(PATH_GEN_INFO)
    print(f"  Generation rows: {len(gen):,}")

    if seeds:
        print(f"Applying test seed filter to generation info: {seeds}")
        gen = gen[gen[SEED_COL].isin(seeds)].copy()
        if len(gen) == 0:
            raise ValueError(
                f"No generation_info rows for test seeds {seeds}. "
                "Use seeds present in generation_info.csv (e.g., 43 and 44)."
            )
        print(f"  Generation rows after seed filter: {len(gen):,}")
    print("Enriching analysis rows with experimental factors...")
    df_enriched = enrich_with_experimental_factors(df, gen)
    print(f"  Enriched rows: {len(df_enriched):,}")
    if seeds:
        df_enriched = df_enriched[df_enriched[SEED_COL].isin(seeds)].copy()
        df_enriched = select_one_metric_per_dimension(df_enriched)
        if len(df_enriched) == 0:
            raise ValueError("No rows left after test filters.")
        print(f"  Rows after test filters: {len(df_enriched):,}")

    print("Building per-seed tables (pre, setting-change, process-change)...")
    pre_seed = _build_pre_seed_table(df_enriched)
    setting_seed = _build_setting_change_seed(pre_seed)
    process_seed = _build_process_change_seed(df_enriched)
    print(
        f"  Seed tables: pre={len(pre_seed):,}, "
        f"setting={len(setting_seed):,}, process={len(process_seed):,}"
    )

    stability_seed = pre_seed.copy()
    stability_seed["stability_variance"] = stability_seed["Variance Pre"]
    stability_seed["stability_CV"] = _safe_ratio(stability_seed["Sample Std"], stability_seed["Mean Value"].abs())
    stability_seed["stability_inverse_CV"] = _safe_ratio(stability_seed["Mean Value"].abs(), stability_seed["Sample Std"])
    setting_seed_no_none = setting_seed[setting_seed[NOISE_LEVEL_COL] != "None"].copy()

    median_df = _build_median_across_all_obs(
        stability_seed=stability_seed,
        setting_seed=setting_seed,
        process_seed=process_seed,
    )
    median_counts_df = _build_median_across_all_obs_counts(
        stability_seed=stability_seed,
        setting_seed=setting_seed,
        process_seed=process_seed,
    )
    _save_median_across_all_obs_bundle(median_df, median_counts_df)
    _plot_median_scatter(median_df)

    if only_median_overview:
        print("Only median overview requested; skipping all other aggregate tables.")
        print("\nAnalysis complete.")
        return

    noise_order = ["None", "Low", "High"]
    complexity_order = ["Simple", "Middle", "Complex"]
    order_map_noise = {NOISE_LEVEL_COL: noise_order, MODEL_COMPLEXITY_COL: complexity_order}
    order_map_process = {
        EDIT_OPERATIONS_COL: CHANGE_OPERATION_ORDER,
        NOISE_LEVEL_COL: noise_order,
        MODEL_COMPLEXITY_COL: complexity_order,
    }

    specs: list[dict[str, Any]] = [
        {
            "name": "stability_variance",
            "seed_df": stability_seed,
            "group_keys": ["Metric", NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
            "column_levels": [NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
            "order_map": order_map_noise,
            "caption": "Variance in pre-drift part of log.",
            "label": "tab:stability-variance",
        },
        {
            "name": "stability_CV",
            "seed_df": stability_seed,
            "group_keys": ["Metric", NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
            "column_levels": [NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
            "order_map": order_map_noise,
            "caption": "Coefficient of variation in pre-drift part of log.",
            "label": "tab:stability-cv",
        },
        {
            "name": "stability_inverse_CV",
            "seed_df": stability_seed,
            "group_keys": ["Metric", NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
            "column_levels": [NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
            "order_map": order_map_noise,
            "caption": "Inverse coefficient of variation in pre-drift part of log.",
            "label": "tab:stability-inverse-cv",
        },
    ]

    for name in [
        "settingChange_median_absChange",
        "settingChange_mean_absChange",
        "settingChange_median_relChange",
        "settingChange_mean_relChange",
        "settingChange_median_invRelChange",
        "settingChange_mean_invRelChange",
    ]:
        specs.append(
            {
                "name": name,
                "seed_df": setting_seed_no_none,
                "group_keys": ["Metric", NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
                "column_levels": [NOISE_LEVEL_COL, MODEL_COMPLEXITY_COL, WINDOW_SIZE_COL],
                "order_map": order_map_noise,
                "caption": f"{name} aggregated across seeds.",
                "label": f"tab:{name.replace('_', '-')}",
            }
        )

    for name in [
        "processChange_median_absChange",
        "processChange_mean_absChange",
        "processChange_median_relChange",
        "processChange_mean_relChange",
        "processChange_median_invRelChange",
        "processChange_mean_invRelChange",
        "processChange_median_cohensD",
        "processChange_mean_cohensD",
    ]:
        specs.append(
            {
                "name": name,
                "seed_df": process_seed,
                "group_keys": [
                    "Metric",
                    CHANGE_MAGNITUDE_COL,
                    EDIT_OPERATIONS_COL,
                    MODEL_COMPLEXITY_COL,
                    NOISE_LEVEL_COL,
                    WINDOW_SIZE_COL,
                ],
                "column_levels": [
                    CHANGE_MAGNITUDE_COL,
                    EDIT_OPERATIONS_COL,
                    MODEL_COMPLEXITY_COL,
                    NOISE_LEVEL_COL,
                    WINDOW_SIZE_COL,
                ],
                "order_map": order_map_process,
                "caption": f"{name} aggregated across seeds.",
                "label": f"tab:{name.replace('_', '-')}",
            }
        )

    total_specs = len(specs)
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{total_specs}] Aggregating {spec['name']}...")
        value_wide, count_wide = _aggregate_measure(
            df_seed=spec["seed_df"],
            measure_col=spec["name"],
            group_keys_without_seed=spec["group_keys"],
            column_levels=spec["column_levels"],
            order_map=spec["order_map"],
            across_seed_aggregation=across_seed_aggregation,
        )
        _save_bundle(
            stem=spec["name"],
            value_wide=value_wide,
            count_wide=count_wide,
            caption=spec["caption"],
            label=spec["label"],
        )

    grouped_views = [
        {
            "stem_suffix": "byChangeOperation",
            "group_col": EDIT_OPERATIONS_COL,
            "order": CHANGE_OPERATION_ORDER,
            "filter_col": None,
            "filter_val": None,
            "caption_suffix": "by change operation.",
            "label_suffix": "by-changeoperation",
        },
        {
            "stem_suffix": "byNoise",
            "group_col": NOISE_LEVEL_COL,
            "order": noise_order,
            "filter_col": EDIT_OPERATIONS_COL,
            "filter_val": "mixed",
            "caption_suffix": "by noise (Edit Operations=mixed).",
            "label_suffix": "by-noise",
        },
        {
            "stem_suffix": "byEvolutionProportion",
            "group_col": CHANGE_MAGNITUDE_COL,
            "order": None,
            "filter_col": EDIT_OPERATIONS_COL,
            "filter_val": "mixed",
            "caption_suffix": "by evolution proportion (Edit Operations=mixed).",
            "label_suffix": "by-evolutionproportion",
        },
    ]
    grouped_metrics = [
        {
            "measure_col": "processChange_mean_cohensD",
            "stem_prefix": "processChange_mean_cohensD",
            "caption_prefix": "processChange_mean_cohensD",
            "label_prefix": "processchange-mean-cohensd",
        },
        {
            "measure_col": "processChange_median_cohensD",
            "stem_prefix": "processChange_median_cohensD",
            "caption_prefix": "processChange_median_cohensD",
            "label_prefix": "processchange-median-cohensd",
        },
    ]
    total_extra = len(grouped_views) * len(grouped_metrics)
    extra_i = 0
    for metric_spec in grouped_metrics:
        for view in grouped_views:
            extra_i += 1
            stem = f"{metric_spec['stem_prefix']}_{view['stem_suffix']}"
            print(f"[extra {extra_i}/{total_extra}] Aggregating {stem}...")
            value_wide, count_wide = _aggregate_grouped_single_level(
                df_seed=process_seed,
                measure_col=metric_spec["measure_col"],
                group_col=view["group_col"],
                column_order=view["order"],
                filter_col=view["filter_col"],
                filter_val=view["filter_val"],
                across_seed_aggregation=across_seed_aggregation,
            )
            _save_bundle(
                stem=stem,
                value_wide=value_wide,
                count_wide=count_wide,
                caption=f"{metric_spec['caption_prefix']} {view['caption_suffix']}",
                label=f"tab:{metric_spec['label_prefix']}-{view['label_suffix']}",
            )

    print("\nAnalysis complete.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified signal/noise output generation.")
    parser.add_argument(
        "--across-seed-aggregation",
        choices=["mean", "median"],
        default="median",
        help="Final cross-seed aggregation function.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode: exactly 2 seeds + one metric per dimension.",
    )
    parser.add_argument(
        "--only-median-overview",
        action="store_true",
        help="Generate only median_across_all_obs (+counts) and median_scatter outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        run(
            across_seed_aggregation=args.across_seed_aggregation,
            test_mode=args.test,
            test_seeds=DEFAULT_TEST_SEEDS,
            only_median_overview=args.only_median_overview,
        )
    if caught_warnings:
        print(f"\n  {len(caught_warnings)} runtime warning(s) captured.")
