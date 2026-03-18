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
    d = d.sort_values(["_dim_ord", "Metric"]).drop(columns=["_dim_ord"])
    cols = ["Dimension", "Metric"] + [c for c in d.columns if c not in {"Dimension", "Metric"}]
    return d[cols]


def _add_dimension_and_sort_wide(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dimension = d.index.map(lambda m: METRIC_DIMENSION_MAP.get(m, "Other"))
    d.index = pd.MultiIndex.from_arrays([dimension, d.index], names=["Dimension", "Metric"])
    dim_order = list(DIMENSIONS_ORDER) + ["Other"]
    desired = []
    for dim in dim_order:
        in_dim = sorted([idx for idx in d.index if idx[0] == dim], key=lambda x: x[1])
        desired.extend(in_dim)
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
    value_wide = long.pivot_table(index=idx, columns=column_levels, values="Value", aggfunc="first")
    count_wide = long.pivot_table(index=idx, columns=column_levels, values="N Used", aggfunc="first").astype("Int64")
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


def run(
    *,
    across_seed_aggregation: Literal["mean", "median"] = "median",
    test_mode: bool = False,
    test_seeds: list[int] | None = None,
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
                "seed_df": setting_seed,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        run(
            across_seed_aggregation=args.across_seed_aggregation,
            test_mode=args.test,
            test_seeds=DEFAULT_TEST_SEEDS,
        )
    if caught_warnings:
        print(f"\n  {len(caught_warnings)} runtime warning(s) captured.")
