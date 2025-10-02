#!/usr/bin/env python3
"""
Compare dependent (overlapping) correlations across experiments using Williams/Steiger test.

We compare r(X, Y_a) vs r(X, Y_b) where X = 'sample_size' and Y_* are the same
measure computed under two different experiments on the *same samples*.
Requires:
  - numpy, pandas, scipy  (already in your environment)

Folder layout (example):
root/
├── synthetic_base/
│   ├── O2C_S/measures.csv
│   ├── LOAN_S/measures.csv
│   ...
├── synthetic_normalized/
│   ├── O2C_S/measures.csv
│   ...

Usage (examples):
python compare_dependent_correlations.py \
  --experiments synthetic_base synthetic_normalized synthetic_normalized_and_population \
  --pairs synthetic_base:synthetic_normalized synthetic_normalized:synthetic_normalized_and_population synthetic_base:synthetic_normalized_and_population \
  --out results_williams.csv

Optionally restrict to measures:
  --measures "Sequence Entropy" "Normalized Sequence Entropy" "Variant Entropy"

References:
- Steiger, J. H. (1980). Tests for comparing elements of a correlation matrix. Psychological Bulletin, 87, 245–251.
- Counsell, A. & Cribbie, R. A. (2015). Equivalence tests for comparing correlation and regression coefficients.
  (t-formula for Williams’ test with df = n-3; |R| term = 1 - r12^2 - r13^2 - r23^2 + 2 r12 r13 r23)
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon

from utils import constants

#  run with
# python -m scripts.bias_study.compare_results --experiments synthetic_base synthetic_normalized synthetic_normalized_and_population --pairs synthetic_base:synthetic_normalized synthetic_normalized:synthetic_normalized_and_population synthetic_base:synthetic_normalized_and_population --out results_williams.csv


# ----------------------------- Williams test (overlapping) ----------------------------- #
def williams_overlapping_t(
    r12: float, r13: float, r23: float, n: int
) -> Tuple[float, int, float]:
    """
    Williams' t-test for comparing two dependent, overlapping correlations:
      compare r12 vs r13, given their common variable correlation r23, sample size n.

    r12 = corr(X, Y1) from experiment A
    r13 = corr(X, Y2) from experiment B
    r23 = corr(Y1, Y2) across the same samples
    n   = number of paired observations

    Returns (t_stat, df, p_two_sided).

    Formula (δ=0 case) with df = n - 3:
      t = (r12 - r13) * sqrt( (n - 1) * (1 + r23) /
            ( 2 * (n - 1)/(n - 3) * |R| + ((r12 + r13)**2 / 4.0) * (1 - r23)**3 ) )

      where |R| = 1 - r12^2 - r13^2 - r23^2 + 2*r12*r13*r23

    Sources: Steiger (1980); Counsell & Cribbie (2015 §5.3).
    """
    if n < 4:
        return (np.nan, n - 3, np.nan)

    R_det = 1.0 - r12**2 - r13**2 - r23**2 + 2.0 * r12 * r13 * r23
    df = n - 3
    # Guard against tiny negative due to numeric jitter
    R_det = max(R_det, 0.0)

    num = (r12 - r13) * math.sqrt((n - 1.0) * (1.0 + r23))
    den = math.sqrt(
        2.0 * ((n - 1.0) / (n - 3.0)) * R_det
        + ((r12 + r13) ** 2 / 4.0) * (1.0 - r23) ** 3
    )

    if den == 0.0:
        return (np.nan, df, np.nan)

    t_stat = num / den
    p_two = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))
    return (t_stat, df, p_two)


# ----------------------------- FDR (Benjamini–Hochberg) ----------------------------- #
def fdr_bh(pvals: pd.Series, alpha: float = 0.05) -> pd.Series:
    """
    Benjamini–Hochberg FDR correction.
    Returns adjusted p-values in the original index order.
    """
    p = pvals.values.astype(float)
    n = np.sum(~np.isnan(p))
    adj = np.full_like(p, np.nan, dtype=float)
    if n == 0:
        return pd.Series(adj, index=pvals.index)

    # rank non-nan p-values
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p) + 1)

    # BH: p_adj = p * m / rank ; then monotone decreasing when sorted
    p_bh = p * len(p) / ranks
    # enforce monotonicity on sorted values
    p_sorted = p_bh[order]
    for i in range(len(p_sorted) - 2, -1, -1):
        p_sorted[i] = min(p_sorted[i], p_sorted[i + 1])
    adj[order] = np.minimum(p_sorted, 1.0)
    return pd.Series(adj, index=pvals.index)


# ----------------------------- Data loading & comparison ----------------------------- #
EXCLUDE_COLS = {"sample_size", "sample_id"}


def discover_datasets(root: Path, experiments: List[str]) -> List[str]:
    """Datasets are subfolders present in ALL experiment folders (intersect)."""
    sets = None
    for exp in experiments:
        ds = sorted([p.name for p in (root / exp).iterdir() if p.is_dir()])
        sets = set(ds) if sets is None else (sets & set(ds))
    return sorted(list(sets or []))


def load_measures(root: Path, experiment: str, dataset: str) -> pd.DataFrame:
    """Load measures.csv; keep sample_id, sample_size and all other numeric measure columns."""
    path = root / experiment / dataset / "measures.csv"
    df = pd.read_csv(path)
    # standardize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    # Ensure we have the necessary keys
    if "sample_id" not in df.columns:
        # fall back: infer a row index as sample_id if needed
        df = df.reset_index().rename(columns={"index": "sample_id"})
    if "sample_size" not in df.columns:
        raise ValueError(f"'sample_size' column missing in {path}")
    return df


def get_measure_columns(
    df: pd.DataFrame, user_measures: Optional[List[str]]
) -> List[str]:
    if user_measures:
        missing = [m for m in user_measures if m not in df.columns]
        if missing:
            raise ValueError(f"Requested measures not found: {missing}")
        return user_measures
    # auto: all numeric columns excluding keys
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDE_COLS]


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r with NaN-safe fallback; returns np.nan if <2 valid points or zero variance."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    if np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return float("nan")
    result = stats.pearsonr(a[mask], b[mask])
    return float(result.statistic)


def compare_pair(
    df_a: pd.DataFrame, df_b: pd.DataFrame, measure: str
) -> Dict[str, float]:
    """
    Align by sample_id; compute r_a = corr(sample_size, measure_a),
    r_b = corr(sample_size, measure_b), r_ab = corr(measure_a, measure_b),
    and run Williams test.
    """
    # inner-join on sample_id; also keep sample_size from both to sanity-check equal
    merged = (
        df_a[["sample_id", "sample_size", measure]]
        .rename(columns={measure: f"{measure}__A", "sample_size": "sample_size_A"})
        .merge(
            df_b[["sample_id", "sample_size", measure]].rename(
                columns={measure: f"{measure}__B", "sample_size": "sample_size_B"}
            ),
            on="sample_id",
            how="inner",
        )
    )

    # If sample_size differs across A/B (shouldn't), take the mean; but warn via variance check
    if not np.allclose(
        merged["sample_size_A"], merged["sample_size_B"], equal_nan=False
    ):
        # Average them (rare), but they should be identical in your setup
        merged["sample_size"] = merged[["sample_size_A", "sample_size_B"]].mean(axis=1)
    else:
        merged["sample_size"] = merged["sample_size_A"]

    n = int(merged.shape[0])

    x = merged["sample_size"].to_numpy(dtype=float)
    y_a = merged[f"{measure}__A"].to_numpy(dtype=float)
    y_b = merged[f"{measure}__B"].to_numpy(dtype=float)

    r_a = pearson_safe(x, y_a)
    r_b = pearson_safe(x, y_b)
    r_ab = pearson_safe(y_a, y_b)

    t_stat, df, p_two = williams_overlapping_t(r_a, r_b, r_ab, n)

    return {
        "n": n,
        "r_a": r_a,
        "r_b": r_b,
        "r_ab": r_ab,
        "t_stat": t_stat,
        "df": df,
        "p_two": p_two,
        "delta_r": (r_b - r_a) if (not np.isnan(r_a) and not np.isnan(r_b)) else np.nan,
        "improved_abs_r": (
            (abs(r_b) < abs(r_a)) if (np.isfinite(r_a) and np.isfinite(r_b)) else np.nan
        ),
    }


# ----------------------------- Measure-level summary (aggregated across datasets) ----------------------------- #
def measure_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate improvements per measure across datasets.
    For each measure × (exp_a, exp_b) comparison:
      - compute mean/median delta_r
      - count how many datasets improved (r_b > r_a)
      - Wilcoxon signed-rank test of delta_r (H0: median delta_r = 0)
    Returns tidy DataFrame.
    """
    results = []
    grouped = df.groupby(["measure", "exp_a", "exp_b"])
    for (measure, exp_a, exp_b), sub in grouped:
        deltas = sub["delta_r"].dropna().values
        improved = np.sum(sub["r_b"] > sub["r_a"])
        total = sub.shape[0]

        # Wilcoxon signed-rank test (two-sided), only if ≥2 datasets
        if len(deltas) >= 2 and np.any(deltas != 0):
            try:
                stat, pval = wilcoxon(deltas, alternative="greater")
            except ValueError:
                pval = np.nan
        else:
            pval = np.nan

        results.append(
            {
                "measure": measure,
                "exp_a": exp_a,
                "exp_b": exp_b,
                "mean_delta_r": np.nanmean(deltas) if len(deltas) else np.nan,
                "median_delta_r": np.nanmedian(deltas) if len(deltas) else np.nan,
                "improved_count": int(improved),
                "total_datasets": int(total),
                "p_wilcoxon": pval,
            }
        )

    res_df = pd.DataFrame(results)
    # Apply FDR correction across all rows
    res_df["p_adj_bh"] = fdr_bh(res_df["p_wilcoxon"])
    res_df["significant_FDR"] = res_df["p_adj_bh"] <= 0.05
    return res_df


# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare dependent correlations across experiments (Williams/Steiger)."
    )
    ap.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="Experiment folder names to include (e.g., synthetic_base synthetic_normalized ...)",
    )
    ap.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        required=True,
        help="Pairs to compare, format EXP_A:EXP_B (must be in --experiments). "
        "Example: synthetic_base:synthetic_normalized",
    )
    ap.add_argument(
        "--measures",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of measure column names to analyze. If omitted, auto-detect numeric measures.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of dataset names (subfolders). If omitted, intersect across experiments.",
    )
    ap.add_argument(
        "--out", type=str, default="williams_results.csv", help="Output CSV file path"
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for FDR reporting (not used in computation)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = constants.BIAS_STUDY_RESULTS_DIR
    exps = args.experiments
    pair_specs = [tuple(p.split(":")) for p in args.pairs]
    for a, b in pair_specs:
        if a not in exps or b not in exps:
            raise ValueError(f"Pair {a}:{b} not in provided --experiments {exps}")

    # datasets
    datasets = args.datasets or discover_datasets(root, exps)
    if not datasets:
        raise ValueError("No datasets found in the intersection of experiment folders.")
    print(f"Datasets discovered: {datasets}")

    # Preload one measures.csv to get measure columns (if not provided)
    probe_df = load_measures(root, exps[0], datasets[0])
    measures = get_measure_columns(probe_df, args.measures)
    print(f"Analyzing {len(measures)} measures: {measures}")

    rows = []
    # Cache loaded frames to avoid re-reading
    cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    for ds in datasets:
        for exp in exps:
            cache[(exp, ds)] = load_measures(root, exp, ds)

        for exp_a, exp_b in pair_specs:
            df_a = cache[(exp_a, ds)]
            df_b = cache[(exp_b, ds)]

            # ensure requested measures exist in both
            for m in measures:
                if m not in df_a.columns or m not in df_b.columns:
                    continue  # skip silently if a measure is missing in one exp

                res = compare_pair(df_a, df_b, m)
                rows.append(
                    {"dataset": ds, "measure": m, "exp_a": exp_a, "exp_b": exp_b, **res}
                )

    out_df = pd.DataFrame(rows)
    # FDR per (pair) across all datasets x measures, or across all rows overall; here: overall
    out_df["p_adj_bh"] = fdr_bh(out_df["p_two"])

    # Convenience flags
    out_df["significant_FDR"] = out_df["p_adj_bh"] <= args.alpha
    out_df["direction"] = np.where(
        out_df["delta_r"].gt(0),
        "r_b > r_a",
        np.where(out_df["delta_r"].lt(0), "r_b < r_a", "no change"),
    )

    # Sort for readability
    out_df = out_df.sort_values(
        by=["dataset", "measure", "exp_a", "exp_b"]
    ).reset_index(drop=True)

    # Save
    out_path = Path(args.out).resolve()
    out_df.to_csv(out_path, index=False)
    print(f"Wrote results to: {out_path}")

    # Quick console preview
    preview_cols = [
        "dataset",
        "measure",
        "exp_a",
        "exp_b",
        "n",
        "r_a",
        "r_b",
        "delta_r",
        "t_stat",
        "df",
        "p_two",
        "p_adj_bh",
        "significant_FDR",
        "improved_abs_r",
        "direction",
    ]
    with pd.option_context(
        "display.max_rows", 50, "display.width", 180, "display.precision", 6
    ):
        print(out_df[preview_cols].head(30))

    # Generate and save measure-level summary
    summary_df = measure_level_summary(out_df)
    summary_df.to_csv("measure_level_summary.csv", index=False)

    print("Saved measure-level summary to measure_level_summary.csv")
    with pd.option_context(
        "display.max_rows", 30, "display.width", 160, "display.precision", 4
    ):
        print(summary_df.head(20))


if __name__ == "__main__":
    main()
