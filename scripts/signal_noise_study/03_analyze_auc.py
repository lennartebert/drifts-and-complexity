"""Compute AUC as a measure of how well complexity scores separate
pre-drift (no-change) and post-drift (change) windows.

Procedure:
1. Use the complexity score as the detection statistic S(W).
2. Treat pre-drift windows as negative class (label = 0),
   post-drift windows as positive class (label = 1).
3. Compute the ROC curve by sweeping a threshold over all unique score values.
4. Compute AUC as the area under the ROC curve.

Implementation notes:
- If either class is empty, return NaN.
- Drop rows with missing scores.
- AUC values lie in [0.5, 1.0], where 0.5 indicates no separability
  and 1.0 indicates perfect separability (direction-invariant via max(AUC, 1-AUC)).

Reuses functions and constants from 02_analyze_complexity_values.py via importlib.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Import script 02 (numeric prefix prevents normal import)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
_script_02 = importlib.import_module("02_analyze_complexity_values")

# ---------------------------------------------------------------------------
# Reused constants from script 02
# ---------------------------------------------------------------------------
PATH_GEN_INFO: str = _script_02.PATH_GEN_INFO
DIR_CSV: str = _script_02.DIR_CSV
DIR_LATEX: str = _script_02.DIR_LATEX

SPLIT_NAME_COL: str = _script_02.SPLIT_NAME_COL
LOG_ID_COL: str = _script_02.LOG_ID_COL
WINDOW_SIZE_COL: str = _script_02.WINDOW_SIZE_COL
SEED_COL: str = _script_02.SEED_COL
MODEL_COMPLEXITY_COL: str = _script_02.MODEL_COMPLEXITY_COL
NOISE_LEVEL_COL: str = _script_02.NOISE_LEVEL_COL
CHANGE_MAGNITUDE_COL: str = _script_02.CHANGE_MAGNITUDE_COL
EDIT_OPERATIONS_COL: str = _script_02.EDIT_OPERATIONS_COL
CHANGE_OPERATION_ORDER: list[str] = _script_02.CHANGE_OPERATION_ORDER

# Reused functions from script 02
load_generation_info = _script_02.load_generation_info
normalize_split_name = _script_02.normalize_split_name
enrich_with_experimental_factors = _script_02.enrich_with_experimental_factors
_aggregate_by_group = _script_02._aggregate_by_group
_write_latex_table = _script_02._write_latex_table
_add_dimension_and_sort_long = _script_02._add_dimension_and_sort_long

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATH_PER_SAMPLE = "results/signal_noise_study/per_sample_metrics.csv"

# Output paths (parallel to SNR outputs)
PATH_AUC_LONG = f"{DIR_CSV}/auc_long.csv"
PATH_AUC_BY_OP = f"{DIR_CSV}/auc_by_operation.csv"
PATH_AUC_BY_EVOL = f"{DIR_CSV}/auc_by_evolution_proportion.csv"
PATH_AUC_BY_NOISE = f"{DIR_CSV}/auc_by_noise.csv"
DIR_PLOTS = "results/signal_noise_study/plots"

# Column name for the per-log AUC value (parallel to "SNR Seed" in script 02)
AUC_VALUE_COL = "AUC Seed"

# 10 randomly selected (log_id, window_size) combos for example ROC plots
# (seed=42 from aggregate_analysis.csv, frozen here for reproducibility)
ROC_PLOT_EXAMPLES: list[dict[str, object]] = [
    dict(log_id="log_1032_1770121643.xes.gz", window_size=100),
    dict(log_id="log_1033_1770121671.xes.gz", window_size=50),
    dict(log_id="log_1768_1770122034.xes.gz", window_size=50),
    dict(log_id="log_2372_1770121976.xes.gz", window_size=150),
    dict(log_id="log_2588_1770121942.xes.gz", window_size=50),
    dict(log_id="log_2958_1770122591.xes.gz", window_size=50),
    dict(log_id="log_3197_1770122531.xes.gz", window_size=50),
    dict(log_id="log_366_1770121509.xes.gz", window_size=150),
    dict(log_id="log_3914_1770122935.xes.gz", window_size=150),
    dict(log_id="log_4778_1770123053.xes.gz", window_size=200),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_per_sample_metrics(path: str) -> pd.DataFrame:
    """Load and validate the per-sample metrics CSV.

    Renames raw columns to match the naming conventions used in script 02.

    Parameters
    ----------
    path
        Path to per_sample_metrics.csv.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with renamed columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(path, low_memory=False)
    required = ["Metric", "Value", "split_name", "log_id", "window_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df.rename(
        columns={
            "split_name": SPLIT_NAME_COL,
            "log_id": LOG_ID_COL,
            "window_size": WINDOW_SIZE_COL,
        }
    )
    return df


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------
def compute_auc_per_log(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AUC for each (Metric, Log ID, Window Size) group.

    Labels pre_drift windows as 0 (negative) and post_drift windows as 1
    (positive). Uses the ``Value`` column as the detection statistic.
    Returns ``max(AUC, 1 - AUC)`` so values lie in [0.5, 1.0]
    (direction-invariant separability).

    Parameters
    ----------
    df
        Per-sample DataFrame with columns ``Metric``, ``LOG_ID_COL``,
        ``WINDOW_SIZE_COL``, ``SPLIT_NAME_COL``, and ``Value``.

    Returns
    -------
    pd.DataFrame
        One row per (Metric, Log ID, Window Size) with column ``AUC Seed``.
    """
    records: list[dict] = []
    group_cols = ["Metric", LOG_ID_COL, WINDOW_SIZE_COL]

    for keys, group in df.groupby(group_cols, sort=False):
        metric, log_id, ws = keys

        # Drop rows with missing scores
        sub = group.dropna(subset=["Value"])
        if len(sub) == 0:
            auc = np.nan
        else:
            labels = (sub[SPLIT_NAME_COL] == "post_drift").astype(int)
            scores = sub["Value"].values

            # Need both classes present
            if labels.nunique() < 2:
                auc = np.nan
            else:
                raw_auc = roc_auc_score(labels, scores)
                auc = max(raw_auc, 1.0 - raw_auc)

        records.append(
            {
                "Metric": metric,
                LOG_ID_COL: log_id,
                WINDOW_SIZE_COL: ws,
                AUC_VALUE_COL: auc,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation helpers (thin wrappers around script 02's _aggregate_by_group)
# ---------------------------------------------------------------------------
def aggregate_auc_by_operation(df_auc_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AUC by metric and change operation.

    Parameters
    ----------
    df_auc_log
        Per-log AUC DataFrame with experimental factors.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric x change_operation.
    """
    return _aggregate_by_group(
        df_auc_log,
        AUC_VALUE_COL,
        EDIT_OPERATIONS_COL,
        CHANGE_OPERATION_ORDER,
    )


def aggregate_auc_by_evolution_proportion(
    df_auc_log: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate AUC by metric and evolution proportion (Edit Operations=mixed).

    Parameters
    ----------
    df_auc_log
        Per-log AUC DataFrame with experimental factors.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric x evolution_proportion.
    """
    return _aggregate_by_group(
        df_auc_log,
        AUC_VALUE_COL,
        CHANGE_MAGNITUDE_COL,
        None,
        filter_col=EDIT_OPERATIONS_COL,
        filter_val="mixed",
    )


def aggregate_auc_by_noise(df_auc_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AUC by metric and noise level (Edit Operations=mixed).

    Parameters
    ----------
    df_auc_log
        Per-log AUC DataFrame with experimental factors.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with metric x noise level.
    """
    noise_order = ["None", "Low", "High"]
    return _aggregate_by_group(
        df_auc_log,
        AUC_VALUE_COL,
        NOISE_LEVEL_COL,
        noise_order,
        filter_col=EDIT_OPERATIONS_COL,
        filter_val="mixed",
    )


# ---------------------------------------------------------------------------
# ROC example plots
# ---------------------------------------------------------------------------
def plot_example_roc_curves(
    df: pd.DataFrame,
    examples: list[dict[str, object]],
    output_dir: str,
) -> None:
    """Plot example ROC curves for selected (log_id, window_size) combos.

    For each example, creates one figure with a grid of subplots (one per
    metric). Each subplot shows the ROC curve, the diagonal chance line, and
    the AUC value in the legend.

    Parameters
    ----------
    df
        Per-sample DataFrame (after enrichment with experimental factors and
        split-name normalization) containing ``Metric``, ``Value``,
        ``SPLIT_NAME_COL``, ``LOG_ID_COL``, ``WINDOW_SIZE_COL``.
    examples
        List of dicts with ``log_id`` and ``window_size`` keys.
    output_dir
        Directory where PDFs are saved.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics = sorted(df["Metric"].unique())
    n_metrics = len(metrics)
    if n_metrics == 0:
        print("  No metrics found for ROC plots.")
        return

    n_cols = min(4, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    for ex in examples:
        log_id = ex["log_id"]
        ws = ex["window_size"]

        mask = (df[LOG_ID_COL] == log_id) & (df[WINDOW_SIZE_COL] == ws)
        df_ex = df[mask]

        if len(df_ex) == 0:
            print(f"  Skipping ROC plot for {log_id} ws={ws}: no data.")
            continue

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes_flat = np.array(axes).flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            sub = df_ex[df_ex["Metric"] == metric].dropna(subset=["Value"])
            labels = (sub[SPLIT_NAME_COL] == "post_drift").astype(int)
            scores = sub["Value"].values

            if len(sub) == 0 or labels.nunique() < 2:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(metric, fontsize=8)
                continue

            fpr, tpr, _ = roc_curve(labels, scores)
            raw_auc = roc_auc_score(labels, scores)
            auc_val = max(raw_auc, 1.0 - raw_auc)

            # If raw AUC < 0.5, flip the curve for display
            if raw_auc < 0.5:
                fpr, tpr, _ = roc_curve(labels, -scores)

            ax.plot(fpr, tpr, linewidth=1.2, label=f"AUC = {auc_val:.3f}")
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel("FPR", fontsize=7)
            ax.set_ylabel("TPR", fontsize=7)
            ax.set_title(metric, fontsize=8)
            ax.legend(fontsize=7, loc="lower right")
            ax.tick_params(labelsize=6)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Sanitize log_id for filename (remove .xes.gz)
        log_stem = log_id.replace(".xes.gz", "").replace(".xes", "")
        fig.suptitle(f"ROC Curves: {log_id}  (ws={ws})", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fname = out_path / f"roc_{log_stem}_ws{ws}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved ROC plot: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Main entry point for the AUC analysis."""
    print("=" * 60)
    print("AUC SEPARABILITY ANALYSIS")
    print("=" * 60)

    # --- Load data ---
    print("Loading per-sample metrics...")
    df = load_per_sample_metrics(PATH_PER_SAMPLE)
    print(f"  Loaded {len(df)} rows, {df['Metric'].nunique()} metrics.")

    print("Normalizing split names...")
    df = normalize_split_name(df)

    print("Loading generation info...")
    gen_info = load_generation_info(PATH_GEN_INFO)

    print("Enriching with experimental factors...")
    df_enriched = enrich_with_experimental_factors(df, gen_info)

    # --- Compute AUC per log ---
    print("Computing AUC per (Metric, Log ID, Window Size)...")
    df_auc_log = compute_auc_per_log(df_enriched)
    n_nan = df_auc_log[AUC_VALUE_COL].isna().sum()
    print(
        f"  Computed {len(df_auc_log)} AUC values "
        f"({n_nan} NaN due to missing classes)."
    )

    # --- Re-merge experimental factors ---
    print("Re-merging experimental factors...")
    factor_cols = [
        LOG_ID_COL,
        SEED_COL,
        MODEL_COMPLEXITY_COL,
        NOISE_LEVEL_COL,
        CHANGE_MAGNITUDE_COL,
        EDIT_OPERATIONS_COL,
    ]
    factors = df_enriched[factor_cols].drop_duplicates()
    df_auc_log = df_auc_log.merge(factors, on=LOG_ID_COL, how="left")

    # --- Aggregate and save ---
    Path(DIR_CSV).mkdir(parents=True, exist_ok=True)
    Path(DIR_LATEX).mkdir(parents=True, exist_ok=True)

    # Long format (sorted by dimension)
    print("Saving AUC long format...")
    df_auc_long = _add_dimension_and_sort_long(df_auc_log)
    df_auc_long.to_csv(PATH_AUC_LONG, index=False)
    print(f"  Saved to {PATH_AUC_LONG}")

    # By operation
    print("Aggregating AUC by operation...")
    df_auc_by_op = aggregate_auc_by_operation(df_auc_log)
    df_auc_by_op.to_csv(PATH_AUC_BY_OP)
    print(f"  Saved to {PATH_AUC_BY_OP}")

    # By evolution proportion (mixed only)
    print("Aggregating AUC by evolution proportion (Edit Operations=mixed)...")
    df_auc_by_evol = aggregate_auc_by_evolution_proportion(df_auc_log)
    df_auc_by_evol.to_csv(PATH_AUC_BY_EVOL)
    print(f"  Saved to {PATH_AUC_BY_EVOL}")

    # By noise (mixed only)
    print("Aggregating AUC by noise (Edit Operations=mixed)...")
    df_auc_by_noise = aggregate_auc_by_noise(df_auc_log)
    df_auc_by_noise.to_csv(PATH_AUC_BY_NOISE)
    print(f"  Saved to {PATH_AUC_BY_NOISE}")

    # --- LaTeX tables ---
    print("Writing LaTeX tables...")
    _AUC_LATEX_CONFIGS = [
        (
            "auc_by_operation",
            df_auc_by_op,
            "AUC by change operation.",
            "tab:auc-by-operation",
        ),
        (
            "auc_by_evolution_proportion",
            df_auc_by_evol,
            "AUC by evolution proportion (Edit Operations=mixed).",
            "tab:auc-by-evolution",
        ),
        (
            "auc_by_noise",
            df_auc_by_noise,
            "AUC by noise level (Edit Operations=mixed).",
            "tab:auc-by-noise",
        ),
    ]
    for stem, frame, caption, label in _AUC_LATEX_CONFIGS:
        if len(frame) == 0:
            print(f"  Skipping LaTeX for {stem}: empty DataFrame.")
            continue
        _write_latex_table(
            frame,
            stem,
            caption=caption,
            label=label,
            index=True,
            heatmap=False,
            rank_highlight=True,
        )
        print(f"  Wrote LaTeX: {DIR_LATEX}/{stem}.tex")

    # --- Example ROC plots ---
    print("Plotting example ROC curves...")
    plot_example_roc_curves(df_enriched, ROC_PLOT_EXAMPLES, DIR_PLOTS)

    print("\nAUC analysis complete!")


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        main()
    if caught_warnings:
        print(
            f"\n  {len(caught_warnings)} runtime warning(s) "
            f"(e.g. invalid value in subtract)."
        )
