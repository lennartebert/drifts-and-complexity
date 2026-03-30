"""Plotting helpers for intralog size-vs-complexity correlation analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.ioff()


def _parse_measure_column(col: str) -> Tuple[Optional[str], str]:
    if not col.startswith("measure_"):
        raise ValueError(f"Expected measure_* column, got {col!r}")
    rest = col[len("measure_") :]
    if "::" in rest:
        adapter, metric = rest.split("::", 1)
        return adapter, metric
    return None, rest


def _safe_filename_fragment(name: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip(), flags=re.UNICODE)
    s = s.strip("_") or "metric"
    return s[:max_len]


def plot_intralog_scatter_with_fit(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    fig_format: str = "pdf",
    y_log: bool = False,
) -> Dict[str, float]:
    """Plot one scatter+linear-fit figure per measure column.

    x-axis is window ``size`` (traces in month), y-axis is the complexity measure value.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "size" not in df.columns:
        raise ValueError("DataFrame must include a 'size' column.")

    measure_cols = [c for c in df.columns if c.startswith("measure_")]
    if not measure_cols:
        print("    [intralog_correlation] No measure_* columns; skipping plots.")
        return {}

    slopes: Dict[str, float] = {}

    for col in measure_cols:
        _, metric_name = _parse_measure_column(col)
        x = pd.to_numeric(df["size"], errors="coerce")
        y = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if not mask.any():
            print(f"    [intralog_correlation] Skipping {col}: no valid points.")
            continue

        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(xv, yv, alpha=0.75, s=22, label="windows")

        if len(xv) >= 2:
            slope, intercept = np.polyfit(xv, yv, deg=1)
            slopes[metric_name] = float(slope)
            xfit = np.linspace(float(np.min(xv)), float(np.max(xv)), 200)
            yfit = slope * xfit + intercept
            ax.plot(
                xfit,
                yfit,
                linewidth=1.8,
                linestyle="-",
                label=f"fit (y={slope:.3g}x+{intercept:.3g})",
            )
        else:
            print(
                f"    [intralog_correlation] Warning: {col} has <2 points; skipping linear fit."
            )

        ax.set_xlabel("Window size (=traces in month)")
        ax.set_ylabel(metric_name)

        if y_log and np.all(yv > 0):
            ax.set_yscale("log")

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fname = _safe_filename_fragment(metric_name) + f".{fig_format}"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    Intralog correlation plots written to {out_dir}")
    return slopes


def plot_intralog_normalized_spread_boxplot(
    normalized_values_by_metric: Dict[str, List[float]],
    out_path: Path,
    *,
    fig_format: str = "pdf",
) -> None:
    """Create a boxplot of normalized monthly values (value/mean) per metric."""
    if not normalized_values_by_metric:
        print("    [intralog_correlation] No normalized values for spread boxplot.")
        return

    metrics = [m for m, vals in normalized_values_by_metric.items() if vals]
    if not metrics:
        print("    [intralog_correlation] No non-empty metric distributions for boxplot.")
        return

    values = [normalized_values_by_metric[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 0.45), 6))
    ax.boxplot(values, labels=metrics, showfliers=True)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Monthly value / metric mean")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != f".{fig_format.lower()}":
        out_path = out_path.with_suffix(f".{fig_format}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
