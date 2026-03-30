"""Line plots: complexity measures vs prefix window size (growing-prefix study)."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

# Non-interactive backend before pyplot (matches utils/plotting/complexity.py)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

plt.ioff()


def _parse_measure_column(col: str) -> Tuple[Optional[str], str]:
    """Split ``measure_*`` column into optional adapter key and display metric name."""
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


def plot_prefix_growth_complexity(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dataset_key: str = "",
    fig_format: str = "png",
    y_log: bool = False,
) -> None:
    """One figure per logical metric: x = window ``size``, y = measure value.

    When columns use ``measure_<adapter>::<Metric Name>`` (``include_adapter_name=True``),
    draws one line per adapter for the same metric. Otherwise one line per ``measure_*`` column.

    Args:
        df: Materialized complexity DataFrame (must include ``size``).
        out_dir: Directory to create (e.g. ``.../BPIC12/plots``).
        dataset_key: Optional label for figure title.
        fig_format: Image extension (default png).
        y_log: If True, use log scale on y when all positive values.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "size" not in df.columns:
        raise ValueError("DataFrame must include a 'size' column for prefix-growth plots.")

    measure_cols = [c for c in df.columns if c.startswith("measure_")]
    if not measure_cols:
        print("    [prefix_growth] No measure_* columns; skipping plots.")
        return

    # metric_name -> list of (column_name, legend_label)
    grouped: DefaultDict[str, List[Tuple[str, str]]] = defaultdict(list)
    for col in measure_cols:
        adapter, metric_name = _parse_measure_column(col)
        label = adapter if adapter else col
        grouped[metric_name].append((col, label))

    for metric_name, series_list in grouped.items():
        fig, ax = plt.subplots(figsize=(12, 5))
        x_base = pd.to_numeric(df["size"], errors="coerce")

        for col, legend_label in sorted(series_list, key=lambda t: t[1]):
            y = pd.to_numeric(df[col], errors="coerce")
            mask = x_base.notna() & y.notna()
            if not mask.any():
                continue
            ax.plot(
                x_base[mask].values,
                y[mask].values,
                marker="o",
                linewidth=1.5,
                label=legend_label,
            )

        ax.set_xlabel("Window size (traces)")
        ax.set_ylabel(metric_name)
        title_parts = []
        if dataset_key:
            title_parts.append(dataset_key)
        title_parts.append(metric_name)
        ax.set_title(" — ".join(title_parts))
        if len(series_list) > 1:
            ax.legend(loc="best", fontsize=9)

        if y_log:
            y_vals = []
            for col, _ in series_list:
                y_vals.extend(
                    pd.to_numeric(df[col], errors="coerce").dropna().tolist()
                )
            if y_vals and min(y_vals) > 0:
                ax.set_yscale("log")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = _safe_filename_fragment(metric_name) + f".{fig_format}"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    Prefix-growth plots written to {out_dir}")


def plot_prefix_growth_dataset_comparison(
    data_by_dataset: Dict[str, pd.DataFrame],
    out_dir: Path,
    *,
    fig_format: str = "png",
    y_log: bool = False,
) -> None:
    """One figure per metric column with one line per dataset.

    Args:
        data_by_dataset: Mapping of dataset key -> complexity DataFrame.
        out_dir: Target output directory for comparison figures.
        fig_format: Image extension (default png).
        y_log: If True, use log scale when all plotted values are positive.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_by_dataset:
        raise ValueError("No dataset data provided for comparison plotting.")

    # Group all measure columns by display metric name.
    metric_to_columns: DefaultDict[str, List[str]] = defaultdict(list)
    for df in data_by_dataset.values():
        for col in df.columns:
            if not col.startswith("measure_"):
                continue
            _, metric_name = _parse_measure_column(col)
            if col not in metric_to_columns[metric_name]:
                metric_to_columns[metric_name].append(col)

    if not metric_to_columns:
        raise ValueError("No measure_* columns found in provided complexity CSV data.")

    for metric_name in sorted(metric_to_columns.keys()):
        fig, ax = plt.subplots(figsize=(12, 5))
        plotted_series = 0
        y_vals_all: List[float] = []
        columns_for_metric = sorted(metric_to_columns[metric_name])
        has_multiple_metric_columns = len(columns_for_metric) > 1

        for mcol in columns_for_metric:
            adapter, _ = _parse_measure_column(mcol)
            for dataset_key in sorted(data_by_dataset.keys()):
                df = data_by_dataset[dataset_key]
                if "size" not in df.columns or mcol not in df.columns:
                    continue
                x = pd.to_numeric(df["size"], errors="coerce")
                y = pd.to_numeric(df[mcol], errors="coerce")
                mask = x.notna() & y.notna()
                if not mask.any():
                    continue
                plotted_series += 1
                y_vals_all.extend(y[mask].tolist())
                legend_label = dataset_key
                if has_multiple_metric_columns and adapter:
                    legend_label = f"{dataset_key} ({adapter})"
                ax.plot(
                    x[mask].values,
                    y[mask].values,
                    marker="o",
                    linewidth=1.5,
                    label=legend_label,
                )

        if plotted_series == 0:
            print(
                f"    [prefix_growth comparison] Skipping {metric_name}: no dataset had plottable values."
            )
            plt.close(fig)
            continue
        if plotted_series < len(data_by_dataset):
            print(
                f"    [prefix_growth comparison] Warning: {metric_name} has "
                "partial dataset coverage."
            )

        ax.set_xlabel("Window size (traces)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Dataset comparison — {metric_name}")
        if plotted_series > 1:
            ax.legend(loc="best", fontsize=9)
        if y_log and y_vals_all and min(y_vals_all) > 0:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = _safe_filename_fragment(metric_name) + f".{fig_format}"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    Prefix-growth dataset comparison plots written to {out_dir}")
