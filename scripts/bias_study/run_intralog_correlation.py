#!/usr/bin/env python
"""Intralog correlation analysis using monthly time windows.

Per dataset, this script computes complexity per monthly window and creates
scatter plots with a linear best-fit line:
    x-axis: window size (number of traces in month)
    y-axis: complexity measure value
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import constants, helpers
from utils.complexity.metrics_adapters.metrics_adapter import get_adapters
from utils.drift_io import load_xes_log
from utils.plotting.intralog_correlation import (
    plot_intralog_normalized_spread_boxplot,
    plot_intralog_scatter_with_fit,
)
from utils.windowing.helpers import split_log_into_fixed_time_windows
from utils.windowing.window import Window

DEFAULT_DATASETS: Sequence[str] = ("BPIC12", "RTFMP")
DEFAULT_SCENARIO = "run_intralog_correlation"
ADAPTER_NAMES = ["local", "vidgof_sample"]
TEST_MODE_INCLUDE_METRICS = ["Avg. Trace Length"]
OUTPUT_FIG_FORMAT = "pdf"
TIME_WINDOW_CONFIG: Dict[str, Any] = {
    "window_size": 1,
    "offset": 1,
    "unit": "month",
    "align_first_window": False,
    "include_incomplete_windows": False,
}


def _parse_measure_column(col: str) -> Tuple[str, str]:
    if not col.startswith("measure_"):
        raise ValueError(f"Expected measure_* column, got {col!r}")
    rest = col[len("measure_") :]
    if "::" in rest:
        adapter, metric = rest.split("::", 1)
        return adapter, metric
    return "", rest


def _metric_sort_key(metric_name: str) -> Tuple[int, int, str]:
    dim = constants.METRIC_DIMENSION_MAP.get(metric_name)
    try:
        dim_idx = constants.DIMENSIONS_ORDER.index(dim) if dim is not None else 10_000
    except ValueError:
        dim_idx = 10_000
    try:
        metric_idx = constants.ALL_METRIC_NAMES.index(metric_name)
    except ValueError:
        metric_idx = 10_000
    return (dim_idx, metric_idx, metric_name)


def _reorder_complexity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Order measure/info columns by global metric ordering config."""
    base_cols = [c for c in df.columns if not (c.startswith("measure_") or c.startswith("info_"))]
    measure_cols = [c for c in df.columns if c.startswith("measure_")]
    info_cols = [c for c in df.columns if c.startswith("info_")]

    def sort_measure(col: str) -> Tuple[Tuple[int, int, str], str]:
        _, metric = _parse_measure_column(col)
        return (_metric_sort_key(metric), col)

    def sort_info(col: str) -> Tuple[Tuple[int, int, str], str]:
        rest = col[len("info_") :]
        metric_guess = rest.split("::", 1)[-1]
        return (_metric_sort_key(metric_guess), col)

    ordered = base_cols + sorted(measure_cols, key=sort_measure) + sorted(
        info_cols, key=sort_info
    )
    return df[ordered]


def _metric_series_from_df(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Get one numeric series per metric name in global metric order."""
    measure_cols = [c for c in df.columns if c.startswith("measure_")]
    metric_to_col: Dict[str, str] = {}
    for col in measure_cols:
        _, metric_name = _parse_measure_column(col)
        if metric_name not in metric_to_col:
            metric_to_col[metric_name] = col
    ordered_metrics = sorted(metric_to_col.keys(), key=_metric_sort_key)
    out: Dict[str, pd.Series] = {}
    for metric_name in ordered_metrics:
        col = metric_to_col[metric_name]
        out[metric_name] = pd.to_numeric(df[col], errors="coerce")
    return out


def _resolve_log_path(dataset_info: dict) -> Path:
    p = Path(dataset_info["path"])
    return p if p.is_absolute() else constants.ROOT / p


def _complexity_csv_path(scenario: str, dataset_key: str) -> Path:
    return constants.BIAS_STUDY_RESULTS_DIR / scenario / dataset_key / "complexity.csv"


def _flatten_adapter_results(
    windows: List[Window],
    per_adapter: List[Tuple[str, Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]]],
    *,
    include_adapter_name: bool = True,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {w.id: {} for w in windows}
    for adapter_name, result in per_adapter:
        for w in windows:
            metrics, info = result[w.id]

            def _k(pfx: str, key: str) -> str:
                if include_adapter_name:
                    return f"{pfx}{adapter_name}::{key}"
                return f"{pfx}{key}"

            out[w.id].update({_k("measure_", k): v for k, v in metrics.items()})
            out[w.id].update({_k("info_", k): v for k, v in info.items()})
    return out


def _compute_complexity_df(
    windows: List[Window],
    adapter_names: Iterable[str],
    *,
    include_metrics: List[str] | None = None,
) -> pd.DataFrame:
    per_adapter: List[Tuple[str, Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]]] = []
    for adapter in get_adapters(list(adapter_names)):
        adapter_results = adapter.compute_measures_for_windows(
            windows, include_metrics=include_metrics
        )
        converted: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        for w in windows:
            store, info = adapter_results.get(w.id, (None, {}))
            metrics = store.to_visible_dict() if store is not None else {}
            converted[w.id] = (metrics, info)
        per_adapter.append((adapter.name, converted))

    merged = _flatten_adapter_results(windows, per_adapter, include_adapter_name=True)
    rows = [{**w.to_dict(), **merged[w.id]} for w in windows]
    return pd.DataFrame(rows)


def _load_existing_complexity_data(
    datasets: List[str], scenario: str
) -> Dict[str, pd.DataFrame]:
    missing: List[Path] = []
    data: Dict[str, pd.DataFrame] = {}
    for dataset_key in datasets:
        p = _complexity_csv_path(scenario, dataset_key)
        if not p.exists():
            missing.append(p)
            continue
        data[dataset_key] = pd.read_csv(p)
    if missing:
        msg = "\n".join(f"  - {str(p)}" for p in missing)
        raise FileNotFoundError(
            "Missing complexity.csv for plot-only mode:\n"
            f"{msg}\nRun without --plot-only first (or generate missing datasets)."
        )
    return data


def _validate_datasets(datasets: List[str]) -> Dict[str, Any]:
    data_dictionary = helpers.load_data_dictionary(
        constants.get_data_dictionary_path(), get_real=True, get_synthetic=False
    )
    missing = [d for d in datasets if d not in data_dictionary]
    if missing:
        print(f"Error: unknown or filtered-out dataset keys: {missing}", file=sys.stderr)
        print(f"Available (real): {sorted(data_dictionary.keys())}", file=sys.stderr)
        sys.exit(1)
    return data_dictionary


def _save_correlations_csv(
    scenario: str, datasets: List[str], slopes_by_dataset: Dict[str, Dict[str, float]]
) -> Path:
    metrics = sorted({m for v in slopes_by_dataset.values() for m in v.keys()}, key=_metric_sort_key)
    rows: List[Dict[str, Any]] = []
    for metric_name in metrics:
        row: Dict[str, Any] = {"metric": metric_name}
        for dataset_key in datasets:
            row[dataset_key] = slopes_by_dataset.get(dataset_key, {}).get(metric_name)
        rows.append(row)

    out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "correlations.csv"
    out_df = pd.DataFrame(rows)
    for dataset_key in datasets:
        if dataset_key in out_df.columns:
            out_df[dataset_key] = out_df[dataset_key].map(
                lambda v: f"{float(v):.10f}" if pd.notna(v) else ""
            )
    out_df.to_csv(out_path, index=False)
    return out_path


def _save_variance_outputs_for_log(
    scenario: str,
    dataset_key: str,
    df: pd.DataFrame,
) -> None:
    """Save variance table and normalized spread boxplot for one log."""
    metric_series = _metric_series_from_df(df)
    rows: List[Dict[str, Any]] = []
    normalized_by_metric: Dict[str, List[float]] = {}

    for metric_name, s in metric_series.items():
        vals = s.dropna().astype(float)
        if vals.empty:
            continue
        mean_val = float(vals.mean())
        std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(
            {
                "dimension": constants.METRIC_DIMENSION_MAP.get(metric_name, "Unknown"),
                "metric": metric_name,
                "mean": mean_val,
                "standard_deviation": std_val,
            }
        )
        # Intentionally always normalize by mean as requested.
        normalized = (vals / mean_val).replace([np.inf, -np.inf], np.nan).dropna()
        normalized_by_metric[metric_name] = normalized.tolist()

    variance_df = pd.DataFrame(rows)
    if not variance_df.empty:
        variance_df["__sort_dim"] = variance_df["dimension"].apply(
            lambda d: constants.DIMENSIONS_ORDER.index(d)
            if d in constants.DIMENSIONS_ORDER
            else 10_000
        )
        variance_df["__sort_metric"] = variance_df["metric"].apply(
            lambda m: constants.ALL_METRIC_NAMES.index(m)
            if m in constants.ALL_METRIC_NAMES
            else 10_000
        )
        variance_df = variance_df.sort_values(
            ["__sort_dim", "__sort_metric", "metric"], kind="stable"
        ).drop(columns=["__sort_dim", "__sort_metric"])

    out_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    variance_path = out_dir / "variance_summary.csv"
    variance_df.to_csv(variance_path, index=False, float_format="%.10f")

    plots_dir = out_dir / "plots"
    plot_intralog_normalized_spread_boxplot(
        normalized_by_metric,
        plots_dir / "normalized_metric_spread_boxplot.pdf",
        fig_format=OUTPUT_FIG_FORMAT,
    )


def run(
    datasets: List[str],
    scenario: str,
    y_log: bool,
    *,
    plot_only: bool = False,
    test_mode: bool = False,
) -> None:
    slopes_by_dataset: Dict[str, Dict[str, float]] = {}

    if plot_only:
        data = _load_existing_complexity_data(datasets, scenario)
        for dataset_key, df in data.items():
            df = _reorder_complexity_columns(df)
            plots_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario / dataset_key / "plots"
            print(f"Plot-only: generating plots for {dataset_key} ...")
            slopes_by_dataset[dataset_key] = plot_intralog_scatter_with_fit(
                df, plots_dir, fig_format=OUTPUT_FIG_FORMAT, y_log=y_log
            )
            _save_variance_outputs_for_log(scenario, dataset_key, df)
        corr_path = _save_correlations_csv(scenario, datasets, slopes_by_dataset)
        print(f"Saved correlations to {corr_path}")
        return

    data_dictionary = _validate_datasets(datasets)
    base_out = constants.BIAS_STUDY_RESULTS_DIR / scenario
    base_out.mkdir(parents=True, exist_ok=True)

    include_metrics = TEST_MODE_INCLUDE_METRICS if test_mode else None
    if test_mode:
        print(f"[TEST MODE] Restricting metrics to: {include_metrics}")

    for dataset_key in datasets:
        info = data_dictionary[dataset_key]
        log_path = _resolve_log_path(info)
        activity_key = info.get("activity_key", "concept:name")
        print(f"Loading {dataset_key} from {log_path} ...")
        traces_sorted = load_xes_log(log_path, activity_key=activity_key)

        print(
            "  Computing monthly windows with config "
            f"{TIME_WINDOW_CONFIG} and adapters {ADAPTER_NAMES} ..."
        )
        windows = split_log_into_fixed_time_windows(
            traces_sorted,
            window_size=TIME_WINDOW_CONFIG["window_size"],
            offset=TIME_WINDOW_CONFIG["offset"],
            unit=TIME_WINDOW_CONFIG["unit"],
            align_first_window=TIME_WINDOW_CONFIG["align_first_window"],
            include_incomplete_windows=TIME_WINDOW_CONFIG["include_incomplete_windows"],
        )
        df = _compute_complexity_df(
            windows,
            ADAPTER_NAMES,
            include_metrics=include_metrics,
        )
        df = _reorder_complexity_columns(df)

        out_dir = base_out / dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "complexity.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path} ({len(df)} rows)")

        plots_dir = out_dir / "plots"
        slopes_by_dataset[dataset_key] = plot_intralog_scatter_with_fit(
            df, plots_dir, fig_format=OUTPUT_FIG_FORMAT, y_log=y_log
        )
        _save_variance_outputs_for_log(scenario, dataset_key, df)

    corr_path = _save_correlations_csv(scenario, datasets, slopes_by_dataset)
    print(f"Saved correlations to {corr_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intralog monthly-window complexity correlation analysis."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help=f"Dataset keys from data dictionary (default: {list(DEFAULT_DATASETS)}).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        help=f"Subdirectory under results/bias_study/ (default: {DEFAULT_SCENARIO}).",
    )
    parser.add_argument(
        "--y-log",
        action="store_true",
        help="Use log scale on y-axis for strictly positive metric values.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip recomputation and generate plots from existing complexity.csv files.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Compute only a very small metric subset for a quick smoke run.",
    )
    args = parser.parse_args()
    run(
        datasets=list(args.datasets),
        scenario=args.scenario,
        y_log=args.y_log,
        plot_only=args.plot_only,
        test_mode=args.test_mode,
    )


if __name__ == "__main__":
    main()
