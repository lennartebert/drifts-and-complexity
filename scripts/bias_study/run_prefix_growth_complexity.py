#!/usr/bin/env python
"""Growing-prefix window complexity for bias analysis (local vs Vidgof).

Loads event logs via the data dictionary, builds nested prefix windows from trace 0
(sizes increment, 2*increment, …), runs ``local`` and ``vidgof_sample`` adapters,
writes ``complexity.csv`` and per-measure line plots (window size vs measure).

Run from repository root with the project conda environment (do not rely on ad-hoc pip installs)::

    conda activate drifts-and-complexity
    python scripts/bias_study/run_prefix_growth_complexity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from utils import constants, helpers
from utils.complexity.assessors import build_growing_prefix_complexity_dataframe
from utils.drift_io import load_xes_log
from utils.plotting.prefix_growth import (
    plot_prefix_growth_complexity,
    plot_prefix_growth_dataset_comparison,
)

# -----------------------------------------------------------------------------
# Windowing for this analysis (same semantics as `growing_prefix_trace_windows` in
# scripts/change_study/window_config.yml). Change defaults here; CLI flags override.
# -----------------------------------------------------------------------------
PREFIX_GROWTH_WINDOW_CONFIG: Dict[str, Any] = {
    "type": "growing_prefix_trace_windows",
    "name": "prefix_growth_bias_study",
    "params": {
        "increment": 500,
        "start_index": 0,
    },
}

DEFAULT_DATASETS: Sequence[str] = ("BPIC12", "RTFMP")
DEFAULT_INCREMENT = int(PREFIX_GROWTH_WINDOW_CONFIG["params"]["increment"])
DEFAULT_START_INDEX = int(PREFIX_GROWTH_WINDOW_CONFIG["params"]["start_index"])
DEFAULT_SCENARIO = "prefix_growth"
ADAPTER_NAMES = ["local", "vidgof_sample"]


def _resolve_log_path(dataset_info: dict) -> Path:
    p = Path(dataset_info["path"])
    if p.is_absolute():
        return p
    return constants.ROOT / p


def _dataset_complexity_csv_path(scenario: str, dataset_key: str) -> Path:
    return constants.BIAS_STUDY_RESULTS_DIR / scenario / dataset_key / "complexity.csv"


def _load_existing_complexity_data(
    datasets: List[str], scenario: str, *, fail_on_missing: bool = True
) -> Dict[str, pd.DataFrame]:
    missing: List[Path] = []
    out: Dict[str, pd.DataFrame] = {}
    for dataset_key in datasets:
        p = _dataset_complexity_csv_path(scenario, dataset_key)
        if not p.exists():
            missing.append(p)
            continue
        out[dataset_key] = pd.read_csv(p)

    if missing and fail_on_missing:
        msg = "\n".join(f"  - {str(p)}" for p in missing)
        raise FileNotFoundError(
            "Missing complexity.csv for plot-only mode:\n"
            f"{msg}\nRun without --plot-only first (or generate missing datasets)."
        )
    return out


def _plot_dataset_comparison_from_data(
    data_by_dataset: Dict[str, pd.DataFrame], scenario: str, y_log: bool
) -> None:
    comparison_dir = constants.BIAS_STUDY_RESULTS_DIR / scenario / "comparison_plots"
    print(f"Creating cross-dataset comparison plots in {comparison_dir} …")
    plot_prefix_growth_dataset_comparison(
        data_by_dataset,
        comparison_dir,
        y_log=y_log,
    )


def run(
    datasets: List[str],
    increment: int,
    scenario: str,
    start_index: int,
    y_log: bool,
    *,
    plot_only: bool = False,
) -> None:
    if plot_only:
        data_by_dataset = _load_existing_complexity_data(
            datasets, scenario, fail_on_missing=True
        )
        _plot_dataset_comparison_from_data(data_by_dataset, scenario, y_log)
        return

    data_dictionary = helpers.load_data_dictionary(
        constants.get_data_dictionary_path(),
        get_real=True,
        get_synthetic=False,
    )
    missing = [d for d in datasets if d not in data_dictionary]
    if missing:
        print(f"Error: unknown or filtered-out dataset keys: {missing}", file=sys.stderr)
        print(f"Available (real): {sorted(data_dictionary.keys())}", file=sys.stderr)
        sys.exit(1)

    base_out = constants.BIAS_STUDY_RESULTS_DIR / scenario
    base_out.mkdir(parents=True, exist_ok=True)
    data_by_dataset: Dict[str, pd.DataFrame] = {}

    for dataset_key in datasets:
        info = data_dictionary[dataset_key]
        log_path = _resolve_log_path(info)
        activity_key = info.get("activity_key", "concept:name")

        print(f"Loading {dataset_key} from {log_path} …")
        traces_sorted = load_xes_log(log_path, activity_key=activity_key)

        print(
            f"  Computing prefix growth (increment={increment}, start_index={start_index}) "
            f"with adapters {ADAPTER_NAMES} …"
        )
        df = build_growing_prefix_complexity_dataframe(
            traces_sorted,
            increment,
            ADAPTER_NAMES,
            start_index=start_index,
            add_prefix=True,
            include_adapter_name=True,
        )

        out_dir = base_out / dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "complexity.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path} ({len(df)} rows)")
        data_by_dataset[dataset_key] = df

        plots_dir = out_dir / "plots"
        plot_prefix_growth_complexity(
            df,
            plots_dir,
            dataset_key=dataset_key,
            y_log=y_log,
        )

    _plot_dataset_comparison_from_data(data_by_dataset, scenario, y_log)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Growing-prefix complexity (local + vidgof) for selected datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help=f"Dataset keys from the data dictionary (default: {list(DEFAULT_DATASETS)}).",
    )
    parser.add_argument(
        "--increment",
        type=int,
        default=DEFAULT_INCREMENT,
        help=(
            "Trace step between prefix sizes "
            f"(default: {DEFAULT_INCREMENT} from PREFIX_GROWTH_WINDOW_CONFIG)."
        ),
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        help=f"Subdirectory under results/bias_study/ (default: {DEFAULT_SCENARIO}).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help=(
            "First trace index for every prefix window "
            f"(default: {DEFAULT_START_INDEX} from PREFIX_GROWTH_WINDOW_CONFIG)."
        ),
    )
    parser.add_argument(
        "--y-log",
        action="store_true",
        help="Use log scale on y-axis when all plotted values are positive.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Skip recomputation and generate cross-dataset comparison plots from "
            "existing results/bias_study/<scenario>/<dataset>/complexity.csv files."
        ),
    )
    args = parser.parse_args()
    if args.increment < 1:
        print("--increment must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.start_index < 0:
        print("--start-index must be >= 0", file=sys.stderr)
        sys.exit(1)

    run(
        datasets=list(args.datasets),
        increment=args.increment,
        scenario=args.scenario,
        start_index=args.start_index,
        y_log=args.y_log,
        plot_only=args.plot_only,
    )


if __name__ == "__main__":
    main()
