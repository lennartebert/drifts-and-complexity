from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from pm4py.objects.log.importer.xes import importer as xes_importer

from utils import constants, helpers
from utils.complexity.assessors import run_adapters
from utils.windowing.windowing import Window


def _sample_random_trace_sets_no_replacement(
    event_log: Iterable[Any],
    min_size: int = 10,
    max_size: Optional[int] = 500,
    num_sizes: int = 10,
    samples_per_size: int = 10,
    spacing: Literal["linear", "log"] = "linear",
    random_state: Optional[int] = None,
) -> List[Tuple[int, str, List[Any]]]:
    """
    Build random, non-consecutive trace sets WITHOUT replacement across the whole log.

    We first derive a vector of window sizes (linearly or logarithmically spaced)
    that respects the total "budget" of available traces for the requested number
    of samples per size. Then, for each sample_id and each size, we sample indices
    from the remaining pool (without replacement) and collect those traces.

    Returns
    -------
    list[tuple[int, str, list]]
        Tuples of (window_size, sample_id, trace_list).
    """
    # Wrap in list for len/index while accepting general iterables
    event_log = list(event_log)
    n_traces = len(event_log)
    if n_traces < min_size:
        return []

    # RNG (avoid touching global state)
    rng = np.random.default_rng(seed=random_state)

    # Determine max size bound
    max_size = min(max_size if max_size is not None else n_traces, n_traces)

    # Per-repetition budget
    if samples_per_size <= 0:
        return []
    budget = n_traces // samples_per_size
    if budget < min_size:
        return []

    feasible_max_size = min(max_size, budget)

    # Create size grid
    if num_sizes <= 0:
        num_sizes = 1
    if num_sizes == 1 or feasible_max_size <= min_size:
        sizes = np.array([min_size], dtype=int)
    else:
        if spacing == "log":
            sizes = np.geomspace(min_size, feasible_max_size, num_sizes).astype(int)
        else:
            sizes = np.linspace(min_size, feasible_max_size, num_sizes).astype(int)

    sizes = np.clip(sizes, min_size, feasible_max_size)
    sizes = np.maximum.accumulate(sizes)

    # Ensure budget is not exceeded (very conservative trim)
    while sizes.sum() > budget and sizes[-1] > min_size:
        sizes[-1] -= 1

    # Sampling without replacement across the full pool
    available = list(range(n_traces))
    results: List[Tuple[int, str, List[Any]]] = []

    for sample_id in range(samples_per_size):
        for size in sizes:
            if len(available) < int(size):
                break
            idxs = rng.choice(available, size=int(size), replace=False).tolist()
            # Remove chosen indices from availability
            chosen_set = set(idxs)
            available = [i for i in available if i not in chosen_set]
            chosen_traces = [event_log[i] for i in idxs]
            results.append((int(size), str(sample_id), chosen_traces))

        if len(available) < min_size:
            break

    return results


def _get_measures_per_sample_per_dataset(
    data_dictionary: Dict[str, Dict[str, Any]],
    adapters: Optional[List[str]] = None,
    random_state: Optional[int] = 0,
    debug_first_only: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    For each dataset in the dictionary, draw samples and compute complexity measures
    via the provided adapters. Returns a DataFrame per dataset.
    """
    if adapters is None:
        adapters = ["vidgof_sample", "population_inext", "population_simple"]

    out: Dict[str, pd.DataFrame] = {}

    for dataset, dataset_info in data_dictionary.items():
        print(f"Processing {dataset}")

        # Load log
        log_path = Path(dataset_info["path"])
        pm4py_log = xes_importer.apply(str(log_path))

        # Sample trace sets
        samples = _sample_random_trace_sets_no_replacement(pm4py_log, random_state=random_state)

        # Prepare windows for the adapters
        windows_map: Dict[str, Window] = {
            str(idx): Window(id=str(idx), size=size, traces=traces)
            for idx, (size, _sid, traces) in enumerate(samples)
        }

        # Compute measures via adapters (returns dict keyed by window.id)
        dictionaries_all_windows = run_adapters(windows_map.values(), adapters)

        # Flatten into rows
        rows: List[Dict[str, Any]] = []
        for win_id, measures_dict in dictionaries_all_windows.items():
            # keep only measure_* keys, strip prefix
            clean_measures = {
                k.removeprefix("measure_"): v
                for k, v in measures_dict.items()
                if k.startswith("measure_")
            }
            rows.append(
                {
                    **clean_measures,
                    "window_size": windows_map[win_id].size,
                    "sample_id": windows_map[win_id].id,
                }
            )

        df = pd.DataFrame(rows)
        out[dataset] = df

        if debug_first_only:
            break

    return out


def _test_measures_for_correlation(
    measures_per_sample_per_dataset: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    For each dataset, compute Pearson r between each numeric measure and window_size.
    Returns a dict of DataFrames with columns: measure, n, pearson_r, p_value.
    """
    results: Dict[str, pd.DataFrame] = {}

    for dataset, df in measures_per_sample_per_dataset.items():
        if "window_size" not in df.columns:
            raise KeyError(f"'window_size' missing for dataset {dataset}")

        # Identify numeric measure columns (exclude identifiers)
        measure_cols = [
            c
            for c in df.columns
            if c not in {"window_size", "sample_id"} and pd.api.types.is_numeric_dtype(df[c])
        ]

        ws = pd.to_numeric(df["window_size"], errors="coerce")
        rows: List[Dict[str, Any]] = []

        for col in measure_cols:
            x = pd.to_numeric(df[col], errors="coerce")

            pair = (
                pd.DataFrame({"x": x, "y": ws})
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
                r, p = np.nan, np.nan
                n = len(pair)
            else:
                r, p = pearsonr(pair["x"].to_numpy(), pair["y"].to_numpy())
                n = len(pair)

            rows.append({"measure": col, "n": n, "pearson_r": r, "p_value": p})

        results[dataset] = (
            pd.DataFrame(rows).sort_values("measure").reset_index(drop=True)
        )

    return results


def plot_correlation_results(
    correlation_results: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """
    Pure-matplotlib boxplot of Pearson r per measure across datasets.
    """
    # Flatten
    records: List[Dict[str, Any]] = []
    for dataset, df in correlation_results.items():
        for _, row in df.iterrows():
            records.append(
                {
                    "dataset": dataset,
                    "measure": row["measure"],
                    "pearson_r": row["pearson_r"],
                }
            )
    corr_df = pd.DataFrame(records)
    if corr_df.empty:
        print("No correlation results to plot.")
        return

    measures = sorted(corr_df["measure"].unique())
    data_by_measure = [corr_df.loc[corr_df["measure"] == m, "pearson_r"].dropna().values for m in measures]

    plt.figure(figsize=(20, 10))
    plt.boxplot(data_by_measure, labels=measures, showfliers=False)
    plt.axhline(0, linewidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel("Pearson r")
    plt.xlabel("Measure")
    plt.title("Distribution of Pearson r Across Datasets by Measure")
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)
    plt.close()


def summarize_correlation_results(
    correlation_results: Dict[str, pd.DataFrame],
    out_path: Path,
) -> pd.DataFrame:
    """
    Summarize per-measure effect sizes and significance rates across datasets.
    Writes CSV and returns the summary DataFrame.
    """
    # Collect all unique measures
    all_measures = sorted(
        {row["measure"] for df in correlation_results.values() for _, row in df.iterrows()}
    )

    summary_rows: List[Dict[str, Any]] = []
    for measure in all_measures:
        r_values: List[float] = []
        p_values: List[float] = []
        sig_count = 0
        total_count = 0

        for _dataset, df in correlation_results.items():
            row = df[df["measure"] == measure]
            if row.empty:
                continue
            r = row.iloc[0]["pearson_r"]
            p = row.iloc[0]["p_value"]
            if pd.notna(r) and pd.notna(p):
                r_values.append(abs(float(r)))
                p_values.append(float(p))
                total_count += 1
                if p <= 0.05:
                    sig_count += 1

        if total_count > 0:
            summary_rows.append(
                {
                    "measure": measure,
                    "median_abs_r": float(np.median(r_values)),
                    "median_p_value": float(np.median(p_values)),
                    "pct_significant_(p<0.05)": (sig_count / total_count) * 100.0,
                    "n_datasets": total_count,
                }
            )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("median_abs_r", kind="mergesort")
        .reset_index(drop=True)
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    return summary_df


def main() -> None:
    DEBUG = False

    # Load data dictionary
    data_dictionary = helpers.load_data_dictionary(constants.DATA_DICTIONARY_FILE_PATH)

    # Compute measures per sample per dataset (debug: only first dataset by default)
    measures = _get_measures_per_sample_per_dataset(
        data_dictionary=data_dictionary,
        adapters=["vidgof_sample", "population_inext", "population_simple"],
        random_state=0,
        debug_first_only=DEBUG,
    )

    # Correlation tests
    correlation_results = _test_measures_for_correlation(measures)

    # Outputs
    out_dir = Path(constants.CORRELATION_RESULTS_DIR)
    plot_correlation_results(correlation_results, out_dir / "correlation_results.png")
    summarize_correlation_results(
        correlation_results, out_dir / "correlation_results_summarized.csv"
    )


if __name__ == "__main__":
    main()
