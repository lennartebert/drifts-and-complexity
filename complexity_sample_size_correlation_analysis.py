from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer
from scipy.stats import pearsonr

from utils import constants, helpers
from utils.complexity.assessors import run_metric_adapters
from utils.windowing.helpers import Window

ReplacementPolicy = Literal[
    "within_and_across",  # (1) full replacement: duplicates allowed within a sample and across samples
    "within_only",        # (2) no duplicates within a sample; samples are independent (replacement across samples)
    "none"                # (3) no replacement at all across all samples; each trace used at most once globally
]

def sample_random_traces(
    event_log: Iterable[Any],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    policy: ReplacementPolicy = "within_and_across",
    random_state: Optional[int] = None
) -> List[Tuple[int, str, List[Any]]]:
    """
    Sample random trace sets from an event log under different replacement policies.

    Parameters
    ----------
    event_log
        Iterable of traces (will be materialized to a list).
    sizes
        Iterable of positive integers indicating sample sizes to draw.
    samples_per_size
        Number of samples to draw per size.
    policy
        Replacement policy:
          - "within_and_across": draw with replacement for each sample (duplicates allowed within & across samples).
          - "within_only":        draw without replacement within a sample, but with replacement across samples
                                  (i.e., each sample is an independent no-replacement draw from the full log).
          - "none":               draw without replacement globally across all samples and sizes.
    random_state
        Seed for reproducible sampling.

    Returns
    -------
    List[Tuple[int, str, List[Any]]]
        List of (window_size, sample_id, trace_list).
    """
    # Materialize and validate
    event_log = list(event_log)
    n_traces = len(event_log)

    if n_traces == 0 or samples_per_size <= 0:
        return []

    sizes = [int(s) for s in sizes if int(s) > 0]
    if not sizes:
        return []

    rng = np.random.default_rng(seed=random_state)
    results: List[Tuple[int, str, List[Any]]] = []

    if policy == "within_and_across":
        # (1) Full replacement: allow repeats within a sample and across samples.
        for sample_id in range(samples_per_size):
            for s in sizes:
                idxs = rng.integers(low=0, high=n_traces, size=s).tolist()
                chosen_traces = [event_log[i] for i in idxs]
                results.append((s, str(sample_id), chosen_traces))
        return results

    if policy == "within_only":
        # (2) No replacement within a sample; across samples independent.
        #     Each sample requires s <= n_traces.
        max_s = max(sizes)
        if max_s > n_traces:
            raise ValueError(
                f"Requested size {max_s} exceeds number of traces {n_traces} for policy 'within_only'."
            )
        indices = np.arange(n_traces)
        for sample_id in range(samples_per_size):
            for s in sizes:
                # independent sample each time, without replacement
                idxs = rng.choice(indices, size=s, replace=False).tolist()
                chosen_traces = [event_log[i] for i in idxs]
                results.append((s, str(sample_id), chosen_traces))
        return results

    if policy == "none":
        # (3) Global no replacement across all samples and sizes.
        total_needed = int(samples_per_size) * int(sum(sizes))
        if total_needed > n_traces:
            raise ValueError(
                f"Insufficient traces ({n_traces}) for global no-replacement sampling "
                f"of {samples_per_size} * sum(sizes) = {total_needed}."
            )

        # Shuffle a pool of all indices once, then carve it sequentially per (sample_id, size)
        pool = rng.permutation(n_traces).tolist()
        cursor = 0
        for sample_id in range(samples_per_size):
            for s in sizes:
                take = pool[cursor: cursor + s]
                cursor += s
                chosen_traces = [event_log[i] for i in take]
                results.append((s, str(sample_id), chosen_traces))
        return results

    raise ValueError(f"Unknown policy: {policy!r}")


# --- Backwards-compatible wrappers for sampling --------------------------

def sample_random_traces_with_replacement(
    event_log: Iterable[Any],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, List[Any]]]:
    """(1) Duplicates allowed within and across samples."""
    return sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="within_and_across",
        random_state=random_state,
    )

def sample_random_trace_sets_no_replacement_within_only(
    event_log: Iterable[Any],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, List[Any]]]:
    """(2) No duplicates within a sample; samples are independent across runs."""
    return sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="within_only",
        random_state=random_state,
    )

def sample_random_trace_sets_no_replacement_global(
    event_log: Iterable[Any],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, List[Any]]]:
    """(3) No replacement globally across all samples and sizes."""
    return sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="none",
        random_state=random_state,
    )


def compute_metrics_for_samples(
    samples: List[Tuple[int, str, List[Any]]],
    adapters: List[str]) -> pd.DataFrame:

    # Prepare windows for the adapters
    windows_map: Dict[str, Window] = {
        str(idx): Window(id=str(idx), size=size, traces=traces)
        for idx, (size, _sid, traces) in enumerate(samples)
    }
    
    # Compute measures via adapters (returns dict keyed by window.id)
    dictionaries_all_windows = run_metric_adapters(windows_map.values(), adapters)

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
    return df


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
        samples = sample_random_trace_sets_no_replacement(pm4py_log, random_state=random_state)

        df = compute_metrics_for_samples(samples, adapters)
        out[dataset] = df

        if debug_first_only:
            break

    return out


def _test_measures_for_correlation(
    measures_per_sample_per_dataset: Dict[str, pd.DataFrame],
    constant_policy = "nan",  # implemnets "nan" or "zero"
) -> Dict[str, pd.DataFrame]:
    """
    For each dataset, compute Pearson r between each numeric measure and window_size.
    Returns a dict of DataFrames with columns: measure, n, pearson_r, p_value, status.
    status element of {"ok","insufficient_n","constant_x","constant_y","constant_both"}.
    """
    results: Dict[str, pd.DataFrame] = {}

    for dataset, df in measures_per_sample_per_dataset.items():
        if "window_size" not in df.columns:
            raise KeyError(f"'window_size' missing for dataset {dataset}")

        measure_cols = [
            c for c in df.columns
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
            n = len(pair)

            if n < 2:
                rows.append({
                    "measure": col, "n": n,
                    "pearson_r": np.nan, "p_value": np.nan,
                    "status": "insufficient_n",
                })
                continue

            x_unique = pair["x"].nunique()
            y_unique = pair["y"].nunique()

            if x_unique < 2 or y_unique < 2:
                if x_unique < 2 and y_unique < 2:
                    status = "constant_both"
                elif x_unique < 2:
                    status = "constant_x"
                else:
                    status = "constant_y"

                if constant_policy == "nan":
                    r, p = np.nan, np.nan
                else:  # "zero": explicit fallback, but keep p undefined
                    r, p = 0.0, np.nan

                rows.append({
                    "measure": col, "n": n,
                    "pearson_r": r, "p_value": p,
                    "status": status,
                })
                continue

            # Regular case
            r, p = pearsonr(pair["x"].to_numpy(), pair["y"].to_numpy())
            rows.append({
                "measure": col, "n": n,
                "pearson_r": r, "p_value": p,
                "status": "ok",
            })

        results[dataset] = (
            pd.DataFrame(rows)
            .sort_values(["status", "measure"])
            .reset_index(drop=True)
        )

    return results

def _flatten_correlation_results(correlation_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Flatten correlation results across datasets into a single DataFrame.
    """
    records: List[Dict[str, Any]] = []
    for dataset, df in correlation_results.items():
        for _, row in df.iterrows():
            records.append(
                {
                    "dataset": dataset,
                    "measure": row["measure"],
                    "n": row["n"],
                    "pearson_r": row["pearson_r"],
                    "p_value": row["p_value"],
                    "status": row["status"],
                }
            )
    return pd.DataFrame(records)


def plot_correlation_results(
    flat_correlation_results: pd.DataFrame,
    out_path: Path,
    plot_type: str = "box",  # "box" or "dot"
) -> None:
    """
    Plot Pearson r per measure across datasets.

    Parameters
    ----------
    flat_correlation_results : pd.DataFrame
        Must contain columns ["measure", "pearson_r", "dataset"].
    out_path : Path
        Path where the figure will be saved.
    plot_type : str, optional
        Type of plot: "box" (default) or "dot".
    """

    measures = sorted(flat_correlation_results["measure"].unique())
    datasets = sorted(flat_correlation_results["dataset"].unique())
    cmap = plt.get_cmap("tab10")  # categorical color map
    dataset_colors = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}

    plt.figure(figsize=(20, 10))

    if plot_type == "box":
        # --- Boxplot ---
        data_by_measure = [
            flat_correlation_results.loc[
                flat_correlation_results["measure"] == m, "pearson_r"
            ].dropna().values
            for m in measures
        ]
        plt.boxplot(data_by_measure, labels=measures, showfliers=False)

    elif plot_type == "dot":
        # --- Dot plot with dataset colors ---
        added_labels = set()  # track which datasets already got a legend label
        for i, m in enumerate(measures, start=1):
            df_m = flat_correlation_results.loc[
                flat_correlation_results["measure"] == m
            ]
            for ds in datasets:
                values = df_m.loc[df_m["dataset"] == ds, "pearson_r"].dropna().values
                if len(values) == 0:
                    continue
                x_jittered = np.random.normal(loc=i, scale=0.05, size=len(values))
                plt.scatter(
                    x_jittered,
                    values,
                    alpha=0.7,
                    color=dataset_colors[ds],
                    label=ds if ds not in added_labels else None,
                )
                added_labels.add(ds)  # mark dataset as labeled
        plt.xticks(range(1, len(measures) + 1), measures, rotation=90)
        plt.legend(title="Dataset", loc="upper right")

    else:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Use 'box' or 'dot'.")

    # Common formatting
    plt.axhline(0, linewidth=0.8, color="grey")
    plt.ylabel("Pearson r")
    plt.xticks(rotation=90)
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


def main(analysis_name=constants.DEFAULT_CORRELATION_ANLAYSIS_NAME, include_real=False, exclude_synthetic=False, debug_first_only=False) -> None:
    if not include_real and exclude_synthetic:
        print("No datasets selected for analysis (both real and synthetic excluded). Exiting.")
        return

    # Load data dictionary
    data_dictionary = helpers.load_data_dictionary(constants.DATA_DICTIONARY_FILE_PATH, get_real=include_real, get_synthetic=(not exclude_synthetic))

    # Compute measures per sample per dataset (debug: only first dataset by default)
    measures = _get_measures_per_sample_per_dataset(
        data_dictionary=data_dictionary,
        adapters=["vidgof_sample", "population_inext", "population_simple"],
        random_state=0,
        debug_first_only=debug_first_only,
    )

    # Correlation tests
    correlation_results = _test_measures_for_correlation(measures)
    flat_correlation_results = _flatten_correlation_results(correlation_results)

    # Outputs
    out_dir = Path(constants.CORRELATION_RESULTS_DIR) / analysis_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # save flat complexity_results
    flat_correlation_results.to_csv(out_dir / "correlation_results_detailed.csv", index=False)

    plot_correlation_results(flat_correlation_results, out_dir / "correlation_results_box.png", 'box')
    plot_correlation_results(flat_correlation_results, out_dir / "correlation_results_dot.png", 'dot')
    summarize_correlation_results(
        correlation_results, out_dir / "correlation_results_summarized.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze correlations of complexity figures to sample size.")
    parser.add_argument(
        "--analysis-name",
        default=constants.DEFAULT_CORRELATION_ANLAYSIS_NAME,
        help=f"Name of analysis. Default '{constants.DEFAULT_CORRELATION_ANLAYSIS_NAME}'."
    )
    parser.add_argument(
        "--include-real",
        action="store_true",
        help="If set, includes real datasets from analysis."
    )
    parser.add_argument(
        "--exclude-synthetic",
        action="store_true",
        help="If set, excludes synthetic datasets from analysis."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, only analysis one dataset."
    )

    args = parser.parse_args()

    main(analysis_name=args.analysis_name, include_real=args.include_real, exclude_synthetic=args.exclude_synthetic, debug_first_only=args.debug)
