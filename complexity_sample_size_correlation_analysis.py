from matplotlib import pyplot as plt
from utils import constants, helpers
import random
from typing import List, Tuple, Optional
from pathlib import Path
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from utils.complexity.assessors import run_adapters
from utils.windowing.windowing import Window

DEBUG = True

def _sample_random_trace_sets_no_replacement(
    event_log,
    min_size: int = 10,
    max_size: int = 500,       
    num_sizes: int = 10, 
    samples_per_size: int = 10,
    spacing: str = "linear",              # "linear" or "log"
    random_state: Optional[int] = None
) -> List[Tuple[int, str, list]]:
    """
    Create random, non-consecutive trace sets without replacement,
    automatically choosing sizes based on log length, min_size, and max_size.

    Returns: list of tuples (window_size, sample_id, trace_list)
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    n_traces = len(event_log)
    if n_traces < min_size:
        return []

    # Apply user-defined max_size cap
    if max_size is not None:
        max_size = min(max_size, n_traces)
    else:
        max_size = n_traces

    # Budget = how many traces we can "spend" per repetition
    budget = n_traces // samples_per_size
    if budget < min_size:
        return []

    # The largest feasible size under budget and max_size
    feasible_max_size = min(max_size, budget)

    # --- Generate size list ---
    if spacing.lower() == "linear":
        if num_sizes <= 1 or feasible_max_size <= min_size:
            sizes = np.array([min_size] * max(1, num_sizes), dtype=int)
        else:
            sizes = np.linspace(min_size, feasible_max_size, num_sizes).astype(int)
    else:  # "log" spacing
        if num_sizes <= 1 or feasible_max_size <= min_size:
            sizes = np.array([min_size] * max(1, num_sizes), dtype=int)
        else:
            sizes = np.geomspace(min_size, feasible_max_size, num_sizes).astype(int)

    # Ensure monotonic increase and cap at feasible_max_size
    sizes = np.clip(sizes, min_size, feasible_max_size)
    sizes = np.maximum.accumulate(sizes)

    # Safety: shrink sizes from the end if total budget exceeded
    while sizes.sum() > budget and sizes[-1] > min_size:
        sizes[-1] -= 1

    # --- Sampling without replacement ---
    available = list(range(n_traces))
    results: List[Tuple[int, str, list]] = []

    for sample_id in range(samples_per_size):
        for size in sizes:
            if len(available) < size:
                break
            idxs = random.sample(available, int(size))
            for idx in idxs:
                available.remove(idx)
            chosen_traces = [event_log[i] for i in idxs]
            results.append((int(size), str(sample_id), chosen_traces))

        if len(available) < min_size:
            break
    
    return results

def _get_measures_per_sample_per_dataset(data_dictionary):
    measures_per_sample_per_dataset = {}
    for dataset, dataset_info in data_dictionary.items():
        print(f"Processing {dataset}")
        # load the log
        log_path = Path(dataset_info["path"])
        pm4py_log = xes_importer.apply(str(log_path))

        sample_sizes = range(100, 1001, 100)
        samples_per_size = 10

        samples = _sample_random_trace_sets_no_replacement(
            pm4py_log,
            random_state=0
        )

        # compute complexity for each sample
        measures_per_sample = []  # rows
        windows_map = {str(id): Window(id=str(id), size=size, traces=traces) for id, (size, _, traces) in enumerate(samples)}
        adapters = ['vidgof_sample', 'population_inext', 'population_simple']
        dictionaries_all_windows = run_adapters(windows_map.values(), adapters)

        for i, dictionary_per_window in dictionaries_all_windows.items():
            measures = dictionary_per_window
            # remove "measures_" prefix and remove all none-measures
            measures = {measure.removeprefix('measure_'):value for measure, value in measures.items() if measure.startswith('measure_')}

            measures_per_sample.append({
                **measures,
                "window_size": windows_map[i].size,
                "sample_id": windows_map[i].id
            })
        
        df = pd.DataFrame(measures_per_sample)

        measures_per_sample_per_dataset[dataset] = df
        if DEBUG: break
    return measures_per_sample_per_dataset

def _test_measures_for_correlation(measures_per_sample_per_dataset):
    correlation_results = {}
    for dataset, df in measures_per_sample_per_dataset.items():
        if "window_size" not in df.columns:
            raise KeyError(f"'window_size' missing for dataset {dataset}")

        # choose numeric measure columns, exclude identifiers
        measure_cols = [
            c for c in df.columns
            if c not in ["window_size", "sample_id"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        ws = pd.to_numeric(df["window_size"], errors="coerce")
        rows = []
        for col in measure_cols:
            x = pd.to_numeric(df[col], errors="coerce")
            pair = (pd.DataFrame({"x": x, "y": ws})
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna())
            if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(pair["x"].to_numpy(), pair["y"].to_numpy())
            rows.append({"measure": col, "n": len(pair), "pearson_r": r, "p_value": p})

        correlation_results[dataset] = (pd.DataFrame(rows)
                                        .sort_values("measure")
                                        .reset_index(drop=True))
    return correlation_results
    
def main():
    # load data dictionary
    data_dictionary = helpers.load_data_dictionary(constants.DATA_DICTIONARY_FILE_PATH)

    # get measures per sample per dataset
    measures_per_sample_per_dataset = _get_measures_per_sample_per_dataset(data_dictionary)
    
    # test for correlation between measure and size
    correlation_results = _test_measures_for_correlation(measures_per_sample_per_dataset)
    # correlation_results[dataset] = DataFrame with columns: measure, n, pearson_r, p_value

    out_dir = constants.CORRELATION_RESULTS_DIR
    plot_correlation_results(correlation_results, out_dir / 'correlation_results.png')
    summarize_correlation_results(correlation_results, out_dir / 'correlation_results_summarized.csv')


def plot_correlation_results(correlation_results, out_path):
    # Flatten correlation_results into one long DataFrame
    all_corrs = []
    for dataset, df in correlation_results.items():
        for _, row in df.iterrows():
            all_corrs.append({
                "dataset": dataset,
                "measure": row["measure"],
                "pearson_r": row["pearson_r"]
            })

    corr_df = pd.DataFrame(all_corrs)

    # Sort measures for consistent x-axis order
    order = sorted(corr_df["measure"].unique())

    # Create Seaborn boxplot
    plt.figure(figsize=(20, 10))
    sns.boxplot(
        data=corr_df,
        x="measure",
        y="pearson_r",
        order=order
    )

    # Formatting
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel("Pearson r")
    plt.xlabel("Measure")
    plt.title("Distribution of Pearson r Across Datasets by Measure")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)


def summarize_correlation_results(correlation_results, out_path):
    summary_rows = []

    for measure in sorted({row["measure"] for df in correlation_results.values() for _, row in df.iterrows()}):
        r_values = []
        p_values = []
        sig_count = 0
        total_count = 0
        
        for dataset, df in correlation_results.items():
            row = df[df["measure"] == measure]
            if not row.empty:
                r = row["pearson_r"].values[0]
                p = row["p_value"].values[0]
                if pd.notna(r) and pd.notna(p):
                    r_values.append(abs(r))
                    p_values.append(p)
                    total_count += 1
                    if p <= 0.05:
                        sig_count += 1
        
        if total_count > 0:
            summary_rows.append({
                "measure": measure,
                "median_abs_r": np.median(r_values),
                "median_p_value": np.median(p_values),
                "pct_significant_(p<0.05)": (sig_count / total_count) * 100,
                "n_datasets": total_count
            })

    summary_df = pd.DataFrame(summary_rows).sort_values("median_abs_r").reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path)

if __name__ == "__main__":
    main()
