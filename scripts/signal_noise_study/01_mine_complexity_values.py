"""Mine complexity values from synthetic logs with sudden drifts.

This script processes synthetic event logs that contain sudden drifts, splits them
at the drift point, and computes complexity metrics for different window sizes
on both pre-drift and post-drift segments.

Supports parallelization and resume capability.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog

from utils import sampling_helper
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
from utils.complexity.metrics_adapters.vidgof_metrics_adapter import (
    VidgofMetricsAdapter,
)
from utils.parallel import run_parallel
from utils.pipeline.compute import (
    compute_analysis_for_metrics,
    compute_metrics_for_samples,
)
from utils.population.extractors.naive_population_extractor import (
    NaivePopulationExtractor,
)
from utils.sample_confidence_interval_extractor import (
    SampleConfidenceIntervalExtractor,
)
from utils.sample_standard_deviation_extractor import (
    SampleStandardDeviationExtractor,
)

# Constants
PATH_TO_LOGS = Path("data") / "synthetic" / "sudden_drifts"
GEN_INFO_FILE_NAME = "generation_info.csv"
DRIFT_POINT_IN_LOGS = (
    5000  # First 5000 traces are one process version, second 5000 traces are the other
)

# Study specific constants
WINDOW_SIZES = [50, 100, 150, 200]
SAMPLES_PER_SIZE = 200
RANDOM_STATE = 321

# Output directory
OUTPUT_DIR = Path("results") / "signal_noise_study"
INTERMEDIARY_DIR = OUTPUT_DIR / "intermediary"
BATCH_SIZE = 10


@dataclass
class ProcessingTask:
    """Task definition for processing a single log (all splits and window sizes)."""

    log_id: str
    log_file_path: Path


@dataclass
class SplitResult:
    """Result for a single split/window_size combination."""

    log_id: str
    split_name: str
    window_size: int
    metrics_df: pd.DataFrame
    analysis_df: pd.DataFrame


def split_event_log(event_log: EventLog, drift_point: int) -> Dict[str, EventLog]:
    """
    Split an event log into pre-drift and post-drift parts.

    Parameters
    ----------
    event_log
        The event log to split.
    drift_point
        The trace index at which to split (first drift_point traces are pre-drift).

    Returns
    -------
    Dict[str, EventLog]
        Dictionary with 'pre_drift' and 'post_drift' keys containing the split logs.
    """
    pre_drift_log = EventLog(event_log[:drift_point])
    post_drift_log = EventLog(event_log[drift_point:])

    return {"pre_drift": pre_drift_log, "post_drift": post_drift_log}


def generate_tasks(generated_files_df: pd.DataFrame) -> List[ProcessingTask]:
    """
    Generate all processing tasks from the CSV metadata.

    Creates one task per log, which will process both splits and all window sizes.

    Parameters
    ----------
    generated_files_df
        DataFrame with log_id as index.

    Returns
    -------
    List[ProcessingTask]
        List of all tasks to process (one per log).
    """
    tasks = []

    for log_id in generated_files_df.index:
        log_file_path = PATH_TO_LOGS / log_id

        if not log_file_path.exists():
            print(f"  Warning: Log file not found: {log_file_path}, skipping...")
            continue

        # Create one task per log (will process both splits and all window sizes)
        task = ProcessingTask(
            log_id=log_id,
            log_file_path=log_file_path,
        )
        tasks.append(task)

    return tasks


def get_intermediary_file_paths(
    log_id: str, split_name: str, window_size: int
) -> Tuple[Path, Path]:
    """
    Get paths for intermediary output files for a split/window_size combination.

    Parameters
    ----------
    log_id
        The log identifier.
    split_name
        The split name ('pre_drift' or 'post_drift').
    window_size
        The window size.

    Returns
    -------
    Tuple[Path, Path]
        Paths for per-sample metrics and analysis files.
    """
    # Sanitize log_id for filename (remove .xes.gz extension if present)
    log_id_safe = log_id.replace(".xes.gz", "").replace(".xes", "").replace("/", "_")
    base_name = f"{log_id_safe}_{split_name}_{window_size}"

    metrics_path = INTERMEDIARY_DIR / f"{base_name}.csv"
    analysis_path = INTERMEDIARY_DIR / f"{base_name}_analysis.csv"

    return metrics_path, analysis_path


def check_resume_status(tasks: List[ProcessingTask]) -> List[ProcessingTask]:
    """
    Check which tasks are already completed and filter them out.

    First checks if final files exist (complete run), then checks intermediary files.
    A task is considered complete if all split/window_size combinations for that log are done.

    Parameters
    ----------
    tasks
        List of all tasks to potentially process.

    Returns
    -------
    List[ProcessingTask]
        Filtered list of tasks that still need processing.
    """
    # Check if final files exist and are complete
    final_per_sample_path = OUTPUT_DIR / "per_sample_metrics.csv"
    final_analysis_path = OUTPUT_DIR / "aggregate_analysis.csv"

    if final_per_sample_path.exists() and final_analysis_path.exists():
        # Check if files are non-empty
        try:
            per_sample_df = pd.read_csv(final_per_sample_path)
            analysis_df = pd.read_csv(final_analysis_path)
            if not per_sample_df.empty and not analysis_df.empty:
                print(
                    "Final output files already exist and are non-empty. "
                    "Skipping all processing."
                )
                return []
        except Exception as e:
            print(
                f"Warning: Could not read final files: {e}. Proceeding with processing."
            )

    # Check intermediary files
    remaining_tasks = []
    completed_count = 0
    total_combinations = len(WINDOW_SIZES) * 2  # 2 splits * N window sizes

    for task in tasks:
        # Check if all split/window_size combinations for this log are complete
        all_complete = True
        completed_combinations = 0

        for split_name in ["pre_drift", "post_drift"]:
            for window_size in WINDOW_SIZES:
                metrics_path, analysis_path = get_intermediary_file_paths(
                    task.log_id, split_name, window_size
                )

                if metrics_path.exists() and analysis_path.exists():
                    # Check if files are non-empty
                    try:
                        metrics_df = pd.read_csv(metrics_path)
                        analysis_df = pd.read_csv(analysis_path)
                        if not metrics_df.empty and not analysis_df.empty:
                            completed_combinations += 1
                            continue
                    except Exception:
                        # If file is corrupted or empty, reprocess
                        pass

                all_complete = False

        if all_complete:
            completed_count += 1
        else:
            remaining_tasks.append(task)

    if completed_count > 0:
        print(
            f"Resuming: {completed_count} logs already completed, "
            f"{len(remaining_tasks)} logs remaining."
        )

    return remaining_tasks


def process_single_task(
    task: ProcessingTask,
    n_jobs: int = 1,
) -> Optional[List[SplitResult]]:
    """
    Process a single task (one log file).

    Loads the log once and processes both pre-drift and post-drift splits
    for all window sizes. Parallelization happens at the window sample level
    within compute_metrics_for_samples.

    Parameters
    ----------
    task
        The processing task to execute.
    n_jobs
        Number of parallel workers for computing metrics across window samples.

    Returns
    -------
    Optional[List[SplitResult]]
        List of results for each split/window_size combination if successful, None if failed.
    """
    try:
        # Load the event log once
        event_log = xes_importer.apply(str(task.log_file_path))

        # Check if log has enough traces
        if len(event_log) < DRIFT_POINT_IN_LOGS:
            print(
                f"  Warning: Log {task.log_id} has only {len(event_log)} traces, "
                f"need at least {DRIFT_POINT_IN_LOGS}, skipping..."
            )
            return None

        # Split the log into pre-drift and post-drift parts
        split_logs = split_event_log(event_log, DRIFT_POINT_IN_LOGS)
        pre_drift_log = split_logs["pre_drift"]
        post_drift_log = split_logs["post_drift"]

        # Set up pipeline components (created in worker to avoid pickling issues)
        population_extractor = NaivePopulationExtractor()
        metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
        bootstrap_sampler = None
        normalizers = None
        include_metrics = None
        sample_confidence_interval_extractor = SampleConfidenceIntervalExtractor(
            conf_level=0.95
        )
        sample_standard_deviation_extractor = SampleStandardDeviationExtractor(ddof=1)

        results = []

        # Process both splits
        for split_name, split_log in [
            ("pre_drift", pre_drift_log),
            ("post_drift", post_drift_log),
        ]:
            # Process all window sizes for this split
            for window_size in WINDOW_SIZES:
                # Check if split has enough traces for this window size
                if len(split_log) < window_size:
                    print(
                        f"  Warning: Split {split_name} for log {task.log_id} "
                        f"has only {len(split_log)} traces, need at least {window_size}, skipping..."
                    )
                    continue

                # Perform sampling
                window_samples = (
                    sampling_helper.sample_consecutive_trace_windows_with_replacement(
                        split_log,
                        sizes=[window_size],
                        samples_per_size=SAMPLES_PER_SIZE,
                        random_state=RANDOM_STATE,
                    )
                )

                # Compute raw metrics (parallelized over window samples)
                metrics_df = compute_metrics_for_samples(
                    window_samples,
                    population_extractor=population_extractor,
                    metric_adapters=metric_adapters,
                    bootstrap_sampler=bootstrap_sampler,
                    normalizers=normalizers,
                    include_metrics=include_metrics,
                    n_jobs=n_jobs,
                )

                # Reset index to access columns
                metrics_df = metrics_df.reset_index()

                # Add log_id, split_name, and window_size columns
                metrics_df["log_id"] = task.log_id
                metrics_df["split_name"] = split_name
                metrics_df["window_size"] = window_size

                # Compute aggregates (mean values, CIs, std, correlations, plateau)
                analysis_df = compute_analysis_for_metrics(
                    metrics_df,
                    sample_confidence_interval_extractor=sample_confidence_interval_extractor,
                    sample_standard_deviation_extractor=sample_standard_deviation_extractor,
                    include_metrics=include_metrics,
                )

                # Reset index to access columns
                analysis_df = analysis_df.reset_index()

                # Add log_id, split_name, and window_size columns
                analysis_df["log_id"] = task.log_id
                analysis_df["split_name"] = split_name
                analysis_df["window_size"] = window_size

                results.append(
                    SplitResult(
                        log_id=task.log_id,
                        split_name=split_name,
                        window_size=window_size,
                        metrics_df=metrics_df,
                        analysis_df=analysis_df,
                    )
                )

        return results if results else None

    except Exception as e:
        print(f"Error processing task {task.log_id}: {e}")
        return None


def write_batch_results(
    results_batch: List[Tuple[ProcessingTask, List[SplitResult]]],
) -> None:
    """
    Write a batch of task results to intermediary files.

    Parameters
    ----------
    results_batch
        List of tuples (task, list of split_results) to write.
    """
    INTERMEDIARY_DIR.mkdir(parents=True, exist_ok=True)

    for task, split_results in results_batch:
        for split_result in split_results:
            metrics_path, analysis_path = get_intermediary_file_paths(
                split_result.log_id,
                split_result.split_name,
                split_result.window_size,
            )

            try:
                split_result.metrics_df.to_csv(metrics_path, index=False)
                split_result.analysis_df.to_csv(analysis_path, index=False)
            except Exception as e:
                print(
                    f"Error writing results for {split_result.log_id}/{split_result.split_name}/{split_result.window_size}: {e}"
                )


def aggregate_final_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate all intermediary results into final DataFrames.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (combined_per_sample_df, combined_aggregate_df).
    """
    INTERMEDIARY_DIR.mkdir(parents=True, exist_ok=True)

    all_per_sample_dfs = []
    all_analysis_dfs = []

    # Find all metrics files (those that don't end with _analysis.csv)
    all_csv_files = list(INTERMEDIARY_DIR.glob("*.csv"))
    metrics_files = [f for f in all_csv_files if not f.name.endswith("_analysis.csv")]

    # For each metrics file, find corresponding analysis file
    for metrics_file in metrics_files:
        analysis_file = metrics_file.with_name(metrics_file.stem + "_analysis.csv")

        if not analysis_file.exists():
            print(f"Warning: Analysis file not found for {metrics_file}, skipping...")
            continue

        try:
            metrics_df = pd.read_csv(metrics_file)
            analysis_df = pd.read_csv(analysis_file)

            if not metrics_df.empty:
                all_per_sample_dfs.append(metrics_df)
            if not analysis_df.empty:
                all_analysis_dfs.append(analysis_df)
        except Exception as e:
            print(
                f"Error reading intermediary files {metrics_file}/{analysis_file}: {e}"
            )

    # Combine all results
    if all_per_sample_dfs:
        combined_per_sample_df = pd.concat(all_per_sample_dfs, ignore_index=True)
    else:
        combined_per_sample_df = pd.DataFrame()

    if all_analysis_dfs:
        combined_aggregate_df = pd.concat(all_analysis_dfs, ignore_index=True)
    else:
        combined_aggregate_df = pd.DataFrame()

    return combined_per_sample_df, combined_aggregate_df


def main(n_jobs: int | None = None) -> None:
    """
    Main execution function with parallelization and resume support.

    Parameters
    ----------
    n_jobs
        Number of parallel workers. If None, defaults to number of CPU cores.
    """
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    # Construct paths
    gen_info_file_path = PATH_TO_LOGS / GEN_INFO_FILE_NAME

    # Load generation info CSV
    print(f"Loading generation info from {gen_info_file_path}")
    generated_files_df = pd.read_csv(gen_info_file_path, sep=";", index_col="log_id")

    print(f"Found {len(generated_files_df)} logs to process")

    # Generate all tasks
    print("\nGenerating tasks...")
    all_tasks = generate_tasks(generated_files_df)
    print(f"Generated {len(all_tasks)} tasks (one per log)")

    # Check resume status
    print("\nChecking resume status...")
    tasks_to_process = check_resume_status(all_tasks)

    if not tasks_to_process:
        print("All tasks already completed. Aggregating final results...")
        combined_per_sample_df, combined_aggregate_df = aggregate_final_results()

        # Save final results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not combined_per_sample_df.empty:
            per_sample_path = OUTPUT_DIR / "per_sample_metrics.csv"
            combined_per_sample_df.to_csv(per_sample_path, index=False)
            print(f"\nSaved per-sample metrics to: {per_sample_path}")
        if not combined_aggregate_df.empty:
            aggregate_path = OUTPUT_DIR / "aggregate_analysis.csv"
            combined_aggregate_df.to_csv(aggregate_path, index=False)
            print(f"Saved aggregate analysis to: {aggregate_path}")
        print("\nDone!")
        return

    print(
        f"Processing {len(tasks_to_process)} logs sequentially (inner parallelism)..."
    )

    # Set BLAS safety (prevent oversubscription from NumPy/SciPy)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

    # Process logs sequentially; parallelization happens inside compute_metrics_for_samples
    results_batch = []
    successful_count = 0
    failed_count = 0
    total_split_results = 0

    for i, task in enumerate(tasks_to_process, start=1):
        print(f"  [{i}/{len(tasks_to_process)}] Processing {task.log_id}...")
        try:
            result = process_single_task(task, n_jobs=n_jobs)
            if result is not None:
                results_batch.append((task, result))
                successful_count += 1
                total_split_results += len(result)

                # Write batch when buffer is full
                if len(results_batch) >= BATCH_SIZE:
                    write_batch_results(results_batch)
                    print(
                        f"    Written batch of {len(results_batch)} logs "
                        f"({total_split_results} split/window combinations)..."
                    )
                    results_batch = []
            else:
                failed_count += 1
        except Exception as e:
            print(f"    Error processing {task.log_id}: {e}")
            failed_count += 1

    # Write remaining results
    if results_batch:
        write_batch_results(results_batch)
        print(f"    Written final batch of {len(results_batch)} logs...")

    print(
        f"\nProcessing complete: {successful_count} logs successful ({total_split_results} split/window combinations), {failed_count} failed"
    )

    # Aggregate final results
    print("\nAggregating final results...")
    combined_per_sample_df, combined_aggregate_df = aggregate_final_results()

    # Save final results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not combined_per_sample_df.empty:
        per_sample_path = OUTPUT_DIR / "per_sample_metrics.csv"
        combined_per_sample_df.to_csv(per_sample_path, index=False)
        print(f"\nSaved per-sample metrics to: {per_sample_path}")
    else:
        print("\nNo per-sample metrics to save")

    if not combined_aggregate_df.empty:
        aggregate_path = OUTPUT_DIR / "aggregate_analysis.csv"
        combined_aggregate_df.to_csv(aggregate_path, index=False)
        print(f"Saved aggregate analysis to: {aggregate_path}")
    else:
        print("No aggregate analysis to save")

    # Clean up intermediary folder to ensure re-computation on next run
    if INTERMEDIARY_DIR.exists():
        shutil.rmtree(INTERMEDIARY_DIR)
        print(f"Deleted intermediary folder: {INTERMEDIARY_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mine complexity values from synthetic logs with sudden drifts."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers. Default: number of CPU cores.",
    )
    args = parser.parse_args()
    main(n_jobs=args.n_jobs)
