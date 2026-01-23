"""Mine complexity values from synthetic logs with sudden drifts.

This script processes synthetic event logs that contain sudden drifts, splits them
at the drift point, and computes complexity metrics for different window sizes
on both pre-drift and post-drift segments.

Supports parallelization and resume capability.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog

from utils import sampling_helper
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
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

# Constants
PATH_TO_LOGS = Path("data") / "synthetic" / "sudden_drifts"
GEN_INFO_FILE_NAME = "generation_info.csv"
DRIFT_POINT_IN_LOGS = (
    1000  # First 1000 traces are one process version, second 1000 traces are the other
)

# Study specific constants
WINDOW_SIZES = [50, 100, 150, 200]
SAMPLES_PER_SIZE = 100
RANDOM_STATE = 321

# Output directory
OUTPUT_DIR = Path("results") / "signal_noise_study"
INTERMEDIARY_DIR = OUTPUT_DIR / "intermediary"
BATCH_SIZE = 10


@dataclass
class ProcessingTask:
    """Task definition for processing a single log/split/window_size combination."""

    log_id: str
    split_name: str
    window_size: int
    log_file_path: Path
    split_start_idx: int  # Start index for split (0 for pre_drift, DRIFT_POINT_IN_LOGS for post_drift)
    split_end_idx: int  # End index for split (DRIFT_POINT_IN_LOGS for pre_drift, None for post_drift)


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

    Parameters
    ----------
    generated_files_df
        DataFrame with log_id as index.

    Returns
    -------
    List[ProcessingTask]
        List of all tasks to process.
    """
    tasks = []

    for log_id in generated_files_df.index:
        log_file_path = PATH_TO_LOGS / log_id

        if not log_file_path.exists():
            print(f"  Warning: Log file not found: {log_file_path}, skipping...")
            continue

        # Create tasks for both splits and all window sizes
        for split_name in ["pre_drift", "post_drift"]:
            if split_name == "pre_drift":
                split_start_idx = 0
                split_end_idx = DRIFT_POINT_IN_LOGS
            else:  # post_drift
                split_start_idx = DRIFT_POINT_IN_LOGS
                split_end_idx = None  # To end of log

            for window_size in WINDOW_SIZES:
                task = ProcessingTask(
                    log_id=log_id,
                    split_name=split_name,
                    window_size=window_size,
                    log_file_path=log_file_path,
                    split_start_idx=split_start_idx,
                    split_end_idx=split_end_idx,
                )
                tasks.append(task)

    return tasks


def get_intermediary_file_paths(task: ProcessingTask) -> Tuple[Path, Path]:
    """
    Get paths for intermediary output files for a task.

    Parameters
    ----------
    task
        The processing task.

    Returns
    -------
    Tuple[Path, Path]
        Paths for per-sample metrics and analysis files.
    """
    # Sanitize log_id for filename (remove .xes.gz extension if present)
    log_id_safe = (
        task.log_id.replace(".xes.gz", "").replace(".xes", "").replace("/", "_")
    )
    base_name = f"{log_id_safe}_{task.split_name}_{task.window_size}"

    metrics_path = INTERMEDIARY_DIR / f"{base_name}.csv"
    analysis_path = INTERMEDIARY_DIR / f"{base_name}_analysis.csv"

    return metrics_path, analysis_path


def check_resume_status(tasks: List[ProcessingTask]) -> List[ProcessingTask]:
    """
    Check which tasks are already completed and filter them out.

    First checks if final files exist (complete run), then checks intermediary files.

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

    for task in tasks:
        metrics_path, analysis_path = get_intermediary_file_paths(task)

        if metrics_path.exists() and analysis_path.exists():
            # Check if files are non-empty
            try:
                metrics_df = pd.read_csv(metrics_path)
                analysis_df = pd.read_csv(analysis_path)
                if not metrics_df.empty and not analysis_df.empty:
                    completed_count += 1
                    continue
            except Exception:
                # If file is corrupted or empty, reprocess
                pass

        remaining_tasks.append(task)

    if completed_count > 0:
        print(
            f"Resuming: {completed_count} tasks already completed, "
            f"{len(remaining_tasks)} tasks remaining."
        )

    return remaining_tasks


def process_single_task(
    task: ProcessingTask,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Process a single task (worker function for parallel execution).

    Parameters
    ----------
    task
        The processing task to execute.

    Returns
    -------
    Optional[Tuple[pd.DataFrame, pd.DataFrame]]
        Tuple of (metrics_df, analysis_df) if successful, None if failed.
    """
    try:
        # Load the event log
        event_log = xes_importer.apply(str(task.log_file_path))

        # Check if log has enough traces
        if len(event_log) < DRIFT_POINT_IN_LOGS:
            print(
                f"  Warning: Log {task.log_id} has only {len(event_log)} traces, "
                f"need at least {DRIFT_POINT_IN_LOGS}, skipping..."
            )
            return None

        # Extract the split
        if task.split_end_idx is None:
            split_log = EventLog(event_log[task.split_start_idx :])
        else:
            split_log = EventLog(event_log[task.split_start_idx : task.split_end_idx])

        # Check if split has enough traces for this window size
        if len(split_log) < task.window_size:
            print(
                f"  Warning: Split {task.split_name} for log {task.log_id} "
                f"has only {len(split_log)} traces, need at least {task.window_size}, skipping..."
            )
            return None

        # Set up pipeline components (created in worker to avoid pickling issues)
        population_extractor = NaivePopulationExtractor()
        metric_adapters = [LocalMetricsAdapter()]
        bootstrap_sampler = None
        normalizers = None
        include_metrics = None
        sample_confidence_interval_extractor = SampleConfidenceIntervalExtractor(
            conf_level=0.95
        )

        # Perform sampling
        window_samples = (
            sampling_helper.sample_consecutive_trace_windows_with_replacement(
                split_log,
                sizes=[task.window_size],
                samples_per_size=SAMPLES_PER_SIZE,
                random_state=RANDOM_STATE,
            )
        )

        # Compute raw metrics
        metrics_df = compute_metrics_for_samples(
            window_samples,
            population_extractor=population_extractor,
            metric_adapters=metric_adapters,
            bootstrap_sampler=bootstrap_sampler,
            normalizers=normalizers,
            include_metrics=include_metrics,
        )

        # Reset index to access columns
        metrics_df = metrics_df.reset_index()

        # Add log_id, split_name, and window_size columns
        metrics_df["log_id"] = task.log_id
        metrics_df["split_name"] = task.split_name
        metrics_df["window_size"] = task.window_size

        # Compute aggregates (mean values, CIs, correlations, plateau)
        analysis_df = compute_analysis_for_metrics(
            metrics_df,
            sample_confidence_interval_extractor=sample_confidence_interval_extractor,
            include_metrics=include_metrics,
        )

        # Reset index to access columns
        analysis_df = analysis_df.reset_index()

        # Add log_id, split_name, and window_size columns
        analysis_df["log_id"] = task.log_id
        analysis_df["split_name"] = task.split_name
        analysis_df["window_size"] = task.window_size

        return (metrics_df, analysis_df)

    except Exception as e:
        print(
            f"Error processing task {task.log_id}/{task.split_name}/{task.window_size}: {e}"
        )
        return None


def write_batch_results(
    results_batch: List[Tuple[ProcessingTask, pd.DataFrame, pd.DataFrame]],
) -> None:
    """
    Write a batch of task results to intermediary files.

    Parameters
    ----------
    results_batch
        List of tuples (task, metrics_df, analysis_df) to write.
    """
    INTERMEDIARY_DIR.mkdir(parents=True, exist_ok=True)

    for task, metrics_df, analysis_df in results_batch:
        metrics_path, analysis_path = get_intermediary_file_paths(task)

        try:
            metrics_df.to_csv(metrics_path, index=False)
            analysis_df.to_csv(analysis_path, index=False)
        except Exception as e:
            print(
                f"Error writing results for task {task.log_id}/{task.split_name}/{task.window_size}: {e}"
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
    print(f"Generated {len(all_tasks)} tasks")

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

    print(f"Processing {len(tasks_to_process)} tasks in parallel...")

    # Process tasks in parallel and write results as they complete
    results_batch = []
    successful_count = 0
    failed_count = 0

    # Create a mapping from future to task for tracking
    task_future_map = {}

    # Set BLAS safety (prevent oversubscription)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

    # Use ProcessPoolExecutor to process results as they complete
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        for task in tasks_to_process:
            future = executor.submit(process_single_task, task)
            task_future_map[future] = task

        # Process results as they complete
        for future in as_completed(task_future_map):
            task = task_future_map[future]
            try:
                result = future.result()
                if result is not None:
                    metrics_df, analysis_df = result
                    results_batch.append((task, metrics_df, analysis_df))
                    successful_count += 1

                    # Write batch when buffer is full
                    if len(results_batch) >= BATCH_SIZE:
                        write_batch_results(results_batch)
                        print(
                            f"  Written batch of {len(results_batch)} results... ({successful_count}/{len(tasks_to_process)} completed)"
                        )
                        results_batch = []
                else:
                    failed_count += 1
            except Exception as e:
                print(
                    f"  Error getting result for task {task.log_id}/{task.split_name}/{task.window_size}: {e}"
                )
                failed_count += 1

    # Write remaining results
    if results_batch:
        write_batch_results(results_batch)
        print(f"  Written final batch of {len(results_batch)} results...")

    print(
        f"\nProcessing complete: {successful_count} successful, {failed_count} failed"
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

    print("\nDone!")


if __name__ == "__main__":
    main()
