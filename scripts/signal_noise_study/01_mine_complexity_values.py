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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


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


def load_event_log(task: ProcessingTask) -> Optional[EventLog]:
    """
    Load an event log from disk (I/O bound operation).

    This function is designed to run in a background thread for prefetching.

    Parameters
    ----------
    task
        The processing task containing the log file path.

    Returns
    -------
    Optional[EventLog]
        The loaded event log, or None if loading failed.
    """
    try:
        return xes_importer.apply(str(task.log_file_path))
    except Exception as e:
        print(f"    Error loading {task.log_id}: {e}")
        return None


def process_loaded_log(
    task: ProcessingTask,
    event_log: EventLog,
    n_jobs: int = 1,
) -> Optional[List[SplitResult]]:
    """
    Process an already-loaded event log (CPU bound operation).

    Batches all window samples across splits and window sizes, then processes
    them in a single parallel call to maximize CPU utilization.

    Parameters
    ----------
    task
        The processing task metadata.
    event_log
        The pre-loaded event log.
    n_jobs
        Number of parallel workers for computing metrics across window samples.

    Returns
    -------
    Optional[List[SplitResult]]
        List of results for each split/window_size combination if successful, None if failed.
    """
    try:
        # Check if log has enough traces
        if len(event_log) < DRIFT_POINT_IN_LOGS:
            print(
                f"    Warning: Log {task.log_id} has only {len(event_log)} traces, "
                f"need at least {DRIFT_POINT_IN_LOGS}, skipping..."
            )
            return None

        # Split the log into pre-drift and post-drift parts
        split_logs = split_event_log(event_log, DRIFT_POINT_IN_LOGS)
        pre_drift_log = split_logs["pre_drift"]
        post_drift_log = split_logs["post_drift"]

        # Set up pipeline components
        population_extractor = NaivePopulationExtractor()
        metric_adapters = [LocalMetricsAdapter(), VidgofMetricsAdapter()]
        bootstrap_sampler = None
        normalizers = None
        include_metrics = None
        sample_confidence_interval_extractor = SampleConfidenceIntervalExtractor(
            conf_level=0.95
        )
        sample_standard_deviation_extractor = SampleStandardDeviationExtractor(ddof=1)

        # =================================================================
        # Phase 1: Collect ALL samples across splits and window sizes
        # =================================================================
        all_samples = []
        sample_id_to_split: Dict[int, str] = {}  # global_sample_id -> split_name
        global_sample_id = 0

        for split_name, split_log in [
            ("pre_drift", pre_drift_log),
            ("post_drift", post_drift_log),
        ]:
            for window_size in WINDOW_SIZES:
                # Check if split has enough traces for this window size
                if len(split_log) < window_size:
                    print(
                        f"    Warning: Split {split_name} for log {task.log_id} "
                        f"has only {len(split_log)} traces, need at least {window_size}, skipping..."
                    )
                    continue

                # Collect samples and remap to globally unique IDs
                samples = list(
                    sampling_helper.sample_consecutive_trace_windows_with_replacement(
                        split_log,
                        sizes=[window_size],
                        samples_per_size=SAMPLES_PER_SIZE,
                        random_state=RANDOM_STATE,
                    )
                )

                for ws, _orig_sample_id, window in samples:
                    all_samples.append((ws, global_sample_id, window))
                    sample_id_to_split[global_sample_id] = split_name
                    global_sample_id += 1

        if not all_samples:
            return None

        # =================================================================
        # Phase 2: Compute ALL metrics in ONE parallel call
        # =================================================================
        metrics_df = compute_metrics_for_samples(
            all_samples,
            population_extractor=population_extractor,
            metric_adapters=metric_adapters,
            bootstrap_sampler=bootstrap_sampler,
            normalizers=normalizers,
            include_metrics=include_metrics,
            n_jobs=n_jobs,
        )

        metrics_df = metrics_df.reset_index()

        # Add split_name based on the sample ID mapping
        metrics_df["split_name"] = metrics_df["Sample ID"].map(sample_id_to_split)
        metrics_df["log_id"] = task.log_id

        # =================================================================
        # Phase 3: Group by split/window_size and compute analysis for each
        # =================================================================
        results = []

        for (split_name, window_size), group_df in metrics_df.groupby(
            ["split_name", "Sample Size"]
        ):
            # Add window_size column for consistency with original output
            group_df = group_df.copy()
            group_df["window_size"] = window_size

            # Compute aggregates (mean values, CIs, std, correlations, plateau)
            analysis_df = compute_analysis_for_metrics(
                group_df,
                sample_confidence_interval_extractor=sample_confidence_interval_extractor,
                sample_standard_deviation_extractor=sample_standard_deviation_extractor,
                include_metrics=include_metrics,
            )

            analysis_df = analysis_df.reset_index()
            analysis_df["log_id"] = task.log_id
            analysis_df["split_name"] = split_name
            analysis_df["window_size"] = window_size

            results.append(
                SplitResult(
                    log_id=task.log_id,
                    split_name=str(split_name),
                    window_size=int(window_size),
                    metrics_df=group_df,
                    analysis_df=analysis_df,
                )
            )

        return results if results else None

    except Exception as e:
        print(f"    Error processing {task.log_id}: {e}")
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

    # Set BLAS safety (prevent oversubscription from NumPy/SciPy)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

    # Hybrid parallelism: process multiple logs concurrently, each with fewer workers
    # This overlaps sequential phases (sampling, analysis) of different logs
    concurrent_logs = min(4, max(1, n_jobs // 4))  # 1-4 concurrent logs
    workers_per_log = max(1, n_jobs // concurrent_logs)

    print(
        f"Processing {len(tasks_to_process)} logs with hybrid parallelism "
        f"({concurrent_logs} concurrent logs × {workers_per_log} workers each, "
        f"total: {concurrent_logs * workers_per_log} CPUs)..."
    )

    # Process logs with hybrid parallelism
    results_batch: List[Tuple[ProcessingTask, List[SplitResult]]] = []
    successful_count = 0
    failed_count = 0
    total_split_results = 0
    total_tasks = len(tasks_to_process)

    # Timing tracking
    start_time = time.time()
    log_times: List[float] = []

    def load_and_process_log(
        task: ProcessingTask, workers: int
    ) -> Tuple[ProcessingTask, Optional[List[SplitResult]], float]:
        """Load and process a single log, returning task, result, and elapsed time."""
        log_start = time.time()
        event_log = load_event_log(task)
        if event_log is None:
            return task, None, time.time() - log_start
        result = process_loaded_log(task, event_log, n_jobs=workers)
        return task, result, time.time() - log_start

    # Use ThreadPoolExecutor for concurrent log processing
    # (each log's inner parallelism uses ProcessPoolExecutor)
    with ThreadPoolExecutor(max_workers=concurrent_logs) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(load_and_process_log, task, workers_per_log): i
            for i, task in enumerate(tasks_to_process)
        }

        # Process results as they complete
        completed_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            task = tasks_to_process[idx]
            completed_count += 1

            try:
                task, result, elapsed = future.result()
                log_times.append(elapsed)

                # Progress info with timing
                if len(log_times) > 1:
                    avg_time = sum(log_times) / len(log_times)
                    remaining = total_tasks - completed_count
                    eta_seconds = (
                        avg_time * remaining / concurrent_logs
                    )  # Adjust for concurrency
                    eta_str = _format_duration(eta_seconds)
                    avg_str = _format_duration(avg_time)
                    print(
                        f"  [{completed_count}/{total_tasks}] Completed {task.log_id} "
                        f"in {_format_duration(elapsed)} (avg: {avg_str}/log, ETA: {eta_str})"
                    )
                else:
                    print(
                        f"  [{completed_count}/{total_tasks}] Completed {task.log_id} "
                        f"in {_format_duration(elapsed)}"
                    )

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

    total_elapsed = time.time() - start_time
    avg_time_str = (
        _format_duration(sum(log_times) / len(log_times)) if log_times else "N/A"
    )
    total_time_str = _format_duration(total_elapsed)

    print(
        f"\nProcessing complete: {successful_count} logs successful "
        f"({total_split_results} split/window combinations), {failed_count} failed"
    )
    print(f"Total time: {total_time_str}, avg: {avg_time_str}/log")

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
