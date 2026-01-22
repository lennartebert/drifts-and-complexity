"""Mine complexity values from synthetic logs with sudden drifts.

This script processes synthetic event logs that contain sudden drifts, splits them
at the drift point, and computes complexity metrics for different window sizes
on both pre-drift and post-drift segments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog

from utils import sampling_helper
from utils.complexity.metrics_adapters.local_metrics_adapter import LocalMetricsAdapter
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
SAMPLES_PER_SIZE = 10
RANDOM_STATE = 321

# Output directory
OUTPUT_DIR = Path("results") / "signal_noise_study"


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


def main() -> None:
    """Main execution function."""
    # Construct paths
    gen_info_file_path = PATH_TO_LOGS / GEN_INFO_FILE_NAME

    # Load generation info CSV
    print(f"Loading generation info from {gen_info_file_path}")
    generated_files_df = pd.read_csv(gen_info_file_path, sep=";", index_col="log_id")

    print(f"Found {len(generated_files_df)} logs to process")

    # Set up minimal pipeline components
    population_extractor = NaivePopulationExtractor()
    metric_adapters = [LocalMetricsAdapter()]
    bootstrap_sampler = None  # Skip bootstrap to minimize dependencies
    normalizers = None  # Skip normalization
    include_metrics = None  # Compute all metrics
    sample_confidence_interval_extractor = SampleConfidenceIntervalExtractor(
        conf_level=0.95
    )

    # Collect results
    all_per_sample_metrics = []
    all_aggregate_analysis = []

    # Process each log
    for log_id in generated_files_df.index:
        print(f"\nProcessing log: {log_id}")

        # Construct log file path
        log_file_path = PATH_TO_LOGS / log_id

        if not log_file_path.exists():
            print(f"  Warning: Log file not found: {log_file_path}, skipping...")
            continue

        # Load the event log
        print(f"  Loading event log from {log_file_path}")
        try:
            event_log = xes_importer.apply(str(log_file_path))
        except Exception as e:
            print(f"  Error loading log: {e}, skipping...")
            continue

        # Check if log has enough traces
        if len(event_log) < DRIFT_POINT_IN_LOGS:
            print(
                f"  Warning: Log has only {len(event_log)} traces, "
                f"need at least {DRIFT_POINT_IN_LOGS}, skipping..."
            )
            continue

        # Split the event log into two parts
        split_logs = split_event_log(event_log, DRIFT_POINT_IN_LOGS)

        # Process each split
        for split_name, split_log in split_logs.items():
            print(f"  Processing {split_name} split ({len(split_log)} traces)")

            # Process each window size
            for window_size in WINDOW_SIZES:
                print(f"    Processing window size: {window_size}")

                # Check if split has enough traces for this window size
                if len(split_log) < window_size:
                    print(
                        f"      Warning: Split has only {len(split_log)} traces, "
                        f"need at least {window_size}, skipping..."
                    )
                    continue

                # Perform sampling
                window_samples = (
                    sampling_helper.sample_consecutive_trace_windows_with_replacement(
                        split_log,
                        sizes=[window_size],  # Pass as list
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
                metrics_df["log_id"] = log_id
                metrics_df["split_name"] = split_name
                metrics_df["window_size"] = window_size

                # Append to collection
                all_per_sample_metrics.append(metrics_df)

                # Compute aggregates (mean values, CIs, correlations, plateau)
                analysis_df = compute_analysis_for_metrics(
                    metrics_df,
                    sample_confidence_interval_extractor=sample_confidence_interval_extractor,
                    include_metrics=include_metrics,
                )

                # Reset index to access columns
                analysis_df = analysis_df.reset_index()

                # Add log_id, split_name, and window_size columns
                analysis_df["log_id"] = log_id
                analysis_df["split_name"] = split_name
                analysis_df["window_size"] = window_size

                # Append to collection
                all_aggregate_analysis.append(analysis_df)

    # Combine all results
    print("\nCombining results...")
    if all_per_sample_metrics:
        combined_per_sample_df = pd.concat(all_per_sample_metrics, ignore_index=True)
    else:
        print("  Warning: No per-sample metrics collected")
        combined_per_sample_df = pd.DataFrame()

    if all_aggregate_analysis:
        combined_aggregate_df = pd.concat(all_aggregate_analysis, ignore_index=True)
    else:
        print("  Warning: No aggregate analysis collected")
        combined_aggregate_df = pd.DataFrame()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save results
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
