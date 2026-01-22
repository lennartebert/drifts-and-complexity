"""Preferential attachment analysis script.

Can work with event logs (extracts attachments) or pre-computed attachment data.
Performs PA analysis including graph generation.
This script is node-type-agnostic - it works with any attachment data.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from pm4py.objects.log.obj import EventLog

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rfc_study.generate_rfc_graphs import load_event_log
from scripts.rfc_study.pa_measurement import run_pa_analysis_from_csv
from utils import constants, helpers


def extract_trace_attachments(event_log: EventLog) -> pd.DataFrame:
    """Extract trace attachments from an event log.

    Traces are sorted by their completion timestamp (attachment_time), and attachment indices
    are assigned based on this sorted order (smallest to highest).

    Traces without timestamps are skipped.

    Args:
        event_log: PM4Py EventLog object.

    Returns:
        DataFrame with columns ['attachment_time', 'attachment_index', 'node_id'] where:
        - attachment_time: UTF-8 timestamp string (ISO format) of trace completion
        - attachment_index: sequential trace number (0, 1, 2, ...) based on completion time order
        - node_id: trace variant identifier (tuple of activity names, serialized as string)
    """
    # Collect all traces with their completion timestamp
    trace_data = []

    for trace in event_log:
        if not trace or len(trace) == 0:
            continue

        # Get the completion timestamp (last event timestamp)
        last_event = trace[-1]
        completion_timestamp = last_event.get("time:timestamp")

        if completion_timestamp is None:
            # Skip traces without timestamps
            continue

        # Get the variant sequence (tuple of activity names)
        variant_sequence = tuple(ev.get("concept:name", "") for ev in trace)
        node_id_str = str(variant_sequence)

        trace_data.append(
            {
                "completion_timestamp": completion_timestamp,
                "node_id": node_id_str,
            }
        )

    # Sort by completion timestamp (smallest to highest)
    trace_data.sort(key=lambda x: x["completion_timestamp"])

    # Create attachments with indices based on sorted order
    attachments = []
    for trace_counter, data in enumerate(trace_data):
        # Convert completion timestamp to UTF-8 string (ISO format)
        completion_ts = data["completion_timestamp"]
        if isinstance(completion_ts, datetime):
            timestamp_str = completion_ts.isoformat()
        else:
            timestamp_str = str(completion_ts)

        attachments.append(
            {
                "attachment_time": timestamp_str,
                "attachment_index": trace_counter,
                "node_id": data["node_id"],
            }
        )

    return pd.DataFrame(attachments)


def load_attachments(attachments_path: Path) -> pd.DataFrame:
    """Load pre-computed attachment data from CSV.

    Args:
        attachments_path: Path to CSV file with columns ['attachment_time', 'attachment_index', 'node_id'].

    Returns:
        DataFrame with attachment data, sorted by attachment_index.
    """
    df = pd.read_csv(attachments_path)

    # Validate required columns
    required_cols = ["attachment_time", "attachment_index", "node_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure attachment_index is numeric and sort by it
    df["attachment_index"] = pd.to_numeric(df["attachment_index"])
    df = df.sort_values("attachment_index").reset_index(drop=True)

    return df


def generate_analysis_name(
    first_t1_start: int,
    gap: int,
    step: int,
    delta_t: int,
    max_windows: Optional[int],
    unit: str,
) -> str:
    """Generate a short analysis name from parameters.

    Args:
        first_t1_start: First t1_start value
        gap: Gap between t0_end and t1_start
        step: Step size between windows
        delta_t: Length of measurement window
        max_windows: Maximum number of windows (None means unlimited)
        unit: Unit for parameters (shortened: i=indices, m=months, y=years)

    Returns:
        Analysis name string, e.g., "1_0_1_1_y_5w" or "1_0_1_1_y_allw" if max_windows is None
    """
    # Shorten unit names
    unit_short = {"indices": "i", "months": "m", "years": "y"}.get(unit, unit[0])

    # Handle None max_windows
    if max_windows is None:
        windows_str = "allw"
    else:
        windows_str = f"{max_windows}w"

    name_parts = [
        str(first_t1_start),
        str(gap),
        str(step),
        str(delta_t),
        unit_short,
        windows_str,
    ]
    return "_".join(name_parts)


def save_analysis_settings(
    settings_path: Path,
    first_t1_start: int,
    gap: int,
    step: int,
    delta_t: int,
    max_windows: int,
    unit: str,
) -> None:
    """Save analysis settings to a JSON file.

    Args:
        settings_path: Path to save settings file
        first_t1_start: First t1_start value
        gap: Gap between t0_end and t1_start
        step: Step size between windows
        delta_t: Length of measurement window
        max_windows: Maximum number of windows
        unit: Unit for parameters
    """
    settings = {
        "first_t1_start": first_t1_start,
        "gap": gap,
        "step": step,
        "delta_t": delta_t,
        "max_windows": max_windows,
        "unit": unit,
    }
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration data.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_dataset_config(
    dataset_name: str, config: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Get configuration for a specific dataset, merging with defaults.

    Args:
        dataset_name: Name of the dataset.
        config: Full configuration dictionary.
        defaults: Default configuration values.

    Returns:
        Merged configuration for the dataset.
    """
    dataset_config = defaults.copy()

    # Override with dataset-specific config if available
    datasets_config = config.get("datasets", {})
    if dataset_name in datasets_config:
        dataset_specific = datasets_config[dataset_name].copy()
        # Handle None/null values - convert string "None" to actual None
        if "max_windows" in dataset_specific:
            max_w = dataset_specific["max_windows"]
            if max_w == "None" or max_w is None:
                dataset_specific["max_windows"] = None
            else:
                dataset_specific["max_windows"] = int(max_w)
        dataset_config.update(dataset_specific)

    return dataset_config


def main() -> None:
    """Main function for PA analysis."""
    parser = argparse.ArgumentParser(
        description="Preferential attachment analysis - loads attachments from file if available, otherwise extracts from event logs"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names to process (as defined in data dictionary) - loads attachments from file if available, otherwise extracts from event logs",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PA analysis results (default: results/rfc_study/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file with dataset-specific settings",
    )
    parser.add_argument(
        "--force-attachment-computation",
        action="store_true",
        help="Force recomputation of attachments from event logs even if attachments.csv already exists",
    )

    # Window specification options (used as defaults if config not provided)
    window_group = parser.add_argument_group("Window specification")
    window_group.add_argument(
        "--first-t1-start",
        type=int,
        default=None,
        help="First t1_start value (start of first measurement window). Overridden by config if provided.",
    )
    window_group.add_argument(
        "--gap",
        type=int,
        default=None,
        help="Gap between t0_end and t1_start. Overridden by config if provided.",
    )
    window_group.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step size between consecutive t1_start values. Overridden by config if provided.",
    )
    window_group.add_argument(
        "--delta-t",
        type=int,
        default=None,
        help="Length of measurement window (delta_t). Overridden by config if provided.",
    )
    window_group.add_argument(
        "--max-windows",
        type=int,
        default=10,
        help="Maximum number of windows to analyze (default: 10). Overridden by config if provided.",
    )
    window_group.add_argument(
        "--units",
        type=str,
        default="indices",
        choices=["indices", "months", "years"],
        help="Unit for window parameters: indices (default), months, or years. Overridden by config if provided.",
    )

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            return
        config = load_config(config_path)

    # Get defaults from config or command-line args
    defaults = config.get("defaults", {})
    if args.first_t1_start is not None:
        defaults["first_t1_start"] = args.first_t1_start
    if args.gap is not None:
        defaults["gap"] = args.gap
    if args.step is not None:
        defaults["step"] = args.step
    if args.delta_t is not None:
        defaults["delta_t"] = args.delta_t
    if args.max_windows is not None:
        defaults["max_windows"] = args.max_windows
    if args.units is not None:
        defaults["units"] = args.units

    # Validate required parameters
    required_params = ["first_t1_start", "gap", "step", "delta_t"]
    missing_params = [p for p in required_params if p not in defaults]
    if missing_params:
        print(f"Error: Missing required parameters: {', '.join(missing_params)}")
        print("  Provide via --config YAML file or command-line arguments")
        return

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "rfc_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate analysis name (use dataset-specific config if available, but for now use defaults)
    # Note: Each dataset may have different max_windows, so we'll generate per-dataset names
    print(
        f"Using configuration from: {args.config if args.config else 'command-line arguments'}"
    )

    # Process datasets (load attachments from file if available, otherwise extract from event logs)
    if args.datasets:
        data_dict_path = constants.get_data_dictionary_path()
        data_dictionary = helpers.load_data_dictionary(
            data_dict_path,
            get_real=True,
            get_synthetic=True,
        )

        # Validate datasets
        invalid_datasets = [ds for ds in args.datasets if ds not in data_dictionary]
        if invalid_datasets:
            print(f"Error: Invalid dataset names: {invalid_datasets}")
            print(f"Available datasets: {sorted(data_dictionary.keys())}")
            sys.exit(1)

        print(f"Processing {len(args.datasets)} dataset(s)...")
        print(f"Output directory: {output_dir}")

        for dataset_name in args.datasets:
            print(f"\nProcessing {dataset_name}...")
            dataset_info = data_dictionary[dataset_name]
            log_path = PROJECT_ROOT / dataset_info["path"]

            if not log_path.exists():
                print(f"  Warning: Log file not found: {log_path}")
                continue

            # Create dataset-specific directory (for attachments.csv)
            dataset_dir = output_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Get dataset-specific configuration for analysis name
            dataset_config = get_dataset_config(dataset_name, config, defaults)

            # Generate dataset-specific analysis name
            dataset_analysis_name = generate_analysis_name(
                first_t1_start=dataset_config["first_t1_start"],
                gap=dataset_config["gap"],
                step=dataset_config["step"],
                delta_t=dataset_config["delta_t"],
                max_windows=dataset_config.get("max_windows"),
                unit=dataset_config.get("units", "indices"),
            )

            # Create analysis-specific directory (for results)
            analysis_dir = dataset_dir / dataset_analysis_name
            analysis_dir.mkdir(parents=True, exist_ok=True)

            # Save settings to analysis directory
            settings_path = analysis_dir / "settings.json"
            save_analysis_settings(
                settings_path=settings_path,
                first_t1_start=dataset_config["first_t1_start"],
                gap=dataset_config["gap"],
                step=dataset_config["step"],
                delta_t=dataset_config["delta_t"],
                max_windows=dataset_config.get("max_windows", 10),
                unit=dataset_config.get("units", "indices"),
            )
            print(
                f"  Saved settings: {dataset_name}/{dataset_analysis_name}/settings.json"
            )

            # Load or create attachments
            attachment_path = dataset_dir / "attachments.csv"
            attachments_df = None

            try:
                # By default, try to load from file if it exists (unless force flag is set)
                if not args.force_attachment_computation and attachment_path.exists():
                    print(
                        f"  Loading attachments from file: {dataset_name}/attachments.csv"
                    )
                    attachments_df = load_attachments(attachment_path)
                    print(f"  Loaded {len(attachments_df)} trace attachments")
                    print(f"  Unique variants: {attachments_df['node_id'].nunique()}")
                else:
                    # Extract attachments from event log
                    if args.force_attachment_computation and attachment_path.exists():
                        print(
                            f"  Force recomputing attachments (existing file will be overwritten)..."
                        )
                    print(f"  Loading log from {log_path}...")
                    event_log = load_event_log(log_path)
                    print(f"  Loaded {len(event_log)} traces")

                    # Extract attachments
                    attachments_df = extract_trace_attachments(event_log)
                    print(f"  Extracted {len(attachments_df)} trace attachments")
                    print(f"  Unique variants: {attachments_df['node_id'].nunique()}")

                    # Save to CSV in dataset folder
                    attachments_df.to_csv(attachment_path, index=False)
                    print(f"  Saved: {dataset_name}/attachments.csv")

                # Get dataset-specific configuration
                dataset_config = get_dataset_config(dataset_name, config, defaults)

                print(f"  Configuration:")
                print(f"    first_t1_start: {dataset_config['first_t1_start']}")
                print(f"    gap: {dataset_config['gap']}")
                print(f"    step: {dataset_config['step']}")
                print(f"    delta_t: {dataset_config['delta_t']}")
                max_windows_val = dataset_config.get("max_windows")
                print(
                    f"    max_windows: {max_windows_val if max_windows_val is not None else 'unlimited'}"
                )
                print(f"    units: {dataset_config.get('units', 'indices')}")

                # Save attachments to temporary CSV if needed (for run_pa_analysis_from_csv)
                # Note: attachments_df is already saved to attachment_path above

                # Run PA analysis using the main interface from pa_measurement
                run_pa_analysis_from_csv(
                    attachments_csv_path=attachment_path,
                    output_dir=analysis_dir,
                    first_t1_start=dataset_config["first_t1_start"],
                    gap=dataset_config["gap"],
                    step=dataset_config["step"],
                    delta_t=dataset_config["delta_t"],
                    max_windows=dataset_config.get("max_windows"),
                    unit=dataset_config.get("units", "indices"),
                    dataset_name=dataset_name,
                )

            except Exception as e:
                print(f"  Error processing {dataset_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(f"\nAll PA analysis outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
