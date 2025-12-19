"""Standalone script to process ground truth drift information from CSV files.

This script reads drift ground truth information from a CSV file and converts it
to the format expected by the drift detection results CSV.
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure project root is on sys.path so local imports work when run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import constants, helpers


def parse_trace_index(value: str) -> List[int]:
    """Parse trace index string like '[1086, 1956]' or '[1303]' into list of integers.

    Parameters
    ----------
    value
        String representation of a list of integers.

    Returns
    -------
    List[int]
        List of integers parsed from the string.
    """
    try:
        # Use ast.literal_eval to safely parse the string representation
        parsed = ast.literal_eval(value)
        if isinstance(parsed, int):
            return [parsed]
        elif isinstance(parsed, list):
            return [int(x) for x in parsed]
        else:
            raise ValueError(f"Unexpected type: {type(parsed)}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse trace index '{value}': {e}")


def normalize_log_name(log_name: str) -> str:
    """Normalize log name to match CSV format.

    The CSV uses .xes.gz extension, but the log_name parameter might be
    provided with just .xes. This function ensures we match the CSV format.

    Parameters
    ----------
    log_name
        Log name as provided (e.g., 'log_1_1692952162.xes' or 'log_1_1692952162.xes.gz').

    Returns
    -------
    str
        Normalized log name (with .gz if it ends with .xes).
    """
    if log_name.endswith(".xes") and not log_name.endswith(".xes.gz"):
        return log_name + ".gz"
    return log_name


def get_xes_file_name_from_dataset_key(dataset_key: str) -> str | None:
    """Get XES file name from data dictionary based on dataset key.

    Parameters
    ----------
    dataset_key
        Dataset key from data dictionary (e.g., 'KA25_1_20_S').

    Returns
    -------
    str | None
        XES file name (e.g., 'log_1_1692952162.xes.gz'), or None if not found.
    """
    # Load data dictionary
    data_dictionary = helpers.load_data_dictionary(
        constants.get_data_dictionary_path(),
        get_real=True,
        get_synthetic=True,
    )

    # Find dataset key and extract the XES file name from the path
    if dataset_key in data_dictionary:
        path = data_dictionary[dataset_key].get("path", "")
        if path:
            # Extract just the filename from the path
            return Path(path).name

    return None


def extract_drift_info_from_csv(
    drift_info_csv_path: Path, xes_file_name: str
) -> Dict[str, Dict[str, Any]]:
    """Extract drift information from ground truth CSV for a specific log.

    Parameters
    ----------
    drift_info_csv_path
        Path to the drift_info.csv file.
    xes_file_name
        Name of the XES log file to filter for (e.g., 'log_1_1692952162.xes.gz').

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary where keys are drift_or_noise_id concatenated with drift_attribute,
        and values contain change_type, change_trace_index, change_start, change_end, and drift_id.
    """
    # Read the CSV file
    df = pd.read_csv(drift_info_csv_path)

    # Normalize xes_file_name to match CSV format (CSV uses .xes.gz)
    normalized_log_name = normalize_log_name(xes_file_name)

    # Filter for the specific log_name
    log_df = df[df["log_name"] == normalized_log_name].copy()

    if log_df.empty:
        print(
            f"WARNING: No entries found for XES file '{normalized_log_name}' "
            f"(searched with normalized name from '{xes_file_name}')"
        )
        return {}

    # Filter for drift_sub_attribute = "change_type" or "change_trace_index"
    filtered_df = log_df[
        log_df["drift_sub_attribute"].isin(["change_type", "change_trace_index"])
    ].copy()

    if filtered_df.empty:
        print(
            f"WARNING: No change_type or change_trace_index entries found for log_name '{normalized_log_name}'"
        )
        return {}

    # Create dictionary with key = drift_or_noise_id + drift_attribute
    drift_dict: Dict[str, Dict[str, Any]] = {}

    for _, row in filtered_df.iterrows():
        key = f"{row['drift_or_noise_id']}_{row['drift_attribute']}"
        sub_attr = row["drift_sub_attribute"]
        value = row["value"]

        if key not in drift_dict:
            drift_dict[key] = {
                "drift_or_noise_id": row["drift_or_noise_id"],
                "drift_attribute": row["drift_attribute"],
            }

        if sub_attr == "change_type":
            drift_dict[key]["change_type"] = value
        elif sub_attr == "change_trace_index":
            drift_dict[key]["change_trace_index"] = value

    # Extract additional information: change_start, change_end, and drift_id
    for key, info in drift_dict.items():
        drift_or_noise_id = info["drift_or_noise_id"]
        drift_attribute = info["drift_attribute"]

        # Find change_start and change_end for this drift_attribute
        change_start_rows = log_df[
            (log_df["drift_or_noise_id"] == drift_or_noise_id)
            & (log_df["drift_attribute"] == drift_attribute)
            & (log_df["drift_sub_attribute"] == "change_start")
        ]
        change_end_rows = log_df[
            (log_df["drift_or_noise_id"] == drift_or_noise_id)
            & (log_df["drift_attribute"] == drift_attribute)
            & (log_df["drift_sub_attribute"] == "change_end")
        ]

        if not change_start_rows.empty:
            info["change_start"] = change_start_rows.iloc[0]["value"]
        if not change_end_rows.empty:
            info["change_end"] = change_end_rows.iloc[0]["value"]

        # Find drift_id for this drift_or_noise_id
        drift_id_rows = log_df[
            (log_df["drift_or_noise_id"] == drift_or_noise_id)
            & (log_df["drift_attribute"] == "drift_id")
        ]
        if not drift_id_rows.empty:
            info["drift_id"] = drift_id_rows.iloc[0]["value"]

        # Find drift_type for this drift_or_noise_id
        drift_type_rows = log_df[
            (log_df["drift_or_noise_id"] == drift_or_noise_id)
            & (log_df["drift_attribute"] == "drift_type")
        ]
        if not drift_type_rows.empty:
            info["drift_type"] = drift_type_rows.iloc[0]["value"]

        # Extract process_change_id from drift_attribute (e.g., "change_info_1" -> 1)
        if drift_attribute.startswith("change_info_"):
            try:
                info["process_change_id"] = int(drift_attribute.split("_")[-1])
            except ValueError:
                info["process_change_id"] = None
        else:
            info["process_change_id"] = None

    return drift_dict


def process_gradual_drifts(
    drift_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Process gradual drifts by splitting them into gradual_start and gradual_end.

    Parameters
    ----------
    drift_dict
        Dictionary with drift information.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with gradual drifts split into two entries.
    """
    processed_dict: Dict[str, Dict[str, Any]] = {}
    change_id_counter = 1

    for key, info in drift_dict.items():
        change_type = info.get("change_type")
        change_trace_index_str = info.get("change_trace_index")

        if not change_type or not change_trace_index_str:
            print(f"WARNING: Missing change_type or change_trace_index for key '{key}'")
            continue

        if change_type == "gradual":
            # Parse the trace index tuple
            trace_indices = parse_trace_index(change_trace_index_str)

            if len(trace_indices) != 2:
                print(
                    f"WARNING: Expected 2 trace indices for gradual drift, got {len(trace_indices)} for key '{key}'"
                )
                continue

            cp1, cp2 = trace_indices

            # Create gradual_start entry
            start_key = f"{key}_start"
            processed_dict[start_key] = {
                "change_type": "gradual_start",
                "change_trace_index": cp1,
                "change_start": info.get("change_start"),
                "change_end": info.get("change_end"),
                "drift_id": info.get("drift_id"),
                "drift_type": info.get("drift_type"),
                "process_change_id": info.get("process_change_id"),
                "original_key": key,
                "change_id": change_id_counter,
            }
            change_id_counter += 1

            # Create gradual_end entry
            end_key = f"{key}_end"
            processed_dict[end_key] = {
                "change_type": "gradual_end",
                "change_trace_index": cp2,
                "change_start": info.get("change_start"),
                "change_end": info.get("change_end"),
                "drift_id": info.get("drift_id"),
                "drift_type": info.get("drift_type"),
                "process_change_id": info.get("process_change_id"),
                "original_key": key,
                "change_id": change_id_counter,
            }
            change_id_counter += 1
        else:
            # For non-gradual drifts, use the trace index directly
            trace_indices = parse_trace_index(change_trace_index_str)

            if len(trace_indices) != 1:
                print(
                    f"WARNING: Expected 1 trace index for {change_type} drift, got {len(trace_indices)} for key '{key}'"
                )
                continue

            processed_dict[key] = {
                "change_type": change_type,
                "change_trace_index": trace_indices[0],
                "change_start": info.get("change_start"),
                "change_end": info.get("change_end"),
                "drift_id": info.get("drift_id"),
                "drift_type": info.get("drift_type"),
                "process_change_id": info.get("process_change_id"),
                "original_key": key,
                "change_id": change_id_counter,
            }
            change_id_counter += 1

    return processed_dict


def convert_to_drift_detection_format(
    processed_dict: Dict[str, Dict[str, Any]], dataset_key: str
) -> pd.DataFrame:
    """Convert processed drift dictionary to drift detection CSV format.

    Parameters
    ----------
    processed_dict
        Dictionary with processed drift information.
    dataset_key
        Dataset key from data dictionary (used as log_name in output CSV).

    Returns
    -------
    pd.DataFrame
        DataFrame in the format expected by drift detection results CSV.
    """
    records = []

    # Sort by change_id to ensure proper ordering
    sorted_items = sorted(processed_dict.items(), key=lambda x: x[1]["change_id"])

    for key, info in sorted_items:
        # Determine change_moment based on change_type
        change_type = info["change_type"]
        if change_type == "gradual_start":
            change_moment = info.get("change_start", "")
        elif change_type == "gradual_end":
            change_moment = info.get("change_end", "")
        else:
            # For sudden drifts and other types, use change_start
            change_moment = info.get("change_start", "")

        record = {
            "collection_name": "",
            "eval_noise_level": "",
            "eval_complexity": "",
            "log_size": "",
            "log_name": dataset_key,
            "calc_drift_id": info.get("drift_id", ""),
            "calc_process_change_id": info.get("process_change_id", ""),
            "calc_change_id": info["change_id"],
            "calc_change_index": info["change_trace_index"],
            "calc_change_moment": change_moment,
            "calc_change_type": info["change_type"],
            "calc_drift_type": info.get("drift_type", ""),
            "config_method": "",
            "config_constant_alpha": "",
            "config_similarity": "",
            "config_incremental": "",
            "config_recurring": "",
            "config_steps": "",
        }
        records.append(record)

    if not records:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "collection_name",
                "eval_noise_level",
                "eval_complexity",
                "log_size",
                "log_name",
                "calc_drift_id",
                "calc_process_change_id",
                "calc_change_id",
                "calc_change_index",
                "calc_change_moment",
                "calc_change_type",
                "calc_drift_type",
                "config_method",
                "config_constant_alpha",
                "config_similarity",
                "config_incremental",
                "config_recurring",
                "config_steps",
            ]
        )

    return pd.DataFrame(records)


def main(
    drift_info_csv_path: Path,
    dataset_key: str,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Main function to process ground truth drift information.

    Parameters
    ----------
    drift_info_csv_path
        Path to the drift_info.csv file.
    dataset_key
        Dataset key from data dictionary (e.g., 'KA25_1_20_S').
        This is used to look up the XES file name, which is then used to filter
        the drift_info.csv. The dataset key will be used as the log_name in the output CSV.
    output_path
        Optional path to save the output CSV. If None, prints to stdout.

    Returns
    -------
    pd.DataFrame
        DataFrame with drift detection results in expected format.
    """
    print(f"Processing ground truth drifts for dataset: {dataset_key}")
    print(f"Reading from: {drift_info_csv_path}")

    # Get XES file name from data dictionary using dataset key
    xes_file_name = get_xes_file_name_from_dataset_key(dataset_key)
    if xes_file_name is None:
        print(
            f"ERROR: Could not find XES file name for dataset key '{dataset_key}' in data dictionary."
        )
        return pd.DataFrame()
    else:
        print(f"Found XES file name: {xes_file_name}")

    # Step 1: Extract drift information from CSV (use XES file name to filter CSV)
    drift_dict = extract_drift_info_from_csv(drift_info_csv_path, xes_file_name)

    if not drift_dict:
        print("No drift information found.")
        return pd.DataFrame()

    print(f"Found {len(drift_dict)} drift entries")

    # Step 2: Process gradual drifts
    processed_dict = process_gradual_drifts(drift_dict)

    print(f"After processing gradual drifts: {len(processed_dict)} change points")

    # Step 3: Convert to drift detection format (use dataset_key as log_name)
    result_df = convert_to_drift_detection_format(processed_dict, dataset_key)

    # Save or print results
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        print("\nDrift detection results:")
        print(result_df.to_string(index=False))

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ground truth drift information from CSV files."
    )
    parser.add_argument(
        "--drift_info_csv",
        type=str,
        required=True,
        help="Path to the drift_info.csv file containing ground truth data.",
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        required=True,
        help="Dataset key from data dictionary (e.g., 'KA25_1_20_S'). "
        "This is used to look up the XES file name from the data dictionary, "
        "which is then used to filter drift_info.csv. The dataset key will be "
        "used as the log_name field in the output CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output CSV file. If not provided, prints to stdout.",
    )

    args = parser.parse_args()

    drift_info_path = Path(args.drift_info_csv)
    if not drift_info_path.exists():
        print(f"ERROR: Drift info CSV file not found: {drift_info_path}")
        exit(1)

    output_path = Path(args.output) if args.output else None

    # Workflow: dataset_key -> look up XES file name -> filter drift_info.csv -> use dataset_key as log_name in output
    main(drift_info_path, args.dataset_key, output_path)
