"""Standalone script to process ground truth drift information from CSV files.

This script reads drift ground truth information from a CSV file and converts it
to the format expected by the drift detection results CSV.
"""

import argparse
import ast
import gzip
import json
import shutil
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


def ensure_xes_gz_exists(xes_file_path: Path) -> bool:
    """Ensure XES file exists as .xes.gz, converting from .xes if needed.

    Parameters
    ----------
    xes_file_path
        Path to the expected .xes.gz file.

    Returns
    -------
    bool
        True if file exists (or was successfully converted), False otherwise.
    """
    if xes_file_path.exists():
        return True

    # Check if .xes version exists (without .gz)
    if xes_file_path.suffix == ".gz":
        xes_path_no_gz = xes_file_path.with_suffix("")
        if xes_path_no_gz.exists():
            print(f"Converting {xes_path_no_gz.name} to {xes_file_path.name}...")
            try:
                with (
                    xes_path_no_gz.open("rb") as f_in,
                    gzip.open(xes_file_path, "wb") as f_out,
                ):
                    shutil.copyfileobj(f_in, f_out)
                # Delete original .xes file after successful compression
                xes_path_no_gz.unlink()
                print(f"✓ Successfully converted to {xes_file_path.name}")
                return True
            except Exception as e:
                print(
                    f"ERROR: Failed to convert {xes_path_no_gz.name} to {xes_file_path.name}: {e}"
                )
                return False

    return False


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
    xes_file_name: str | None = None,
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
    xes_file_name
        Optional XES file name. If provided, skips data dictionary lookup.
        If None, attempts to look up from data dictionary using dataset_key.

    Returns
    -------
    pd.DataFrame
        DataFrame with drift detection results in expected format.
    """
    print(f"Processing ground truth drifts for dataset: {dataset_key}")
    print(f"Reading from: {drift_info_csv_path}")

    # Get XES file name from data dictionary using dataset key, or use provided one
    if xes_file_name is None:
        xes_file_name = get_xes_file_name_from_dataset_key(dataset_key)
        if xes_file_name is None:
            print(
                f"ERROR: Could not find XES file name for dataset key '{dataset_key}' in data dictionary."
            )
            return pd.DataFrame()

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


def get_dataset_key_from_xes_file_name(
    xes_file_name: str, drift_info_csv_path: Path | None = None
) -> str | None:
    """Get dataset key from data dictionary based on XES file name.

    Since the same XES file name can appear in multiple folders (no noise, 20pct, 40pct),
    this function returns all matching dataset keys. The caller should determine which
    one to use based on context.

    Parameters
    ----------
    xes_file_name
        Name of the XES file (e.g., 'log_1_1692952162.xes.gz').
    drift_info_csv_path
        Optional path to drift_info.csv file (not used for matching, kept for compatibility).

    Returns
    -------
    str | None
        Dataset key from data dictionary, or None if not found.
        Note: If multiple matches exist, returns the first one found.
        For proper matching, use get_all_dataset_keys_from_xes_file_name().
    """
    # Normalize XES file name
    normalized_xes_name = normalize_log_name(xes_file_name)

    # Load data dictionary
    data_dictionary = helpers.load_data_dictionary(
        constants.get_data_dictionary_path(),
        get_real=True,
        get_synthetic=True,
    )

    # Find all dataset keys where path ends with the XES file name
    matches = []
    for dataset_key, dataset_info in data_dictionary.items():
        path = dataset_info.get("path", "")
        if path.endswith(normalized_xes_name):
            matches.append((dataset_key, path))

    if len(matches) == 1:
        return matches[0][0]
    elif len(matches) > 1:
        # Return the first match (caller should handle multiple matches if needed)
        return matches[0][0]

    return None


def get_all_dataset_keys_from_xes_file_name(
    xes_file_name: str,
) -> List[str]:
    """Get all dataset keys from data dictionary that match an XES file name.

    Parameters
    ----------
    xes_file_name
        Name of the XES file (e.g., 'log_1_1692952162.xes.gz').

    Returns
    -------
    List[str]
        List of all matching dataset keys (e.g., ['KA25_1_0_S', 'KA25_1_20_S', 'KA25_1_40_S']).
    """
    # Normalize XES file name
    normalized_xes_name = normalize_log_name(xes_file_name)

    # Load data dictionary
    data_dictionary = helpers.load_data_dictionary(
        constants.get_data_dictionary_path(),
        get_real=True,
        get_synthetic=True,
    )

    # Find all dataset keys where path ends with the XES file name
    matches = []
    for dataset_key, dataset_info in data_dictionary.items():
        path = dataset_info.get("path", "")
        if path.endswith(normalized_xes_name):
            matches.append(dataset_key)

    return sorted(matches)


def process_all_logs_from_drift_info(
    drift_info_csv_path: Path,
    max_logs: int | None = None,
) -> Dict[str, Path]:
    """Process all logs in a drift_info.csv file and generate output CSVs.

    Parameters
    ----------
    drift_info_csv_path
        Path to the drift_info.csv file.
    max_logs
        Optional limit on number of logs to process (for testing).

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping dataset_key to output CSV path.
    """
    # Read the CSV to get all unique log names
    df = pd.read_csv(drift_info_csv_path)
    unique_log_names = df["log_name"].unique()

    # Normalize log names to .xes.gz format
    unique_log_names = [normalize_log_name(log) for log in unique_log_names]

    # Sort for consistent ordering
    unique_log_names = sorted(unique_log_names)

    # Filter to only logs that exist in file system (check all Kraus et al folders)
    # This allows processing logs 21-100 even if they're not in data dictionary yet
    # Also converts .xes to .xes.gz if needed
    valid_log_names = set()
    synthetic_base = PROJECT_ROOT / "data" / "synthetic"
    kraus_folders = [
        "Kraus et al no noise",
        "Kraus et al 20pct noise",
        "Kraus et al 40pct noise",
    ]
    for folder in kraus_folders:
        folder_path = synthetic_base / folder
        if folder_path.exists():
            # Get all .xes.gz files in this folder
            xes_gz_files = list(folder_path.glob("*.xes.gz"))
            for xes_file in xes_gz_files:
                valid_log_names.add(xes_file.name)

            # Check for .xes files (without .gz) and convert them
            xes_files = list(folder_path.glob("*.xes"))
            # Filter out files that already have .gz versions
            xes_files = [
                f for f in xes_files if not (f.parent / f"{f.name}.gz").exists()
            ]
            for xes_file in xes_files:
                xes_gz_path = xes_file.with_suffix(xes_file.suffix + ".gz")
                if ensure_xes_gz_exists(xes_gz_path):
                    valid_log_names.add(xes_gz_path.name)

    # Filter to only valid log names (that exist in file system)
    unique_log_names = [log for log in unique_log_names if log in valid_log_names]

    # Limit if specified
    if max_logs is not None:
        unique_log_names = unique_log_names[:max_logs]

    print(f"Processing {len(unique_log_names)} log file(s) from {drift_info_csv_path}")

    # Load data dictionary once
    data_dictionary = helpers.load_data_dictionary(
        constants.get_data_dictionary_path(),
        get_real=True,
        get_synthetic=True,
    )

    dataset_to_output_path: Dict[str, Path] = {}

    for log_name in unique_log_names:
        # Get all dataset keys that match this XES file name
        # (same log file appears in multiple noise level folders)
        all_dataset_keys = get_all_dataset_keys_from_xes_file_name(log_name)

        # If no dataset keys found in data dictionary, generate them from log file name
        # Pattern: log_21_1692952246.xes.gz -> KA25_21_0_S, KA25_21_20_S, KA25_21_40_S
        if not all_dataset_keys:
            # Extract log number from filename (e.g., "log_21_1692952246.xes.gz" -> 21)
            import re

            match = re.match(r"log_(\d+)_", log_name)
            if match:
                log_num = match.group(1)
                all_dataset_keys = [
                    f"KA25_{log_num}_0_S",
                    f"KA25_{log_num}_20_S",
                    f"KA25_{log_num}_40_S",
                ]
            else:
                print(
                    f"WARNING: Could not parse log number from '{log_name}'. Skipping."
                )
                continue

        # Process each matching dataset key
        for dataset_key in all_dataset_keys:
            # Determine path based on dataset key pattern
            # KA25_21_0_S -> "Kraus et al no noise"
            # KA25_21_20_S -> "Kraus et al 20pct noise"
            # KA25_21_40_S -> "Kraus et al 40pct noise"
            if dataset_key.endswith("_0_S"):
                folder_name = "Kraus et al no noise"
            elif dataset_key.endswith("_20_S"):
                folder_name = "Kraus et al 20pct noise"
            elif dataset_key.endswith("_40_S"):
                folder_name = "Kraus et al 40pct noise"
            else:
                # Fallback: try to get from data dictionary if it exists
                if dataset_key in data_dictionary:
                    dataset_path = data_dictionary[dataset_key].get("path", "")
                    if dataset_path:
                        xes_file_path = PROJECT_ROOT / dataset_path
                        output_dir = xes_file_path.parent
                        output_dir.mkdir(parents=True, exist_ok=True)
                        log_name_normalized = normalize_log_name(log_name)
                        log_name_base = log_name_normalized.replace(
                            ".xes.gz", ""
                        ).replace(".xes", "")
                        output_path = output_dir / f"ground_truth_{log_name_base}.csv"
                        print(f"\nProcessing {dataset_key} (log: {log_name})...")
                        result_df = main(drift_info_csv_path, dataset_key, output_path)
                        if not result_df.empty:
                            dataset_to_output_path[dataset_key] = output_path
                            print(f"✓ Generated ground truth for {dataset_key}")
                        else:
                            print(f"✗ No drift information found for {dataset_key}")
                continue

            # Construct path based on folder name
            dataset_path = f"data/synthetic/{folder_name}/{log_name}"
            xes_file_path = PROJECT_ROOT / dataset_path

            # Ensure file exists as .xes.gz (convert from .xes if needed)
            if not ensure_xes_gz_exists(xes_file_path):
                print(
                    f"WARNING: XES file not found for {dataset_key}: {xes_file_path}. Skipping."
                )
                continue

            # Determine output path
            output_dir = xes_file_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            log_name_normalized = normalize_log_name(log_name)
            log_name_base = log_name_normalized.replace(".xes.gz", "").replace(
                ".xes", ""
            )
            output_path = output_dir / f"ground_truth_{log_name_base}.csv"

            # Process this log
            print(f"\nProcessing {dataset_key} (log: {log_name})...")
            result_df = main(
                drift_info_csv_path, dataset_key, output_path, xes_file_name=log_name
            )

            if not result_df.empty:
                dataset_to_output_path[dataset_key] = output_path
                print(f"✓ Generated ground truth for {dataset_key}")
            else:
                print(f"✗ No drift information found for {dataset_key}")

    return dataset_to_output_path


def update_data_dictionary_with_ground_truth(
    dataset_to_output_path: Dict[str, Path],
) -> None:
    """Update data_dictionary.json to add ground_truth field for synthetic datasets.
    Creates new entries if they don't exist.

    Parameters
    ----------
    dataset_to_output_path
        Dictionary mapping dataset_key to output CSV path (relative to project root).
    """
    data_dictionary_path = constants.get_data_dictionary_path()

    # Load existing data dictionary
    with open(data_dictionary_path, "r", encoding="utf-8") as f:
        data_dictionary = json.load(f)

    # Update synthetic datasets with ground_truth paths
    updated_count = 0
    created_count = 0
    for dataset_key, output_path in dataset_to_output_path.items():
        # Make path relative to project root
        relative_path = output_path.relative_to(PROJECT_ROOT)
        # Convert to forward slashes for JSON (Windows compatibility)
        relative_path_str = str(relative_path).replace("\\", "/")

        if dataset_key in data_dictionary:
            # Update existing entry
            data_dictionary[dataset_key]["ground_truth"] = relative_path_str
            updated_count += 1
            print(f"Updated {dataset_key} with ground_truth: {relative_path_str}")
        else:
            # Create new entry for KA25 datasets
            if dataset_key.startswith("KA25_"):
                # Parse dataset key: KA25_21_0_S -> log_num=21, noise=0
                import re

                match = re.match(r"KA25_(\d+)_(\d+)_S", dataset_key)
                if match:
                    log_num = match.group(1)
                    noise_level = match.group(2)

                    # Determine folder name and noise description
                    if noise_level == "0":
                        folder_name = "Kraus et al no noise"
                        noise_desc = "no noise"
                    elif noise_level == "20":
                        folder_name = "Kraus et al 20pct noise"
                        noise_desc = "20% noise"
                    elif noise_level == "40":
                        folder_name = "Kraus et al 40pct noise"
                        noise_desc = "40% noise"
                    else:
                        folder_name = f"Kraus et al {noise_level}pct noise"
                        noise_desc = f"{noise_level}% noise"

                    # Extract log name from ground truth path
                    # ground_truth_log_21_1692952246.csv -> log_21_1692952246.xes.gz
                    log_name_base = relative_path_str.split("ground_truth_")[
                        -1
                    ].replace(".csv", "")
                    xes_path = f"data/synthetic/{folder_name}/{log_name_base}.xes.gz"

                    # Create new entry (drift type will be generic, can be updated later if needed)
                    data_dictionary[dataset_key] = {
                        "name": f"Synthetic data by Kraus and van der Aa (2025), log {log_num}, {noise_desc}",
                        "short_name": dataset_key,
                        "type": "synthetic",
                        "path": xes_path,
                        "ground_truth": relative_path_str,
                    }
                    created_count += 1
                    print(
                        f"Created {dataset_key} with ground_truth: {relative_path_str}"
                    )
                else:
                    print(f"WARNING: Could not parse dataset key '{dataset_key}'")
            else:
                print(
                    f"WARNING: Dataset key '{dataset_key}' not found in data dictionary and not a KA25 dataset"
                )

    # Save updated data dictionary
    with open(data_dictionary_path, "w", encoding="utf-8") as f:
        json.dump(data_dictionary, f, indent=2, ensure_ascii=False)

    print(
        f"\n✓ Updated {updated_count} entries and created {created_count} new entries in data_dictionary.json"
    )


def process_all_kraus_datasets(
    test_mode: bool = False, max_logs: int | None = None
) -> None:
    """Process all Kraus et al datasets and update data dictionary.

    Uses the drift_info.csv file in scripts/change_study/data_prep/ to process all logs,
    and updates the data dictionary with ground_truth paths.

    Parameters
    ----------
    test_mode
        If True, only process a small subset for testing.
    max_logs
        If set, limit the number of logs processed (for testing).
    """
    # Drift info CSV is now in scripts/change_study/data_prep/
    drift_info_path = (
        PROJECT_ROOT / "scripts" / "change_study" / "data_prep" / "drift_info.csv"
    )

    if not drift_info_path.exists():
        print(f"ERROR: drift_info.csv not found at {drift_info_path}")
        return

    print(f"\n{'='*80}")
    print(f"Processing drift_info.csv from: {drift_info_path}")
    if test_mode:
        max_logs = max_logs or 3
        print(f"TEST MODE: Processing up to {max_logs} logs")
    print(f"{'='*80}")

    # Process all logs in the drift_info.csv
    all_dataset_to_output_path = process_all_logs_from_drift_info(
        drift_info_path, max_logs=max_logs
    )

    # Update data dictionary with all ground_truth paths
    if all_dataset_to_output_path:
        print(f"\n{'='*80}")
        print("Updating data_dictionary.json with ground_truth paths")
        print(f"{'='*80}")
        update_data_dictionary_with_ground_truth(all_dataset_to_output_path)
        print(f"\n✓ Completed processing {len(all_dataset_to_output_path)} datasets")
    else:
        print("\n✗ No datasets were processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ground truth drift information from CSV files."
    )
    parser.add_argument(
        "--process_all",
        action="store_true",
        help="Process all logs in all Kraus et al folders and update data dictionary.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only a small subset (3 logs from one folder).",
    )
    parser.add_argument(
        "--max_logs",
        type=int,
        default=None,
        help="Maximum number of logs to process (for testing).",
    )
    parser.add_argument(
        "--drift_info_csv",
        type=str,
        default=None,
        help="Path to the drift_info.csv file containing ground truth data. "
        "Required if --process_all is not used.",
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        default=None,
        help="Dataset key from data dictionary (e.g., 'KA25_1_20_S'). "
        "Required if --process_all is not used. "
        "This is used to look up the XES file name from the data dictionary, "
        "which is then used to filter drift_info.csv. The dataset key will be "
        "used as the log_name field in the output CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output CSV file. If not provided, prints to stdout. "
        "Ignored if --process_all is used.",
    )

    args = parser.parse_args()

    if args.process_all:
        # Process all Kraus datasets
        process_all_kraus_datasets(test_mode=args.test, max_logs=args.max_logs)
    else:
        # Single dataset processing mode
        if args.drift_info_csv is None or args.dataset_key is None:
            parser.error(
                "--drift_info_csv and --dataset_key are required when --process_all is not used"
            )

        drift_info_path = Path(args.drift_info_csv)
        if not drift_info_path.exists():
            print(f"ERROR: Drift info CSV file not found: {drift_info_path}")
            exit(1)

        output_path = Path(args.output) if args.output else None

        # Workflow: dataset_key -> look up XES file name -> filter drift_info.csv -> use dataset_key as log_name in output
        main(drift_info_path, args.dataset_key, output_path)
