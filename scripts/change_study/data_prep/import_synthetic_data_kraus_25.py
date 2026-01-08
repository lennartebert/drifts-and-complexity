"""Import and prepare synthetic data from Kraus et al. (2025).

This script:
1. Unpacks zip files from plugins/drift_characterization/evaluation_paper/data_collection/datasets_evaluation
2. Converts .xes files to .xes.gz format
3. Adds missing concept:name attributes to events
4. Processes ground truth drift information from CSV
5. Updates data dictionary with all synthetic datasets
"""

import argparse
import ast
import gzip
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.util import xes_constants as xes

# Ensure project root is on sys.path so local imports work when run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import constants, helpers

# Paths
DRIFT_INFO_CSV_PATH = (
    PROJECT_ROOT
    / "plugins"
    / "drift_characterization"
    / "evaluation_paper"
    / "data_collection"
    / "datasets_evaluation"
    / "drift_info.csv"
)

ZIP_BASE_PATH = (
    PROJECT_ROOT
    / "plugins"
    / "drift_characterization"
    / "evaluation_paper"
    / "data_collection"
    / "datasets_evaluation"
)

SYNTHETIC_BASE_PATH = PROJECT_ROOT / "data" / "synthetic"

# Zip file mappings: (source_folder, target_folder)
ZIP_MAPPINGS = [
    ("without_noise", "Kraus et al 00pct noise"),
    ("with_noise_5", "Kraus et al 20pct noise"),
    ("with_noise_10", "Kraus et al 40pct noise"),
]


def unpack_zip_files() -> None:
    """Unpack zip files to appropriate folders if needed."""
    for source_folder, target_folder in ZIP_MAPPINGS:
        target_path = SYNTHETIC_BASE_PATH / target_folder
        target_path.mkdir(parents=True, exist_ok=True)

        # Skip if folder already has enough .gz files
        if len(list(target_path.glob("*.xes.gz"))) >= 90:
            print(f"Skipping {target_folder} - already has files")
            continue

        source_path = ZIP_BASE_PATH / source_folder
        if not source_path.exists():
            continue

        zip_files = sorted(source_path.glob("*.zip"))
        if not zip_files:
            continue

        print(f"Unpacking {len(zip_files)} zip file(s) to {target_folder}...")
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(target_path)


def convert_xes_to_xes_gz(folder_path: Path) -> None:
    """Convert all .xes files to .xes.gz in a folder."""
    xes_files = [
        f for f in folder_path.glob("*.xes") if not (f.parent / f"{f.name}.gz").exists()
    ]

    if not xes_files:
        return

    print(f"  Converting {len(xes_files)} .xes file(s) to .gz...")
    for xes_file in xes_files:
        xes_gz_path = xes_file.with_suffix(xes_file.suffix + ".gz")
        try:
            with open(xes_file, "rb") as xes_f, gzip.open(xes_gz_path, "wb") as gz_f:
                shutil.copyfileobj(xes_f, gz_f)
            xes_file.unlink()
        except Exception as e:
            print(f"    ERROR converting {xes_file.name}: {e}")


def add_concept_name_if_missing(xes_file_path: Path) -> bool:
    """Add concept:name attribute to events if missing. Returns True if file was modified."""
    if not xes_file_path.exists():
        return False

    concept_name_key = xes.DEFAULT_NAME_KEY
    is_gzipped = xes_file_path.name.endswith(".gz")

    # For .gz files, we need a temp .xes file to work with
    if is_gzipped:
        # Remove .gz to get base .xes name
        temp_xes_path = xes_file_path.with_suffix("")
    else:
        temp_xes_path = xes_file_path

    try:
        # Decompress if needed
        if is_gzipped:
            with (
                gzip.open(xes_file_path, "rb") as gz_file,
                open(temp_xes_path, "wb") as xes_file,
            ):
                shutil.copyfileobj(gz_file, xes_file)

        # Load the log
        log = xes_importer.apply(str(temp_xes_path))

        # Check and fix missing concept:name
        needs_fix = False
        fixed_count = 0
        for trace in log:
            for event in trace:
                if concept_name_key not in event:
                    needs_fix = True
                    fixed_count += 1
                    event[concept_name_key] = (
                        event.get("activity")
                        or event.get("Activity")
                        or event.get("name")
                        or event.get("Name")
                        or "UNKNOWN"
                    )

        if not needs_fix:
            # Clean up temp file if we created one
            if is_gzipped and temp_xes_path.exists():
                temp_xes_path.unlink()
            return False

        # Save fixed log to temp .xes file
        xes_exporter.apply(log, str(temp_xes_path))

        # Compress to .gz (always create .gz, even if original wasn't)
        final_gz_path = temp_xes_path.with_suffix(".xes.gz")
        with open(temp_xes_path, "rb") as xes_f, gzip.open(final_gz_path, "wb") as gz_f:
            shutil.copyfileobj(xes_f, gz_f)

        # Clean up temp .xes file
        if temp_xes_path.exists():
            temp_xes_path.unlink()

        # Replace original file with fixed version
        if is_gzipped:
            # Replace the original .gz file
            xes_file_path.unlink()
            final_gz_path.rename(xes_file_path)
        else:
            # Original was .xes, replace it with .gz version
            xes_file_path.unlink()
            final_gz_path.rename(xes_file_path)

        print(
            f"    Fixed {fixed_count} events missing concept:name in {xes_file_path.name}"
        )
        return True
    except Exception as e:
        print(f"    ERROR processing {xes_file_path.name}: {e}")
        import traceback

        traceback.print_exc()
        # Clean up temp files
        if is_gzipped and temp_xes_path.exists() and temp_xes_path != xes_file_path:
            temp_xes_path.unlink()
        return False


def verify_concept_name_exists(xes_file_path: Path) -> bool:
    """Verify that all events in the file have concept:name. Returns True if all events have it."""
    if not xes_file_path.exists():
        return False

    concept_name_key = xes.DEFAULT_NAME_KEY
    is_gzipped = xes_file_path.name.endswith(".gz")

    try:
        if is_gzipped:
            temp_xes_path = xes_file_path.with_suffix("")
            with (
                gzip.open(xes_file_path, "rb") as gz_file,
                open(temp_xes_path, "wb") as xes_file,
            ):
                shutil.copyfileobj(gz_file, xes_file)
            log = xes_importer.apply(str(temp_xes_path))
            temp_xes_path.unlink()
        else:
            log = xes_importer.apply(str(xes_file_path))

        # Check all events
        for trace in log:
            for event in trace:
                if concept_name_key not in event:
                    return False
        return True
    except Exception:
        return False


def normalize_log_name(log_name: str) -> str:
    """Normalize log name to match CSV format (.xes, not .xes.gz)."""
    if log_name.endswith(".xes.gz"):
        return log_name[:-3]
    if log_name.endswith(".xes"):
        return log_name
    return log_name + ".xes"


def parse_trace_index(value: str) -> List[int]:
    """Parse trace index string like '[1086, 1956]' or '[1303]' into list of integers."""
    parsed = ast.literal_eval(value)
    return [parsed] if isinstance(parsed, int) else [int(x) for x in parsed]


def extract_drift_info_from_csv(
    drift_info_csv_path: Path, xes_file_name: str
) -> Dict[str, Dict[str, Any]]:
    """Extract drift information from ground truth CSV for a specific log."""
    df = pd.read_csv(drift_info_csv_path)
    normalized_log_name = normalize_log_name(xes_file_name)
    log_df = df[df["log_name"] == normalized_log_name].copy()

    if log_df.empty:
        return {}

    filtered_df = log_df[
        log_df["drift_sub_attribute"].isin(["change_type", "change_trace_index"])
    ].copy()
    if filtered_df.empty:
        return {}

    drift_dict: Dict[str, Dict[str, Any]] = {}
    for _, row in filtered_df.iterrows():
        key = f"{row['drift_or_noise_id']}_{row['drift_attribute']}"
        if key not in drift_dict:
            drift_dict[key] = {
                "drift_or_noise_id": row["drift_or_noise_id"],
                "drift_attribute": row["drift_attribute"],
            }
        if row["drift_sub_attribute"] == "change_type":
            drift_dict[key]["change_type"] = row["value"]
        elif row["drift_sub_attribute"] == "change_trace_index":
            drift_dict[key]["change_trace_index"] = row["value"]

    # Extract additional information
    for key, info in drift_dict.items():
        drift_or_noise_id = info["drift_or_noise_id"]
        drift_attribute = info["drift_attribute"]
        sub_df = log_df[
            (log_df["drift_or_noise_id"] == drift_or_noise_id)
            & (log_df["drift_attribute"] == drift_attribute)
        ]

        for attr in [
            "change_start",
            "change_end",
            "drift_id",
            "drift_type",
            "activities_deleted",
            "activities_added",
            "activities_moved",
        ]:
            rows = sub_df[sub_df["drift_sub_attribute"] == attr]
            if not rows.empty:
                info[attr] = rows.iloc[0]["value"]

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
    """Process gradual drifts by splitting them into gradual_start and gradual_end."""
    processed_dict: Dict[str, Dict[str, Any]] = {}
    change_id_counter = 1

    for key, info in drift_dict.items():
        change_type = info.get("change_type")
        change_trace_index_str = info.get("change_trace_index")

        if not change_type or not change_trace_index_str:
            continue

        trace_indices = parse_trace_index(change_trace_index_str)

        if change_type == "gradual":
            if len(trace_indices) != 2:
                continue
            cp1, cp2 = trace_indices
            for suffix, cp in [("_start", cp1), ("_end", cp2)]:
                processed_dict[f"{key}{suffix}"] = {
                    "change_type": f"gradual{suffix}",
                    "change_trace_index": cp,
                    "change_start": info.get("change_start"),
                    "change_end": info.get("change_end"),
                    "drift_id": info.get("drift_id"),
                    "drift_type": info.get("drift_type"),
                    "process_change_id": info.get("process_change_id"),
                    "activities_deleted": info.get("activities_deleted"),
                    "activities_added": info.get("activities_added"),
                    "activities_moved": info.get("activities_moved"),
                    "change_id": change_id_counter,
                }
                change_id_counter += 1
        else:
            if len(trace_indices) != 1:
                continue
            processed_dict[key] = {
                "change_type": change_type,
                "change_trace_index": trace_indices[0],
                "change_start": info.get("change_start"),
                "change_end": info.get("change_end"),
                "drift_id": info.get("drift_id"),
                "drift_type": info.get("drift_type"),
                "process_change_id": info.get("process_change_id"),
                "activities_deleted": info.get("activities_deleted"),
                "activities_added": info.get("activities_added"),
                "activities_moved": info.get("activities_moved"),
                "change_id": change_id_counter,
            }
            change_id_counter += 1

    return processed_dict


def convert_to_drift_detection_format(
    processed_dict: Dict[str, Dict[str, Any]], dataset_key: str
) -> pd.DataFrame:
    """Convert processed drift dictionary to drift detection CSV format."""
    if not processed_dict:
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
                "activities_deleted",
                "activities_added",
                "activities_moved",
                "config_method",
                "config_constant_alpha",
                "config_similarity",
                "config_incremental",
                "config_recurring",
                "config_steps",
            ]
        )

    records = []
    for info in sorted(processed_dict.values(), key=lambda x: x["change_id"]):
        change_type = info["change_type"]
        change_moment = (
            info.get("change_end", "")
            if change_type == "gradual_end"
            else info.get("change_start", "")
        )

        records.append(
            {
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
                "calc_change_type": change_type,
                "calc_drift_type": info.get("drift_type", ""),
                "activities_deleted": info.get("activities_deleted", ""),
                "activities_added": info.get("activities_added", ""),
                "activities_moved": info.get("activities_moved", ""),
                "config_method": "",
                "config_constant_alpha": "",
                "config_similarity": "",
                "config_incremental": "",
                "config_recurring": "",
                "config_steps": "",
            }
        )

    return pd.DataFrame(records)


def process_all_logs_from_drift_info(
    drift_info_csv_path: Path, max_logs: int | None = None
) -> Dict[str, Path]:
    """Process all logs in a drift_info.csv file and generate output CSVs."""
    df = pd.read_csv(drift_info_csv_path)
    unique_log_names = sorted(
        [normalize_log_name(log) for log in df["log_name"].unique()]
    )

    # Get valid log names from file system
    valid_log_names = set()
    for _, target_folder in ZIP_MAPPINGS:
        folder_path = SYNTHETIC_BASE_PATH / target_folder
        if folder_path.exists():
            for xes_gz_file in folder_path.glob("*.xes.gz"):
                valid_log_names.add(normalize_log_name(xes_gz_file.name))

    unique_log_names = [log for log in unique_log_names if log in valid_log_names]
    if max_logs is not None:
        unique_log_names = unique_log_names[:max_logs]

    print(f"Processing {len(unique_log_names)} log file(s)")

    dataset_to_output_path: Dict[str, Path] = {}

    for log_name in unique_log_names:
        # Generate dataset keys from log name
        match = re.match(r"log_(\d+)_", log_name)
        if not match:
            continue

        log_num = int(match.group(1))
        all_dataset_keys = [
            f"KA25_{log_num:03d}_00_S",
            f"KA25_{log_num:03d}_20_S",
            f"KA25_{log_num:03d}_40_S",
        ]

        # Map noise level to folder
        noise_to_folder = {
            "00": "Kraus et al 00pct noise",
            "20": "Kraus et al 20pct noise",
            "40": "Kraus et al 40pct noise",
        }

        for dataset_key in all_dataset_keys:
            noise_level = dataset_key.split("_")[2]
            folder_name = noise_to_folder.get(noise_level)
            if not folder_name:
                continue

            xes_file_path = PROJECT_ROOT / f"data/synthetic/{folder_name}/{log_name}.gz"
            if not xes_file_path.exists():
                continue

            output_dir = xes_file_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            log_name_base = log_name.replace(".xes", "")
            output_path = output_dir / f"ground_truth_{log_name_base}.csv"

            # Process ground truth
            drift_dict = extract_drift_info_from_csv(drift_info_csv_path, log_name)
            if not drift_dict:
                continue

            processed_dict = process_gradual_drifts(drift_dict)
            result_df = convert_to_drift_detection_format(processed_dict, dataset_key)

            if not result_df.empty:
                result_df.to_csv(output_path, index=False)
                dataset_to_output_path[dataset_key] = output_path
                print(f"Generated ground truth for {dataset_key}")

    return dataset_to_output_path


def update_data_dictionary_with_ground_truth(
    dataset_to_output_path: Dict[str, Path],
) -> None:
    """Update data dictionary with ground truth paths. Creates new entries if they don't exist."""
    data_dictionary_path = constants.get_data_dictionary_path()

    with open(data_dictionary_path, "r", encoding="utf-8") as f:
        data_dictionary = json.load(f)

    updated_count = 0
    created_count = 0

    for dataset_key, output_path in dataset_to_output_path.items():
        relative_path_str = str(output_path.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        )

        if dataset_key in data_dictionary:
            data_dictionary[dataset_key]["ground_truth"] = relative_path_str
            updated_count += 1
        elif dataset_key.startswith("KA25_"):
            match = re.match(r"KA25_(\d+)_(\d+)_S", dataset_key)
            if match:
                log_num = int(match.group(1))
                noise_level = match.group(2)

                noise_to_folder = {
                    "00": "Kraus et al 00pct noise",
                    "20": "Kraus et al 20pct noise",
                    "40": "Kraus et al 40pct noise",
                }
                noise_to_desc = {"00": "no noise", "20": "20% noise", "40": "40% noise"}

                folder_name = noise_to_folder.get(
                    noise_level, f"Kraus et al {noise_level}pct noise"
                )
                noise_desc = noise_to_desc.get(noise_level, f"{noise_level}% noise")

                log_name_base = relative_path_str.split("ground_truth_")[-1].replace(
                    ".csv", ""
                )
                xes_path = f"data/synthetic/{folder_name}/{log_name_base}.xes.gz"

                data_dictionary[dataset_key] = {
                    "name": f"Synthetic data by Kraus and van der Aa (2025), log {log_num}, {noise_desc}",
                    "short_name": dataset_key,
                    "type": "synthetic",
                    "path": xes_path,
                    "ground_truth": relative_path_str,
                }
                created_count += 1

    with open(data_dictionary_path, "w", encoding="utf-8") as f:
        json.dump(data_dictionary, f, indent=2, ensure_ascii=False)

    print(
        f"\nUpdated {updated_count} entries and created {created_count} new entries in data_dictionary.json"
    )


def process_all_kraus_datasets(
    test_mode: bool = False, max_logs: int | None = None, start_from_step: int = 1
) -> None:
    """Process all Kraus datasets: unpack, convert, fix, and generate ground truth.

    Parameters
    ----------
    test_mode
        If True, use simplified configuration for faster testing.
    max_logs
        Maximum number of logs to process (None = all).
    start_from_step
        Step number to start from (1-4). Steps before this will be skipped.
    """
    print("=== Importing Synthetic Data from Kraus et al. (2025) ===\n")

    if start_from_step > 1:
        print(
            f"Starting from step {start_from_step} (skipping steps 1-{start_from_step-1})\n"
        )

    # Step 1: Unpack .xes files from zip archives
    if start_from_step <= 1:
        print("Step 1: Unpacking .xes files from zip archives...")
        unpack_zip_files()
    else:
        print("Step 1: Skipped (already completed)")

    # Step 2: Convert .xes to .xes.gz
    if start_from_step <= 2:
        print("\nStep 2: Converting .xes to .xes.gz...")
        for _, target_folder in ZIP_MAPPINGS:
            folder_path = SYNTHETIC_BASE_PATH / target_folder
            if folder_path.exists():
                convert_xes_to_xes_gz(folder_path)
    else:
        print("\nStep 2: Skipped (already completed)")

    # Step 3: Ensure concept:name exists in all .xes.gz files
    if start_from_step <= 3:
        print("\nStep 3: Ensuring concept:name exists in all .xes.gz files...")
        for _, target_folder in ZIP_MAPPINGS:
            folder_path = SYNTHETIC_BASE_PATH / target_folder
            if folder_path.exists():
                xes_gz_files = list(folder_path.glob("*.xes.gz"))
                if xes_gz_files:
                    print(
                        f"  Checking {len(xes_gz_files)} .gz files in {target_folder}..."
                    )
                    fixed_count = 0
                    error_count = 0
                    for f in xes_gz_files:
                        try:
                            if add_concept_name_if_missing(f):
                                fixed_count += 1
                            # Verify the file is now correct
                            if not verify_concept_name_exists(f):
                                print(
                                    f"    WARNING: {f.name} still has missing concept:name after fix attempt"
                                )
                                error_count += 1
                        except Exception as e:
                            print(f"    ERROR processing {f.name}: {e}")
                            error_count += 1

                    if error_count > 0:
                        print(f"    WARNING: {error_count} file(s) had errors")
                    if fixed_count > 0:
                        print(f"    Fixed concept:name in {fixed_count} file(s)")
                    else:
                        print(f"    All files already have concept:name")
    else:
        print("\nStep 3: Skipped (already completed)")

    # Step 4: Process ground truth and update data dictionary
    if start_from_step <= 4:
        print(
            "\nStep 4: Processing ground truth drifts and updating data dictionary..."
        )
        if not DRIFT_INFO_CSV_PATH.exists():
            print(f"ERROR: Drift info CSV not found: {DRIFT_INFO_CSV_PATH}")
            return

        dataset_to_output_path = process_all_logs_from_drift_info(
            DRIFT_INFO_CSV_PATH, max_logs=max_logs
        )
        update_data_dictionary_with_ground_truth(dataset_to_output_path)
    else:
        print("\nStep 4: Skipped (already completed)")

    print("\n=== Import complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import and prepare synthetic data from Kraus et al. (2025)."
    )
    parser.add_argument(
        "--process-all", action="store_true", help="Process all Kraus datasets."
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: limit processing."
    )
    parser.add_argument(
        "--max-logs", type=int, default=None, help="Maximum number of logs to process."
    )
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Step number to start from (1=unpack, 2=convert, 3=fix concept:name, 4=ground truth). Steps before this will be skipped.",
    )

    args = parser.parse_args()

    if args.process_all:
        process_all_kraus_datasets(
            test_mode=args.test,
            max_logs=args.max_logs,
            start_from_step=args.start_from_step,
        )
    else:
        parser.print_help()
        print("\nUse --process-all to import all synthetic datasets.")
