import argparse
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from utils import constants, helpers

import argparse
from pathlib import Path
from utils.windowing.loader import load_window_config
from utils.drift_io import (
    drift_info_to_dict, load_xes_log
)
from utils.complexity_assessors import (
    assess_complexity_via_change_point_split,
    assess_complexity_via_fixed_sized_windows,
    assess_complexity_via_window_comparison
)
from utils.plotting.complexity import plot_complexity_measures, plot_delta_measures


## UTILS

def clean_folder_except_gitkeep(folder: Path, delete: bool = False):
    if not folder.exists():
        return

    gitkeep_in_dir = False
    for item in folder.iterdir():
        if item.name != ".gitkeep":
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        else:
            gitkeep_in_dir = True
    
    if not gitkeep_in_dir:
        try:
            shutil.rmtree(folder)
        except Exception as e:
            # prin  t(f"Failed to delete folder {folder}: {e}")
            pass

def flatten_measurements(window_results):
    flat_records = []
    for w in window_results:
        flat_record = w.copy()
        del flat_record["measurements"]

        meas = w["measurements"].copy()

        # Prefix remaining measurement keys with "measure_"
        meas = {f"measure_{k}": v for k, v in meas.items()}

        flat_record.update(meas)
        flat_records.append(flat_record)

    return flat_records


## MAIN FUNCTIONS

def concept_drift_characterization(dataset_key, dataset_info):
    print(f"## Running concept drift characterization ##")
    local_dataset_path = Path(dataset_info["path"])
    target_dataset_filename = Path(f"{dataset_key}.xes.gz")
    drift_characterization_input_file_path = constants.DRIFT_CHARACTERIZATION_TEMP_INPUT_DIR / dataset_key / target_dataset_filename
    drift_characterization_output_dir_path = constants.DRIFT_CHARACTERIZATION_TEMP_OUTPUT_DIR / dataset_key

    # Clean input and output directories
    clean_folder_except_gitkeep(drift_characterization_input_file_path.parent)
    clean_folder_except_gitkeep(drift_characterization_output_dir_path)

    # Copy dataset to input
    drift_characterization_input_file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(local_dataset_path, drift_characterization_input_file_path)

    # Run concept drift characterization
    try:
        result = subprocess.run(
            ["python", str(constants.DRIFT_CHARACTERIZATION_SCRIPT),
             "--input_dir", str(drift_characterization_input_file_path.parent.relative_to(constants.DRIFT_CHARACTERIZATION_DIR)),
             "--output_dir", str(drift_characterization_output_dir_path.relative_to(constants.DRIFT_CHARACTERIZATION_DIR))],
            check=True,
            capture_output=True,
            text=True,
            cwd=constants.DRIFT_CHARACTERIZATION_DIR,
            encoding="utf-8"
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Subprocess failed!")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise
    
    # Copy output to results folder
    target_dir = constants.DRIFT_CHARACTERIZATION_RESULTS_DIR / dataset_key

    target_dir.mkdir(parents=True, exist_ok=True)

    results_target_file_paths = []
    for file in drift_characterization_output_dir_path.iterdir():
        if file.is_file() and file.name != ".gitkeep":
            results_target_file_path = target_dir / file.name
            shutil.copy(file, results_target_file_path)
            results_target_file_paths.append(results_target_file_path)
    
    # Clean input and output directories
    clean_folder_except_gitkeep(drift_characterization_input_file_path.parent, delete=True)
    clean_folder_except_gitkeep(drift_characterization_output_dir_path, delete=True)

    return results_target_file_paths


def concept_drift_complexity_assessment(dataset_key, dataset_info, concept_drift_info_path):
    print(f"## Running concept drift complexity assessment ##")

    approaches = load_window_config(constants.WINDOW_CONFIG_FILE_PATH)
    drift_df = pd.read_csv(concept_drift_info_path)
    drift_info_by_id = drift_info_to_dict(drift_df)
    configuration_name = concept_drift_info_path.stem.split("_")[-1]

    traces_sorted = load_xes_log(Path(dataset_info["path"]))
    for apc in approaches:
        name, typ, p = apc["name"], apc["type"], apc.get("params", {}) or {}
        if typ == "change_point_windows":
            df = assess_complexity_via_change_point_split(traces_sorted, drift_info_by_id, dataset_key, configuration_name, name)
            plot_complexity_measures(dataset_key, f"{configuration_name}__{name}", df, drift_info_by_id)
        elif typ == "fixed_size_windows":
            df = assess_complexity_via_fixed_sized_windows(traces_sorted, int(p["window_size"]), int(p["offset"]), dataset_key, configuration_name, name, drift_info_by_id)
            plot_complexity_measures(dataset_key, f"{configuration_name}__{name}", df, drift_info_by_id)
        elif typ == "window_comparison":
            df = assess_complexity_via_window_comparison(traces_sorted, int(p["window_1_size"]), int(p["window_2_size"]), int(p["offset"]), int(p["step"]), dataset_key, configuration_name, name)
            plot_delta_measures(dataset_key, f"{configuration_name}__{name}", df, drift_info_by_id)
        else:
            raise ValueError(f"Unknown approach type: {typ}")

    print("Drift complexity assessment complete.")

def main_per_dataset(dataset_key, dataset_info, mode='all'):
    print(f"### Processing dataset: {dataset_key} ###")
    if mode == 'all' or mode == 'detection_only':
        concept_drift_info_paths = concept_drift_characterization(dataset_key, dataset_info)
    else:
        # search for all csvs in input folder
        input_folder = constants.DRIFT_CHARACTERIZATION_RESULTS_DIR / dataset_key
        concept_drift_info_paths = list(input_folder.glob("*.csv"))

    if mode == 'all' or mode == 'complexity_only':
        for concept_drift_info_path in concept_drift_info_paths:
            concept_drift_complexity_assessment(dataset_key, dataset_info, concept_drift_info_path)

def main(datasets=None, mode='all'):
    print(f"#### Starting drift complextity analysis ####")

    data_dictionary = helpers.load_data_dictionary(constants.DATA_DICTIONARY_FILE_PATH)

    # only keep datasets in data_dictionary that are in the datasets
    if datasets is not None:
        data_dictionary = {k: v for k, v in data_dictionary.items() if k in datasets}

    for dataset_key, dataset_info in data_dictionary.items():
        main_per_dataset(dataset_key, dataset_info, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drift complexity analysis on selected datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset keys to include. If not set, all datasets are used."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='all',
        help="What parts of the script to run. Choose from 'all', 'detection_only', 'complexity_only'"
    )
    args = parser.parse_args()

    main(datasets=args.datasets, mode=args.mode)
