import argparse
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path


DRIFT_INPUT_DIR = Path("concept-drift-characterization/input")
DRIFT_OUTPUT_DIR = Path("concept-drift-characterization/output")
DRIFT_DIR = Path("concept-drift-characterization/")
DRIFT_SCRIPT = Path("main.py")
RESULTS_BASE_DIR = Path("results/drift_detection")
DATA_DICTIONARY_FILE_PATH = Path('configuration/data_dictionary.json')


def load_data_dictionary(path):
    with open(path, 'r') as f:
        return json.load(f)

def clean_folder_except_gitkeep(folder: Path):
    for item in folder.iterdir():
        if item.name != ".gitkeep":
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

def concept_drift_characterization(dataset_key, dataset_info):
    print(f"## Running concept drift characterization ##")
    dataset_path = Path(dataset_info["path"])
    dataset_filename = dataset_path.name
    input_target_path = DRIFT_INPUT_DIR / dataset_filename

    # Copy dataset to input
    shutil.copy(dataset_path, input_target_path)

    # Run concept drift characterization
    try:
        result = subprocess.run(
            ["python", str(DRIFT_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            cwd=DRIFT_DIR
    )
    except subprocess.CalledProcessError as e:
        print("Subprocess failed!")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    # Copy output to results folder
    result_target_dir = RESULTS_BASE_DIR / dataset_key
    result_target_dir.mkdir(parents=True, exist_ok=True)
    for file in DRIFT_OUTPUT_DIR.iterdir():
        if file.name != ".gitkeep":
            shutil.copy(file, result_target_dir / file.name)

    # Cleanup input/output (preserve .gitkeep)
    clean_folder_except_gitkeep(DRIFT_INPUT_DIR)
    clean_folder_except_gitkeep(DRIFT_OUTPUT_DIR)

def concept_drift_complexity_assessment(dataset_key, dataset_info):
    print(f"## Running concept drift complexity assessment ##")

    concept_drift_info_path = RESULTS_BASE_DIR / dataset_key / 'results_adwin_j_input_approach.csv' # TODO make approach and file path adjustable
    concept_drift_info_df = pd.read_csv(concept_drift_info_path)
    


def main_per_dataset(dataset_key, dataset_info, only_complexity=False):
    print(f"### Processing dataset: {dataset_key} ###")
    if not only_complexity:
        concept_drift_characterization(dataset_key, dataset_info)
    concept_drift_complexity_assessment(dataset_key, dataset_info)
    

def main(dataset_list=None, only_complexity=False):
    print(f"#### Starting drift complextity analysis ####")

    data_dictionary = load_data_dictionary(DATA_DICTIONARY_FILE_PATH)

    # only keep datasets in data_dictionary that are in the dataset_list
    if dataset_list is not None:
        data_dictionary = {k: v for k, v in data_dictionary.items() if k in dataset_list}

    for dataset_key, dataset_info in data_dictionary.items():
        main_per_dataset(dataset_key, dataset_info, only_complexity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drift complexity analysis on selected datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset keys to include. If not set, all datasets are used."
    )
    parser.add_argument(
        "--only-complexity",
        type=bool,
        default=False,
        help="Set flag to True to only run complexity detection, not drift characterization."
    )
    args = parser.parse_args()

    main(dataset_list=args.datasets, only_complexity=args.only_complexity)
