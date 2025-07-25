import argparse
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path
# Add the process-complexity folder to the Python path
sys.path.append(str(Path(__file__).resolve().parent / "process-complexity"))
from Complexity import generate_log, build_graph, perform_measurements, graph_complexity, log_complexity
from pm4py.objects.log.importer.xes import importer as xes_importer

from utils import constants, helpers

## UTILS

def clean_folder_except_gitkeep(folder: Path):
    for item in folder.iterdir():
        if item.name != ".gitkeep":
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

def flatten_measurements(window_results):
    flat_records = []
    for w in window_results:
        flat_record = w.copy()
        del flat_record["measurements"]

        meas = w["measurements"].copy()

        if "Trace length" in meas and isinstance(meas["Trace length"], dict):
            tl = meas.pop("Trace length")
            meas["measure_Trace length min"] = tl.get("min", None)
            meas["measure_Trace length avg"] = tl.get("avg", None)
            meas["measure_Trace length max"] = tl.get("max", None)

        # Prefix remaining measurement keys with "measure_"
        meas = {f"measure_{k}": v for k, v in meas.items()}

        flat_record.update(meas)
        flat_records.append(flat_record)

    return flat_records

def save_complexity_csv(dataset_key, flat_data):
    target_dir = constants.COMPLEXITY_RESULTS_DIR / dataset_key
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(flat_data)
    df.to_csv(target_dir / "complexity_per_window.csv", index=False)

def plot_complexity_measures(dataset_key, flat_data, drift_info_by_id):
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    df = pd.DataFrame(flat_data)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    label_map = {
        "sudden": "Sudden",
        "gradual_start": "Gradual start",
        "gradual_end": "Gradual end",
        "start": "Start",
        "end": "End"
    }

    measure_columns = [col for col in df.columns if col.startswith("measure_")]

    for measure_column in measure_columns:
        measure_name = measure_column.removeprefix("measure_")
        plt.figure(figsize=(12, 5))

        y_min, y_max = df[measure_column].min(), df[measure_column].max()

        for _, row in df.iterrows():
            plt.plot(
                [row["start_time"], row["end_time"]],
                [row[measure_column], row[measure_column]],
                color='blue'
            )
            # Trace count
            mid_time = row["start_time"] + (row["end_time"] - row["start_time"]) / 2
            y_pos = row[measure_column] + 0.002 * (y_max - y_min)
            trace_label = f'N={int(row["measure_Support"])}' if "measure_Support" in row else ""
            plt.text(mid_time, y_pos, trace_label, fontsize=7, ha='center', va='bottom')

        label_y_offset = (y_max - y_min) * 0.08  # 8% above the plot

        # Add start line and label
        x_start = df["start_time"].min()
        plt.axvline(x_start, color='black', linestyle='--', linewidth=1)
        plt.text(x_start, y_max + label_y_offset, label_map["start"], fontsize=8,
                 ha='left', va='bottom', rotation=45)

        # Add end line and label
        x_end = df["end_time"].max()
        plt.axvline(x_end, color='black', linestyle='--', linewidth=1)
        plt.text(x_end, y_max + label_y_offset, label_map["end"], fontsize=8,
                 ha='left', va='bottom', rotation=45)

        # Add change point lines and beautified labels
        for cid, info in drift_info_by_id.items():
            x = pd.to_datetime(info["calc_change_moment"])
            change_type = info["calc_change_type"]
            change_label = label_map.get(change_type, change_type)
            plt.axvline(x=x, color='red', linestyle='--', alpha=0.5)
            plt.text(x, y_max + label_y_offset, change_label, rotation=45,
                     fontsize=8, ha='left', va='bottom')

        # Axis formatting
        plt.xlabel("Time")
        plt.ylabel(measure_name)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        output_path = constants.COMPLEXITY_RESULTS_DIR / dataset_key / f"{measure_name}_over_time.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=600)
        plt.close()


## MAIN FUNCTIONS

def concept_drift_characterization(dataset_key, dataset_info):
    print(f"## Running concept drift characterization ##")
    dataset_path = Path(dataset_info["path"])
    dataset_filename = dataset_path.name
    input_target_path = constants.DRIFT_CHARACTERIZATION_TEMP_INPUT_DIR / dataset_filename

    # Clean input ando output directories
    clean_folder_except_gitkeep(constants.DRIFT_CHARACTERIZATION_TEMP_INPUT_DIR)
    clean_folder_except_gitkeep(constants.DRIFT_CHARACTERIZATION_TEMP_OUTPUT_DIR)


    # Copy dataset to input
    shutil.copy(dataset_path, input_target_path)

    # Run concept drift characterization
    try:
        result = subprocess.run(
            ["python", str(constants.DRIFT_CHARACTERIZATION_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            cwd=constants.DRIFT_CHARACTERIZATION_DIR
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Subprocess failed!")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    # Copy output to results folder
    result_target_dir = constants.DRIFT_CHARACTERIZATION_RESULTS_DIR / dataset_key
    result_target_dir.mkdir(parents=True, exist_ok=True)
    for file in constants.DRIFT_CHARACTERIZATION_TEMP_OUTPUT_DIR.iterdir():
        if file.name != ".gitkeep":
            shutil.copy(file, result_target_dir / file.name)
    
    # Cleanup input/output (preserve .gitkeep)
    clean_folder_except_gitkeep(constants.DRIFT_CHARACTERIZATION_TEMP_INPUT_DIR)
    clean_folder_except_gitkeep(constants.DRIFT_CHARACTERIZATION_TEMP_OUTPUT_DIR)


def concept_drift_complexity_assessment(dataset_key, dataset_info):
    print(f"## Running concept drift complexity assessment ##")

    # TODO: Make approach configurable
    concept_drift_info_path = constants.DRIFT_CHARACTERIZATION_RESULTS_DIR / dataset_key / 'results_adwin_j_input_approach.csv'
    concept_drift_info_df = pd.read_csv(concept_drift_info_path)
    
    drift_info_by_id = concept_drift_info_df.set_index('calc_change_id').to_dict(orient='index')

    # calc_change_id == 'na', there is no drift
    if 'na' in drift_info_by_id.keys():
        print("No drifts in dataset. Cannot assess drift.") # this is not entirely true - we could still plot the complexity from start to end. However, this would require major updates for this edge case.
        return None
    
    log_path = Path(dataset_info["path"])
    pm4py_log = xes_importer.apply(str(log_path))
    traces_sorted = sorted(pm4py_log, key=lambda trace: trace[0]['time:timestamp'])

    window_complexity_results = []
    window_id = 0

    sorted_change_ids = sorted(drift_info_by_id.keys())

    for i, end_change_id in enumerate(sorted_change_ids + [None]):
        if end_change_id is None:
            # final window: after last change
            start_change_id = sorted_change_ids[-1]
            start_index = drift_info_by_id[start_change_id]['calc_change_index']
            end_index = len(traces_sorted)
            start_time = pd.to_datetime(drift_info_by_id[start_change_id]['calc_change_moment'])
            end_time = traces_sorted[-1][-1]['time:timestamp']
            window_change = 'stable'
        else:
            start_change_id = None if i == 0 else sorted_change_ids[i - 1]
            start_index = 0 if i == 0 else drift_info_by_id[start_change_id]['calc_change_index']
            end_index = drift_info_by_id[end_change_id]['calc_change_index']
            start_time = traces_sorted[0][0]['time:timestamp'] if i == 0 else pd.to_datetime(drift_info_by_id[start_change_id]['calc_change_moment'])
            end_time = pd.to_datetime(drift_info_by_id[end_change_id]['calc_change_moment'])
            window_change = 'gradual' if drift_info_by_id[end_change_id]['calc_change_type'] == 'gradual_end' else 'stable'

        # Build window and compute complexity
        window_traces = traces_sorted[start_index:end_index]
        log = generate_log(window_traces, verbose=False)
        pa = build_graph(log, verbose=False, accepting=False)
        measurements = perform_measurements('all', log, window_traces, pa, quiet=True, verbose=False)
        var_ent = graph_complexity(pa)
        seq_ent = log_complexity(pa)
        measurements['Variant Entropy'] = var_ent[0]
        measurements['Normalized Variant Entropy'] = var_ent[1]
        measurements['Trace Entropy'] = seq_ent[0]
        measurements['Normalized Trace Entropy'] = seq_ent[1]

        window_complexity_results.append({
            "window_id": window_id,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": pd.to_datetime(start_time).tz_convert(None),
            "end_time": pd.to_datetime(end_time).tz_convert(None),
            "start_change_id": start_change_id,
            "end_change_id": end_change_id,
            "window_change": window_change,
            'traces_in_window': end_index - start_index,
            "measurements": measurements
        })

        window_id += 1

    flat_measurements_per_window = flatten_measurements(window_complexity_results)
    save_complexity_csv(dataset_key, flat_measurements_per_window)
    plot_complexity_measures(dataset_key, flat_measurements_per_window, drift_info_by_id)

    print("Drift complexity assessment complete.")

def main_per_dataset(dataset_key, dataset_info, only_complexity=False):
    print(f"### Processing dataset: {dataset_key} ###")
    if not only_complexity:
        concept_drift_characterization(dataset_key, dataset_info)
    concept_drift_complexity_assessment(dataset_key, dataset_info)
    

def main(datasets=None, only_complexity=False):
    print(f"#### Starting drift complextity analysis ####")

    data_dictionary = helpers.load_data_dictionary(constants.DATA_DICTIONARY_FILE_PATH)

    # only keep datasets in data_dictionary that are in the datasets
    if datasets is not None:
        data_dictionary = {k: v for k, v in data_dictionary.items() if k in datasets}

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
        "-oc", "--only-complexity",
        action="store_true",
        help="Only run complexity detection, not drift characterization."
    )
    args = parser.parse_args()

    main(datasets=args.datasets, only_complexity=args.only_complexity)
