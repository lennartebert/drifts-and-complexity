import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from utils import constants, helpers

# Local configuration path
WINDOW_CONFIG_FILE_PATH = Path(__file__).parent.parent / "window_config.yml"
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.attributes.log import get as attributes_get

# pm4py extractors for building abundances per species
from pm4py.util import xes_constants as xes

# NEW: assessor import (singular)
from utils.complexity.assessors import (
    assess_complexity_via_change_point_split,
    assess_complexity_via_fixed_sized_windows,
    assess_complexity_via_window_comparison,
)
from utils.drift_io import drift_info_to_dict, load_xes_log
from utils.plotting.complexity import (
    plot_complexity_via_change_point_split,
    plot_complexity_via_fixed_sized_windows,
    plot_delta_measures,
)
from utils.plotting.coverage_curves import plot_coverage_curves_for_cp_windows
from utils.windowing.helpers import split_log_into_windows_by_change_points
from utils.windowing.loader import load_window_config

# ------------------- UTILS -------------------


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
        except Exception:
            pass


# ------------------- MAIN ORCHESTRATION -------------------


def concept_drift_characterization(dataset_key, dataset_info):
    print(f"## Running concept drift characterization ##")
    local_dataset_path = Path(dataset_info["path"])
    target_dataset_filename = Path(f"{dataset_key}.xes.gz")
    drift_characterization_input_file_path = (
        constants.DRIFT_CHARACTERIZATION_TEMP_INPUT_DIR
        / dataset_key
        / target_dataset_filename
    )
    drift_characterization_output_dir_path = (
        constants.DRIFT_CHARACTERIZATION_TEMP_OUTPUT_DIR / dataset_key
    )

    # Clean input and output directories
    clean_folder_except_gitkeep(drift_characterization_input_file_path.parent)
    clean_folder_except_gitkeep(drift_characterization_output_dir_path)

    # Copy dataset to input
    drift_characterization_input_file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(local_dataset_path, drift_characterization_input_file_path)

    # Run concept drift characterization
    try:
        result = subprocess.run(
            [
                "python",
                str(constants.DRIFT_CHARACTERIZATION_SCRIPT),
                "--input_dir",
                str(
                    drift_characterization_input_file_path.parent.relative_to(
                        constants.DRIFT_CHARACTERIZATION_DIR
                    )
                ),
                "--output_dir",
                str(
                    drift_characterization_output_dir_path.relative_to(
                        constants.DRIFT_CHARACTERIZATION_DIR
                    )
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=constants.DRIFT_CHARACTERIZATION_DIR,
            encoding="utf-8",
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Subprocess failed!")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    # Copy output to results folder
    target_dir = constants.CHANGE_STUDY_RESULTS_DIR / "drift_detection" / dataset_key
    target_dir.mkdir(parents=True, exist_ok=True)

    results_target_file_paths = []
    for file in drift_characterization_output_dir_path.iterdir():
        if file.is_file() and file.name != ".gitkeep":
            results_target_file_path = target_dir / file.name
            shutil.copy(file, results_target_file_path)
            results_target_file_paths.append(results_target_file_path)

    # Clean input and output directories
    clean_folder_except_gitkeep(
        drift_characterization_input_file_path.parent, delete=True
    )
    clean_folder_except_gitkeep(drift_characterization_output_dir_path, delete=True)

    return results_target_file_paths


def concept_drift_complexity_assessment(
    dataset_key, dataset_info, concept_drift_info_path, plot_coverage_curves: bool
):
    """
    Orchestrator:
    - loads window approaches from YAML
    - computes complexity per approach
    - renders approach-specific plots
    """
    print("## Running concept drift complexity assessment ##")

    # Load approaches & drift info
    approaches = load_window_config(WINDOW_CONFIG_FILE_PATH)
    drift_df = pd.read_csv(concept_drift_info_path)
    drift_info_by_id = drift_info_to_dict(drift_df)

    # Used for output subfolder names
    configuration_name = concept_drift_info_path.stem.split("_")[-1]

    # Load and sort the log once
    traces_sorted = load_xes_log(Path(dataset_info["path"]))

    for apc in approaches:
        name = apc["name"]
        typ = apc["type"]
        title = apc.get("title", None)
        p = apc.get("params", {}) or {}

        # Optional plotting knobs from YAML
        y_log = bool(p.get("y_log", False))
        fig_format = p.get("fig_format", "png")
        headroom = float(p.get("headroom", 0.10))
        point_position = p.get(
            "point_position", "end_w2"
        )  # for window_comparison plots

        cfg_with_approach = f"{configuration_name}__{name}"

        if typ == "change_point_windows":
            # --- Run both population adapters (simple + iNEXT) (+ vidgof if desired) ---
            adapter_names = ["vidgof_sample", "population_simple", "population_inext"]

            df = assess_complexity_via_change_point_split(
                traces_sorted,
                drift_info_by_id,
                dataset_key,
                configuration_name,
                name,
                adapter_names,
            )

            plot_complexity_via_change_point_split(
                dataset_key,
                cfg_with_approach,
                df,
                drift_info_by_id,
                y_log=y_log,
                fig_format=fig_format,
                headroom=0.2,
                title=title,
            )

            # --- Optionally plot coverage curves (iNEXT) ---
            if plot_coverage_curves:
                # Rebuild CP windows (same logic as in assessor)
                cp_info = {k: v for k, v in drift_info_by_id.items() if k != "na"}
                cps = (
                    [
                        (
                            cp_info[i]["calc_change_index"],
                            i,
                            cp_info[i]["calc_change_type"],
                        )
                        for i in sorted(cp_info.keys())
                    ]
                    if cp_info
                    else []
                )
                windows = split_log_into_windows_by_change_points(traces_sorted, cps)

                # save next to other complexity plots
                out_dir = (
                    constants.CHANGE_STUDY_RESULTS_DIR
                    / "complexity_assessment"
                    / dataset_key
                    / cfg_with_approach
                    / "coverage_curves"
                )
                plot_coverage_curves_for_cp_windows(
                    windows, out_dir, q_orders=(0,), xlim=(0, 1.0)
                )

        elif typ == "fixed_size_windows":
            # --- Only simple population adapter here (+ vidgof if desired) ---
            adapter_names = ["vidgof_sample", "population_simple"]

            window_size = int(p["window_size"])
            offset = int(p["offset"])

            df = assess_complexity_via_fixed_sized_windows(
                traces_sorted,
                window_size,
                offset,
                dataset_key,
                configuration_name,
                name,
                adapter_names=adapter_names,
                drift_info_by_id=drift_info_by_id,
            )

            plot_complexity_via_fixed_sized_windows(
                dataset_key,
                cfg_with_approach,
                df,
                drift_info_by_id,
                window_size=window_size,
                offset=offset,
                y_log=y_log,
                fig_format=fig_format,
                headroom=headroom,
                title=title,
            )

        elif typ == "window_comparison":
            # --- Only simple population adapter here (+ vidgof if desired) ---
            adapter_names = ["vidgof_sample", "population_simple"]

            df = assess_complexity_via_window_comparison(
                traces_sorted,
                int(p["window_1_size"]),
                int(p["window_2_size"]),
                int(p["offset"]),
                int(p["step"]),
                dataset_key,
                configuration_name,
                name,
                adapter_names=adapter_names,
            )

            plot_delta_measures(
                dataset_key,
                cfg_with_approach,
                df,
                drift_info_by_id,
                point_position=point_position,
                y_log=y_log,
                fig_format=fig_format,
                headroom=headroom,
                title=title,
            )
        else:
            raise ValueError(f"Unknown approach type: {typ}")

    print("Drift complexity assessment complete.")


def main_per_dataset(
    dataset_key, dataset_info, mode="all", plot_coverage_curves: bool = False
):
    print(f"### Processing dataset: {dataset_key} ###")
    if mode == "all" or mode == "detection_only":
        concept_drift_info_paths = concept_drift_characterization(
            dataset_key, dataset_info
        )
    else:
        # search for all csvs in input folder
        input_folder = (
            constants.CHANGE_STUDY_RESULTS_DIR / "drift_detection" / dataset_key
        )
        concept_drift_info_paths = list(input_folder.glob("*.csv"))

    if mode == "all" or mode == "complexity_only":
        for concept_drift_info_path in concept_drift_info_paths:
            concept_drift_complexity_assessment(
                dataset_key,
                dataset_info,
                concept_drift_info_path,
                plot_coverage_curves=plot_coverage_curves,
            )


def main(datasets=None, mode="all", plot_coverage_curves: bool = False):
    print(f"#### Starting drift complexity analysis ####")

    data_dictionary = helpers.load_data_dictionary(constants.get_data_dictionary_path())

    # only keep datasets in data_dictionary that are in the datasets
    if datasets is not None:
        data_dictionary = {k: v for k, v in data_dictionary.items() if k in datasets}

    for dataset_key, dataset_info in data_dictionary.items():
        main_per_dataset(dataset_key, dataset_info, mode, plot_coverage_curves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run drift complexity analysis on selected datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset keys to include. If not set, all datasets are used.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Choose from 'all', 'detection-only', 'complexity-only'",
    )
    # NEW: plot coverage curves for change-point windows
    parser.add_argument(
        "--plot-coverage-curves",
        action="store_true",
        help="If set, saves iNEXT coverage curves for change-point windows.",
    )
    args = parser.parse_args()

    main(
        datasets=args.datasets,
        mode=args.mode,
        plot_coverage_curves=args.plot_coverage_curves,
    )
