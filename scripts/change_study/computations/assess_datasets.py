import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Set matplotlib to use non-interactive backend before importing plotting modules
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import pandas as pd

# Ensure project root is on sys.path so local imports work when run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import constants, helpers

# Local configuration path
WINDOW_CONFIG_FILE_PATH = Path(__file__).parent.parent / "window_config.yml"

# Assessor imports
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
from utils.windowing.loader import load_window_config

# ------------------- UTILS -------------------


def get_adapter_names(test_mode: bool) -> list[str]:
    """Get adapter names based on test mode.

    Parameters
    ----------
    test_mode
        If True, returns only vidgof_sample for speed. Otherwise returns both local and vidgof_sample.

    Returns
    -------
    list[str]
        List of adapter names to use.
    """
    return ["vidgof_sample"] if test_mode else ["local", "vidgof_sample"]


def run_with_error_handling(operation_name: str, func, *args, **kwargs):
    """Run a function with standardized error handling.

    Parameters
    ----------
    operation_name
        Human-readable name of the operation (e.g., "complexity computation").
    func
        Function to execute.
    *args, **kwargs
        Arguments to pass to the function.

    Returns
    -------
    Any
        Return value of the function.

    Raises
    ------
    Exception
        Re-raises any exception that occurs, after printing error message and traceback.
    """
    try:
        result = func(*args, **kwargs)
        print(f"  {operation_name} complete.")
        return result
    except Exception as e:
        print(f"  ERROR during {operation_name}: {e}")
        import traceback

        traceback.print_exc()
        raise


def clean_folder_except_gitkeep(folder: Path, delete: bool = False) -> None:
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


def normalize_mode(mode: str) -> str:
    """
    Normalize and validate the CLI mode parameter.

    Parameters
    ----------
    mode:
        Raw mode argument from the CLI.

    Returns
    -------
    str
        Normalized mode string (hyphenated).

    Raises
    ------
    ValueError
        If the provided mode is not supported.
    """
    normalized = (mode or "all").lower().replace("_", "-")
    allowed = {"all", "detection-only", "complexity-only"}
    if normalized not in allowed:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {sorted(allowed)}.")
    return normalized


def concept_drift_characterization(
    dataset_key: str, dataset_info: dict[str, Any], test_mode: bool = False
) -> list[Path]:
    """
    Run concept drift characterization or use existing results in test mode.

    Parameters
    ----------
    dataset_key
        Dataset identifier.
    dataset_info
        Dataset metadata dictionary.
    test_mode
        If True, skip drift characterization and use/create minimal test results.

    Returns
    -------
    list[Path]
        List of paths to drift detection result CSV files.
    """
    target_dir = constants.CHANGE_STUDY_RESULTS_DIR / "drift_detection" / dataset_key
    target_dir.mkdir(parents=True, exist_ok=True)

    # In test mode, check for existing results first to speed up repeated runs
    if test_mode:
        existing_results = list(target_dir.glob("*.csv"))
        if existing_results:
            print(
                "## Test mode: Found existing drift detection results; "
                "skipping new drift characterization run ##"
            )
            return existing_results
        # Fall through when no existing results are found so that
        # drift characterization is still executed in test mode.
        print(
            "## Test mode: No existing results found, "
            "running full drift characterization ##"
        )

    print(f"## Running concept drift characterization ##")
    local_dataset_path = (PROJECT_ROOT / dataset_info["path"]).resolve()
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
            timeout=86400,  # 24 hours timeout for very large datasets
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.TimeoutExpired as e:
        print(
            f"ERROR: Drift characterization subprocess timed out after 24 hours for {dataset_key}"
        )
        print("STDOUT:\n", e.stdout if e.stdout else "(no output)")
        print("STDERR:\n", e.stderr if e.stderr else "(no output)")
        raise
    except subprocess.CalledProcessError as e:
        print("Subprocess failed!")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    # Copy output to results folder
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
    dataset_key: str,
    dataset_info: dict[str, Any],
    concept_drift_info_path: Path,
    test_mode: bool = False,
) -> None:
    """
    Orchestrator:
    - loads window approaches from YAML
    - computes complexity per approach
    - renders approach-specific plots

    Parameters
    ----------
    dataset_key
        Dataset identifier.
    dataset_info
        Dataset metadata dictionary.
    concept_drift_info_path
        Path to drift detection results CSV.
    test_mode
        If True, use simplified configuration for faster testing.
    """
    print("## Running concept drift complexity assessment ##")

    # Load approaches & drift info
    approaches = load_window_config(WINDOW_CONFIG_FILE_PATH)

    # In test mode, use only the first change_point_windows approach
    if test_mode:
        approaches = [
            apc for apc in approaches if apc["type"] == "change_point_windows"
        ][:1]
        print(f"## Test mode: Using only {len(approaches)} approach(es) ##")

    drift_df = pd.read_csv(concept_drift_info_path)
    drift_info_by_id = drift_info_to_dict(drift_df)
    dataset_path = (PROJECT_ROOT / dataset_info["path"]).resolve()

    # Extract configuration name from filename (e.g., "results_DATASET_config.csv" -> "config")
    configuration_name = concept_drift_info_path.stem.rsplit("_", 1)[-1]

    # Load and sort the log once
    traces_sorted = load_xes_log(dataset_path)

    # In test mode, limit to first 1000 traces for faster processing
    if test_mode and len(traces_sorted) > 1000:
        print(
            f"## Test mode: Limiting to first 1000 traces (from {len(traces_sorted)}) ##"
        )
        traces_sorted = traces_sorted[:1000]

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
        adapter_names = get_adapter_names(test_mode)

        if typ == "change_point_windows":
            print(
                f"  Computing complexity for approach: {name} with adapters: {adapter_names}"
            )
            df = run_with_error_handling(
                "complexity computation",
                assess_complexity_via_change_point_split,
                traces_sorted,
                drift_info_by_id,
                dataset_key,
                configuration_name,
                name,
                adapter_names,
            )

            print(f"  Plotting...")
            run_with_error_handling(
                f"plotting for {name}",
                plot_complexity_via_change_point_split,
                dataset_key,
                cfg_with_approach,
                df,
                drift_info_by_id,
                y_log=y_log,
                fig_format=fig_format,
                headroom=headroom,
                title=None,
            )

        elif typ == "fixed_size_windows":
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
    dataset_key: str,
    dataset_info: dict[str, Any],
    mode: str = "all",
    test_mode: bool = False,
) -> None:
    """
    Process a single dataset.

    Parameters
    ----------
    dataset_key
        Dataset identifier.
    dataset_info
        Dataset metadata dictionary.
    mode
        Processing mode: 'all', 'detection-only', or 'complexity-only'.
    test_mode
        If True, use simplified configuration for faster testing.
    """
    print(f"### Processing dataset: {dataset_key} ###")
    normalized_mode = normalize_mode(mode)

    if normalized_mode in {"all", "detection-only"}:
        concept_drift_info_paths = concept_drift_characterization(
            dataset_key, dataset_info, test_mode=test_mode
        )
    else:
        # search for all csvs in input folder
        input_folder = (
            constants.CHANGE_STUDY_RESULTS_DIR / "drift_detection" / dataset_key
        )
        concept_drift_info_paths = list(input_folder.glob("*.csv"))
        if not concept_drift_info_paths:
            print(
                f"WARNING: No drift detection results found for {dataset_key} in {input_folder}"
            )
            print(f"  Skipping complexity assessment for this dataset.")
            return

    if normalized_mode in {"all", "complexity-only"}:
        for concept_drift_info_path in concept_drift_info_paths:
            concept_drift_complexity_assessment(
                dataset_key,
                dataset_info,
                concept_drift_info_path,
                test_mode=test_mode,
            )


def main(
    datasets: list[str] | None = None,
    mode: str = "all",
    test_mode: bool = False,
) -> None:
    """
    Main entry point for drift complexity analysis.

    Parameters
    ----------
    datasets
        List of dataset keys to process. If None, processes all datasets.
    mode
        Processing mode: 'all', 'detection-only', or 'complexity-only'.
    test_mode
        If True, use simplified configuration for faster testing.
    """
    print(f"#### Starting drift complexity analysis ####")
    if test_mode:
        print("#### TEST MODE: Using simplified configuration ####")

    normalized_mode = normalize_mode(mode)

    data_dictionary = helpers.load_data_dictionary(
        constants.get_data_dictionary_path(),
        get_real=True,
        get_synthetic=True,
    )

    # only keep datasets in data_dictionary that are in the datasets
    if datasets is not None:
        available_keys = set(data_dictionary.keys())
        requested_keys = set(datasets)
        missing_keys = requested_keys - available_keys

        if missing_keys:
            print(
                f"WARNING: The following requested dataset(s) were not found in the data dictionary: {sorted(missing_keys)}"
            )
            print(f"Available datasets are: {sorted(available_keys)}")

        data_dictionary = {k: v for k, v in data_dictionary.items() if k in datasets}

        if not data_dictionary:
            print("ERROR: No matching datasets found after filtering. Exiting.")
            return

    for dataset_key, dataset_info in data_dictionary.items():
        main_per_dataset(
            dataset_key,
            dataset_info,
            normalized_mode,
            test_mode=test_mode,
        )


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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a lightweight test using the TEST_BPIC12 dataset. "
        "Prefers existing drift detection results when available, "
        "uses minimal config, limits traces.",
    )
    args = parser.parse_args()

    selected_datasets = ["TEST_BPIC12"] if args.test else args.datasets

    main(
        datasets=selected_datasets,
        mode=args.mode,
        test_mode=args.test,
    )
