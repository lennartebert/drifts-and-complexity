import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import constants 

# Define drift type order
drift_type_order = [
    "Sudden (before to after)",
    "Gradual (before to after)",
    "Gradual (before to during)",
    "Gradual (during to after)"
]

def load_complexity_per_window_dict(datasets):
    all_files = [constants.COMPLEXITY_RESULTS_DIR / dataset / "complexity_per_window.csv" for dataset in datasets]
    complexity_per_window_df_dict = {}
    for f in all_files:
        dataset = f.parent.name
        df = pd.read_csv(f)
        df["dataset"] = dataset
        df["start_change_id"] = df["start_change_id"].astype("Int64")
        df["end_change_id"] =  df["end_change_id"].astype("Int64")
        df["window_id"] = df["window_id"].astype(int)
        complexity_per_window_df_dict[dataset] = df
    return complexity_per_window_df_dict


def load_drift_info(complexity_per_window_df_dict, change_point_detector=constants.DEFAULT_CHANGE_POINT_DETECTOR):
    drift_info_by_dataset = {}
    for dataset in complexity_per_window_df_dict.keys():
        path = constants.DRIFT_CHARACTERIZATION_RESULTS_DIR / dataset / f"results_{change_point_detector}_input_approach.csv"
        drift_info = pd.read_csv(path)
        drift_info["calc_change_id"] = drift_info["calc_change_id"].astype("Int64")
        drift_info_by_dataset[dataset] = drift_info.set_index("calc_change_id").to_dict(orient="index")
    return drift_info_by_dataset

def get_drift_info_summary_table(drift_info_by_dataset):
    drift_info_summary_dict = {}

    for dataset, drift_info_dict in drift_info_by_dataset.items():
        result = {}
        sudden_changes = []
        gradual_changes = []

        # iterate over dict items in order of keys (already sorted)
        items = list(drift_info_dict.items())
        for i, (change_id, row) in enumerate(items):
            change_type = row.get("calc_change_type")
            change_index = int(row.get("calc_change_index"))

            if change_type == "sudden":
                sudden_changes.append(change_index)

            if change_type == "gradual_start":
                start_index = change_index
                end_index = None
                # look at next item if exists
                if i + 1 < len(items):
                    end_index = int(items[i + 1][1].get("calc_change_index"))
                gradual_changes.append((start_index, end_index))

        result["# Total Changes"] = len(sudden_changes) + len(gradual_changes)
        result["# Sudden Changes"] = len(sudden_changes)
        result["Sudden Change Points"] = ", ".join(map(str, sudden_changes))
        result["# Gradual Changes"] = len(gradual_changes)
        result["Gradual Change Points"] = ", ".join(
            f"({s}, {e})" for s, e in gradual_changes
        )

        drift_info_summary_dict[dataset] = result

    # convert dict to DataFrame
    drift_info_summary_df = pd.DataFrame.from_dict(drift_info_summary_dict, orient="index")

    # add total row
    total_row = {
        '# Total Changes': drift_info_summary_df["# Total Changes"].sum(),
        "# Sudden Changes": drift_info_summary_df["# Sudden Changes"].sum(),
        "Sudden Change Points": "",
        "# Gradual Changes": drift_info_summary_df["# Gradual Changes"].sum(),
        "Gradual Change Points": "",
    }
    drift_info_summary_df.loc["Total"] = total_row

    return drift_info_summary_df

def compute_complexity_deltas(window_dict, drift_info_by_dataset):
    results = []
    for dataset, window_df in window_dict.items():
        drift_info = drift_info_by_dataset.get(dataset, {})
        measure_columns = [col for col in window_df.columns if col.startswith("measure_")]

        for change_id, info in drift_info.items():
            change_type = info["calc_change_type"]
            window_before_change_point = window_df[window_df["end_change_id"] == change_id].iloc[0]
            window_after_change_point = window_df[window_df["start_change_id"] == change_id].iloc[0]

            # Assert that window_before_change and window_after_change are not empty - there always needs to be a window before/after a change point
            assert not window_before_change_point.empty, f"window_before_change is empty for dataset {dataset}, change_id {change_id}, change_type {change_type}"
            assert not window_after_change_point.empty, f"window_after_change is empty for dataset {dataset}, change_id {change_id}, change_type {change_type}"

            if change_type == "sudden":
                deltas = (window_after_change_point[measure_columns] - window_before_change_point[measure_columns]).to_dict()
                results.append({"change_type": "Sudden (before to after)", **{k.replace("measure_", ""): v for k, v in deltas.items()}})

            elif change_type == "gradual_start":
                deltas = (window_after_change_point[measure_columns] - window_before_change_point[measure_columns]).to_dict()
                results.append({"change_type": "Gradual (before to during)", **{k.replace("measure_", ""): v for k, v in deltas.items()}})

                # also record the before to after change
                window_after_gradual_end = window_df[window_df["start_change_id"] == change_id + 1].iloc[0]
                assert not window_after_gradual_end.empty, f"window_after_gradual_end is empty for dataset {dataset}, change_id {change_id}, change_type {change_type}"
                deltas = (window_after_gradual_end[measure_columns] - window_before_change_point[measure_columns]).to_dict()
                results.append({"change_type": "Gradual (before to after)", **{k.replace("measure_", ""): v for k, v in deltas.items()}})

            elif change_type == "gradual_end":
                deltas = (window_after_change_point[measure_columns] - window_before_change_point[measure_columns]).to_dict()
                results.append({"change_type": "Gradual (during to after)", **{k.replace("measure_", ""): v for k, v in deltas.items()}})
                       
            else: 
                raise ValueError("Unknown change type")

    return pd.DataFrame(results)


def format_number(x, include_plus=False):
    # Use scientific notation if abs(x) >= 1000, otherwise fixed-point
    if abs(x) >= 1000:
        if include_plus:
            return f"{x:+.1e}"  # scientific notation
        else: 
            return f"{x:.1e}"  # scientific notation
    else:
        if include_plus:
            return f"{x:+.2f}"  # fixed-point with 2 decimals
        else:
            return f"{x:.2f}"  # fixed-point with 2 decimals
    

def save_aggregated_table(results_df):
    results_df_clean = results_df.dropna()
    change_types = results_df_clean["change_type"].unique()

    measure_cols = [col for col in results_df_clean.columns if col != "change_type"]
    records = []
    for measure in measure_cols:
        for change_type in change_types:
            subset = results_df_clean[results_df_clean["change_type"] == change_type][measure]
            records.append({
                "measure": measure,
                "change_type": change_type,
                "mean": subset.mean(),
                "min": subset.min(),
                "max": subset.max(),
                "std": subset.std(),
                "count": subset.count()
            })

    summary_df = pd.DataFrame(records)
    summary_df.set_index(["measure", "change_type"], inplace=True)
    summary_df = summary_df.reorder_levels(["change_type", "measure"]).sort_index()
    summary_df.to_csv(constants.COMBINED_RESULTS_TABLE_DIR / "complexity_delta_aggregated.csv")
    return summary_df

def save_simple_aggregated_table(aggregated_table_df):
    # Ensure index is MultiIndex
    if not isinstance(aggregated_table_df.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex with levels (change_type, measure)")

    # Prepare output structure
    rows = []
    measures = aggregated_table_df.index.get_level_values("measure").unique()

    for change_type in drift_type_order:
        row = {"Change Type": change_type}

        if change_type in aggregated_table_df.index.get_level_values("change_type"):
            subset = aggregated_table_df.loc[change_type]
            row["Instances"] = int(subset["count"].iloc[0]) if "count" in subset.columns else None

            for measure in measures:
                if measure in subset.index:
                    mean = subset.loc[measure]["mean"]
                    std = subset.loc[measure]["std"]
                    formatted = f"{format_number(mean, include_plus=True)} ({format_number(std, include_plus=False)})"
                    row[measure] = formatted
                else:
                    row[measure] = ""
        else:
            row["Instances"] = 0
            for measure in measures:
                row[measure] = ""

        rows.append(row)

    # Create DataFrame and save
    final_df = pd.DataFrame(rows)
    final_df.set_index("Change Type", inplace=True)
    final_df.to_csv(constants.COMBINED_RESULTS_TABLE_DIR / "complexity_delta_simple.csv")

    return final_df


def save_boxplots(results_df):
    measure_names = [col for col in results_df.columns if col != "change_type"]

    for measure in measure_names:
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=results_df,
            x="change_type",
            y=measure,
            order=drift_type_order
        )
        plt.xlabel("Change Type")
        plt.ylabel(measure.replace("_", " ").title())
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(constants.COMBINED_RESULTS_BOXPLOT_DIR / f"{measure}_boxplot.png", dpi=300)
        plt.close()


def main(datasets=None, change_point_detector=constants.DEFAULT_CHANGE_POINT_DETECTOR):
    print("#### Starting to combine drift analysis results ####")
    if datasets is None:
        # Get all folder names (1st level child) under folder results/complexity_assessment
        datasets = [
            d.name for d in (constants.COMPLEXITY_RESULTS_DIR).iterdir()
            if d.is_dir()
        ]

    window_dict = load_complexity_per_window_dict(datasets)
    if not window_dict:
        print("No datasets with complexity_per_window.csv found.")
        return
    drift_info_by_dataset = load_drift_info(window_dict, change_point_detector)

    # get summary of drift info
    drift_info_summary_table_df = get_drift_info_summary_table(drift_info_by_dataset=drift_info_by_dataset)
    drift_info_summary_table_df.to_csv(constants.COMBINED_RESULTS_TABLE_DIR / "drift_info_summary.csv")


    results_df = compute_complexity_deltas(window_dict, drift_info_by_dataset)
    # save_boxplots(results_df)

    aggregated_table = save_aggregated_table(results_df)
    print(aggregated_table)
    simple_aggregated_table = save_simple_aggregated_table(aggregated_table)
    print(simple_aggregated_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine drift complexity analysis detection results.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset keys to include. If not set, all datasets are used."
    )
    parser.add_argument(
        "--change-point-detector",
        default=constants.DEFAULT_CHANGE_POINT_DETECTOR,
        help="Defauld change point detection approach. Choose from 'emd', 'prodrift', 'zheng', 'bose_j', 'process_graphs', 'lcdd', 'adwin_j'."
    )

    args = parser.parse_args()

    main(datasets=args.datasets, change_point_detector=args.change_point_detector)

