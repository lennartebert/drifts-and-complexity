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
    "Gradual (during to after)",
    "Incremental (before to after)"
]

def load_complexity_per_window_dict():
    all_files = list(Path("results/complexity_assessment").glob("*/complexity_per_window.csv"))
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


def load_drift_info(complexity_per_window_df_dict):
    drift_info_by_dataset = {}
    for dataset in complexity_per_window_df_dict.keys():
        path = Path("results/drift_detection") / dataset / "results_adwin_j_input_approach.csv"
        if path.exists():
            drift_info = pd.read_csv(path)
            drift_info["calc_change_id"] = drift_info["calc_change_id"].astype("Int64")
            drift_info_by_dataset[dataset] = drift_info.set_index("calc_change_id").to_dict(orient="index")
    return drift_info_by_dataset


def compute_complexity_deltas(window_dict, drift_info_by_dataset):
    results = []
    for dataset, df in window_dict.items():
        drift_info = drift_info_by_dataset.get(dataset, {})
        measure_columns = [col for col in df.columns if col.startswith("measure_")]
        for change_id, info in drift_info.items():
            change_type = info["calc_change_type"]
            prev_row = df[df["end_change_id"] == change_id - 1]
            curr_row = df[df["end_change_id"] == change_id]
            next_row = df[df["end_change_id"] == change_id + 1]

            if change_type == "sudden":
                if not prev_row.empty and not curr_row.empty:
                    deltas = (curr_row.iloc[0][measure_columns] - prev_row.iloc[0][measure_columns]).to_dict()
                    results.append({"change_type": "Sudden (before to after)", **{k.replace("measure_", ""): v for k, v in deltas.items()}})

            elif change_type == "gradual_start":
                if not prev_row.empty and not curr_row.empty:
                    deltas = (curr_row.iloc[0][measure_columns] - prev_row.iloc[0][measure_columns]).to_dict()
                    results.append({"change_type": "Gradual (before to during)", **{k.replace("measure_", ""): v for k, v in deltas.items()}})

            elif change_type == "gradual_end":
                during_row = df[df["end_change_id"] == change_id - 1]
                after_row = df[df["end_change_id"] == change_id]
                if not during_row.empty and not after_row.empty and not prev_row.empty:
                    before = prev_row.iloc[0][measure_columns]
                    during = during_row.iloc[0][measure_columns]
                    after = after_row.iloc[0][measure_columns]

                    results.append({"change_type": "Gradual (before to after)", **{k.replace("measure_", ""): v for k, v in (after - before).items()}})
                    results.append({"change_type": "Gradual (during to after)", **{k.replace("measure_", ""): v for k, v in (after - during).items()}})
            
            elif change_type == "incremental":
                if not prev_row.empty and not curr_row.empty:
                    deltas = (curr_row.iloc[0][measure_columns] - prev_row.iloc[0][measure_columns]).to_dict()
                    results.append({"change_type": "Incremental (before to after)", **{k.replace("measure_", ""): v for k, v in deltas.items()}})

    return pd.DataFrame(results)


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
                    formatted = f"{mean:+.3f} ({std:.3f})"
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


def main():
    print("#### Starting to combine drift analysis results ####")
    window_dict = load_complexity_per_window_dict()
    if not window_dict:
        print("No datasets with complexity_per_window.csv found.")
        return
    drift_info_by_dataset = load_drift_info(window_dict)
    results_df = compute_complexity_deltas(window_dict, drift_info_by_dataset)
    save_boxplots(results_df)

    aggregated_table = save_aggregated_table(results_df)
    print(aggregated_table)
    simple_aggregated_table = save_simple_aggregated_table(aggregated_table)
    print(simple_aggregated_table)

if __name__ == "__main__":
    main()
