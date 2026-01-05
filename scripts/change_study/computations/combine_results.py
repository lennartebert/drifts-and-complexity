import argparse
import os
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is on sys.path so local imports work when run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from utils import constants

# Define drift type order
drift_type_order = [
    "Sudden (before to after)",
    "Gradual (before to after)",
    "Gradual (before to during)",
    "Gradual (during to after)",
]


def load_complexity_per_window_dict(datasets, complexity_window_string):
    """"""
    dataset_file_dict = {
        dataset: constants.CHANGE_STUDY_RESULTS_DIR
        / "complexity_assessment"
        / dataset
        / complexity_window_string
        / "complexity.csv"
        for dataset in datasets
    }
    complexity_per_window_df_dict = {}
    for dataset, f in dataset_file_dict.items():
        if not f.exists() or not f.is_file():
            continue
        df = pd.read_csv(f)
        df["dataset"] = dataset
        df["start_change_point"] = df["start_change_point"].astype("Int64")
        df["end_change_point"] = df["end_change_point"].astype("Int64")
        df["id"] = df["id"].astype(int)
        complexity_per_window_df_dict[dataset] = df
    return complexity_per_window_df_dict


def load_drift_info(
    complexity_per_window_df_dict,
    cp_parameter_setting=constants.DEFAULT_CHANGE_POINT_PARAMETER_SETTING,
):
    drift_info_by_dataset = {}
    for dataset in complexity_per_window_df_dict.keys():
        path = (
            constants.CHANGE_STUDY_RESULTS_DIR
            / "drift_detection"
            / dataset
            / f"results_{dataset}_{cp_parameter_setting}.csv"
        )
        # Initialize with empty dict - will be populated if valid change points exist
        drift_info_by_dataset[dataset] = {}

        # Check if file exists
        if not path.exists():
            continue

        drift_info = pd.read_csv(path)
        # drift info may be empty for some datasets, or all change_ids may be "na"
        if drift_info.empty:
            continue

        # Check if all change_ids are "na" (no valid change points)
        if "calc_change_id" in drift_info.columns:
            if drift_info["calc_change_id"].eq("na").all():
                continue

        # If we get here, there are valid change points
        drift_info["calc_change_id"] = drift_info["calc_change_id"].astype("Int64")
        drift_info_by_dataset[dataset] = drift_info.set_index("calc_change_id").to_dict(
            orient="index"
        )
    return drift_info_by_dataset


def get_drift_info_summary_table(drift_info_by_dataset):
    drift_info_summary_dict = {}

    for dataset, drift_info_dict in drift_info_by_dataset.items():
        result = {}
        sudden_changes = []
        gradual_changes = []

        # Handle datasets with no drifts found
        if drift_info_dict == {}:
            result["# Total Changes"] = 0
            result["# Sudden Changes"] = 0
            result["Sudden Change Points"] = ""
            result["# Gradual Changes"] = 0
            result["Gradual Change Points"] = ""
            drift_info_summary_dict[dataset] = result
            continue

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
    drift_info_summary_df = pd.DataFrame.from_dict(
        drift_info_summary_dict, orient="index"
    )

    # Handle empty DataFrame (no change points detected)
    if drift_info_summary_df.empty:
        return drift_info_summary_df

    # Sort datasets alphabetically by index (dataset names)
    drift_info_summary_df = drift_info_summary_df.sort_index()

    # add total row
    total_row = {
        "# Total Changes": drift_info_summary_df["# Total Changes"].sum(),
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
        measure_columns = [
            col for col in window_df.columns if col.startswith("measure_")
        ]

        for change_id, info in drift_info.items():
            change_type = info["calc_change_type"]
            window_before_change_point = window_df[
                window_df["end_change_point"] == change_id
            ].iloc[0]
            window_after_change_point = window_df[
                window_df["start_change_point"] == change_id
            ].iloc[0]

            # Assert that window_before_change and window_after_change are not empty - there always needs to be a window before/after a change point
            assert (
                not window_before_change_point.empty
            ), f"window_before_change is empty for dataset {dataset}, change_id {change_id}, change_type {change_type}"
            assert (
                not window_after_change_point.empty
            ), f"window_after_change is empty for dataset {dataset}, change_id {change_id}, change_type {change_type}"

            if change_type == "sudden":
                # Compute relative differences: (after - before) / before
                before_vals = window_before_change_point[measure_columns]
                after_vals = window_after_change_point[measure_columns]
                # Replace zeros in before_vals with NaN to avoid division by zero
                before_vals_safe = before_vals.replace(0, np.nan)
                deltas = ((after_vals - before_vals) / before_vals_safe).to_dict()
                results.append(
                    {
                        "change_type": "Sudden (before to after)",
                        **{k.replace("measure_", ""): v for k, v in deltas.items()},
                    }
                )

            elif change_type == "gradual_start":
                # Compute relative differences: (after - before) / before
                before_vals = window_before_change_point[measure_columns]
                after_vals = window_after_change_point[measure_columns]
                before_vals_safe = before_vals.replace(0, np.nan)
                deltas = ((after_vals - before_vals) / before_vals_safe).to_dict()
                results.append(
                    {
                        "change_type": "Gradual (before to during)",
                        **{k.replace("measure_", ""): v for k, v in deltas.items()},
                    }
                )

                # also record the before to after change
                window_after_gradual_end = window_df[
                    window_df["start_change_point"] == change_id + 1
                ].iloc[0]
                assert (
                    not window_after_gradual_end.empty
                ), f"window_after_gradual_end is empty for dataset {dataset}, change_id {change_id}, change_type {change_type}"
                after_end_vals = window_after_gradual_end[measure_columns]
                before_vals_safe = before_vals.replace(0, np.nan)
                deltas = ((after_end_vals - before_vals) / before_vals_safe).to_dict()
                results.append(
                    {
                        "change_type": "Gradual (before to after)",
                        **{k.replace("measure_", ""): v for k, v in deltas.items()},
                    }
                )

            elif change_type == "gradual_end":
                # Compute relative differences: (after - before) / before
                before_vals = window_before_change_point[measure_columns]
                after_vals = window_after_change_point[measure_columns]
                before_vals_safe = before_vals.replace(0, np.nan)
                deltas = ((after_vals - before_vals) / before_vals_safe).to_dict()
                results.append(
                    {
                        "change_type": "Gradual (during to after)",
                        **{k.replace("measure_", ""): v for k, v in deltas.items()},
                    }
                )

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


def save_all_change_points(results_df, cp_parameter_setting, results_subfolder="real"):
    """Save all individual change points with their delta values (not aggregated).

    Parameters
    ----------
    results_df
        DataFrame with all change points and their complexity deltas.
    cp_parameter_setting
        Change point parameter setting name.
    results_subfolder
        Subfolder name within combined_results directory.
    """
    output_dir = (
        constants.CHANGE_STUDY_RESULTS_DIR
        / "combined_results"
        / results_subfolder
        / "tables"
        / cp_parameter_setting
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the full results DataFrame
    results_df.to_csv(output_dir / "complexity_delta_all.csv", index=False)
    print(f"Saved all change points to: {output_dir / 'complexity_delta_all.csv'}")


def save_aggregated_table(results_df, cp_parameter_setting, results_subfolder="real"):
    """Save aggregated statistics including one-sample t-test results.

    Parameters
    ----------
    results_df
        DataFrame with all change points and their complexity deltas.
    cp_parameter_setting
        Change point parameter setting name.
    results_subfolder
        Subfolder name within combined_results directory.

    Returns
    -------
    pd.DataFrame
        Aggregated table with statistics and t-test results.
    """
    # Handle empty DataFrame (no change points detected)
    if results_df.empty or "change_type" not in results_df.columns:
        # Create empty summary DataFrame with expected structure
        summary_df = pd.DataFrame(
            columns=[
                "mean",
                "min",
                "max",
                "std",
                "count",
                "t_statistic",
                "p_value",
                "cohens_d",
                "ci_lower",
                "ci_upper",
                "significance",
            ]
        )
        summary_df = summary_df.set_index(
            pd.MultiIndex.from_tuples([], names=["change_type", "measure"])
        )
        output_dir = (
            constants.CHANGE_STUDY_RESULTS_DIR
            / "combined_results"
            / results_subfolder
            / "tables"
            / cp_parameter_setting
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "complexity_delta_aggregated.csv")
        return summary_df

    results_df_clean = results_df.dropna()
    change_types = results_df_clean["change_type"].unique()

    measure_cols = [col for col in results_df_clean.columns if col != "change_type"]
    records = []
    alpha = 0.05  # Significance level

    for measure in measure_cols:
        for change_type in change_types:
            subset = results_df_clean[results_df_clean["change_type"] == change_type][
                measure
            ]

            # Basic statistics
            mean_val = subset.mean()
            min_val = subset.min()
            max_val = subset.max()
            std_val = subset.std()
            count_val = subset.count()

            # One-sample t-test (testing if mean is significantly different from 0)
            t_statistic = np.nan
            p_value = np.nan
            cohens_d = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            significance = "not significant"

            if count_val >= 2:  # Need at least 2 observations for t-test
                try:
                    # Perform one-sample t-test
                    t_result = stats.ttest_1samp(subset, 0)
                    t_statistic = t_result.statistic
                    p_value = t_result.pvalue

                    # Cohen's d effect size
                    if std_val > 0:
                        cohens_d = mean_val / std_val

                    # 95% confidence interval
                    ci = stats.t.interval(
                        0.95, df=count_val - 1, loc=mean_val, scale=stats.sem(subset)
                    )
                    ci_lower = ci[0]
                    ci_upper = ci[1]

                    # Determine significance interpretation
                    if p_value < alpha:
                        if mean_val > 0:
                            significance = "significant positive change"
                        else:
                            significance = "significant negative change"
                    else:
                        significance = "not significant"
                except Exception as e:
                    # If t-test fails, leave as NaN
                    pass

            records.append(
                {
                    "measure": measure,
                    "change_type": change_type,
                    "mean": mean_val,
                    "min": min_val,
                    "max": max_val,
                    "std": std_val,
                    "count": count_val,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "significance": significance,
                }
            )

    summary_df = pd.DataFrame(records)
    summary_df.set_index(["measure", "change_type"], inplace=True)
    summary_df = summary_df.reorder_levels(["change_type", "measure"]).sort_index()
    # Ensure the output directory exists
    output_dir = (
        constants.CHANGE_STUDY_RESULTS_DIR
        / "combined_results"
        / results_subfolder
        / "tables"
        / cp_parameter_setting
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "complexity_delta_aggregated.csv")
    return summary_df


def save_avg_table(aggregated_table_df, cp_parameter_setting, results_subfolder="real"):
    """Save average table with mean and std (same format as old complexity_delta_simple).

    Parameters
    ----------
    aggregated_table_df
        Aggregated table DataFrame with MultiIndex (change_type, measure).
    cp_parameter_setting
        Change point parameter setting name.
    results_subfolder
        Subfolder name within combined_results directory.

    Returns
    -------
    pd.DataFrame
        Average table with mean (std) format.
    """
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
            row["Instances"] = (
                int(subset["count"].iloc[0]) if "count" in subset.columns else None
            )

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
    final_df.to_csv(
        constants.CHANGE_STUDY_RESULTS_DIR
        / "combined_results"
        / results_subfolder
        / "tables"
        / cp_parameter_setting
        / "complexity_delta_avg.csv"
    )

    return final_df


def save_ttest_table(
    aggregated_table_df, cp_parameter_setting, results_subfolder="real"
):
    """Save t-test significance table with interpretation.

    Parameters
    ----------
    aggregated_table_df
        Aggregated table DataFrame with MultiIndex (change_type, measure).
    cp_parameter_setting
        Change point parameter setting name.
    results_subfolder
        Subfolder name within combined_results directory.

    Returns
    -------
    pd.DataFrame
        T-test table with significance interpretation.
    """
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
            row["Instances"] = (
                int(subset["count"].iloc[0]) if "count" in subset.columns else None
            )

            for measure in measures:
                if measure in subset.index:
                    # Get significance interpretation from aggregated table
                    significance = subset.loc[measure].get(
                        "significance", "not significant"
                    )
                    row[measure] = significance
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
    final_df.to_csv(
        constants.CHANGE_STUDY_RESULTS_DIR
        / "combined_results"
        / results_subfolder
        / "tables"
        / cp_parameter_setting
        / "complexity_delta_ttest.csv"
    )

    return final_df


def save_boxplots(results_df, cp_parameter_setting, results_subfolder="real"):
    measure_names = [col for col in results_df.columns if col != "change_type"]

    for measure in measure_names:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=results_df, x="change_type", y=measure, order=drift_type_order)
        plt.xlabel("Change Type")
        plt.ylabel(measure.replace("_", " ").title())
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        # Ensure the output directory exists
        output_dir = (
            constants.CHANGE_STUDY_RESULTS_DIR
            / "combined_results"
            / results_subfolder
            / "boxplots"
            / cp_parameter_setting
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{measure}_boxplot.png", dpi=300)
        plt.close()


def main(
    datasets=None,
    cp_parameter_setting=constants.DEFAULT_CHANGE_POINT_PARAMETER_SETTING,
    complexity_window_setting=constants.DEFAULT_COMPLEXITY_WINDOW_SETTING,
    results_subfolder="real",
):
    print("#### Starting to combine drift analysis results ####")
    if datasets is None:
        # Get all folder names (1st level child) under change_study/complexity_assessment
        # Exclude TEST_BPIC12 by default
        datasets = [
            d.name
            for d in (
                constants.CHANGE_STUDY_RESULTS_DIR / "complexity_assessment"
            ).iterdir()
            if d.is_dir() and d.name != "TEST_BPIC12"
        ]

    complexity_window_string = f"{cp_parameter_setting}__{complexity_window_setting}"

    window_dict = load_complexity_per_window_dict(datasets, complexity_window_string)
    if not window_dict:
        print(
            f"No datasets with complexity_per_window.csv for {cp_parameter_setting} found."
        )
        return
    drift_info_by_dataset = load_drift_info(window_dict, cp_parameter_setting)

    # get summary of drift info
    drift_info_summary_table_df = get_drift_info_summary_table(
        drift_info_by_dataset=drift_info_by_dataset
    )
    # Ensure the output directory exists
    output_dir = (
        constants.CHANGE_STUDY_RESULTS_DIR
        / "combined_results"
        / results_subfolder
        / "tables"
        / complexity_window_string
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    drift_info_summary_table_df.to_csv(output_dir / "drift_info_summary.csv")

    results_df = compute_complexity_deltas(window_dict, drift_info_by_dataset)

    # Save all individual change points (not aggregated)
    save_all_change_points(results_df, complexity_window_string, results_subfolder)

    if not results_df.empty:
        # Create boxplots for normality checking
        save_boxplots(results_df, complexity_window_string, results_subfolder)

        # Save aggregated table with t-test statistics
        aggregated_table = save_aggregated_table(
            results_df, complexity_window_string, results_subfolder
        )
        print("\nAggregated table with t-test statistics:")
        print(aggregated_table)

        # Save average table (mean and std)
        avg_table = save_avg_table(
            aggregated_table, complexity_window_string, results_subfolder
        )
        print("\nAverage table (mean (std)):")
        print(avg_table)

        # Save t-test significance table
        ttest_table = save_ttest_table(
            aggregated_table, complexity_window_string, results_subfolder
        )
        print("\nT-test significance table:")
        print(ttest_table)
    else:
        print("No change points detected - skipping delta calculations and boxplots")
        # Still create empty aggregated table for consistency
        aggregated_table = save_aggregated_table(
            results_df, complexity_window_string, results_subfolder
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine drift complexity analysis detection results."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset keys to include. If not set, all datasets are used.",
    )
    parser.add_argument(
        "--cp-parameter-setting",
        default=constants.DEFAULT_CHANGE_POINT_PARAMETER_SETTING,
        help="Name of change point parameter setting (e.g., processGraphsPDefaultWDefault)",
    )
    parser.add_argument(
        "--complexity-window-setting",
        default=constants.DEFAULT_COMPLEXITY_WINDOW_SETTING,
        help="Name of complexity window setting (e.g., cp_default)",
    )
    parser.add_argument(
        "--results-subfolder",
        default="real",
        help="Subfolder name within 'combined_results' directory to store results (e.g., 'real', 'synthetic')",
    )

    args = parser.parse_args()

    main(
        datasets=args.datasets,
        cp_parameter_setting=args.cp_parameter_setting,
        complexity_window_setting=args.complexity_window_setting,
        results_subfolder=args.results_subfolder,
    )
