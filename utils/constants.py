"""Constants and configuration paths for the drifts-and-complexity project."""

from __future__ import annotations

from pathlib import Path

### DIRECTORIES

ROOT = Path(__file__).resolve().parents[1]

# configs
DATA_DICTIONARY_FILE_PATH = ROOT / "data" / "data_dictionary.json"

# drift characterization
DRIFT_CHARACTERIZATION_DIR = ROOT / "plugins" / "drift_characterization"
DRIFT_CHARACTERIZATION_TEMP_INPUT_DIR = DRIFT_CHARACTERIZATION_DIR / "input"
DRIFT_CHARACTERIZATION_TEMP_OUTPUT_DIR = DRIFT_CHARACTERIZATION_DIR / "output"
DRIFT_CHARACTERIZATION_SCRIPT = Path(
    "main.py"
)  # relative to DRIFT_CHARACTERIZATION_DIR

# results
RESULTS_DIR = ROOT / "results"
DRIFT_CHARACTERIZATION_RESULTS_DIR = RESULTS_DIR / "drift_detection"
COMPLEXITY_RESULTS_DIR = RESULTS_DIR / "complexity_assessment"
COMPLEXITY_PLOTS_DIR = COMPLEXITY_RESULTS_DIR
COMBINED_RESULTS_DIR = RESULTS_DIR / "combined_results"
COMBINED_RESULTS_TABLE_DIR = COMBINED_RESULTS_DIR / "tables"
COMBINED_RESULTS_BOXPLOT_DIR = COMBINED_RESULTS_DIR / "boxplots"
CORRELATION_RESULTS_DIR = RESULTS_DIR / "correlations"

# study-specific results
BIAS_STUDY_RESULTS_DIR = RESULTS_DIR / "bias_study"
CHANGE_STUDY_RESULTS_DIR = RESULTS_DIR / "change_study"

### CONSTANT CONFIGURATIONS
DEFAULT_CHANGE_POINT_PARAMETER_SETTING = "processGraphsPDefaultWDefault"
DEFAULT_COMPLEXITY_WINDOW_SETTING = "cp_default"
DEFAULT_CORRELATION_ANALYSIS_NAME = "default"

### METRIC NAMES
# All available complexity metrics in the system
ALL_METRIC_NAMES = [
    "Number of Events",
    "Number of Distinct Activities",
    "Number of Traces",
    "Number of Distinct Traces",
    "Number of Distinct Activity Transitions",
    "Min. Trace Length",
    "Avg. Trace Length",
    "Max. Trace Length",
    "Percentage of Distinct Traces",
    "Average Distinct Activities per Trace",
    "Structure",
    "Estimated Number of Acyclic Paths",
    "Number of Ties in Paths to Goal",
    "Lempel-Ziv Complexity",
    "Average Affinity",
    "Deviation from Random",
    "Average Edit Distance",
    "Sequence Entropy",
    "Normalized Sequence Entropy",
    "Variant Entropy",
    "Normalized Variant Entropy",
]

# Metrics that have both trace-based and distribution-based implementations
DUAL_VARIANT_METRIC_NAMES = [
    "Number of Distinct Traces",
    "Number of Distinct Activities",
    "Number of Distinct Activity Transitions",
]

METRIC_BASIS_MAP = {
    # Size (counts)
    "Number of Events": "OC",
    "Number of Traces": "OC",
    # Population counts (richness-style)
    "Number of Distinct Activities": "PC",
    "Number of Distinct Traces": "PC",
    "Number of Distinct Activity Transitions": "PC",
    "Structure": "PC",
    "Estimated Number of Acyclic Paths": "PC",
    "Percentage of Distinct Traces": "PC",
    # Population distributions (moments / summaries)
    "Min. Trace Length": "PD",
    "Avg. Trace Length": "PD",
    "Max. Trace Length": "PD",
    "Average Distinct Activities per Trace": "PD",
    # Concrete trace–based (sequence/graph derived)
    "Number of Ties in Paths to Goal": "CT",
    "Lempel-Ziv Complexity": "CT",
    "Average Affinity": "CT",
    "Deviation from Random": "CT",
    "Average Edit Distance": "CT",
    "Sequence Entropy": "CT",
    "Normalized Sequence Entropy": "CT",
    "Variant Entropy": "CT",
    "Normalized Variant Entropy": "CT",
}

METRIC_DIMENSION_MAP = {
    "Number of Events": "Size",
    "Number of Distinct Activities": "Size",
    "Number of Traces": "Size",
    "Number of Distinct Traces": "Size",
    "Number of Distinct Activity Transitions": "Size",
    "Min. Trace Length": "Size",
    "Avg. Trace Length": "Size",
    "Max. Trace Length": "Size",
    "Percentage of Distinct Traces": "Variation",
    "Average Distinct Activities per Trace": "Variation",
    "Structure": "Variation",
    "Estimated Number of Acyclic Paths": "Variation",
    "Number of Ties in Paths to Goal": "Variation",
    "Lempel-Ziv Complexity": "Variation",
    "Average Affinity": "Distance",
    "Deviation from Random": "Distance",
    "Average Edit Distance": "Distance",
    "Sequence Entropy": "Graph Entropy",
    "Normalized Sequence Entropy": "Graph Entropy",
    "Variant Entropy": "Graph Entropy",
    "Normalized Variant Entropy": "Graph Entropy",
}

PC_METRICS = [metric for metric, basis in METRIC_BASIS_MAP.items() if basis == "PC"]

# Basis ordering for LaTeX tables
BASIS_ORDER = ["OC", "PC", "PD", "CT"]

# Column name mapping for LaTeX display (CSV column name -> LaTeX display name)
# When using this map, column names are NOT escaped (they're already formatted)
COLUMN_NAME_MAP = {
    # Basic identifiers
    "Metric": "Metric",
    "Basis": "Basis",
    "Log": "Log",
    # Correlation columns
    "Pearson_Rho": "Pearson $\\rho$",
    "Spearman_Rho": "Spearman $\\rho$",
    "Delta_PearsonSpearman": "$\\Delta$ Pearson-Spearman",
    "Chosen_Rho_before": "Chosen $\\rho$ (before)",
    "Chosen_Rho_after": "Chosen $\\rho$ (after)",
    "Abs_Delta_Chosen_Rho": "$|\\Delta|$ Chosen $\\rho$",
    "Delta_Pearson": "$\\Delta$ Pearson",
    "Delta_Spearman": "$\\Delta$ Spearman",
    # Shape and correlation type
    "Shape": "Shape",
    "Shape_before": "Shape (before)",
    "Shape_after": "Shape (after)",
    "Preferred_Correlation": "Preferred Correlation",
    "Preferred_Correlation_before": "Preferred Correlation (before)",
    "Preferred_Correlation_after": "Preferred Correlation (after)",
    "Chosen_Correlation": "Chosen Correlation",
    # Statistical tests
    "Z_test_stat": "$Z$ statistic",
    "Z_test_p": "$Z$ test $p$-value",
    "z_type": "$z$ type",
    "Significant_Improvement": "Significant Improvement",
    # P-values
    "Pearson_P": "Pearson $p$",
    "Spearman_P": "Spearman $p$",
    # CI and Plateau
    "RelCI_50": "RelCI (50)",
    "RelCI_250": "RelCI (250)",
    "RelCI_500": "RelCI (500)",
    "Plateau_n": "Plateau $n$",
    "RelCI_50_before": "RelCI (50, before)",
    "RelCI_50_after": "RelCI (50, after)",
    "RelCI_50_delta": "$\\Delta$ RelCI (50)",
    "RelCI_250_before": "RelCI (250, before)",
    "RelCI_250_after": "RelCI (250, after)",
    "RelCI_250_delta": "$\\Delta$ RelCI (250)",
    "RelCI_500_before": "RelCI (500, before)",
    "RelCI_500_after": "RelCI (500, after)",
    "RelCI_500_delta": "$\\Delta$ RelCI (500)",
    "Plateau_n_before": "Plateau $n$ (before)",
    "Plateau_n_after": "Plateau $n$ (after)",
    "Plateau_n_delta": "$\\Delta$ Plateau $n$",
    # Other
    "FPC_used": "FPC Used",
    "Row_Status": "Row Status",
    "ts": "Timestamp",
}


# Metrics that are trace-based only
TRACE_ONLY_METRIC_NAMES = [
    name for name in ALL_METRIC_NAMES if name not in DUAL_VARIANT_METRIC_NAMES
]

# Metric shorthand handles for command line usage
METRIC_SHORTHAND = {
    # Basic counts
    "events": "Number of Events",
    "distinct_activities": "Number of Distinct Activities",
    "traces": "Number of Traces",
    "distinct_traces": "Number of Distinct Traces",
    "transitions": "Number of Distinct Activity Transitions",
    # Trace length metrics
    "min_length": "Min. Trace Length",
    "avg_length": "Avg. Trace Length",
    "max_length": "Max. Trace Length",
    "pct_distinct": "Percentage of Distinct Traces",
    "avg_activities": "Average Distinct Activities per Trace",
    # Structural metrics
    "structure": "Structure",
    "acyclic_paths": "Estimated Number of Acyclic Paths",
    "ties": "Number of Ties in Paths to Goal",
    "lz": "Lempel-Ziv Complexity",
    "affinity": "Average Affinity",
    "deviation": "Deviation from Random",
    "edit_distance": "Average Edit Distance",
    # Entropy metrics
    "seq_entropy": "Sequence Entropy",
    "norm_seq_entropy": "Normalized Sequence Entropy",
    "var_entropy": "Variant Entropy",
    "norm_var_entropy": "Normalized Variant Entropy",
}
