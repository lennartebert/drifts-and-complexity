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
    "Number of Traces",
    "Number of Distinct Activities",
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
    "Number of Events": "Length",
    "Number of Traces": "Length",
    "Number of Distinct Activities": "Size",
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

# Dimensions ordering for LaTeX tables
DIMENSIONS_ORDER = ["Length", "Size", "Variation", "Distance", "Graph Entropy"]

# Column name mapping for LaTeX display (CSV column name -> LaTeX display name)
# When using this map, column names are NOT escaped (they're already formatted)
# Consolidated mapping from internal column names (with spaces) to LaTeX display names
COLUMN_NAMES_TO_LATEX_MAP = {
    # Basic identifiers
    "Metric": "Measure",  # LaTeX uses "Measure" instead of "Metric"
    "Basis": "Basis",
    "Dimension": "Dimension",
    "Log": "Log",
    "Sample Size": "Sample Size",
    "Sample ID": "Sample ID",
    "Value": "Value",
    # Analysis result columns
    "Mean Value": "Mean Value",
    "Sample CI Low": "Sample CI Low",
    "Sample CI High": "Sample CI High",
    "Sample CI Rel Width": "Sample CI Rel Width",
    "Bootstrap CI Low": "Bootstrap CI Low",
    "Bootstrap CI High": "Bootstrap CI High",
    # Correlation columns
    "Pearson Rho": "Pearson $\\rho$",
    "Pearson P": "Pearson $p$",
    "Spearman Rho": "Spearman $\\rho$",
    "Spearman P": "Spearman $p$",
    "Delta Pearson Spearman": "$\\Delta$ Pearson-Spearman",
    "Chosen Rho Before": "Chosen $\\rho$ (before)",
    "Chosen Rho After": "Chosen $\\rho$ (after)",
    "Abs Delta Chosen Rho": "$|\\Delta|$ Chosen $\\rho$",
    "Delta Pearson": "$\\Delta$ Pearson",
    "Delta Spearman": "$\\Delta$ Spearman",
    # Shape and correlation type
    "Shape": "Shape",
    "Shape Before": "Shape (before)",
    "Shape After": "Shape (after)",
    "Preferred Correlation": "Preferred Correlation",
    "Preferred Correlation Before": "Preferred Correlation (before)",
    "Preferred Correlation After": "Preferred Correlation (after)",
    "Chosen Correlation": "Chosen Correlation",
    # Statistical tests
    "Z Test Stat": "$Z$ statistic",
    "Z Test P": "$Z$ test $p$-value",
    "Z Type": "$z$ type",
    "Significant Improvement": "Significant Improvement",
    # CI and Plateau
    "RelCI 50": "RelCI (50)",
    "RelCI 250": "RelCI (250)",
    "RelCI 500": "RelCI (500)",
    "Plateau n": "Plateau $n$",
    "Plateau Found": "Plateau Found",
    "Plateau Reached": "Plateau Reached",
    "RelCI 50 Before": "RelCI (50, before)",
    "RelCI 50 After": "RelCI (50, after)",
    "RelCI 50 Delta": "$\\Delta$ RelCI (50)",
    "RelCI 250 Before": "RelCI (250, before)",
    "RelCI 250 After": "RelCI (250, after)",
    "RelCI 250 Delta": "$\\Delta$ RelCI (250)",
    "RelCI 500 Before": "RelCI (500, before)",
    "RelCI 500 After": "RelCI (500, after)",
    "RelCI 500 Delta": "$\\Delta$ RelCI (500)",
    "Plateau n Before": "Plateau $n$ (before)",
    "Plateau n After": "Plateau $n$ (after)",
    "Plateau n Delta": "$\\Delta$ Plateau $n$",
    # Sample and population sizes
    "n": "$n$",
    "N Pop": "$N$",
    "n Before": "$n$ (before)",
    "n After": "$n$ (after)",
    "N Pop Before": "$N$ (before)",
    "N Pop After": "$N$ (after)",
    # Other
    "FPC Used": "FPC Used",
    "Row Status": "Row Status",
    "Timestamp": "Timestamp",
    # Improvement columns
    "Pearson Improvement": "Pearson Improvement",
    "Spearman Improvement": "Spearman Improvement",
    "Delta Rho": "Delta Rho",
}

# Float columns that need special formatting in CSV/DataFrame operations
FLOAT_COLUMNS = [
    "Pearson Rho",
    "Spearman Rho",
    "Pearson P",
    "Spearman P",
    "Delta Pearson Spearman",
    "RelCI 50",
    "RelCI 250",
    "RelCI 500",
    "Plateau n",
]


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

# Mapping from metric names to LaTeX display names
# Applies abbreviations: "Average" → "Avg.", "Number of" → "\#", "Percentage of" → "%"
METRIC_NAMES_TO_LATEX_MAP = {
    "Number of Events": "\\# Events",
    "Number of Distinct Activities": "\\# Distinct Activities",
    "Number of Traces": "\\# Traces",
    "Number of Distinct Traces": "\\# Distinct Traces",
    "Number of Distinct Activity Transitions": "\\# Distinct Activity Transitions",
    "Min. Trace Length": "Min. Trace Length",
    "Avg. Trace Length": "Avg. Trace Length",
    "Max. Trace Length": "Max. Trace Length",
    "Percentage of Distinct Traces": "\\% Distinct Traces",
    "Average Distinct Activities per Trace": "Avg. Distinct Activities per Trace",
    "Structure": "Structure",
    "Estimated Number of Acyclic Paths": "Estimated \\# Acyclic Paths",
    "Number of Ties in Paths to Goal": "\\# Ties in Paths to Goal",
    "Lempel-Ziv Complexity": "Lempel-Ziv Complexity",
    "Average Affinity": "Avg. Affinity",
    "Deviation from Random": "Deviation from Random",
    "Average Edit Distance": "Avg. Edit Distance",
    "Sequence Entropy": "Sequence Entropy",
    "Normalized Sequence Entropy": "Normalized Sequence Entropy",
    "Variant Entropy": "Variant Entropy",
    "Normalized Variant Entropy": "Normalized Variant Entropy",
}
