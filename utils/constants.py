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
