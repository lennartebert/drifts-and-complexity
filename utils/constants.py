from pathlib import Path

DATA_DICTIONARY_FILE_PATH = Path('configuration/data_dictionary.json')

DRIFT_CHARACTERIZATION_DIR = Path("concept-drift-characterization/")
DRIFT_CHARACTERIZATION_TEMP_INPUT_DIR = DRIFT_CHARACTERIZATION_DIR / "input"
DRIFT_CHARACTERIZATION_TEMP_OUTPUT_DIR = DRIFT_CHARACTERIZATION_DIR / "output"
DRIFT_CHARACTERIZATION_SCRIPT = Path("main.py")
DRIFT_CHARACTERIZATION_RESULTS_DIR = Path("results/drift_detection")

COMPLEXITY_RESULTS_DIR = Path("results/complexity_assessment")

COMBINED_RESULTS_DIR = Path("results/combined_results")

COMBINED_RESULTS_TABLE_DIR = COMBINED_RESULTS_DIR / "tables"
COMBINED_RESULTS_BOXPLOT_DIR = COMBINED_RESULTS_DIR / "boxplots"

# create all directories
DRIFT_CHARACTERIZATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COMPLEXITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COMPLEXITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_RESULTS_BOXPLOT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHANGE_POINT_DETECTOR = 'prodrift'
