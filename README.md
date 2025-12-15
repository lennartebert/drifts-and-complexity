# Drifts and Complexity - An Empirical Study

This repository contains the code and data for empirical studies on concept drift and process complexity in event logs.

**Author:** Lennart Ebert (lennart.ebert@hu-berlin.de)

## Setup

### Prerequisites

- Python 3.10 or 3.11
- Conda (recommended) or pip

### Installation

1. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate drifts-and-complexity
   ```

2. **Initialize required git submodule (plugin):**
   ```bash
   # Initialize and clone the required vidgof_complexity plugin
   git submodule update --init plugins/vidgof_complexity
   ```

   **Optional plugins:** The following plugins are optional and not installed by default:
   - `drift_characterization`: Used for concept drift detection in the main analysis pipeline. To install:
     ```bash
     git submodule update --init plugins/drift_characterization
     ```
   - `cdrift_evaluation`: Used for drift evaluation analyses. To install:
     ```bash
     git submodule update --init plugins/cdrift_evaluation
     ```

   **Note:** If you encounter authentication errors when installing *optional* plugins, the repositories may be private. Contact the repository maintainer for access if needed.

3. **Verify installation:**
   ```bash
   python --version  # Should be 3.10.x or 3.11.x
   python -c "import pm4py; import pandas; import numpy; print('All packages installed successfully')"
   ```

4. **Data (optional):**
   - The repository already includes all necessary data (BPIC12, RTFMP, and all synthetic logs)
   - The default data dictionary (`data/default_data_dictionary.json`) is configured and ready to use
   - To add your own datasets:
     - Copy `data/default_data_dictionary.json` to `data/data_dictionary.json`
     - Place your XES files in `data/real/<dataset_name>/` or `data/synthetic/<dataset_name>/`
     - Add entries to `data/data_dictionary.json` following the same structure as the default file

## Run/Replicate Analysis

### Sampling-Window Bias: Threats to Construct Validity and Reliability in Process Complexity Measurement

To replicate the bias study analysis, run the bias study script with scenarios 0 and 3:

```bash
python scripts/bias_study/run_bias_study.py 0 3
```

**Scenario descriptions:**
- **Scenario 0** (`synthetic_base`): Synthetic datasets with base configuration
- **Scenario 3** (`real_base`): Real-world datasets (BPIC12, RTFMP) with base configuration

**Additional options:**
- `--test`: Run in test mode with reduced parameters (faster, for testing)
- `--metrics METRIC1 METRIC2 ...`: Calculate only specified metrics (use shorthand names or full names)

**Output:** Bias study results are saved in `results/bias_study/<scenario_name>/`

**Submitted results:** Permanent (submitted) results for this study are saved under `results/perm/CAiSE2026/...`

### Drifts and Complexity Analysis Pipeline

The main analysis pipeline consists of two steps:

1. **Assess complexity in dataset windows:**
   ```bash
   python scripts/change_study/computations/assess_datasets.py
   ```

2. **Combine results across datasets:**
   ```bash
   python scripts/change_study/computations/combine_results.py
   ```

**Options for `assess_datasets.py`:**
- `--datasets DATASET1 DATASET2 ...`: Process only specified datasets
- `--mode {all,detection-only,complexity-only}`: Control which steps to run
- `--plot-coverage-curves`: Generate iNEXT coverage curves

**Output locations:**
- Complexity results: `results/complexity_assessment/<dataset_name>/`
- Combined results: `results/combined_results/`

## Project Structure

```
drifts-and-complexity/
├── data/                    # Event log data (real and synthetic)
├── utils/                   # Core analysis utilities
├── scripts/                 # Analysis scripts
│   ├── change_study/       # Main analysis scripts
│   └── bias_study/         # Bias analysis scripts
├── results/                 # Analysis outputs
│   └── perm/                # Permanent (submitted) results
├── plugins/                 # External tools and plugins (git submodules)
│   ├── vidgof_complexity/       # Process complexity metrics plugin (required)
│   ├── drift_characterization/  # Concept drift detection plugin (optional)
│   └── cdrift_evaluation/       # Drift evaluation tools plugin (optional)
└── tests/                   # Unit and integration tests
```

**Note:** The `plugins/` directory contains git submodules. The required `vidgof_complexity` plugin is initialized during installation (see Installation section). Optional plugins can be installed separately if needed.

## Features

### Complexity Metrics

The project implements a wide range of process complexity metrics, including:
- **Size-based metrics**: Number of events, traces, activities, variants
- **Variation metrics**: Entropy measures, distinct trace percentage
- **Structure metrics**: Lempel-Ziv complexity, path-based measures
- **Distance metrics**: Deviation from random, edit distances

### Drift Detection

Integration with concept drift detection methods to identify:
- Sudden changes in process behavior
- Gradual transitions between process variants
- Change points in event logs

### Statistical Analysis

- Bootstrap-based confidence intervals
- Population size estimation (Chao1, naive methods)
- Correlation analysis between drift and complexity
- Normalization techniques for fair metric comparison

## Data

### Included Datasets

This repository includes the following datasets ready to use:

**Real-World Event Logs:**
- **BPIC12 (BPI Challenge 2012)**: van Dongen, B. 2012. BPI Challenge 2012, Media types: application/x-gzip, text/xml, Eindhoven University of Technology, April 23. (https://doi.org/10.4121/UUID:3926DB30-F712-4394-AEBC-75976070E91F).
- **RTFMP (Road Traffic Fine Management Process)**: de Leoni, M. (Massimiliano); Mannhardt, Felix (2015): Road Traffic Fine Management Process. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5

**Synthetic Event Logs:**
All synthetic event logs are based on process models (BPMN) from:

Dumas, M., La Rosa, M., Mendling, J., and Reijers, H. A. 2018. Fundamentals of Business Process Management, Berlin, Heidelberg: Springer. (https://doi.org/10.1007/978-3-662-56509-4).

For each BPMN model, 10,000 simulation runs were performed using the BIMP simulator (https://bimp.cs.ut.ee/simulator). Simulation settings can be loaded into BIMP by importing the `*_scenario.bpmn` files (e.g., `10-ch9_loan5_scenario.bpmn`, `4-ch3_PurchaseOrder2_scenario.bpmn`) into BIMP. The BIMP outputs were converted from MXML to XES using Fluxicon Disco (https://fluxicon.com/disco/).

### Event Log Format

All event logs must be in **XES format** (eXtensible Event Stream), the standard format for process mining event logs:
- **File extension**: `.xes` or `.xes.gz` (gzipped)
- **Standard**: XES 2.0 (IEEE 1849-2016)

### Adding Your Own Datasets

To add custom datasets:

1. **Copy the default data dictionary:**
   ```bash
   cp data/default_data_dictionary.json data/data_dictionary.json
   ```

2. **Place your XES files** in the appropriate directory:
   - Real-world logs: `data/real/<dataset_name>/`
   - Synthetic logs: `data/synthetic/<dataset_name>/`

3. **Add entries to `data/data_dictionary.json`** following the same structure as the default file. Each entry requires:
   - `name`: Full descriptive name
   - `short_name`: Abbreviated name for displays
   - `type`: `"real"` or `"synthetic"`
   - `path`: Relative path to the XES file

The code automatically uses `data/data_dictionary.json` if it exists, otherwise it falls back to `data/default_data_dictionary.json`.

## Development

### Quick Development Setup

```bash
# Run type checking
mypy utils

# Run tests
pytest

# Format code
black utils tests
isort utils tests
```

### Development Tools

- **Type Checking**: MyPy configuration in `mypy.ini`
- **Code Formatting**: Black, isort, flake8 configured in `pyproject.toml`
- **Pre-commit Hooks**: Automated quality checks in `.pre-commit-config.yaml`
- **Testing**: Pytest configuration in `pytest.ini`


## License

See [LICENSE.txt](LICENSE.txt) for details.
