# Drifts and Complexity - An Empirical Study
By Lennart Ebert (lennart.ebert@hu-berlin.de)

## Getting started

1. In a python 3.11 environment, install dependencies (´´´conda env create -f environment.yml´´´, ´´´conda activate drifts-and-complexity´´´).
2. Install R dependencies by ´´´Rscript install_r_packages.R´´´    
2. Place your data in folder /data/...
3. Create a local copy of the data dictionary and configure for your datasets. The data_dictionary should be saved at data/data_dictionary.json
4. Run assess_datasets.py (e.g., python assess_dataset.py) for assessement of complexity in dataset windows
5. Run combine_results.py (e.g., pyhton combine_results.py) for computing aggregate tables across datasets

## Development

Development tools and configuration are organized in the `dev/` directory. See `dev/README.md` for detailed information.

### Quick Development Setup

```bash
# Run type checking
mypy --config-file=dev/config/mypy.ini utils

# Run tests
pytest --config-file=dev/config/pytest.ini

# Fix type errors (if fix-types.py exists)
python dev/scripts/fix-types.py
```

### Development Tools

- **Type Checking**: MyPy configuration in `dev/config/mypy.ini`
- **Code Formatting**: Black, isort, flake8 configured in `dev/config/`
- **Pre-commit Hooks**: Automated quality checks in `dev/config/.pre-commit-config.yaml`
- **Testing**: Pytest configuration in `dev/config/pytest.ini`