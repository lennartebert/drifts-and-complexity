# Dependency Management

This project uses **conda** as the primary dependency management system, with **pip** as a fallback for users who prefer it.

## Recommended Approach: Conda

### Why Conda?
- **Complete environment management**: Handles Python, system libraries, and Python packages
- **Better dependency resolution**: Avoids conflicts between system and Python dependencies
- **Reproducible environments**: Exact package versions across different systems
- **Scientific computing optimized**: Better support for numpy, scipy, and related packages

### Setup with Conda
```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate drifts-and-complexity

# Or use the Make command
make install-conda
```

### Updating Dependencies
```bash
# Update the environment
conda env update -f environment.yml

# Or use the Make command
make update-conda
```

## Alternative Approach: Pip

If you prefer pip or need to use it for specific reasons:

### Setup with Pip
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## File Structure

### `environment.yml` (Primary)
- **Complete conda environment** with all dependencies
- **Runtime dependencies**: numpy, pandas, scipy, etc.
- **Development tools**: mypy, black, isort, flake8, pre-commit
- **Type stubs**: types-PyYAML, types-requests, etc.
- **PM4Py ecosystem**: All PM4Py-related packages

### `pyproject.toml` (Secondary)
- **Project metadata** and build configuration
- **Tool configurations**: mypy, black, isort, pytest settings
- **Pip dependencies**: For users who prefer pip over conda
- **Development dependencies**: Available as `[dev]` extra

## Why Not Both?

Having both `requirements-dev.txt` and `environment.yml` creates:
- **Maintenance overhead**: Need to keep both files in sync
- **Version conflicts**: Different version specifications
- **Confusion**: Users don't know which file to use
- **Redundancy**: Same information in multiple places

## Best Practices

1. **Use conda as primary**: Most users should use `environment.yml`
2. **Keep pyproject.toml minimal**: Only essential pip dependencies
3. **Document the approach**: Clear instructions for both methods
4. **Test both approaches**: Ensure both work in CI/CD

## Migration from requirements-dev.txt

The project previously had both files, but this was consolidated:

### What was moved:
- Development tools (mypy, black, isort, flake8, pre-commit)
- Type stubs (types-PyYAML, types-requests, types-setuptools)
- IDE tools (jupyter, ipython)

### What was removed:
- `requirements-dev.txt` file (redundant)

### What was kept:
- `environment.yml` (primary dependency management)
- `pyproject.toml` (project metadata and pip fallback)

## Troubleshooting

### Conda Issues
```bash
# If environment creation fails
conda clean --all
conda env create -f environment.yml

# If packages conflict
conda env update -f environment.yml --prune
```

### Pip Issues
```bash
# If installation fails
pip install --upgrade pip
pip install -e ".[dev]"

# If type stubs are missing
pip install types-PyYAML types-requests types-setuptools
```

### Mixed Environment Issues
```bash
# If you have both conda and pip packages
conda list
pip list
# Check for conflicts and remove conflicting packages
```

## CI/CD Considerations

The GitHub Actions workflow uses conda for consistency:
```yaml
- name: Set up Python ${{ matrix.python-version }}
  uses: actions/setup-python@v4
  with:
    python-version: ${{ matrix.python-version }}

- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
```

This ensures both conda and pip users can run the same CI checks.
