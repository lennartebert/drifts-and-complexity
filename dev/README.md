# Development Tools & Configuration

This directory contains all development-related tools, configuration files, and documentation for the drifts-and-complexity project.

## 📁 Directory Structure

```
dev/
├── config/           # Development configuration files
│   ├── mypy.ini      # MyPy type checking configuration
│   ├── pyproject.toml # Project metadata and tool configurations
│   ├── pytest.ini    # Pytest configuration
│   ├── .pre-commit-config.yaml # Pre-commit hooks
│   └── Makefile      # Development commands
├── scripts/          # Development utility scripts
│   ├── fix-types.py  # Type error analysis and suggestions (if exists)
│   ├── mypy-lenient.ini # Lenient MyPy configuration
│   ├── reorganize_submodules.py # Submodule reorganization script
│   └── gitmodules_template # Git submodules template
└── docs/             # Development documentation
    ├── TYPE_CHECKING.md # Type checking guide
    ├── MYPY_SETUP.md    # MyPy setup and troubleshooting
    ├── DEPENDENCY_MANAGEMENT.md # Dependency management guide
    └── SUBMODULE_REORGANIZATION.md # Submodule reorganization guide
```

## 🚀 Quick Start

1. **Run type checking:**
   ```bash
   # Using mypy directly with dev config
   mypy --config-file=dev/config/mypy.ini utils
   
   # Strict type checking
   mypy --config-file=dev/config/mypy.ini --strict utils
   ```

2. **Run tests:**
   ```bash
   # Using pytest with dev config
   pytest --config-file=dev/config/pytest.ini
   
   # Or directly
   pytest
   ```

## 🔧 Configuration Files

### MyPy (`config/mypy.ini`)
- Strict type checking configuration
- Excludes `plugins/` directory from checking
- Configured for Python 3.10+

### PyProject (`config/pyproject.toml`)
- Project metadata and dependencies
- Tool configurations (black, isort, mypy)
- Development dependencies

### Pre-commit (`config/.pre-commit-config.yaml`)
- Automated code quality checks
- Excludes `plugins/` directory
- Runs on every commit

### Makefile (`config/Makefile`)
- Convenient development commands
- Type checking, testing, installation

## 📝 Development Scripts

- **`fix-types.py`** - Analyze type errors and suggest fixes (if exists)
- **`reorganize_submodules.py`** - Automated submodule reorganization

## 📚 Documentation

- **Type Checking Guide** - Comprehensive guide to type hints and MyPy
- **MyPy Setup** - Setup instructions and troubleshooting
- **Dependency Management** - How dependencies are managed in this project
- **Submodule Reorganization** - Guide for reorganizing git submodules

## 🎯 Benefits of This Organization

1. **Clean Root Directory** - Main project files are not cluttered with dev tools
2. **Centralized Dev Tools** - All development configuration in one place
3. **Easy Maintenance** - Update dev tools without affecting main project
4. **Team Consistency** - Everyone uses the same development setup
5. **Version Control** - Dev tools are still tracked but organized

## 🔄 Updating Configuration

To update any development configuration:

1. Edit the file in `dev/config/`
2. Run `python dev-setup.py` to update symlinks
3. Commit the changes

This ensures all developers get the updated configuration while keeping the main project clean.