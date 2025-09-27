# MyPy Type Checking Setup

This document describes the MyPy type checking setup for the drifts-and-complexity project.

## Overview

MyPy is configured to provide comprehensive static type checking for the `utils/` directory. The setup includes:

- **Strict type checking** for production code
- **Lenient checking** for tests and scripts
- **Third-party library handling** with missing import ignores
- **CI/CD integration** with GitHub Actions
- **Development tools** including pre-commit hooks

## Configuration Files

### 1. `mypy.ini` - Main Configuration
- Strict type checking settings
- Module-specific overrides
- Import handling for third-party libraries

### 2. `pyproject.toml` - Project Metadata
- Project dependencies and metadata
- Tool configurations (mypy, black, isort, pytest)
- Development dependencies (for pip users)

### 3. `environment.yml` - Conda Environment
- Complete conda environment with all dependencies
- Includes both runtime and development tools
- Primary dependency management for the project

### 4. `scripts/mypy-lenient.ini` - Lenient Configuration
- Less strict settings for initial setup
- Useful for gradual migration to strict typing

## Quick Start

### Basic Type Checking
```bash
# Run basic type checking
mypy utils

# Or use the convenience script
python scripts/typecheck.py
```

### Lenient Type Checking
```bash
# Run with lenient settings
mypy --config-file=scripts/mypy-lenient.ini utils
```

### Strict Type Checking
```bash
# Run strict type checking
mypy --strict utils
```

## Current Status

### ✅ Completed
- [x] MyPy configuration files
- [x] Project metadata and dependencies
- [x] CI/CD integration
- [x] Development tools setup
- [x] Basic type annotations added
- [x] Core protocol fixes

### 🔄 In Progress
- [ ] Fix remaining type errors (34 errors remaining)
- [ ] Add missing type stubs
- [ ] Resolve import issues
- [ ] Fix method signature incompatibilities

### 📋 Remaining Issues

#### High Priority
1. **Missing Imports** (2 errors)
   - `utils.population.estimators`
   - `utils.population.inext_adapter`

2. **Type Stub Issues** (1 error)
   - `yaml` library stubs

3. **Method Signature Incompatibilities** (1 error)
   - `LocalMetricsAdapter.compute_measures_for_windows`

#### Medium Priority
4. **Union Type Handling** (6 errors)
   - `Measure | None` attribute access
   - Optional value handling

5. **Unused Type Ignores** (8 errors)
   - Remove unnecessary `# type: ignore` comments

#### Low Priority
6. **Minor Type Issues** (16 errors)
   - Unreachable code
   - Argument type mismatches
   - Variable annotations

## Error Categories

### 1. Import Errors
```
Cannot find implementation or library stub for module named "utils.population.estimators"
```
**Solution**: Create missing modules or add to ignore list

### 2. Type Stub Errors
```
Library stubs not installed for "yaml"
```
**Solution**: Install type stubs or add to ignore list

### 3. Union Type Errors
```
Item "None" of "Measure | None" has no attribute "value"
```
**Solution**: Add proper null checks or use type guards

### 4. Signature Incompatibility
```
Signature of "compute_measures_for_windows" incompatible with supertype
```
**Solution**: Align method signatures with protocol

## Development Workflow

### 1. Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 2. IDE Integration

#### VS Code
1. Install Python extension
2. Install MyPy extension
3. Add to `settings.json`:
```json
{
    "python.linting.mypyEnabled": true,
    "python.linting.mypyArgs": ["--config-file=mypy.ini"]
}
```

#### PyCharm
1. Go to Settings → Languages & Frameworks → Python → Type Checking
2. Enable "Use mypy"
3. Set mypy executable path
4. Add configuration file: `mypy.ini`

### 3. Make Commands
```bash
# Type checking
make typecheck          # Basic type checking
make typecheck-strict   # Strict type checking

# All checks
make check-all          # Format, lint, type, test
make ci                 # Full CI pipeline
```

## Troubleshooting

### Common Issues

1. **"Module has no attribute"**
   - Check if the module is properly imported
   - Verify the attribute exists in the module

2. **"Incompatible return type"**
   - Check return type annotations
   - Ensure all code paths return the same type

3. **"Argument has incompatible type"**
   - Check function parameter types
   - Verify argument types match expected types

### Getting Help

1. Check MyPy documentation: https://mypy.readthedocs.io/
2. Use `--show-error-codes` for specific error codes
3. Run with `--verbose` for detailed output
4. Check the project's type checking configuration

## Best Practices

1. **Start with lenient settings** and gradually increase strictness
2. **Use type aliases** for complex types
3. **Prefer protocols over ABCs** for duck typing
4. **Use `from __future__ import annotations`** for forward references
5. **Keep type annotations close to the code** they describe
6. **Use type guards** for runtime type checking
7. **Document complex type relationships** in docstrings

## Next Steps

1. **Fix missing imports** by creating stub modules or adding to ignore list
2. **Install missing type stubs** for third-party libraries
3. **Resolve method signature incompatibilities** in adapter classes
4. **Add proper null checks** for union types
5. **Remove unused type ignore comments**
6. **Gradually increase strictness** as issues are resolved

## Resources

- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 526 - Variable Annotations](https://peps.python.org/pep-0526/)
- [PEP 563 - Postponed Evaluation of Annotations](https://peps.python.org/pep-0563/)
