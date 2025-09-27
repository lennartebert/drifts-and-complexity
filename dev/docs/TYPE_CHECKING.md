# Type Checking with MyPy

This project uses [MyPy](https://mypy.readthedocs.io/) for static type checking to ensure code quality and catch potential bugs early.

## Quick Start

### Basic Type Checking
```bash
# Run basic type checking
mypy utils

# Or use the convenience script
python scripts/typecheck.py
```

### Strict Type Checking
```bash
# Run strict type checking
mypy --strict utils

# Or use the convenience script
python scripts/typecheck.py --mode strict
```

### CI Mode (Most Comprehensive)
```bash
# Run all type checks as in CI
python scripts/typecheck.py --mode ci
```

## Configuration

### MyPy Configuration Files

1. **`mypy.ini`** - Main configuration file with project-specific settings
2. **`pyproject.toml`** - Project metadata and tool configurations

### Key Configuration Features

- **Strict Mode**: Enforces comprehensive type checking
- **Import Handling**: Ignores missing imports for third-party libraries
- **Module-specific Rules**: Different rules for tests vs. main code
- **Error Reporting**: Detailed error messages with context

## Development Workflow

### Pre-commit Hooks
Install pre-commit hooks to automatically run type checking on commit:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### IDE Integration

#### VS Code
1. Install the Python extension
2. Install the MyPy extension
3. Add to your `settings.json`:
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

### Make Commands

```bash
# Type checking
make typecheck          # Basic type checking
make typecheck-strict   # Strict type checking

# All checks
make check-all          # Format, lint, type, test
make ci                 # Full CI pipeline
```

## Common Type Checking Issues

### 1. Missing Type Annotations
```python
# ❌ Bad
def process_data(data):
    return data.upper()

# ✅ Good
def process_data(data: str) -> str:
    return data.upper()
```

### 2. Optional Types
```python
# ❌ Bad
def get_value(key: str) -> str:
    return cache.get(key)  # Could return None

# ✅ Good
def get_value(key: str) -> Optional[str]:
    return cache.get(key)
```

### 3. Generic Types
```python
# ❌ Bad
def process_items(items):
    return [item.upper() for item in items]

# ✅ Good
def process_items(items: List[str]) -> List[str]:
    return [item.upper() for item in items]
```

### 4. Protocol Types
```python
# ✅ Good - Use protocols for duck typing
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def render(obj: Drawable) -> None:
    obj.draw()
```

## Ignoring Type Checks

### File-level Ignores
```python
# mypy: ignore-errors
```

### Line-level Ignores
```python
result = some_function()  # type: ignore[return-value]
```

### Function-level Ignores
```python
def legacy_function():  # type: ignore[misc]
    pass
```

## Continuous Integration

The project includes GitHub Actions workflows that run type checking:

- **Basic CI**: Runs on every push/PR
- **Strict Type Check**: Separate job for comprehensive checking
- **Matrix Testing**: Tests on Python 3.10 and 3.11

## Troubleshooting

### Common Errors

1. **"Module has no attribute"**
   - Check if the module is properly imported
   - Verify the attribute exists in the module

2. **"Incompatible return type"**
   - Check return type annotations
   - Ensure all code paths return the same type

3. **"Argument 1 has incompatible type"**
   - Check function parameter types
   - Verify argument types match expected types

### Getting Help

1. Check MyPy documentation: https://mypy.readthedocs.io/
2. Use `--show-error-codes` for specific error codes
3. Run with `--verbose` for detailed output
4. Check the project's type checking configuration

## Best Practices

1. **Start with basic type checking** and gradually increase strictness
2. **Use type aliases** for complex types
3. **Prefer protocols over ABCs** for duck typing
4. **Use `from __future__ import annotations`** for forward references
5. **Keep type annotations close to the code** they describe
6. **Use type guards** for runtime type checking
7. **Document complex type relationships** in docstrings

## Examples

### Type Aliases
```python
from typing import TypeAlias

TraceList: TypeAlias = List[Trace]
MeasureDict: TypeAlias = Dict[str, float]
```

### Generic Classes
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
    
    def get(self) -> T:
        return self.value
```

### Union Types
```python
from typing import Union

def process_data(data: Union[str, int]) -> str:
    return str(data)
```

### Type Guards
```python
from typing import TypeGuard

def is_string(value: object) -> TypeGuard[str]:
    return isinstance(value, str)

def process(value: Union[str, int]) -> str:
    if is_string(value):
        return value.upper()  # MyPy knows this is str
    else:
        return str(value)
```
