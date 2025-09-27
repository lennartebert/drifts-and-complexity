# Population Module Tests

This directory contains comprehensive tests for the population analysis utilities.

## Test Structure

The tests follow the same structure as the source code:

```
tests/utils/population/
├── __init__.py
├── README.md
├── test_population_distribution.py      # Core distribution class tests
├── test_population_distributions.py     # Distributions container tests
└── extractors/
    ├── __init__.py
    ├── test_population_extractor.py          # Abstract base class tests
    ├── test_naive_population_extractor.py    # Naive estimator tests
    └── test_chao1_population_extractor.py    # Chao1 estimator tests
```

## Test Coverage

### PopulationDistribution (`test_population_distribution.py`)

Tests the core `PopulationDistribution` class which handles probability distributions with caching:

- **Initialization**: Valid parameters, edge cases, validation errors
- **Caching System**: Cache invalidation on mutation, cache persistence
- **Edge Cases**: Empty distributions, pure observed/unseen distributions, mixed distributions
- **Numerical Stability**: Probability rescaling, tiny values, floating-point precision
- **Properties**: Thread-safe property access, copy semantics to prevent mutation
- **Complex Data**: Support for complex label structures (trace variants, etc.)

### PopulationDistributions (`test_population_distributions.py`)

Tests the `PopulationDistributions` dataclass container:

- **Dataclass Functionality**: Proper initialization, field access, equality
- **Integration**: Works with different types of `PopulationDistribution` instances
- **Modification**: Post-creation access and property queries
- **Edge Cases**: Empty distributions, various distribution configurations

### PopulationExtractor (`test_population_extractor.py`)

Tests the abstract base class:

- **Abstract Class Behavior**: Cannot be instantiated directly
- **Contract Enforcement**: Concrete implementations must implement `apply`
- **Inheritance**: Proper inheritance hierarchy and method signatures
- **Documentation**: Docstrings and API contract documentation

### NaivePopulationExtractor (`test_naive_population_extractor.py`)

Tests the naive estimation strategy (assumes sample = population):

#### Helper Functions
- `_counts_activities()`: Activity frequency counting
- `_counts_dfg_edges()`: DFG edge frequency counting  
- `_counts_trace_variants()`: Trace variant frequency counting
- `_build_naive_distribution_from_counts()`: Distribution construction

#### Main Class
- **Basic Functionality**: Simple trace processing, distribution creation
- **Edge Cases**: Empty traces, single traces, complex trace patterns
- **Assumptions**: Full coverage (p0=0), no unseen categories
- **Integration**: Proper window modification, multiple applications

### Chao1PopulationExtractor (`test_chao1_population_extractor.py`)

Tests the Chao1 statistical estimation strategy:

#### Statistical Functions
- `_chao1_S_hat_from_counts()`: Richness estimation with/without doubletons
- `_coverage_hat()`: Sample coverage estimation
- `_build_chao_distribution_from_counts()`: Chao1 distribution construction

#### Main Class
- **Statistical Properties**: Richness estimates ≥ observed, valid coverage values
- **Data Diversity**: High diversity (many singletons) vs low diversity scenarios
- **Consistency**: Deterministic results across multiple applications
- **Edge Cases**: Empty data, no singletons, statistical edge cases
- **Validation**: Probability conservation, reasonable parameter ranges

## Test Data

Tests use fixtures from `conftest.py`:

- `empty_traces`: Empty trace list
- `single_trace`: Single trace for edge cases
- `simple_traces`: Basic trace patterns (A→B→C, A→B→D, A→B→C)
- `complex_traces`: Various patterns for comprehensive testing
- `standard_test_log`: Carefully designed log with predictable metrics

## Key Testing Principles

1. **Statistical Validity**: Chao1 tests verify mathematically sound results
2. **Edge Case Coverage**: Empty data, single elements, extreme values
3. **Integration Testing**: Components work together correctly
4. **Deterministic Results**: Consistent outputs for same inputs
5. **Error Handling**: Proper validation and error messages
6. **Performance**: Caching behavior and efficiency

## Running the Tests

```bash
# All population tests
python -m pytest tests/utils/population/ -v

# Specific test files
python -m pytest tests/utils/population/test_population_distribution.py -v
python -m pytest tests/utils/population/extractors/test_chao1_population_extractor.py -v

# With coverage
python -m pytest tests/utils/population/ --cov=utils.population --cov-report=term-missing
```

## Test Dependencies

The tests rely on:
- `pytest` for test framework
- `pm4py` for trace/event objects  
- `conftest.py` fixtures for test data
- Standard library modules (`collections.Counter`, etc.)

Note: Tests are designed to work with the existing project test architecture and fixtures.