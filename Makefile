.PHONY: help install install-dev test test-unit test-integration lint typecheck format clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode (for pip users)
	pip install -e .

install-dev: ## Install development dependencies (for pip users)
	pip install -e ".[dev]"

install-conda: ## Install conda environment (recommended)
	conda env create -f environment.yml
	conda activate drifts-and-complexity

update-conda: ## Update conda environment
	conda env update -f environment.yml

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest -m "not integration"

test-integration: ## Run integration tests only
	pytest -m integration

test-slow: ## Run slow tests
	pytest -m slow

lint: ## Run linting (flake8)
	flake8 utils tests

typecheck: ## Run mypy type checking
	mypy utils

typecheck-strict: ## Run mypy with strict settings
	mypy --strict utils

format: ## Format code with black and isort
	black utils tests
	isort utils tests

format-check: ## Check if code is formatted correctly
	black --check utils tests
	isort --check-only utils tests

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check-all: format-check lint typecheck test ## Run all checks (format, lint, type, test)

ci: check-all ## Run all CI checks
