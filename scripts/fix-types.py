#!/usr/bin/env python3
"""Script to help fix common MyPy type issues."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_mypy(config_file: str = "mypy.ini") -> list[str]:
    """Run mypy and return error lines."""
    try:
        result = subprocess.run(
            ["mypy", "--config-file", config_file, "utils", "--show-error-codes"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout.split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error running mypy: {e}")
        return []


def analyze_errors(error_lines: list[str]) -> dict[str, list[str]]:
    """Analyze mypy errors and categorize them."""
    categories = {
        "import_not_found": [],
        "import_untyped": [],
        "union_attr": [],
        "unused_ignore": [],
        "override": [],
        "other": []
    }
    
    for line in error_lines:
        if not line.strip():
            continue
            
        if "import-not-found" in line:
            categories["import_not_found"].append(line)
        elif "import-untyped" in line:
            categories["import_untyped"].append(line)
        elif "union-attr" in line:
            categories["union_attr"].append(line)
        elif "unused-ignore" in line:
            categories["unused_ignore"].append(line)
        elif "override" in line:
            categories["override"].append(line)
        else:
            categories["other"].append(line)
    
    return categories


def print_summary(categories: dict[str, list[str]]) -> None:
    """Print a summary of error categories."""
    print("MyPy Error Summary")
    print("=" * 50)
    
    total_errors = sum(len(errors) for errors in categories.values())
    print(f"Total errors: {total_errors}")
    print()
    
    for category, errors in categories.items():
        if errors:
            print(f"{category.replace('_', ' ').title()}: {len(errors)} errors")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more")
            print()


def suggest_fixes(categories: dict[str, list[str]]) -> None:
    """Suggest fixes for common error types."""
    print("Suggested Fixes")
    print("=" * 50)
    
    if categories["import_not_found"]:
        print("1. Missing Imports:")
        print("   - Create missing modules or add to ignore list")
        print("   - Check if modules exist in the project")
        print()
    
    if categories["import_untyped"]:
        print("2. Missing Type Stubs:")
        print("   - Install type stubs: pip install types-PyYAML")
        print("   - Or add to ignore list in mypy.ini")
        print()
    
    if categories["union_attr"]:
        print("3. Union Type Issues:")
        print("   - Add null checks before attribute access")
        print("   - Use type guards for runtime type checking")
        print("   - Consider using Optional[T] instead of T | None")
        print()
    
    if categories["unused_ignore"]:
        print("4. Unused Type Ignores:")
        print("   - Remove unnecessary # type: ignore comments")
        print("   - Fix the underlying type issues instead")
        print()
    
    if categories["override"]:
        print("5. Method Signature Issues:")
        print("   - Align method signatures with superclass")
        print("   - Use protocols for flexible method signatures")
        print("   - Consider using overloads for different signatures")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze and suggest fixes for MyPy errors")
    parser.add_argument(
        "--config",
        default="mypy.ini",
        help="MyPy configuration file to use"
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Use lenient configuration"
    )
    
    args = parser.parse_args()
    
    if args.lenient:
        config_file = "scripts/mypy-lenient.ini"
    else:
        config_file = args.config
    
    print(f"Running MyPy with config: {config_file}")
    print()
    
    error_lines = run_mypy(config_file)
    if not error_lines:
        print("No errors found!")
        return 0
    
    categories = analyze_errors(error_lines)
    print_summary(categories)
    suggest_fixes(categories)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
