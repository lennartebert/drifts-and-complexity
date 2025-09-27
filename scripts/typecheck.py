#!/usr/bin/env python3
"""Type checking script with different configurations."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run mypy type checking with different configurations")
    parser.add_argument(
        "--mode",
        choices=["basic", "strict", "ci"],
        default="basic",
        help="Type checking mode (default: basic)"
    )
    parser.add_argument(
        "--target",
        default="utils",
        help="Target directory to check (default: utils)"
    )
    parser.add_argument(
        "--show-error-codes",
        action="store_true",
        help="Show mypy error codes"
    )
    parser.add_argument(
        "--install-types",
        action="store_true",
        help="Install missing type stubs"
    )
    
    args = parser.parse_args()
    
    # Base mypy command
    cmd = ["mypy"]
    
    # Add target
    cmd.append(args.target)
    
    # Add mode-specific options
    if args.mode == "strict":
        cmd.extend(["--strict", "--warn-return-any"])
    elif args.mode == "ci":
        cmd.extend([
            "--strict",
            "--warn-return-any",
            "--warn-unused-configs",
            "--disallow-untyped-defs",
            "--disallow-incomplete-defs",
            "--check-untyped-defs",
            "--disallow-untyped-decorators",
            "--no-implicit-optional",
            "--warn-redundant-casts",
            "--warn-unused-ignores",
            "--warn-no-return",
            "--warn-unreachable",
            "--strict-equality"
        ])
    
    # Add common options
    if args.show_error_codes:
        cmd.append("--show-error-codes")
    
    cmd.extend([
        "--show-column-numbers",
        "--show-error-context",
        "--pretty",
        "--color-output",
        "--error-summary"
    ])
    
    # Install types if requested
    if args.install_types:
        print("Installing missing type stubs...")
        install_cmd = ["mypy", "--install-types", "--non-interactive"]
        if not run_command(install_cmd, "Installing type stubs"):
            print("Warning: Failed to install some type stubs")
    
    # Run type checking
    success = run_command(cmd, f"Type checking ({args.mode} mode)")
    
    if success:
        print(f"\n🎉 Type checking completed successfully!")
        return 0
    else:
        print(f"\n💥 Type checking found issues!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
