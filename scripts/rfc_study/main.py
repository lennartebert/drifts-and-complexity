"""Single CLI entrypoint for the RFC study pipeline.

Pipeline order:
1) extract_attachments.py
2) static_rfc_analysis.py
3) dynamic_rfc_analysis.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# scripts/rfc_study/ -> project root (run from repository root; PYTHONPATH set below for child scripts)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def _subprocess_env() -> Dict[str, str]:
    """Ensure repository root is on PYTHONPATH so `import utils` works for invoked scripts."""
    root = str(REPO_ROOT)
    prev = os.environ.get("PYTHONPATH", "")
    merged = root if not prev else f"{root}{os.pathsep}{prev}"
    return {**os.environ, "PYTHONPATH": merged}


def _run_command(cmd: List[str], dry_run: bool) -> None:
    """Run one subprocess command or print it in dry-run mode."""
    pretty = " ".join(cmd)
    print(f"$ {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=_subprocess_env())


def _build_attachment_inputs(datasets: List[str], output_dir: str | None, concept: str) -> List[str]:
    """Build <dataset>=<attachments.csv> input pairs for analysis scripts."""
    output_root = Path(output_dir) if output_dir else Path("results") / "rfc_study"
    return [f"{dataset}={output_root / concept / dataset / 'attachments.csv'}" for dataset in datasets]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for full pipeline execution."""
    parser = argparse.ArgumentParser(description="Run full RFC study pipeline")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names from data dictionary")
    parser.add_argument(
        "--analysis-name",
        type=str,
        default=None,
        help="Subfolder for static/dynamic outputs (default: same as --concept; omit dataset from this label)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output root directory (default: results/rfc_study)")
    parser.add_argument(
        "--concept",
        type=str,
        default="variants",
        choices=["variants", "activities", "dfrs"],
        help="Concept used for extraction and static analysis inputs",
    )
    parser.add_argument("--split-ratio", type=float, default=0.75, help="Split ratio for dynamic PA analysis")
    parser.add_argument("--lambda-reg", type=float, default=0.1, help="Lambda regularization for dynamic PA analysis")
    parser.add_argument("--skip-static", action="store_true", help="Skip static RFC stage")
    parser.add_argument("--skip-extract", action="store_true", help="Skip attachment extraction stage")
    parser.add_argument("--skip-dynamic", action="store_true", help="Skip dynamic RFC stage")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    return parser.parse_args()


def main() -> None:
    """Execute the selected stages in sequence."""
    args = parse_args()
    python_exec = sys.executable
    analysis_name = (args.analysis_name or args.concept).strip() or "run"

    # Build shared dataset arg list once so all stages stay consistent.
    dataset_args = ["--datasets", *args.datasets]
    output_args = ["--output-dir", args.output_dir] if args.output_dir else []

    if not args.skip_extract:
        extract_cmd = [
            python_exec,
            str(SCRIPT_DIR / "extract_attachments.py"),
            *dataset_args,
            "--concepts",
            args.concept,
            *output_args,
        ]
        _run_command(extract_cmd, args.dry_run)

    if not args.skip_static:
        static_inputs = _build_attachment_inputs(args.datasets, args.output_dir, args.concept)
        static_cmd = [
            python_exec,
            str(SCRIPT_DIR / "static_rfc_analysis.py"),
            "--inputs",
            *static_inputs,
            "--analysis-name",
            analysis_name,
            *output_args,
        ]
        _run_command(static_cmd, args.dry_run)

    if not args.skip_dynamic:
        dynamic_inputs = _build_attachment_inputs(args.datasets, args.output_dir, args.concept)
        dynamic_cmd = [
            python_exec,
            str(SCRIPT_DIR / "dynamic_rfc_analysis.py"),
            "--inputs",
            *dynamic_inputs,
            "--analysis-name",
            analysis_name,
            "--split-ratio",
            str(args.split_ratio),
            "--lambda-reg",
            str(args.lambda_reg),
            *output_args,
        ]
        _run_command(dynamic_cmd, args.dry_run)

    print("\nRFC study pipeline completed.")


if __name__ == "__main__":
    main()
