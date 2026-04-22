"""Extract concept-specific attachment tables from event logs.

Run from the repository root. Dataset paths in the data dictionary and default
`results/` output are relative to the current working directory. Importing
`utils` requires the repo root on PYTHONPATH (e.g. set by `main.py` or
`export PYTHONPATH="$PWD"` before calling this script directly).

Outputs are written as:
`<output-root>/<concept>/<dataset>/attachments.csv`
where concept is one of: variants, activities, dfrs.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog

from utils import constants, helpers


def load_event_log(log_path: Path) -> EventLog:
    """Load one XES/XES.GZ event log sorted by timestamp."""
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    return xes_importer.apply(str(log_path), variant=variant, parameters=parameters)


def _trace_completion_data(event_log: EventLog) -> List[Dict[str, object]]:
    """Return traces with completion timestamp and activity sequence."""
    trace_data: List[Dict[str, object]] = []
    for trace in event_log:
        if not trace:
            continue

        # Use completion timestamp from the last event.
        completion_timestamp = trace[-1].get("time:timestamp")
        if completion_timestamp is None:
            # Skip traces without timestamps because ordering would be undefined.
            continue

        activity_sequence = tuple(event.get("concept:name", "") for event in trace)
        trace_data.append(
            {
                "completion_timestamp": completion_timestamp,
                "activity_sequence": activity_sequence,
            }
        )
    return trace_data


def _node_ids_for_concept(activity_sequence: Tuple[str, ...], concept: str) -> List[str]:
    """Create node ids contributed by one trace under a concept."""
    if concept == "variants":
        return [str(activity_sequence)]
    if concept == "activities":
        return [str(activity) for activity in activity_sequence]
    # concept == "dfrs"
    return [str((source, target)) for source, target in zip(activity_sequence, activity_sequence[1:])]


def extract_attachments(event_log: EventLog, concept: str) -> pd.DataFrame:
    """Extract attachment rows with time, index, and concept-specific node_id."""
    trace_data = _trace_completion_data(event_log)
    # Sort globally by completion timestamp before assigning attachment index.
    trace_data.sort(key=lambda item: item["completion_timestamp"])
    return extract_attachments_from_trace_data(trace_data, concept)


def extract_attachments_from_trace_data(trace_data: List[Dict[str, object]], concept: str) -> pd.DataFrame:
    """Extract attachment rows for one concept from precomputed trace data."""

    attachments = []
    attachment_index = 0
    for row in trace_data:
        completion_ts = row["completion_timestamp"]
        timestamp_str = completion_ts.isoformat() if isinstance(completion_ts, datetime) else str(completion_ts)
        activity_sequence = row["activity_sequence"]
        for node_id in _node_ids_for_concept(activity_sequence, concept):
            attachments.append(
                {
                    "attachment_time": timestamp_str,
                    "attachment_index": attachment_index,
                    "node_id": node_id,
                }
            )
            attachment_index += 1
    return pd.DataFrame(attachments)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for attachment extraction."""
    parser = argparse.ArgumentParser(description="Extract trace attachments to CSV files")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names from data dictionary")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results/rfc_study)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute attachments even when attachments.csv already exists",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["variants"],
        choices=["variants", "activities", "dfrs"],
        help="Concepts for attachment extraction",
    )
    return parser.parse_args()


def main() -> None:
    """Run extraction for all requested datasets."""
    args = parse_args()
    output_root = Path(args.output_dir) if args.output_dir else Path("results") / "rfc_study"
    output_root.mkdir(parents=True, exist_ok=True)

    # Load project dataset registry and validate requested names.
    data_dict_path = constants.get_data_dictionary_path()
    data_dictionary = helpers.load_data_dictionary(data_dict_path, get_real=True, get_synthetic=True)
    invalid_datasets: List[str] = [ds for ds in args.datasets if ds not in data_dictionary]
    if invalid_datasets:
        print(f"Error: Invalid dataset names: {invalid_datasets}")
        print(f"Available datasets: {sorted(data_dictionary.keys())}")
        sys.exit(1)

    print(f"Processing {len(args.datasets)} dataset(s)...")
    for dataset_name in args.datasets:
        log_path = Path(data_dictionary[dataset_name]["path"])
        if not log_path.exists():
            print(f"Warning: Log file missing for {dataset_name}: {log_path}")
            continue

        concept_outputs = []
        for concept in args.concepts:
            dataset_concept_dir = output_root / concept / dataset_name
            dataset_concept_dir.mkdir(parents=True, exist_ok=True)
            attachment_path = dataset_concept_dir / "attachments.csv"
            concept_outputs.append((concept, attachment_path))

        pending_outputs = []
        for concept, attachment_path in concept_outputs:
            if attachment_path.exists() and not args.force:
                print(f"  Skipping {concept}/{dataset_name}: {attachment_path} already exists")
                continue
            pending_outputs.append((concept, attachment_path))

        if not pending_outputs:
            print(f"Skipping dataset {dataset_name}: all requested concepts already extracted")
            continue

        print(f"Extracting {dataset_name} from {log_path}...")
        event_log = load_event_log(log_path)
        trace_data = _trace_completion_data(event_log)
        # Sort once and reuse for all requested concepts to avoid repeated log traversal.
        trace_data.sort(key=lambda item: item["completion_timestamp"])
        for concept, attachment_path in pending_outputs:
            attachments_df = extract_attachments_from_trace_data(trace_data, concept)
            attachments_df.to_csv(attachment_path, index=False)
            unique_nodes = attachments_df["node_id"].nunique() if not attachments_df.empty else 0
            print(f"  Saved {concept}/{dataset_name}/attachments.csv (rows={len(attachments_df)}, nodes={unique_nodes})")

    print(f"\nAttachment extraction complete. Output root: {output_root}")


if __name__ == "__main__":
    main()
