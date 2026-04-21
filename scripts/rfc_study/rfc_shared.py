"""Shared helpers for RFC/PA scripts using attachment CSV inputs."""

from pathlib import Path
from typing import List, Tuple

import pandas as pd


REQUIRED_ATTACHMENT_COLUMNS = ["attachment_time", "attachment_index", "node_id"]


def parse_dataset_input(raw_value: str) -> Tuple[str, Path]:
    """Parse one --input value in the form <dataset>=<attachments_csv_path>."""
    if "=" not in raw_value:
        raise ValueError(f"Invalid --input value '{raw_value}'. Expected <dataset>=<attachments_csv_path>.")
    dataset_name, csv_path = raw_value.split("=", 1)
    dataset_name = dataset_name.strip()
    csv_path = csv_path.strip()
    if not dataset_name or not csv_path:
        raise ValueError(f"Invalid --input value '{raw_value}'. Expected non-empty dataset and file path.")
    return dataset_name, Path(csv_path)


def parse_dataset_inputs(raw_values: List[str]) -> List[Tuple[str, Path]]:
    """Parse all --inputs and guard against duplicated dataset names."""
    pairs = [parse_dataset_input(raw) for raw in raw_values]
    names = [name for name, _ in pairs]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"duplicated dataset names in --inputs: {duplicated}")
    return pairs


def load_attachments(attachments_path: Path) -> pd.DataFrame:
    """Load and validate one attachments CSV."""
    df = pd.read_csv(attachments_path)
    missing = [col for col in REQUIRED_ATTACHMENT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {attachments_path}: {missing}")
    df["attachment_index"] = pd.to_numeric(df["attachment_index"])
    return df.sort_values("attachment_index").reset_index(drop=True)

