"""Window configuration loader and validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..helpers import load_yaml

# Window approach types accepted from window_config.yml.
# Note: `fixed_size_windows` was renamed to `fixed_trace_windows` (strict rename).
_ALLOWED = {"change_point_windows", "fixed_trace_windows", "fixed_time_windows", "window_comparison"}


def _req_int(params: Dict[str, Any], key: str, min_val: int) -> int:
    """Require an integer parameter with minimum value.

    Args:
        params: Parameter dictionary.
        key: Parameter key.
        min_val: Minimum allowed value.

    Returns:
        Integer value.

    Raises:
        ValueError: If parameter is missing, not an integer, or below minimum.
    """
    if key not in params:
        raise ValueError(f"Missing param '{key}'.")
    try:
        val = int(params[key])
    except Exception:
        raise ValueError(f"Param '{key}' must be integer.")
    if val < min_val:
        raise ValueError(f"Param '{key}' must be ≥ {min_val}.")
    return val


def validate_window_approaches(approaches: List[Dict[str, Any]]) -> None:
    """Validate window approach configurations.

    Args:
        approaches: List of approach configuration dictionaries.

    Raises:
        ValueError: If validation fails.
    """
    if not approaches or not isinstance(approaches, list):
        raise ValueError("'approaches' must be a non-empty list")
    seen = set()
    for a in approaches:
        name = str(a.get("name", "")).strip()
        typ = str(a.get("type", "")).strip()
        params = a.get("params", {}) or {}
        if not name:
            raise ValueError("Approach requires a 'name'")
        if name in seen:
            raise ValueError(f"Duplicate approach name: {name}")
        seen.add(name)
        if typ not in _ALLOWED:
            raise ValueError(f"Unknown approach type: {typ}")
        if typ == "fixed_trace_windows":
            _req_int(params, "window_size", 1)
            _req_int(params, "offset", 1)
        elif typ == "fixed_time_windows":
            _req_int(params, "window_size", 1)
            _req_int(params, "offset", 1)

            unit = params.get("unit", None)
            if unit not in {"day", "month", "year"}:
                raise ValueError(
                    "fixed_time_windows requires params.unit in {'day','month','year'}."
                )

            align = params.get("align_first_window", None)
            if not isinstance(align, bool):
                raise ValueError(
                    "fixed_time_windows requires params.align_first_window to be a boolean."
                )
        elif typ == "window_comparison":
            _req_int(params, "window_1_size", 1)
            _req_int(params, "window_2_size", 1)
            _req_int(params, "offset", 0)
            _req_int(params, "step", 1)


def load_window_config(path: Path) -> List[Dict[str, Any]]:
    """Load and validate window configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        List of validated approach configurations.

    Raises:
        ValueError: If validation fails.
    """
    cfg = load_yaml(path)
    approaches = cfg.get("approaches", [])
    validate_window_approaches(approaches)
    return approaches
