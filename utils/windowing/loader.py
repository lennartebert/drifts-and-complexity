from typing import Dict, List
from pathlib import Path
from ..helpers import load_yaml

_ALLOWED = {"change_point_windows", "fixed_size_windows", "window_comparison"}

def _req_int(params: Dict, key: str, min_val: int) -> int:
    if key not in params: raise ValueError(f"Missing param '{key}'.")
    try: val = int(params[key])
    except Exception: raise ValueError(f"Param '{key}' must be integer.")
    if val < min_val: raise ValueError(f"Param '{key}' must be ≥ {min_val}.")
    return val

def validate_window_approaches(approaches: List[Dict]) -> None:
    if not approaches or not isinstance(approaches, list):
        raise ValueError("'approaches' must be a non-empty list")
    seen = set()
    for a in approaches:
        name = str(a.get("name","")).strip()
        typ  = str(a.get("type","")).strip()
        params = a.get("params", {}) or {}
        if not name: raise ValueError("Approach requires a 'name'")
        if name in seen: raise ValueError(f"Duplicate approach name: {name}")
        seen.add(name)
        if typ not in _ALLOWED: raise ValueError(f"Unknown approach type: {typ}")
        if typ == "fixed_size_windows":
            _req_int(params, "window_size", 1); _req_int(params, "offset", 1)
        elif typ == "window_comparison":
            _req_int(params, "window_1_size", 1)
            _req_int(params, "window_2_size", 1)
            _req_int(params, "offset", 0)
            _req_int(params, "step", 1)

def load_window_config(path: Path) -> List[Dict]:
    cfg = load_yaml(path)
    approaches = cfg.get("approaches", [])
    validate_window_approaches(approaches)
    return approaches
