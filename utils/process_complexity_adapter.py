# process_complexity_adapter.py
from __future__ import annotations
from pathlib import Path
import sys
from typing import Dict, Any, List

# Make the forked dependency importable:
# repo layout assumption: this file sits next to the 'process-complexity' folder
THIS_DIR = Path(__file__).resolve().parent.parent
PROC_COMPLEXITY_DIR = THIS_DIR / "process-complexity"
if PROC_COMPLEXITY_DIR.exists():
    sys.path.insert(0, str(PROC_COMPLEXITY_DIR))

# Imports from Vidgof’s package (your fork)
from Complexity import (
    generate_log,
    build_graph,
    perform_measurements,
    graph_complexity,
    log_complexity,
)

def _snake(s: str) -> str:
    return s.replace(" ", "_").replace("-", "_")

def _prefixed(measurements: Dict[str, Any], prefix: str = "measure_") -> Dict[str, Any]:
    """
    Normalize keys to snake_case and prefix with 'measure_' to match plotting code.
    """
    out: Dict[str, Any] = {}
    for k, v in measurements.items():
        key = f"{prefix}{_snake(k)}"
        out[key] = v
    return out

def get_measures_for_traces(traces: List) -> Dict[str, Any]:
    """
    Adapter around Vidgof’s complexity functions.
    Returns a dict with 'measure_*' keys to align with your plotting.
    """
    # Edge case: empty input
    if not traces:
        return {
            "measure_Support": 0,
        }

    # Build log + PA
    log = generate_log(traces, verbose=False)
    pa = build_graph(log, verbose=False, accepting=False)

    # Base measurements
    base = perform_measurements('all', log, traces, pa, quiet=True, verbose=False)

    # Entropies
    var_ent = graph_complexity(pa)  # (entropy, normalized_entropy)
    seq_ent = log_complexity(pa)    # (entropy, normalized_entropy)

    # Attach with explicit names
    base["Variant Entropy"] = var_ent[0]
    base["Normalized Variant Entropy"] = var_ent[1]
    base["Trace Entropy"] = seq_ent[0]
    base["Normalized Trace Entropy"] = seq_ent[1]

    # Trace length fields may be a dict; expand MIN/AVG/MAX
    if isinstance(base.get("Trace length"), dict):
        tl = base.pop("Trace length")
        base["Trace length min"] = tl.get("min")
        base["Trace length avg"] = tl.get("avg")
        base["Trace length max"] = tl.get("max")

    # Always include support (= number of traces in the window)
    base["Support"] = len(traces)

    # Prefix + normalize keys so plotting picks them up
    return _prefixed(base, prefix="measure_")
