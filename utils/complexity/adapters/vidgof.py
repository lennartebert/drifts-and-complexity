from __future__ import annotations
from typing import Dict, Any, Tuple
from utils.windowing.windowing import Window
from .base import ComplexityAdapter, Metrics, Info

# fork import sandbox
from pathlib import Path
import sys
THIS_DIR = Path(__file__).resolve().parents[3]
PROC_COMPLEXITY_DIR = THIS_DIR / "process-complexity"
if PROC_COMPLEXITY_DIR.exists():
    sys.path.insert(0, str(PROC_COMPLEXITY_DIR))

from Complexity import (  # type: ignore
    generate_log, build_graph, perform_measurements,
    graph_complexity, log_complexity
)

class VidgofSampleAdapter(ComplexityAdapter):
    name = "vidgof_sample"

    def compute_one(self, window: Window) -> Tuple[Metrics, Info]:
        traces = window.traces
        log = generate_log(traces, verbose=False)
        pa = build_graph(log, verbose=False, accepting=False)

        base: Dict[str, Any] = perform_measurements('all', log, traces, pa, quiet=True, verbose=False)
        var_ent = graph_complexity(pa)  # (entropy, normalized_entropy)
        seq_ent = log_complexity(pa)    # (entropy, normalized_entropy)

        base["Variant Entropy"] = var_ent[0]
        base["Normalized Variant Entropy"] = var_ent[1]
        base["Trace Entropy"] = seq_ent[0]
        base["Normalized Trace Entropy"] = seq_ent[1]

        if isinstance(base.get("Trace Length"), dict):
            tl = base.pop("Trace Length")
            base["Min. Trace Length"] = tl.get("min")
            base["Avg. Trace Length"] = tl.get("avg")
            base["Max. Trace Length"] = tl.get("max")

        metrics: Metrics = {k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in base.items()}
        info: Info = {"Support": len(traces)}
        return metrics, info
