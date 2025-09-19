from __future__ import annotations
from typing import Iterable, Optional, Dict, Tuple, Any, Union, List

from pathlib import Path
import sys

from utils.complexity.measures.measure_store import MeasureStore, Measure
from utils.complexity.metrics_adapters.metrics_adapter import MetricsAdapter
from utils.windowing.window import Window

# --- fork import sandbox (unchanged logic) ------------------------------------
THIS_DIR = Path(__file__).resolve().parents[3]
PROC_COMPLEXITY_DIR = THIS_DIR / "process-complexity"
if PROC_COMPLEXITY_DIR.exists():
    sys.path.insert(0, str(PROC_COMPLEXITY_DIR))

# Third-party lib (Vidgof / Augusto fork)
from Complexity import (  # type: ignore
    generate_log, build_graph, perform_measurements,
    graph_complexity, log_complexity
)


class VidgofMetricsAdapter(MetricsAdapter):
    """
    Adapter wrapping the Vidgof/Augusto 'Complexity' library.

    Produces (canonical) measures:
      - Variant Entropy
      - Normalized Variant Entropy
      - Trace Entropy
      - Normalized Trace Entropy
    and passes through all numeric results from `perform_measurements('all', ...)`
    under whatever keys that lib returns (float-castable only).

    Notes
    -----
    - We always compute once (library doesn't expose granular calls), then filter
      via `include`/`exclude` before writing into the MeasureStore.
    - Existing values in the provided MeasureStore are respected (not overwritten).
    - Meta includes {"source": "vidgof"}.
    """
    name: str = "vidgof"

    def available_metrics(self) -> Iterable[str]:
        metric_names = (
            "Variant Entropy",
            "Normalized Variant Entropy",
            "Trace Entropy",
            "Normalized Trace Entropy"
        )
        return metric_names

    # --- internal helpers -----------------------------------------------------

    @staticmethod
    def _compute_all_from_lib(window: Window) -> Dict[str, Any]:
        traces = window.traces
        log = generate_log(traces, verbose=False)
        pa = build_graph(log, verbose=False, accepting=False)

        # Entropies (tuples: (entropy, normalized_entropy))
        var_ent = graph_complexity(pa)  # (entropy, normalized_entropy)
        seq_ent = log_complexity(pa)    # (entropy, normalized_entropy)
        
        metrics = {}
        metrics["Variant Entropy"] = var_ent[0]
        metrics["Normalized Variant Entropy"] = var_ent[1]
        metrics["Trace Entropy"] = seq_ent[0]
        metrics["Normalized Trace Entropy"] = seq_ent[1]

        return metrics

    @staticmethod
    def _floatify(d: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in d.items():
            if isinstance(v, (int, float)):
                try:
                    out[k] = float(v)
                except Exception:
                    pass
        return out

    # --- API ------------------------------------------------------------------

    def compute_measures_for_window(
        self,
        window: Window,
        measures: Optional[Union[MeasureStore, Dict[str, Measure]]] = None,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Tuple[MeasureStore, Dict[str, Any]]:
        store = measures if isinstance(measures, MeasureStore) else MeasureStore(measures)

        raw = self._compute_all_from_lib(window)
        floats = self._floatify(raw)
        
        include_names = tuple(include) if include is not None else tuple(floats.keys())
        exclude_set = set(exclude) if exclude else set()

        computed: List[str] = []
        skipped_existing: List[str] = []

        for name in include_names:
            if name in exclude_set:
                continue
            if name not in floats:
                continue  # metric not produced by this lib
            if store.has(name):
                skipped_existing.append(name)
                continue
            store.set(name, floats[name], hidden=False, meta={"source": self.name})
            computed.append(name)

        info = {
            "adapter": self.name,
            "computed": computed,
            "skipped_existing": skipped_existing,
            "available_now": tuple(sorted(floats.keys())),
            "support": len(window.traces),
        }
        return store, info
