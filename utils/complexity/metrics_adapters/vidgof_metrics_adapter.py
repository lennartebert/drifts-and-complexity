from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from utils.complexity.measures.measure_store import Measure, MeasureStore
from utils.complexity.metrics_adapters.metrics_adapter import MetricsAdapter
from utils.windowing.window import Window

# --- fork import sandbox (unchanged logic) ------------------------------------
THIS_DIR = Path(__file__).resolve().parents[3]
PROC_COMPLEXITY_DIR = THIS_DIR / "plugins" / "vidgof_complexity"
if PROC_COMPLEXITY_DIR.exists():
    sys.path.insert(0, str(PROC_COMPLEXITY_DIR))

# Third-party lib (Vidgof / Augusto fork)
from Complexity import (  # type: ignore
    build_graph,
    create_c_index,
    generate_log,
    graph_complexity,
    log_complexity,
)


class VidgofMetricsAdapter(MetricsAdapter):
    """
    Adapter wrapping the Vidgof/Augusto 'Complexity' library.

    Produces measures:
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

    def available_metrics(self) -> List[str]:
        metric_names = [
            "Variant Entropy",
            "Normalized Variant Entropy",
            "Trace Entropy",
            "Normalized Trace Entropy",
            "Number of Partitions",
        ]
        return metric_names

    # --- internal helpers -----------------------------------------------------
    @staticmethod
    def _get_num_partitions(pa: Any) -> int:
        # Get number of partitions with same logic used by Vidgof: (re)build c_index, then ignore c=0
        pa.c_index = create_c_index(pa)
        return len(list(pa.c_index.keys())[1:])

    @staticmethod
    def _compute_all_from_lib(window: Window) -> Dict[str, Any]:
        traces = window.traces
        log = generate_log(traces, verbose=False)
        pa = build_graph(log, verbose=False, accepting=False)

        # Entropies (tuples: (entropy, normalized_entropy))
        var_ent = graph_complexity(pa)  # (entropy, normalized_entropy)
        seq_ent = log_complexity(pa)  # (entropy, normalized_entropy)

        metrics = {}
        metrics["Variant Entropy"] = var_ent[0]
        metrics["Normalized Variant Entropy"] = var_ent[1]
        metrics["Sequence Entropy"] = seq_ent[0]
        metrics["Normalized Sequence Entropy"] = seq_ent[1]

        metrics["Number of Partitions"] = VidgofMetricsAdapter._get_num_partitions(pa)

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
    ) -> Tuple[MeasureStore, Dict[str, Any]]:
        store = (
            measures if isinstance(measures, MeasureStore) else MeasureStore(measures)
        )

        raw = self._compute_all_from_lib(window)
        floats = self._floatify(raw)

        for name, value in floats.items():
            store.set(name, value, hidden=False, meta={"source": self.name})

        info = {
            "adapter": self.name,
            "support": len(window.traces),
        }
        return store, info
