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

# Third-party lib (Vidgof / Augusto fork) – robust import for CI/local
try:
    from Complexity import (  # type: ignore
        build_graph,
        create_c_index,
        generate_log,
        graph_complexity,
        log_complexity,
    )
except ModuleNotFoundError:  # Fallback when PYTHONPATH does not include plugins dir
    from plugins.vidgof_complexity.Complexity import (
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
            "Sequence Entropy",
            "Normalized Sequence Entropy",
            "Number of Partitions",
        ]
        return metric_names

    # --- internal helpers -----------------------------------------------------
    @staticmethod
    def _get_num_states(pa) -> int:
        pa.c_index = create_c_index(pa)  # keep Vidgof’s pattern
        return len(pa.nodes) - 1  # |S|, root excluded

    @staticmethod
    def _get_num_partitions(pa) -> int:
        pa.c_index = create_c_index(pa)
        return len(pa.c_index) - 1  # ignore c=0 sentinel

    @staticmethod
    def _compute_from_lib(
        window: Window, metrics_to_compute: List[str]
    ) -> List[Measure]:
        traces = window.traces

        # Filter out events with missing required fields before calling generate_log
        # This handles cases where noise in synthetic logs removes attributes
        # Note: trace.attributes['concept:name'] is optional - generate_log will use trace index as fallback
        from pm4py.objects.log.obj import Trace
        from pm4py.util import xes_constants as xes

        concept_name_key = xes.DEFAULT_NAME_KEY

        # Filter traces to remove events with missing required attributes
        filtered_traces = []
        skipped_events_count = 0

        for trace_idx, trace in enumerate(traces):
            filtered_trace = Trace()
            filtered_trace.attributes = trace.attributes.copy()

            for event_idx, event in enumerate(trace):
                # Skip events missing required attributes
                if concept_name_key not in event or "time:timestamp" not in event:
                    skipped_events_count += 1
                    continue
                filtered_trace.append(event)

            # Only include traces that have at least one valid event
            if len(filtered_trace) > 0:
                filtered_traces.append(filtered_trace)

        if skipped_events_count > 0:
            print(
                f"WARNING: Skipped {skipped_events_count} event(s) with missing required attributes "
                f"('{concept_name_key}' or 'time:timestamp') in window {window.id}"
            )

        if not filtered_traces:
            # If all events were filtered out, return empty measures
            print(
                f"WARNING: All events in window {window.id} were filtered out due to missing attributes. "
                f"Returning empty measures."
            )
            return []

        log = generate_log(filtered_traces, verbose=False)
        pa = build_graph(log, verbose=False, accepting=False)

        measures = []

        # Only compute Number of Partitions as hidden if any other metric is requested
        if any(
            metric in metrics_to_compute
            for metric in [
                "Variant Entropy",
                "Normalized Variant Entropy",
                "Sequence Entropy",
                "Normalized Sequence Entropy",
            ]
        ):
            num_partitions = VidgofMetricsAdapter._get_num_partitions(pa)
            measures.append(
                Measure(
                    name="Number of Partitions",
                    value=float(num_partitions),
                    hidden=True,
                    meta={"source": "vidgof"},
                )
            )

            num_states = VidgofMetricsAdapter._get_num_states(pa)
            measures.append(
                Measure(
                    name="Number of States",
                    value=float(num_states),
                    hidden=True,
                    meta={"source": "vidgof"},
                )
            )

        # Compute requested metrics
        if (
            "Variant Entropy" in metrics_to_compute
            or "Normalized Variant Entropy" in metrics_to_compute
        ):
            var_ent = graph_complexity(pa)  # (entropy, normalized_entropy)

            if "Variant Entropy" in metrics_to_compute:
                measures.append(
                    Measure(
                        name="Variant Entropy",
                        value=float(var_ent[0]),
                        hidden=False,
                        meta={"source": "vidgof"},
                    )
                )

            if "Normalized Variant Entropy" in metrics_to_compute:
                measures.append(
                    Measure(
                        name="Normalized Variant Entropy",
                        value=float(var_ent[1]),
                        hidden=False,
                        meta={"source": "vidgof"},
                    )
                )

        if (
            "Sequence Entropy" in metrics_to_compute
            or "Normalized Sequence Entropy" in metrics_to_compute
        ):
            seq_ent = log_complexity(pa)  # (entropy, normalized_entropy)

            if "Sequence Entropy" in metrics_to_compute:
                measures.append(
                    Measure(
                        name="Sequence Entropy",
                        value=float(seq_ent[0]),
                        hidden=False,
                        meta={"source": "vidgof"},
                    )
                )

            if "Normalized Sequence Entropy" in metrics_to_compute:
                measures.append(
                    Measure(
                        name="Normalized Sequence Entropy",
                        value=float(seq_ent[1]),
                        hidden=False,
                        meta={"source": "vidgof"},
                    )
                )

        return measures

    # --- API ------------------------------------------------------------------

    def compute_measures_for_window(
        self,
        window: Window,
        measure_store: Optional[MeasureStore] = None,
        include_metrics: Optional[Iterable[str]] = None,
        exclude_metrics: Optional[Iterable[str]] = None,
    ) -> Tuple[MeasureStore, Dict[str, Any]]:
        store = measure_store or MeasureStore()

        # Determine which metrics to compute
        metrics_to_compute = self.available_metrics()
        if include_metrics is not None:
            metrics_to_compute = [m for m in metrics_to_compute if m in include_metrics]
        if exclude_metrics is not None:
            metrics_to_compute = [
                m for m in metrics_to_compute if m not in exclude_metrics
            ]

        # Compute only the requested metrics
        computed_measures = self._compute_from_lib(window, metrics_to_compute)

        # Add all computed measures to the store
        for measure in computed_measures:
            store.set(
                measure.name, measure.value, hidden=measure.hidden, meta=measure.meta
            )

        info = {
            "adapter": self.name,
            "support": len(window.traces),
        }
        return store, info
