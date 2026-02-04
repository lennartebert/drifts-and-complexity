"""Check whether pre-drift traces match across multiple logs.

This script loads a set of .xes.gz logs with pm4py and compares the first N traces
trace-by-trace. By default, a trace is compared by its activity sequence
(`concept:name` per event), which is typically what "same trace" means in this
project context.

Run from repo root (Windows / PowerShell):
  conda run -n drifts-and-complexity python scripts/signal_noise_study/check_pre_drift_trace_equality.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pm4py.objects.log.importer.xes import importer as xes_importer

LOG_DIR = Path("data") / "synthetic" / "sudden_drifts"
LOG_IDS = [
    "log_3445_1770066087.xes.gz",
    "log_3985_1770066331.xes.gz",
]
N_PRE_TRACES = 5000


def _event_activity(event: object) -> str:
    """Return the activity label for an event."""
    # pm4py uses dict-like events; concept:name is the standard XES key.
    try:
        return str(event.get("concept:name"))  # type: ignore[attr-defined]
    except Exception:
        return str(event)  # last-resort fallback


def _trace_signature(trace: Iterable[object]) -> tuple[str, ...]:
    """Return a deterministic signature for a trace (activity sequence)."""
    return tuple(_event_activity(e) for e in trace)


def _load_log(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Log not found: {path}")
    return xes_importer.apply(str(path))


def compare_first_n_traces(*, log_a, log_b, name_a: str, name_b: str, n: int) -> bool:
    """Compare first n traces of log_a vs log_b; print first mismatch."""
    if len(log_a) < n or len(log_b) < n:
        print(
            f"ERROR: Not enough traces (need {n}). "
            f"{name_a} has {len(log_a)}, {name_b} has {len(log_b)}."
        )
        return False

    a_pre = log_a[:n]
    b_pre = log_b[:n]

    for i, (ta, tb) in enumerate(zip(a_pre, b_pre)):
        sig_a = _trace_signature(ta)
        sig_b = _trace_signature(tb)
        if sig_a != sig_b:
            print(f"Mismatch at trace index i={i}: {name_a} != {name_b}")
            print(f"  {name_a}: n_events={len(ta)} first10={sig_a[:10]}")
            print(f"  {name_b}: n_events={len(tb)} first10={sig_b[:10]}")
            return False

    return True


def main() -> None:
    paths = [LOG_DIR / log_id for log_id in LOG_IDS]
    print("Loading logs:")
    for p in paths:
        print(f"  - {p}")

    logs = [(_load_log(p), p.name) for p in paths]
    print(f"\nComparing first {N_PRE_TRACES} traces (activity sequences):")

    base_log, base_name = logs[0]
    all_ok = True
    for other_log, other_name in logs[1:]:
        ok = compare_first_n_traces(
            log_a=base_log,
            log_b=other_log,
            name_a=base_name,
            name_b=other_name,
            n=N_PRE_TRACES,
        )
        status = "OK" if ok else "DIFFER"
        print(f"  {base_name} vs {other_name}: {status}")
        all_ok = all_ok and ok

    if all_ok:
        print(
            "\nResult: All three logs have identical first 5000 traces (by activity sequence)."
        )
    else:
        print("\nResult: At least one pair differs in the first 5000 traces.")


if __name__ == "__main__":
    main()
