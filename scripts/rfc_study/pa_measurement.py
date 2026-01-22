"""
Preferential attachment measurement for network growth.

Uses Jeong-Néda-Barabási (2003) style measurement:
- Each node type is a "node" (e.g., trace variants, but can be any unit)
- Each occurrence is an "attachment" to that node
- T0 nodes: nodes existing at snapshot time T0 (with counts k_v(T0))
- Observation window [T1, T1+ΔT]: attachments counted in this interval
- Measures Pi(k) = probability of attachment to nodes with initial count k
- Uses cumulative kappa(k) = sum_{j<=k} Pi(j) to reduce noise
- Only attachments to nodes observed in T0 are counted (limitation of data model)

Windowing modes:
- Indices-based: slide by integer step sizes in attachments (positional)
- Time-based: slide by time using attachment_time timestamps (months, years, days, hours)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Utilities
# -----------------------------


def _prepare_attachments_for_indices_mode(
    df: pd.DataFrame,
    node_col: str,
    order_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare attachments for indices-based windowing.

    Parameters
    ----------
    df : pd.DataFrame
        Input attachments DataFrame.
    node_col : str
        Column name for node identifiers.
    order_col : str, optional
        Column to sort by (if None, preserves current order or uses index).

    Returns
    -------
    pd.DataFrame
        DataFrame with node_col and optionally order_col, sorted by order_col if provided.
    """
    if node_col not in df.columns:
        raise ValueError(f"Missing node_col={node_col!r} in attachments.columns")

    cols_to_keep = [node_col]
    if order_col is not None:
        if order_col not in df.columns:
            raise ValueError(f"Missing order_col={order_col!r} in attachments.columns")
        cols_to_keep.append(order_col)

    out = df[cols_to_keep].copy()
    if order_col is not None:
        out = out.sort_values(order_col, kind="mergesort").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def _prepare_attachments_for_time_mode(
    df: pd.DataFrame,
    node_col: str,
    time_col: str = "attachment_time",
) -> pd.DataFrame:
    """
    Prepare attachments for time-based windowing.

    Requires attachment_time column with parseable timestamps.
    Sorts by attachment_time (UTC) and asserts monotonic increasing.

    Parameters
    ----------
    df : pd.DataFrame
        Input attachments DataFrame.
    node_col : str
        Column name for node identifiers.
    time_col : str
        Column name for timestamps (default: "attachment_time").

    Returns
    -------
    pd.DataFrame
        DataFrame with node_col and time_col, sorted by time_col (UTC).
    """
    if node_col not in df.columns:
        raise ValueError(f"Missing node_col={node_col!r} in attachments.columns")
    if time_col not in df.columns:
        raise ValueError(f"Missing time_col={time_col!r} in attachments.columns")

    out = df[[node_col, time_col]].copy()

    # Parse to datetime UTC
    ts = pd.to_datetime(out[time_col], utc=True, errors="raise")
    out[time_col] = ts

    # Sort by time
    out = out.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    # Assert monotonic increasing
    if not out[time_col].is_monotonic_increasing:
        raise ValueError(
            f"{time_col} is not monotonic increasing after sorting. "
            f"Check for duplicate or out-of-order timestamps."
        )

    return out


def _counts_by_node(series: pd.Series) -> Dict[Any, int]:
    """Count occurrences of each node."""
    return dict(series.value_counts(dropna=False))


def _unit_to_date_offset(unit: str, value: int) -> pd.DateOffset:
    """
    Convert unit value to pandas DateOffset.

    Parameters
    ----------
    unit : str
        Unit: "months", "years", "days", or "hours".
    value : int
        Value to convert.

    Returns
    -------
    pd.DateOffset
        DateOffset object for the specified unit and value.
    """
    value = int(value or 0)
    if unit == "months":
        return pd.DateOffset(months=value)
    if unit == "years":
        return pd.DateOffset(years=value)
    if unit == "days":
        return pd.DateOffset(days=value)
    if unit == "hours":
        return pd.DateOffset(hours=value)
    raise ValueError(f"Unsupported spec.unit='{unit}'. Extend _offset() if needed.")


def compute_pi_and_kappa(
    k_at_T0: Dict[Any, int],
    k_at_T1_filtered: Dict[Any, int],
) -> pd.DataFrame:
    """
    Compute Π(k) and κ(k) exactly following
    Jeong–Néda–Barabási (2003).

    Parameters
    ----------
    k_at_T0 : Dict[Any, int]
        Degree k_v(T0) of each node v at snapshot T0 (non-normalized).
    k_at_T1_filtered : Dict[Any, int]
        Number of attachments Δk_v from T1 nodes to node v,
        filtered to nodes present at T0 (non-normalized).

    Returns
    -------
    pd.DataFrame
        Columns:
        - k        : degree at T0
        - kappa    : cumulative attachment function κ(k)
        - T0_ks    : number of T0-nodes with degree k
        - T1_ks    : number of attachments received by degree-k nodes
        - pi_k     : Π(k)
    """

    # --- Step 1: attach T1 counts to T0 degrees -----------------------------
    rows = []
    for node, k0 in k_at_T0.items():
        delta_k = k_at_T1_filtered.get(
            node, 0
        )  # delta_k is the number of new attachments in t1
        rows.append((k0, delta_k))

    df_nodes = pd.DataFrame(rows, columns=["k", "delta_k"])

    # --- Step 2: aggregate by degree class ----------------------------------
    grouped = (
        df_nodes.groupby("k", as_index=False)
        .agg(
            T0_ks=("k", "size"),  # number of T0-nodes with degree k
            T1_ks=("delta_k", "sum"),  # total attachments to degree-k nodes
        )
        .sort_values("k")
        .reset_index(drop=True)
    )

    # --- Step 3: Π(k) normalization -----------------------------------------
    total_attachments = grouped["T1_ks"].sum()

    if total_attachments == 0:
        grouped["pi_k"] = 0.0
    else:
        grouped["pi_k"] = grouped["T1_ks"] / total_attachments

    # --- Step 4: cumulative κ(k) --------------------------------------------
    grouped["kappa"] = grouped["pi_k"].cumsum()

    # --- Column order (paper-style clarity) ---------------------------------
    return grouped[["k", "kappa", "T0_ks", "T1_ks", "pi_k"]]


# -----------------------------
# Main measurement (single window)
# -----------------------------


@dataclass(frozen=True)
class PAWindowSpec:
    t0_end: int  # snapshot cut: attachments [0, t0_end) define V_T0 and k_v(T0)
    t1_start: (
        int  # measurement start: attachments [t1_start, t1_start+delta_t) are counted
    )
    delta_t: int  # delta T in "attachments" (length of measurement window)
    unit: str = (
        "indices"  # unit for the window spec: "indices", "months", or "years"; if years or months, these are relative to the first timestamp of the attachments
    )

    @property
    def gap(self) -> int:
        return self.t1_start - self.t0_end


def generate_pa_windows(
    first_t1_start: int,
    gap: int,
    step: int,
    delta_t: int,
    max_windows: Optional[int] = None,
    max_t1_end: Optional[int] = None,
    unit: str = "indices",
) -> List[PAWindowSpec]:
    """
    Generate a sequence of PAWindowSpec objects for sliding window analysis.

    Window specs are created with values in the original unit (e.g., years, months, or indices).
    Conversion to attachment indices happens later in compute_pi_kappa_window.

    Parameters
    ----------
    first_t1_start : int
        First t1_start value (start of first measurement window) in the specified unit.
    gap : int
        Gap between t0_end and t1_start (t0_end = t1_start - gap) in the specified unit.
    step : int
        Step size between consecutive t1_start values in the specified unit.
    delta_t : int
        Length of measurement window (delta T) in the specified unit.
    max_windows : int, optional
        Maximum number of windows to generate. If None, generates until data runs out.
    max_t1_end : int, optional
        Maximum t1_end value (t1_start + delta_t) in the specified unit (stops before exceeding this).
        Works for both indices and time-based units.
    unit : str
        Unit for window parameters: "indices", "months", "years", "days", or "hours".

    Returns
    -------
    List[PAWindowSpec]
        List of PAWindowSpec objects with values in the original unit, one per window.

    Raises
    ------
    ValueError
        If no windows can be generated because t1_end would exceed max_t1_end.

    Examples
    --------
    >>> windows = generate_pa_windows(first_t1_start=150, gap=50, step=100, delta_t=200)
    >>> # Window 0: t0_end=100 (150-50), t1_start=150, t1_end=350 (150+200) (in indices)
    >>> windows = generate_pa_windows(first_t1_start=1, gap=0, step=1, delta_t=1, unit="years")
    >>> # Window 0: t0_end=1 (1-0), t1_start=1, t1_end=2 (1+1) (unit="years")
    """
    windows = []
    i = 0
    while True:
        # Calculate window values in the original unit
        t1_start = first_t1_start + i * step
        t0_end = t1_start - gap  # T0 window: [t0_start, t0_end) = [0, t1_start - gap)
        t1_end = (
            t1_start + delta_t
        )  # T1 window: [t1_start, t1_end) = [t1_start, t1_start + delta_t)

        # Check stopping conditions
        if t0_end < 0:
            break  # Can't have negative t0_end
        if max_windows is not None and i >= max_windows:
            break
        # Check if t1_end would exceed max_t1_end
        if max_t1_end is not None and t1_end > max_t1_end:
            break

        windows.append(
            PAWindowSpec(t0_end=t0_end, t1_start=t1_start, delta_t=delta_t, unit=unit)
        )

        i += 1

    # Check if no windows were generated due to t1_end constraint
    if len(windows) == 0 and max_t1_end is not None:
        t1_end_first = first_t1_start + delta_t
        if t1_end_first > max_t1_end:
            raise ValueError(
                f"No windows can be generated: first window t1_end ({t1_end_first} {unit}) "
                f"exceeds max_t1_end ({max_t1_end} {unit})"
            )

    return windows


def compute_pi_kappa_window(
    attachments_sorted: pd.DataFrame,
    node_col: str,
    spec: "PAWindowSpec",
    time_col: str = "attachment_time",
    closed: str = "left",  # [start, end) by default
) -> pd.DataFrame:
    """
    Unified Pi(k)/kappa(k) computation for a single window (Jeong-Néda-Barabási 2003).

    - If spec.unit == "indices": uses positional slicing (fast, deterministic).
    - Else (e.g., months/years/days/hours): uses time-based slicing on `time_col` (no index conversion).

    Window semantics (paper-aligned):
      Indices mode:
        T0 snapshot:  [0, t0_end)  (always starts at first row)
        gap:          [t0_end, t1_start)
        Obs window:   [t1_start, t1_start + delta_t)

      Time-based mode:
        T0 snapshot:  [first_timestamp, T0_end_time)  (always starts at first timestamp)
        T1_start_time = first_timestamp + t1_start offset
        T0_end_time   = T1_start_time - gap offset
        T1_end_time   = T1_start_time + delta_t offset
        Obs window:   [T1_start_time, T1_end_time)

    Only attachments to nodes that appeared in T0 are counted (limitation of data model).
    Returns empty DataFrame if no T0 nodes or no observation attachments.
    """

    # Extract k_at_T0 and k_at_T1_filtered using appropriate method
    if spec.unit == "indices":
        k_at_T0, k_at_T1_filtered, metadata = _extract_k_at_T0_T1_indices(
            attachments_sorted=attachments_sorted,
            node_col=node_col,
            spec=spec,
        )
    else:
        k_at_T0, k_at_T1_filtered, metadata = _extract_k_at_T0_T1_time(
            attachments_sorted=attachments_sorted,
            node_col=node_col,
            spec=spec,
            time_col=time_col,
            closed=closed,
        )

    # Unified computation (agnostic to indices/time basis)
    if not k_at_T0:
        # Empty, well-formed DataFrame
        df = pd.DataFrame({"k": [], "T0_ks": [], "T1_ks": [], "pi_k": [], "kappa": []})
    else:
        df = compute_pi_and_kappa(k_at_T0=k_at_T0, k_at_T1_filtered=k_at_T1_filtered)

    # Add metadata to DataFrame
    for key, value in metadata.items():
        df[key] = value

    return df


# -----------------------------
# Window extraction: indices mode
# -----------------------------
def _extract_k_at_T0_T1_indices(
    attachments_sorted: pd.DataFrame,
    node_col: str,
    spec: "PAWindowSpec",
) -> Tuple[Dict[Any, int], Dict[Any, int], Dict[str, Any]]:
    """
    Extract k_at_T0 and k_at_T1_filtered for indices-based windowing.

    T0 always starts at index 0 (first row). T1 window position is determined by spec.

    Returns
    -------
    k_at_T0 : Dict[Any, int]
        Node counts at T0 snapshot.
    k_at_T1_filtered : Dict[Any, int]
        Node counts in T1 window, filtered to only nodes present in T0.
    metadata : Dict[str, Any]
        Metadata about the window (positions, unit, etc.).
    """
    n = len(attachments_sorted)

    t0_end_idx = int(spec.t0_end)
    t1_start_idx = int(spec.t1_start)
    delta_t_idx = int(spec.delta_t)

    # T0 always starts at 0
    t0_end_abs = t0_end_idx
    t1_start_abs = t1_start_idx
    t1_end_abs = t1_start_abs + delta_t_idx

    if t0_end_abs < 0 or t1_start_abs < 0:
        raise ValueError("Invalid window: negative boundaries.")
    if t0_end_abs > n or t1_start_abs > n or t1_end_abs > n:
        raise ValueError(
            f"Window exceeds available attachments: need {t1_end_abs}, have {n}"
        )

    # Assert no overlap between T0 and T1 windows
    # T0 is [0, t0_end_abs), T1 is [t1_start_abs, t1_end_abs)
    # They don't overlap if t1_start_abs >= t0_end_abs (since both ends are exclusive)
    assert t1_start_abs >= t0_end_abs, (
        f"T0 and T1 windows overlap: T0=[0, {t0_end_abs}), T1=[{t1_start_abs}, {t1_end_abs}). "
        f"T1 must start at or after T0 ends."
    )

    # T0 nodes and their counts (always starts at 0)
    t0_attachments = attachments_sorted.iloc[0:t0_end_abs]
    k_at_T0 = _counts_by_node(t0_attachments[node_col])

    # T1 attachments
    t1_attachments = attachments_sorted.iloc[t1_start_abs:t1_end_abs]
    k_at_T1 = _counts_by_node(t1_attachments[node_col])

    # Only count attachments to nodes present in T0
    k_at_T1_filtered = {node: c for node, c in k_at_T1.items() if node in k_at_T0}

    # Calculate kept_ratio: [total attachments in t1 after filtering] / [total attachments in t1]
    total_t1_attachments = sum(k_at_T1.values())
    total_t1_attachments_filtered = sum(k_at_T1_filtered.values())
    kept_ratio = (
        total_t1_attachments_filtered / total_t1_attachments
        if total_t1_attachments > 0
        else 0.0
    )

    # Metadata (positions in sorted attachments)
    metadata = {
        "t0_start_pos": 0,
        "t0_end_pos": t0_end_abs,
        "t1_start_pos": t1_start_abs,
        "t1_end_pos": t1_end_abs,
        "unit": "indices",
        "kept_ratio": kept_ratio,
    }

    return k_at_T0, k_at_T1_filtered, metadata


# --------------------------
# Window extraction: time mode
# --------------------------
def _extract_k_at_T0_T1_time(
    attachments_sorted: pd.DataFrame,
    node_col: str,
    spec: "PAWindowSpec",
    time_col: str,
    closed: str,
) -> Tuple[Dict[Any, int], Dict[Any, int], Dict[str, Any]]:
    """
    Extract k_at_T0 and k_at_T1_filtered for time-based windowing.

    T0 always starts at the first timestamp (index 0). T1 window position is determined by spec.

    Returns
    -------
    k_at_T0 : Dict[Any, int]
        Node counts at T0 snapshot.
    k_at_T1_filtered : Dict[Any, int]
        Node counts in T1 window, filtered to only nodes present in T0.
    metadata : Dict[str, Any]
        Metadata about the window (times, positions, unit, etc.).
    """
    if time_col not in attachments_sorted.columns:
        raise ValueError(f"attachments_sorted must contain time_col='{time_col}'")

    if closed not in {"left", "both"}:
        raise ValueError("closed must be 'left' ([a,b)) or 'both' ([a,b])")

    # T0 always starts at first timestamp (index 0)
    ts = pd.to_datetime(attachments_sorted[time_col], utc=True, errors="raise")
    attachments_sorted = attachments_sorted.copy()
    attachments_sorted["_ts"] = ts

    if not ts.is_monotonic_increasing:
        raise ValueError(
            f"{time_col} is not monotonic increasing. "
            f"Sort attachments_sorted by {time_col} before calling."
        )

    # First timestamp (T0 always starts here)
    base_time = ts.iloc[0]

    t1_start_time = base_time + _unit_to_date_offset(spec.unit, spec.t1_start)
    # T0 window: [base_time, t0_end_time), T1 window: [t1_start_time, t1_end_time)
    # t0_end = t1_start - gap, so t0_end_time = t1_start_time - gap_offset
    gap_offset = _unit_to_date_offset(spec.unit, spec.gap)
    t0_end_time = t1_start_time - gap_offset
    t1_end_time = t1_start_time + _unit_to_date_offset(spec.unit, spec.delta_t)

    # Ensure timestamps are consistent
    if t0_end_time < base_time:
        raise ValueError(
            "T0 end time is before base time. Reduce gap or increase t1 start time."
        )
    if t1_end_time < t1_start_time:
        raise ValueError(
            "Invalid window: t1_end_time < t1_start_time (delta_t negative?)"
        )

    # Assert no overlap between T0 and T1 windows
    # T0 is [base_time, t0_end_time) (exclusive end), T1 is [t1_start_time, t1_end_time) (exclusive end)
    # They don't overlap if t1_start_time >= t0_end_time (since both ends are exclusive)
    assert t1_start_time >= t0_end_time, (
        f"T0 and T1 windows overlap: T0=[{base_time}, {t0_end_time}), "
        f"T1=[{t1_start_time}, {t1_end_time}). T1 must start at or after T0 ends."
    )

    def _in_range(x: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        if closed == "left":
            return (x >= start) & (x < end)
        return (x >= start) & (x <= end)

    t0_mask = _in_range(attachments_sorted["_ts"], base_time, t0_end_time)
    t1_mask = _in_range(attachments_sorted["_ts"], t1_start_time, t1_end_time)

    t0_attachments = attachments_sorted.loc[t0_mask]
    t1_attachments = attachments_sorted.loc[t1_mask]

    k_at_T0 = _counts_by_node(t0_attachments[node_col]) if len(t0_attachments) else {}

    if len(t1_attachments) and k_at_T0:
        k_at_T1 = _counts_by_node(t1_attachments[node_col])
        k_at_T1_filtered = {node: c for node, c in k_at_T1.items() if node in k_at_T0}
    else:
        k_at_T1 = {}
        k_at_T1_filtered = {}

    # Calculate kept_ratio: [total attachments in t1 after filtering] / [total attachments in t1]
    total_t1_attachments = sum(k_at_T1.values())
    total_t1_attachments_filtered = sum(k_at_T1_filtered.values())
    kept_ratio = (
        total_t1_attachments_filtered / total_t1_attachments
        if total_t1_attachments > 0
        else 0.0
    )

    # Metadata (times + realized counts/positions for auditing)
    metadata = {
        "unit": spec.unit,
        "base_time": base_time,
        "t0_start_time": base_time,
        "t0_end_time": t0_end_time,
        "t1_start_time": t1_start_time,
        "t1_end_time": t1_end_time,
        "t0_n_attachments": int(t0_mask.sum()),
        "obs_n_attachments": int(t1_mask.sum()),
        "kept_ratio": kept_ratio,
    }

    # Positions in full sorted attachments
    if t0_mask.any():
        idxs = np.where(t0_mask.to_numpy())[0]
        metadata["t0_pos_first"] = int(idxs[0])
        metadata["t0_pos_last"] = int(idxs[-1])
    else:
        metadata["t0_pos_first"] = -1
        metadata["t0_pos_last"] = -1

    if t1_mask.any():
        idxs = np.where(t1_mask.to_numpy())[0]
        metadata["obs_pos_first"] = int(idxs[0])
        metadata["obs_pos_last"] = int(idxs[-1])
    else:
        metadata["obs_pos_first"] = -1
        metadata["obs_pos_last"] = -1

    return k_at_T0, k_at_T1_filtered, metadata


# -----------------------------
# Repeated measurements
# -----------------------------


def run_pa_measurements(
    attachments: pd.DataFrame,
    time_col: str,
    order_col: str,
    node_col: str,
    spec: PAWindowSpec,
    step_attachments: int,
    max_windows: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Slide a window spec forward by step_attachments and compute Pi/kappa repeatedly.

    Dispatches to indices-based or time-based implementation based on spec.unit.

    Parameters
    ----------
    attachments : pd.DataFrame
        Attachments DataFrame.
    time_col : str
        Column name for timestamps (used for time-based units).
    order_col : str
        Column name for ordering (used for indices-based units).
    node_col : str
        Column name for node identifiers.
    spec : PAWindowSpec
        Window specification (t0_end, t1_start, delta_t, unit).
    step_attachments : int
        Step size in same units as spec.unit.
    max_windows : int, optional
        Maximum number of windows to compute.

    Returns
    -------
    List[pd.DataFrame]
        List of Pi/kappa DataFrames, one per window.

    Note
    ----
    For more control over window positions, use run_pa_measurements_from_specs()
    with generate_pa_windows().
    """
    # Sanity assertions
    if spec.unit == "indices":
        if spec.t0_end < 0 or spec.t1_start < 0 or spec.delta_t < 0:
            raise ValueError(
                "For indices unit, spec values must be non-negative integers"
            )
        if (
            not isinstance(spec.t0_end, (int, np.integer))
            or not isinstance(spec.t1_start, (int, np.integer))
            or not isinstance(spec.delta_t, (int, np.integer))
        ):
            raise ValueError("For indices unit, spec values must be integers")
        return _run_pa_measurements_indices(
            attachments=attachments,
            order_col=order_col,
            node_col=node_col,
            spec=spec,
            step_attachments=step_attachments,
            max_windows=max_windows,
        )
    else:
        # Time-based units
        if (
            spec.t0_end < 0
            or spec.t1_start < 0
            or spec.delta_t < 0
            or step_attachments < 0
        ):
            raise ValueError(
                "For time-based units, spec values and step must be non-negative"
            )
        if time_col not in attachments.columns:
            raise ValueError(
                f"Time-based units require '{time_col}' column in attachments"
            )
        return _run_pa_measurements_time(
            attachments=attachments,
            node_col=node_col,
            spec=spec,
            step=step_attachments,
            max_windows=max_windows,
        )


def _run_pa_measurements_indices(
    attachments: pd.DataFrame,
    order_col: str,
    node_col: str,
    spec: PAWindowSpec,
    step_attachments: int,
    max_windows: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Run PA measurements with indices-based windowing.

    T0 always starts at index 0. Slides T1 window by step_attachments.
    """
    sorted_attachments = _prepare_attachments_for_indices_mode(
        attachments, node_col=node_col, order_col=order_col
    )
    n_attachments = len(sorted_attachments)
    window_dfs = []
    window_count = 0
    t1_start_offset = 0  # Offset to add to spec.t1_start for sliding

    while True:
        # Create a modified spec with adjusted t1_start for this window
        # Recalculate t0_end to maintain relationship: t0_end = t1_start - gap
        adjusted_t1_start = spec.t1_start + t1_start_offset
        adjusted_t0_end = adjusted_t1_start - spec.gap
        adjusted_spec = PAWindowSpec(
            t0_end=adjusted_t0_end,
            t1_start=adjusted_t1_start,
            delta_t=spec.delta_t,
            unit=spec.unit,
        )

        try:
            window_df = compute_pi_kappa_window(
                sorted_attachments,
                node_col=node_col,
                spec=adjusted_spec,
            )
            window_dfs.append(window_df)
            window_count += 1

            if max_windows is not None and window_count >= max_windows:
                break

            # Slide T1 window by step_attachments
            t1_start_offset += step_attachments

            # Check bounds: T1 window must fit
            t1_start_abs = adjusted_spec.t1_start + step_attachments
            t1_end_abs = t1_start_abs + spec.delta_t
            if t1_end_abs > n_attachments:
                break

        except ValueError:
            break

    return window_dfs


def _run_pa_measurements_time(
    attachments: pd.DataFrame,
    node_col: str,
    spec: PAWindowSpec,
    step: int,
    max_windows: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Run PA measurements with time-based windowing.

    T0 always starts at the first timestamp. Slides T1 window by step in time units.
    No index conversion - operates directly on timestamps.

    Parameters
    ----------
    attachments : pd.DataFrame
        Attachments DataFrame with attachment_time column.
    node_col : str
        Column name for node identifiers.
    spec : PAWindowSpec
        Window specification with time-based unit.
    step : int
        Step size in same units as spec.unit.
    max_windows : int, optional
        Maximum number of windows to compute.

    Returns
    -------
    List[pd.DataFrame]
        List of Pi/kappa DataFrames, one per window.
    """
    sorted_attachments = _prepare_attachments_for_time_mode(
        attachments, node_col=node_col, time_col="attachment_time"
    )
    n = len(sorted_attachments)

    # Precompute timestamps array (numpy datetime64 for efficient searchsorted)
    ts_series = pd.to_datetime(sorted_attachments["attachment_time"], utc=True)
    ts_array = ts_series.values

    # Base time: first timestamp (T0 always starts here)
    base_time0 = ts_series.iloc[0]

    window_dfs = []
    window_count = 0
    i = 0

    while True:
        # Create a modified spec with adjusted t1_start for this window
        # T0 always starts at base_time0, T1 slides by i * step
        # Recalculate t0_end to maintain relationship: t0_end = t1_start - gap
        adjusted_t1_start = spec.t1_start + i * step
        adjusted_t0_end = adjusted_t1_start - spec.gap
        adjusted_spec = PAWindowSpec(
            t0_end=adjusted_t0_end,
            t1_start=adjusted_t1_start,
            delta_t=spec.delta_t,
            unit=spec.unit,
        )

        try:
            window_df = compute_pi_kappa_window(
                sorted_attachments,
                node_col=node_col,
                spec=adjusted_spec,
                time_col="attachment_time",
            )

            # Check if observation window has any attachments
            if (
                len(window_df) == 0
                or window_df.get("obs_n_attachments", pd.Series([0])).iloc[0] == 0
            ):
                # Empty observation window - stop
                break

            window_dfs.append(window_df)
            window_count += 1

            if max_windows is not None and window_count >= max_windows:
                break

            i += 1

        except ValueError:
            # Window exceeds data or is invalid
            break

    return window_dfs


def run_pa_measurements_from_specs(
    attachments: pd.DataFrame,
    time_col: str,
    order_col: str,
    node_col: str,
    specs: List[PAWindowSpec],
) -> List[pd.DataFrame]:
    """
    Compute Pi/kappa for a list of window specs.

    Assumes specs have absolute positions starting at index/ time 0.
    Use generate_pa_windows() for generating windows.

    Parameters
    ----------
    attachments : pd.DataFrame
        Attachments DataFrame.
    time_col : str
        Column name for timestamps (used for time-based units).
    order_col : str
        Column name for ordering (used for indices-based units).
    node_col : str
        Column name for node identifiers.
    specs : List[PAWindowSpec]
        List of window specifications.

    Returns
    -------
    List[pd.DataFrame]
        List of Pi/kappa DataFrames, one per window.
    """
    # Check if any spec needs time-based computation
    needs_time_based = any(spec.unit != "indices" for spec in specs)

    # Prepare attachments based on mode
    if needs_time_based:
        if time_col not in attachments.columns:
            raise ValueError(
                f"Time-based windows require '{time_col}' column in attachments DataFrame. "
                f"Available columns: {list(attachments.columns)}"
            )
        sorted_attachments = _prepare_attachments_for_time_mode(
            attachments, node_col=node_col, time_col=time_col
        )
    else:
        sorted_attachments = _prepare_attachments_for_indices_mode(
            attachments, node_col=node_col, order_col=order_col
        )

    window_dfs = []

    for spec in specs:
        try:
            if spec.unit == "indices":
                # Use indices mode - no time column needed
                window_df = compute_pi_kappa_window(
                    sorted_attachments, node_col=node_col, spec=spec
                )
            else:
                # Use time-based mode - use time_col parameter
                window_df = compute_pi_kappa_window(
                    sorted_attachments,
                    node_col=node_col,
                    spec=spec,
                    time_col=time_col,
                )
            window_dfs.append(window_df)
        except (ValueError, KeyError) as e:
            # Skip windows that exceed available data or have other issues
            # Print error for debugging
            import sys

            print(
                f"    Warning: Skipping window (t1_start={spec.t1_start} {spec.unit}): {e}",
                file=sys.stderr,
            )
            continue

    return window_dfs


# -----------------------------
# alpha estimation from kappa(k)
# -----------------------------


@dataclass(frozen=True)
class AlphaEstimate:
    alpha: float
    slope: float
    intercept: float
    r2: float
    n_points: int
    k_min_used: int
    k_max_used: int


def estimate_alpha_from_kappa(
    df_pi: pd.DataFrame,
    k_min: int = 2,
    k_max: Optional[int] = None,
    min_points: int = 1,
) -> Optional[AlphaEstimate]:
    """
    Estimate alpha from kappa(k) ~ k^{alpha+1}.

    Fits log(kappa) = (alpha+1) * log(k) + c using least squares.
    Returns None if insufficient data points.
    """
    # Filter to valid range
    data = df_pi[(df_pi["k"] >= k_min) & (df_pi["kappa"] > 0)].copy()
    if k_max is not None:
        data = data[data["k"] <= k_max].copy()

    if len(data) < min_points:
        return None

    # Fit log(kappa) = slope * log(k) + intercept
    log_k = np.log(data["k"].to_numpy(dtype=float))
    log_kappa = np.log(data["kappa"].to_numpy(dtype=float))

    # Least squares: y = X * beta
    X = np.vstack([log_k, np.ones_like(log_k)]).T
    beta, _, _, _ = np.linalg.lstsq(X, log_kappa, rcond=None)
    slope = float(beta[0])
    intercept = float(beta[1])

    # Compute R²
    log_kappa_pred = slope * log_k + intercept
    ss_res = float(((log_kappa - log_kappa_pred) ** 2).sum())
    ss_tot = float(((log_kappa - log_kappa.mean()) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # alpha = slope - 1 (since kappa ~ k^{alpha+1})
    alpha = slope - 1.0

    return AlphaEstimate(
        alpha=alpha,
        slope=slope,
        intercept=intercept,
        r2=r2,
        n_points=len(data),
        k_min_used=int(data["k"].min()),
        k_max_used=int(data["k"].max()),
    )


# -----------------------------
# Plotting functions
# -----------------------------


def plot_kappa(
    df_pi: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    k_min: int = 1,
    k_anchor_fit_min: int = 2,
    k_anchor_fit_max: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot κ(k) vs k on log-log scale with simple slope guides.

    Jeong–Néda–Barabási (2003):
      Π(k) ∝ k^α  =>  κ(k) ∝ k^(α+1)

    We draw slope guides κ ∝ k^p for p ∈ {1,2,3}, corresponding to α ∈ {0,1,2}.
    These are visual guides; they do not (and need not) respect κ ≤ 1 everywhere.
    """
    if ax is None:
        _, ax = plt.subplots()

    valid = df_pi[(df_pi["k"] >= k_min) & (df_pi["kappa"] > 0)].copy()

    ax.set_xlabel("k (node count at T0)")
    ax.set_ylabel("kappa(k) (cumulative Pi)")

    if len(valid) == 0:
        ax.set_title("No attachments from T0 observed in observation window")
        ax.grid(True, alpha=0.3)
        return ax

    # Set log-log scale explicitly
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)

    measured_label = label if label is not None else "Measured"
    ax.plot(valid["k"], valid["kappa"], marker="o", linestyle="-", label=measured_label)

    # Add slope guides (κ ∝ k, k^2, k^3) using a simple anchored helper
    _add_slope_guides(
        ax=ax,
        valid_data=valid,
        k_fit_min=k_anchor_fit_min,
        k_fit_max=k_anchor_fit_max,
        slopes=(1, 2, 3),
    )

    ax.legend()

    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)

    return ax


def _draw_slope_guide(
    ax: plt.Axes,
    *,
    k0: float,
    y0: float,
    slope: float,
    k_vals: np.ndarray,
    linestyle: str,
    color: str,
    label: str,
    linewidth: float = 1.3,
) -> None:
    """Draw y = y0 * (k/k0)^slope on log-log axes over the provided k_vals."""
    k_vals = np.asarray(k_vals, dtype=float)
    mask = (k_vals > 0) & (k_vals >= k0)
    if not np.any(mask):
        return

    y = y0 * (k_vals[mask] / k0) ** slope
    ax.plot(
        k_vals[mask],
        y,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        label=label,
    )


def _add_slope_guides(
    ax: plt.Axes,
    *,
    valid_data: pd.DataFrame,
    k_fit_min: int = 2,
    k_fit_max: Optional[int] = None,
    slopes: Tuple[int, ...] = (1, 2, 3),
) -> None:
    """Add κ(k) slope guides using a single anchor point from the measured data.

    Anchor selection: use the smallest k's x and y value in [k_fit_min, k_fit_max].
    If no points exist in the fit range, fall back to all valid points.
    """
    fit = valid_data[valid_data["k"] >= k_fit_min].copy()
    if k_fit_max is not None:
        fit = fit[fit["k"] <= k_fit_max].copy()
    if len(fit) == 0:
        fit = valid_data.copy()
    if len(fit) == 0:
        return

    fit = fit.sort_values("k")
    # Use smallest k's x and y value as anchor point
    k0 = float(fit["k"].iloc[0])
    y0 = float(fit["kappa"].iloc[0])

    if not (k0 > 0 and y0 > 0):
        return

    k_vals = valid_data["k"].to_numpy(dtype=float)

    # Style map (simple, paper-like)
    style = {
        1: dict(linestyle="--", color="gray", label="κ(k) ∝ k (α = 0, random)"),
        2: dict(linestyle="--", color="black", label="κ(k) ∝ k² (α = 1, linear PA)"),
        3: dict(
            linestyle=":", color="black", label="κ(k) ∝ k³ (α = 2, superlinear PA)"
        ),
    }

    for s in slopes:
        st = style.get(
            int(s), dict(linestyle="--", color="black", label=f"κ(k) ∝ k^{s}")
        )
        _draw_slope_guide(
            ax,
            k0=k0,
            y0=y0,
            slope=float(s),
            k_vals=k_vals,
            linestyle=st["linestyle"],
            color=st["color"],
            label=st["label"],
        )


def plot_alpha_over_windows(
    window_dfs: List[pd.DataFrame],
    k_min_fit: int = 2,
    k_max_fit: Optional[int] = None,
    min_points: int = 8,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute alpha per window and plot it over window index.
    Returns a summary DataFrame of estimates.

    Includes all window boundary data: t0_start, t0_end (inclusive), t1_start, t1_end, and unit.
    """
    rows = []
    for i, dfw in enumerate(window_dfs):
        est = estimate_alpha_from_kappa(
            dfw, k_min=k_min_fit, k_max=k_max_fit, min_points=min_points
        )
        if est is None:
            continue

        row: Dict[str, Any] = {
            "window_idx": i,
            "alpha": est.alpha,
            "slope": est.slope,
            "r2": est.r2,
            "n_points": est.n_points,
        }

        # Extract window boundary data based on mode
        if "t0_start_pos" in dfw.columns:
            # Indices mode: use position values, make t0_end and t1_end inclusive
            row["t0_start"] = int(dfw["t0_start_pos"].iloc[0])
            row["t0_end"] = (
                int(dfw["t0_end_pos"].iloc[0]) - 1
            )  # Make inclusive (code uses exclusive)
            row["t1_start"] = int(dfw["t1_start_pos"].iloc[0])
            row["t1_end"] = (
                int(dfw["t1_end_pos"].iloc[0]) - 1
            )  # Make inclusive (code uses exclusive)
            unit_val = dfw["unit"].iloc[0]
            row["unit"] = str(unit_val)
        elif "t0_start_time" in dfw.columns:
            # Time mode: use timestamp values
            row["t0_start"] = dfw["t0_start_time"].iloc[0]
            row["t0_end"] = dfw["t0_end_time"].iloc[0]
            row["t1_start"] = dfw["t1_start_time"].iloc[0]
            row["t1_end"] = dfw["t1_end_time"].iloc[0]
            unit_val = dfw["unit"].iloc[0]
            row["unit"] = str(unit_val)
        else:
            # Fallback: try to get unit at least
            if "unit" in dfw.columns:
                unit_val = dfw["unit"].iloc[0]
                row["unit"] = str(unit_val)

        # Add kept_ratio if available
        if "kept_ratio" in dfw.columns:
            kept_ratio_val = dfw["kept_ratio"].iloc[0]
            row["kept_ratio"] = (
                float(kept_ratio_val) if pd.notna(kept_ratio_val) else 0.0
            )

        rows.append(row)

    summ = pd.DataFrame(rows)
    if len(summ) == 0:
        # No valid estimates, return empty DataFrame
        plt.close("all")  # Clean up
        columns = [
            "window_idx",
            "alpha",
            "slope",
            "r2",
            "n_points",
            "t0_start",
            "t0_end",
            "t1_start",
            "t1_end",
            "unit",
            "kept_ratio",
        ]
        return pd.DataFrame(columns=columns)

    fig, ax = plt.subplots()
    ax.plot(summ["window_idx"], summ["alpha"], marker="o", linestyle="-")
    ax.set_xlabel("window index")
    ax.set_ylabel("alpha (from kappa(k) ~ k^{alpha+1})")
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return summ


# -----------------------------
# Main analysis interface
# -----------------------------


def _calculate_max_t1_end(
    attachments_df: pd.DataFrame,
    unit: str,
) -> int:
    """
    Calculate max_t1_end (maximum end of observation window) for window generation.

    Parameters
    ----------
    attachments_df : pd.DataFrame
        DataFrame with attachments data.
    unit : str
        Unit for window parameters: "indices", "months", "years", "days", or "hours".

    Returns
    -------
    int
        Maximum t1_end value in the specified unit.

    Raises
    ------
    ValueError
        If attachment_time column is missing for time-based units, or unit is unsupported.
    """
    if unit == "indices":
        return len(attachments_df)  # t1_end is the end position
    else:
        # For time-based units, calculate max_t1_end from timestamps
        if "attachment_time" not in attachments_df.columns:
            raise ValueError(
                f"attachment_time column required for time-based unit '{unit}'"
            )
        timestamps = pd.to_datetime(attachments_df["attachment_time"], utc=True)
        first_time = timestamps.iloc[0]
        last_time = timestamps.iloc[-1]
        time_span = last_time - first_time

        # Convert time span to the specified unit
        if unit == "years":
            return int(time_span.days / 365.25)
        elif unit == "months":
            return int(time_span.days / 30)
        elif unit == "days":
            return int(time_span.days)
        elif unit == "hours":
            return int(time_span.total_seconds() / 3600)
        else:
            raise ValueError(f"Unsupported unit: {unit}")


def run_pa_analysis_from_csv(
    attachments_csv_path: str | Path,
    output_dir: str | Path,
    first_t1_start: int,
    gap: int,
    step: int,
    delta_t: int,
    max_windows: Optional[int] = None,
    unit: str = "indices",
    time_col: str = "attachment_time",
    index_col: str = "attachment_index",
    node_col: str = "node_id",
    dataset_name: Optional[str] = None,
) -> None:
    """
    Run PA analysis from a CSV file with attachment data.

    This is the main interface function for PA analysis. It loads attachment data from CSV,
    generates windows, computes measurements, and creates plots.

    Parameters
    ----------
    attachments_csv_path : str | Path
        Path to CSV file with attachment data. Expected columns:
        - time_col (default: "attachment_time"): timestamp column
        - index_col (default: "attachment_index"): sequential index column
        - node_col (default: "node_id"): node identifier column
    output_dir : str | Path
        Directory where analysis results will be saved.
    first_t1_start : int
        First t1_start value for window generation (in specified unit).
    gap : int
        Gap between t0_end and t1_start (in specified unit).
    step : int
        Step size between consecutive t1_start values (in specified unit).
    delta_t : int
        Length of measurement window (in specified unit).
    max_windows : int, optional
        Maximum number of windows to generate. If None, generates all possible windows.
    unit : str, default "indices"
        Unit for window parameters: "indices", "months", "years", "days", or "hours".
    time_col : str, default "attachment_time"
        Column name for timestamps in the CSV.
    index_col : str, default "attachment_index"
        Column name for sequential indices in the CSV.
    node_col : str, default "node_id"
        Column name for node identifiers in the CSV.
    dataset_name : str, optional
        Name of the dataset (for logging). If None, inferred from CSV path.
    """
    # Convert paths
    attachments_csv_path = Path(attachments_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer dataset name if not provided
    if dataset_name is None:
        dataset_name = attachments_csv_path.stem

    # Load attachments from CSV
    print(f"Loading attachments from {attachments_csv_path}...")
    attachments_df = pd.read_csv(attachments_csv_path)

    # Validate required columns
    required_cols = [time_col, index_col, node_col]
    missing_cols = [col for col in required_cols if col not in attachments_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Ensure index_col is numeric and sort by it
    attachments_df[index_col] = pd.to_numeric(attachments_df[index_col])
    attachments_df = attachments_df.sort_values(index_col).reset_index(drop=True)

    print(f"  Total attachments: {len(attachments_df)}")
    print(f"  Unique nodes: {attachments_df[node_col].nunique()}")
    print(f"  Unit: {unit}")

    # Calculate max_t1_end for window generation
    max_t1_end = _calculate_max_t1_end(attachments_df, unit)

    # Generate window specs
    try:
        window_specs = generate_pa_windows(
            first_t1_start=first_t1_start,
            gap=gap,
            step=step,
            delta_t=delta_t,
            max_windows=max_windows,
            max_t1_end=max_t1_end,
            unit=unit,
        )
    except ValueError as e:
        print(f"  Error: {e}")
        return

    print(f"  Generated {len(window_specs)} window specifications")

    if not window_specs:
        print(f"  Error: No window specifications generated")
        return

    try:
        # Run PA measurements
        print(f"  Computing PA measurements...")
        window_dfs = run_pa_measurements_from_specs(
            attachments=attachments_df,
            time_col=time_col,
            order_col=index_col,
            node_col=node_col,
            specs=window_specs,
        )

        if not window_dfs:
            print(f"  Warning: No valid windows generated")
            return

        print(f"  Generated {len(window_dfs)} measurement windows")

        # Create kappa plots for each window
        print(f"  Creating kappa plots...")
        for i, window_df in enumerate(window_dfs):
            # Calculate original t1_start value in the specified unit
            original_t1_start = first_t1_start + i * step
            kappa_path = output_dir / f"k_window_{i:02d}_{original_t1_start:06d}.png"

            # Check if window has any valid kappa > 0 data
            valid_data = (
                len(window_df[(window_df["k"] >= 1) & (window_df["kappa"] > 0)]) > 0
            )
            if valid_data:
                plot_kappa(window_df, save_path=str(kappa_path))
                print(
                    f"    Window {i:02d} (t1_start={original_t1_start} {unit}): Plot created"
                )
            else:
                print(
                    f"    Window {i:02d} (t1_start={original_t1_start} {unit}): No attachments from T0 observed in observation window, skipping plot"
                )

        # Create alpha over windows plot
        print(f"  Creating alpha over windows plot...")
        alpha_csv_path = output_dir / "alpha_over_windows.csv"
        alpha_plot_path = output_dir / "alpha_over_windows.png"

        alpha_summary = plot_alpha_over_windows(
            window_dfs=window_dfs,
            save_path=str(alpha_plot_path),
        )

        # Save alpha summary to CSV
        alpha_summary.to_csv(alpha_csv_path, index=False)
        print(f"  Saved alpha summary: {alpha_csv_path.name}")

        # Report final window alpha if available
        if window_dfs:
            final_alpha = estimate_alpha_from_kappa(window_dfs[-1])
            if final_alpha:
                print(
                    f"  Final window alpha: {final_alpha.alpha:.4f} (R²={final_alpha.r2:.4f})"
                )

        print(f"  All PA outputs saved to {output_dir}")

    except Exception as e:
        print(f"  Error in PA analysis: {e}")
        import traceback

        traceback.print_exc()


# -----------------------------
# Tests
# -----------------------------


def test_pa_measurement_modes() -> None:
    """
    Minimal test: verify indices-based and time-based modes produce identical results
    when time increments correspond 1:1 to indices.

    Creates timestamps at 1-day increments, nodes A/B/C with known degrees in T0
    and known attachments in Obs window. Verifies both modes produce identical results.
    """
    # Create synthetic dataset: timestamps at 1-day increments
    n = 100
    timestamps = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    nodes = ["A", "B", "C"] * (n // 3 + 1)
    nodes = nodes[:n]

    # Create attachments DataFrame
    attachments = pd.DataFrame(
        {
            "node": nodes,
            "attachment_time": timestamps,
        }
    )

    # Sort by time (required for both modes)
    attachments_sorted = attachments.sort_values("attachment_time").reset_index(
        drop=True
    )

    # Window spec: T0=[0, 20), gap=0 (t0_end=20, t1_start=20), Obs=[20, 40)
    # With t0_end = t1_start - gap: 20 = 20 - 0
    spec_indices = PAWindowSpec(t0_end=20, t1_start=20, delta_t=20, unit="indices")
    spec_days = PAWindowSpec(t0_end=20, t1_start=20, delta_t=20, unit="days")

    # Compute with indices mode
    result_indices = compute_pi_kappa_window(
        attachments_sorted=attachments_sorted,
        node_col="node",
        spec=spec_indices,
    )

    # Compute with time mode (days)
    result_time = compute_pi_kappa_window(
        attachments_sorted=attachments_sorted,
        node_col="node",
        spec=spec_days,
        time_col="attachment_time",
    )

    # Results should be identical (same windows, same data)
    assert len(result_indices) == len(result_time), "Result lengths should match"
    if len(result_indices) > 0:
        # Compare core metrics (ignore metadata columns that differ)
        pd.testing.assert_frame_equal(
            result_indices[["k", "T0_ks", "T1_ks", "pi_k", "kappa"]],
            result_time[["k", "T0_ks", "T1_ks", "pi_k", "kappa"]],
            check_exact=False,
            rtol=1e-10,
        )

    # Test sliding windows
    windows_indices = run_pa_measurements(
        attachments=attachments,
        time_col="attachment_time",
        order_col="attachment_index",
        node_col="node",
        spec=spec_indices,
        step_attachments=10,
        max_windows=3,
    )

    windows_time = run_pa_measurements(
        attachments=attachments,
        time_col="attachment_time",
        order_col="attachment_index",
        node_col="node",
        spec=spec_days,
        step_attachments=10,  # 10 days
        max_windows=3,
    )

    assert len(windows_indices) == len(windows_time), "Window counts should match"
    if len(windows_indices) > 0:
        # First window should match
        pd.testing.assert_frame_equal(
            windows_indices[0][["k", "T0_ks", "T1_ks", "pi_k", "kappa"]],
            windows_time[0][["k", "T0_ks", "T1_ks", "pi_k", "kappa"]],
            check_exact=False,
            rtol=1e-10,
        )

    print(
        "✓ Test passed: indices and time modes produce identical results for 1:1 mapping"
    )


if __name__ == "__main__":
    test_pa_measurement_modes()
