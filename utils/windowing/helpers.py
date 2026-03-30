"""Windowing helper functions for creating windows from event logs."""

import bisect
from typing import Any, List, Optional, Tuple

import pandas as pd

from pm4py.objects.log.obj import Trace

from utils.windowing.window import Window


def _trace_start_time(trace: Trace) -> Optional[Any]:
    """Get the start time of a trace.

    Args:
        trace: PM4Py Trace object.

    Returns:
        Start timestamp or None if not available.
    """
    if not trace:
        return None
    return trace[0].get("time:timestamp", None)


def _make_window(
    log: List[Trace],
    i0: int,
    i1: int,
    wid: str,
    scp: Optional[int] = None,
    scpt: Optional[str] = None,
    ecp: Optional[int] = None,
    ecpt: Optional[str] = None,
) -> Window:
    """Create a window from a log slice.

    Args:
        log: List of PM4Py Trace objects.
        i0: Start index.
        i1: End index.
        wid: Window ID.
        scp: Start change point.
        scpt: Start change point type.
        ecp: End change point.
        ecpt: End change point type.

    Returns:
        Window object.
    """
    n = len(log)
    if i0 < 0 or i1 >= n or i0 > i1:
        raise ValueError("Window creation parameters do not fit.")
    sub = log[i0 : i1 + 1]
    # Calculate center moment as the timestamp of the trace at the midpoint
    # For a window from i0 to i1 (inclusive), the midpoint index is (i0 + i1) // 2
    # This represents w/2, e.g., for w=100 starting at 0, center is at trace 50
    center_idx = (i0 + i1) // 2
    center_moment = (
        _trace_start_time(sub[center_idx - i0])
        if sub and 0 <= (center_idx - i0) < len(sub)
        else None
    )

    return Window(
        id=wid,
        first_index=i0,
        last_index=i1,
        size=len(sub),
        start_moment=_trace_start_time(sub[0]) if sub else None,
        end_moment=_trace_start_time(sub[-1]) if sub else None,
        center_moment=center_moment,
        traces=sub,
        start_change_point=scp,
        start_change_point_type=scpt,
        end_change_point=ecp,
        end_change_point_type=ecpt,
    )


def split_log_into_windows_by_change_points(
    log: List[Trace], change_points: List[Tuple[int, int, str]]
) -> List[Window]:
    """Split log into windows based on change points.

    Args:
        log: List of PM4Py Trace objects.
        change_points: List of (index, change_point_id, change_type) tuples.

    Returns:
        List of Window objects.
    """
    n = len(log)
    sorted_points = sorted(change_points, key=lambda x: x[0])
    boundaries = [(0, None, None)] + sorted_points + [(n, None, None)]
    out: List[Window] = []
    for i in range(len(boundaries) - 1):
        i0 = boundaries[i][0]
        i1 = boundaries[i + 1][0] - 1
        scp = boundaries[i][1]
        scpt = boundaries[i][2]
        ecp = boundaries[i + 1][1]
        ecpt = boundaries[i + 1][2]
        out.append(_make_window(log, i0, i1, str(i), scp, scpt, ecp, ecpt))
    return out


def split_log_into_fixed_windows(
    log: List[Trace],
    window_size: int,
    offset: int,
    include_incomplete_windows: bool = True,
) -> List[Window]:
    """Split log into fixed-size windows.

    Args:
        log: List of PM4Py Trace objects.
        window_size: Size of each window.
        offset: Offset between windows.
        include_incomplete_windows: If True, include trailing partial windows by
            truncating to the end of the log. If False, keep only full windows
            of exactly ``window_size`` traces.

    Returns:
        List of Window objects.
    """
    n = len(log)
    out: List[Window] = []
    wid = 0
    i0 = 0
    if window_size <= 0 or offset <= 0 or window_size > n:
        return out
    while i0 < n:
        i1 = i0 + window_size - 1
        if i1 >= n:
            if not include_incomplete_windows:
                break
            i1 = n - 1
        out.append(_make_window(log, i0, i1, str(wid)))
        wid += 1
        i0 += offset
    return out


def split_log_into_growing_prefix_trace_windows(
    log: List[Trace], increment: int, *, start_index: int = 0
) -> List[Window]:
    """Nested prefix windows: each window starts at ``start_index`` and grows by ``increment`` traces.

    Window index ``wid`` (0-based) spans traces with sizes ``increment``, ``2*increment``, …
    i.e. ``last_index = start_index + (wid + 1) * increment - 1``, while that slice fits in the log.

    Args:
        log: List of PM4Py Trace objects (e.g. from ``load_xes_log``, sorted by start time).
        increment: Trace count added per iteration (>= 1).
        start_index: Index of the first trace in every window (default 0).

    Returns:
        List of Window objects. Empty if parameters are invalid or the log is too short
        for at least one full window of size ``increment`` from ``start_index``.
    """
    n = len(log)
    out: List[Window] = []
    if increment <= 0 or start_index < 0 or start_index >= n:
        return out
    k = 1
    while start_index + k * increment <= n:
        i0 = start_index
        i1 = start_index + k * increment - 1
        out.append(_make_window(log, i0, i1, str(len(out))))
        k += 1
    return out


def split_log_into_fixed_time_windows(
    log: List[Trace],
    window_size: int,
    offset: int,
    unit: str,
    align_first_window: bool,
    include_incomplete_windows: bool = True,
) -> List[Window]:
    """
    Split log into fixed time windows (sliding by time).

    Semantics:
    - The input log is assumed to be sorted by trace start timestamp.
    - A trace is assigned to a time window iff its start timestamp is in
      [window_start, window_end) (start inclusive, end exclusive).
    - Because windows are defined over trace start times, each window's
      assigned traces form a contiguous slice in the start-sorted log.

    Args:
        log: List of PM4Py Trace objects, sorted by trace start time.
        window_size: Window length in `unit` (int, >= 1).
        offset: Step between consecutive windows in `unit` (int, >= 1).
        unit: 'day', 'month', or 'year'.
        align_first_window: If True, align the first window start to the
            beginning of the corresponding day/month/year in the first event's
            timestamp.
        include_incomplete_windows: If True, include a trailing partial window.
            If False, exclude windows whose end would be beyond the last trace
            start timestamp.

    Returns:
        List of Window objects. Empty windows (with zero assigned traces) are
        omitted.
    """
    if not log:
        return []
    if window_size <= 0 or offset <= 0:
        return []

    unit = str(unit).lower().strip()
    if unit not in {"day", "month", "year"}:
        raise ValueError("unit must be one of {'day','month','year'}")

    start_times: List[pd.Timestamp] = []
    for tr in log:
        ts = _trace_start_time(tr)
        if ts is None:
            raise ValueError("All traces must have a non-empty time:timestamp")
        start_times.append(pd.to_datetime(ts))

    # Ensure stable ordering for bisect.
    # (We rely on the pipeline load_xes_log sort, but tests may provide
    # unsorted logs, so be defensive.)
    starts = sorted(start_times)
    # If we sorted timestamps but not traces, we would break slice indices.
    # Therefore, we only sort timestamps when the caller already provided a
    # correctly sorted log. We check monotonicity to enforce that.
    if start_times != starts:
        raise ValueError(
            "split_log_into_fixed_time_windows expects log traces sorted by trace start timestamp"
        )

    first_start = start_times[0]
    last_start = start_times[-1]

    # Align first window start if requested.
    if align_first_window:
        if unit == "day":
            window_start = first_start.normalize()
        elif unit == "month":
            window_start = pd.Timestamp(first_start.year, first_start.month, 1)
            if first_start.tz is not None:
                window_start = window_start.tz_localize(first_start.tz)
        else:  # year
            window_start = pd.Timestamp(first_start.year, 1, 1)
            if first_start.tz is not None:
                window_start = window_start.tz_localize(first_start.tz)
    else:
        window_start = first_start

    if unit == "day":
        window_delta = pd.Timedelta(days=window_size)
        step_delta = pd.Timedelta(days=offset)
    elif unit == "month":
        window_delta = pd.DateOffset(months=window_size)
        step_delta = pd.DateOffset(months=offset)
    else:  # year
        window_delta = pd.DateOffset(years=window_size)
        step_delta = pd.DateOffset(years=offset)

    windows: List[Window] = []
    wid = 0
    cur_start = window_start

    # Step until the window start moves beyond the last trace start.
    while cur_start <= last_start:
        cur_end = cur_start + window_delta
        if not include_incomplete_windows and cur_end > last_start:
            break

        # Find the contiguous trace slice whose start times are in
        # [cur_start, cur_end).
        i0 = bisect.bisect_left(start_times, cur_start)
        i1 = bisect.bisect_left(start_times, cur_end)

        if i0 < i1:
            center_moment = cur_start + (cur_end - cur_start) / 2
            windows.append(
                Window(
                    id=str(wid),
                    first_index=i0,
                    last_index=i1 - 1,
                    size=i1 - i0,
                    start_moment=cur_start.to_pydatetime()
                    if hasattr(cur_start, "to_pydatetime")
                    else cur_start,
                    end_moment=cur_end.to_pydatetime()
                    if hasattr(cur_end, "to_pydatetime")
                    else cur_end,
                    center_moment=center_moment.to_pydatetime()
                    if hasattr(center_moment, "to_pydatetime")
                    else center_moment,
                    traces=log[i0:i1],
                )
            )
            wid += 1

        cur_start = cur_start + step_delta

    return windows


def split_log_into_fixed_comparable_windows(
    log: List[Trace], window_1_size: int, window_2_size: int, offset: int, step: int
) -> List[Tuple[Window, Window]]:
    """Split log into comparable window pairs.

    Args:
        log: List of PM4Py Trace objects.
        window_1_size: Size of first window in each pair.
        window_2_size: Size of second window in each pair.
        offset: Offset between windows in a pair.
        step: Step size for window positioning.

    Returns:
        List of (Window, Window) tuples.
    """
    n = len(log)
    pairs: List[Tuple[Window, Window]] = []
    k = 0
    wid = 0
    if min(window_1_size, window_2_size, step) <= 0 or offset < 0:
        return pairs
    while True:
        w1i0 = k * step
        w1i1 = w1i0 + window_1_size - 1
        w2i0 = w1i0 + offset
        w2i1 = w2i0 + window_2_size - 1
        if w1i1 >= n or w2i1 >= n:
            break
        w1 = _make_window(log, w1i0, w1i1, str(wid))
        w2 = _make_window(log, w2i0, w2i1, str(wid + 1))
        pairs.append((w1, w2))
        wid += 2
        k += 1
    return pairs
