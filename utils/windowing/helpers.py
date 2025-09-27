"""Windowing helper functions for creating windows from event logs."""

from typing import List, Tuple, Optional, Any
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
    ecpt: Optional[str] = None
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
    sub = log[i0:i1+1]
    return Window(
        id=wid, first_index=i0, last_index=i1, size=len(sub),
        start_moment=_trace_start_time(sub[0]) if sub else None,
        end_moment=_trace_start_time(sub[-1]) if sub else None,
        traces=sub, start_change_point=scp, start_change_point_type=scpt, end_change_point=ecp, end_change_point_type=ecpt
    )

def split_log_into_windows_by_change_points(log: List[Trace], change_points: List[Tuple[int, int, str]]) -> List[Window]:
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
    for i in range(len(boundaries)-1):
        i0 = boundaries[i][0]
        i1 = boundaries[i+1][0]-1
        scp = boundaries[i][1]
        scpt = boundaries[i][2]
        ecp = boundaries[i+1][1]
        ecpt = boundaries[i+1][2]
        out.append(_make_window(log, i0, i1, str(i), scp, scpt, ecp, ecpt))
    return out

def split_log_into_fixed_windows(log: List[Trace], window_size: int, offset: int) -> List[Window]:
    """Split log into fixed-size windows.
    
    Args:
        log: List of PM4Py Trace objects.
        window_size: Size of each window.
        offset: Offset between windows.
        
    Returns:
        List of Window objects.
    """
    n = len(log)
    out: List[Window] = []
    wid = 0
    i0 = 0
    if window_size <= 0 or offset <= 0 or window_size > n: 
        return out
    while i0 + window_size <= n:
        i1 = i0 + window_size - 1
        out.append(_make_window(log, i0, i1, str(wid)))
        wid += 1
        i0 += offset
    return out

def split_log_into_fixed_comparable_windows(
    log: List[Trace], 
    window_1_size: int, 
    window_2_size: int, 
    offset: int, 
    step: int
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
        w2 = _make_window(log, w2i0, w2i1, str(wid+1))
        pairs.append((w1, w2))
        wid += 2
        k += 1
    return pairs
