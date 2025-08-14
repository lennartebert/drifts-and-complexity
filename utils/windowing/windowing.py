from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from datetime import datetime

@dataclass
class Window:
    id: int
    first_index: int
    last_index: int           # inclusive
    size: int
    start_moment: Optional[datetime]
    end_moment: Optional[datetime]
    traces: list
    start_change_point: Optional[int] = None
    end_change_point: Optional[int] = None
    def to_dict(self): d = asdict(self); d.pop("traces", None); return d

def _trace_start_time(trace):
    if not trace: return None
    return trace[0].get("time:timestamp", None)

def _make_window(log, i0: int, i1: int, wid: int, scp=None, ecp=None) -> Window:
    n = len(log)
    if i0 < 0 or i1 >= n or i0 > i1:   # defensive
        return Window(wid, i0, i1, 0, None, None, [], scp, ecp)
    sub = log[i0:i1+1]
    return Window(
        id=wid, first_index=i0, last_index=i1, size=len(sub),
        start_moment=_trace_start_time(sub[0]) if sub else None,
        end_moment=_trace_start_time(sub[-1]) if sub else None,
        traces=sub, start_change_point=scp, end_change_point=ecp
    )

def split_log_into_windows_by_change_points(log, change_points: List[int], attach_change_points=False) -> List[Window]:
    n = len(log); cps = sorted({cp for cp in change_points if 0 < cp < n})
    boundaries = [0] + cps + [n]
    out: List[Window] = []
    for i in range(len(boundaries)-1):
        i0 = boundaries[i]
        i1 = boundaries[i+1]-1
        scp = i0 if attach_change_points and i0 in cps else None
        ecp = i1 if attach_change_points and boundaries[i+1] in cps else None
        out.append(_make_window(log, i0, i1, i, scp, ecp))
    return out

def split_log_into_fixed_windows(log, window_size: int, offset: int) -> List[Window]:
    n = len(log); out=[]; wid=0; i0=0
    if window_size <= 0 or offset <= 0 or window_size>n: return out
    while i0 + window_size <= n:
        i1 = i0 + window_size - 1
        out.append(_make_window(log, i0, i1, wid)); wid += 1; i0 += offset
    return out

def split_log_into_fixed_comparable_windows(log, window_1_size:int, window_2_size:int, offset:int, step:int) -> List[Tuple[Window,Window]]:
    n=len(log); pairs=[]; k=0; wid=0
    if min(window_1_size, window_2_size, step) <= 0 or offset < 0: return pairs
    while True:
        w1i0 = k*step; w1i1 = w1i0 + window_1_size - 1
        w2i0 = w1i0 + offset; w2i1 = w2i0 + window_2_size - 1
        if w1i1 >= n or w2i1 >= n: break
        w1 = _make_window(log, w1i0, w1i1, wid); w2 = _make_window(log, w2i0, w2i1, wid+1)
        pairs.append((w1, w2)); wid += 2; k += 1
    return pairs
