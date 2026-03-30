from __future__ import annotations

import textwrap

from pm4py.objects.log.obj import Event, Trace

from utils.windowing.helpers import split_log_into_growing_prefix_trace_windows
from utils.windowing.loader import load_window_config


def _trace(ts: str, tid: str) -> Trace:
    tr = Trace()
    tr.attributes["concept:name"] = tid
    tr.append(
        Event(
            {
                "concept:name": "A",
                "time:timestamp": ts,
            }
        )
    )
    return tr


def test_growing_prefix_default_start():
    log = [_trace("2020-01-01T00:00:00Z", f"t{i}") for i in range(10)]
    windows = split_log_into_growing_prefix_trace_windows(log, increment=5)
    assert len(windows) == 2
    assert windows[0].first_index == 0 and windows[0].last_index == 4
    assert windows[0].size == 5
    assert windows[1].first_index == 0 and windows[1].last_index == 9
    assert windows[1].size == 10
    assert [w.id for w in windows] == ["0", "1"]


def test_growing_prefix_too_short_for_one_window():
    log = [_trace("2020-01-01T00:00:00Z", f"t{i}") for i in range(4)]
    windows = split_log_into_growing_prefix_trace_windows(log, increment=5)
    assert windows == []


def test_growing_prefix_exactly_one_window():
    log = [_trace("2020-01-01T00:00:00Z", f"t{i}") for i in range(5)]
    windows = split_log_into_growing_prefix_trace_windows(log, increment=5)
    assert len(windows) == 1
    assert windows[0].size == 5


def test_growing_prefix_with_start_index():
    log = [_trace("2020-01-01T00:00:00Z", f"t{i}") for i in range(10)]
    windows = split_log_into_growing_prefix_trace_windows(
        log, increment=3, start_index=2
    )
    # k=1: [2,4] size 3; k=2: [2,7] size 6; k=3: need 2+9=11 > 10 stop
    assert len(windows) == 2
    assert windows[0].first_index == 2 and windows[0].last_index == 4
    assert windows[1].first_index == 2 and windows[1].last_index == 7


def test_growing_prefix_invalid_increment():
    log = [_trace("2020-01-01T00:00:00Z", "t0")]
    assert split_log_into_growing_prefix_trace_windows(log, increment=0) == []
    assert split_log_into_growing_prefix_trace_windows(log, increment=-1) == []


def test_window_config_loader_growing_prefix(tmp_path):
    cfg = textwrap.dedent(
        """
        approaches:
          - name: grow_inc10
            type: growing_prefix_trace_windows
            params:
              increment: 10
          - name: grow_with_start
            type: growing_prefix_trace_windows
            params:
              increment: 5
              start_index: 1
        """
    )
    p = tmp_path / "window_config.yml"
    p.write_text(cfg, encoding="utf-8")

    approaches = load_window_config(p)
    assert len(approaches) == 2
    assert approaches[0]["params"]["increment"] == 10
    assert approaches[1]["params"]["start_index"] == 1
