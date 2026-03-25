from __future__ import annotations

import textwrap

import pandas as pd
import pytest
from pm4py.objects.log.obj import Event, Trace

from utils.windowing.helpers import split_log_into_fixed_time_windows
from utils.windowing.loader import load_window_config


def _make_trace_start(ts: str, trace_id: str) -> Trace:
    """Create a minimal PM4Py Trace whose first event timestamp is `ts`."""
    trace = Trace()
    trace.attributes["concept:name"] = trace_id
    trace.append(
        Event(
            {
                "concept:name": "A",
                "time:timestamp": ts,
            }
        )
    )
    return trace


def test_fixed_time_windows_month_aligned():
    # Trace start times:
    # - Jan 15
    # - Feb 20
    # - Feb 25
    # - Mar 02
    log = [
        _make_trace_start("2020-01-15T10:00:00+00:00", "t0"),
        _make_trace_start("2020-02-20T10:00:00+00:00", "t1"),
        _make_trace_start("2020-02-25T10:00:00+00:00", "t2"),
        _make_trace_start("2020-03-02T10:00:00+00:00", "t3"),
    ]

    windows = split_log_into_fixed_time_windows(
        log,
        window_size=1,
        offset=1,
        unit="month",
        align_first_window=True,
    )

    assert len(windows) == 3
    assert [w.size for w in windows] == [1, 2, 1]
    assert [(w.first_index, w.last_index) for w in windows] == [
        (0, 0),
        (1, 2),
        (3, 3),
    ]


def test_fixed_time_windows_exact_first_timestamp():
    log = [
        _make_trace_start("2020-01-15T10:00:00+00:00", "t0"),
        _make_trace_start("2020-02-20T10:00:00+00:00", "t1"),
        _make_trace_start("2020-02-25T10:00:00+00:00", "t2"),
        _make_trace_start("2020-03-02T10:00:00+00:00", "t3"),
    ]

    windows = split_log_into_fixed_time_windows(
        log,
        window_size=1,
        offset=1,
        unit="month",
        align_first_window=False,
    )

    # windows:
    # [Jan15, Feb15): includes t0 only
    # [Feb15, Mar15): includes t1, t2, t3
    assert len(windows) == 2
    assert [w.size for w in windows] == [1, 3]
    assert [(w.first_index, w.last_index) for w in windows] == [
        (0, 0),
        (1, 3),
    ]


def test_window_config_loader_fixed_time_windows(tmp_path):
    cfg = textwrap.dedent(
        """
        approaches:
          - name: fixed_t1m_o1m_align_month
            type: fixed_time_windows
            params:
              window_size: 1
              offset: 1
              unit: month
              align_first_window: true
        """
    )
    p = tmp_path / "window_config.yml"
    p.write_text(cfg, encoding="utf-8")

    approaches = load_window_config(p)
    assert len(approaches) == 1
    assert approaches[0]["type"] == "fixed_time_windows"


def test_window_config_loader_rejects_fixed_size_windows(tmp_path):
    cfg = textwrap.dedent(
        """
        approaches:
          - name: fixed_old
            type: fixed_size_windows
            params:
              window_size: 10
              offset: 5
        """
    )
    p = tmp_path / "window_config.yml"
    p.write_text(cfg, encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown approach type"):
        load_window_config(p)

