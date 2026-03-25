"""Tests for drift log I/O helpers."""

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import Event, EventLog, Trace

from utils.drift_io import load_xes_log


def test_load_xes_log_normalizes_custom_activity_key(tmp_path):
    """Custom activity keys are copied to concept:name during log loading."""
    log = EventLog()
    trace = Trace()
    trace.attributes["concept:name"] = "case_1"
    trace.append(
        Event(
            {
                "ACTIVITY_EN": "Create Purchase Order Item",
                "time:timestamp": "2020-01-01T00:00:00.000+00:00",
            }
        )
    )
    trace.append(
        Event(
            {
                "ACTIVITY_EN": "Create Outbound Delivery",
                "time:timestamp": "2020-01-01T01:00:00.000+00:00",
            }
        )
    )
    log.append(trace)

    xes_path = tmp_path / "custom_activity_key.xes"
    xes_exporter.apply(log, str(xes_path))

    loaded = load_xes_log(xes_path, activity_key="ACTIVITY_EN")

    assert loaded[0][0]["concept:name"] == "Create Purchase Order Item"
    assert loaded[0][1]["concept:name"] == "Create Outbound Delivery"
