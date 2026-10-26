"""Push-GPS receiver tests.

GPSLogger for Android calls a URL on every fix. These tests drive the receiver
the same way the phone does: GET with query parameters, form POST, JSON POST,
and the half-filled placeholders the app sends before it has a fix.
"""

from __future__ import annotations

import json
import socket
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from roadscope.geo.gps_sources import PushGpsSource, parse_push_fix
from roadscope.geo.quality import gate_fix


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def receiver():
    port = _free_port()
    fixes = []
    src = PushGpsSource(port=port, host="127.0.0.1")
    src.start(fixes.append)
    yield f"http://127.0.0.1:{port}", fixes, src
    src.stop()


def test_gpslogger_get_query(receiver):
    base, fixes, src = receiver
    q = urlencode(
        {"lat": "12.84", "lon": "80.15", "acc": "6.5", "spd": "10", "dir": "45", "sat": "9"}
    )
    with urlopen(f"{base}/gps?{q}", timeout=3) as resp:
        assert json.loads(resp.read())["ok"] is True
    assert len(fixes) == 1
    fix = fixes[0]
    assert (fix.latitude, fix.longitude) == (12.84, 80.15)
    assert fix.speed_kmh == pytest.approx(36.0)  # 10 m/s
    assert fix.accuracy_m == 6.5 and fix.n_sats == 9 and fix.heading == 45
    assert src.received == 1


def test_form_and_json_post(receiver):
    base, fixes, _ = receiver
    urlopen(Request(f"{base}/gps", data=b"lat=12.8&lon=80.1", method="POST"), timeout=3).read()
    body = json.dumps({"latitude": 12.9, "longitude": 80.2}).encode()
    req = Request(f"{base}/gps", data=body, headers={"Content-Type": "application/json"})
    urlopen(req, timeout=3).read()
    assert [(f.latitude, f.longitude) for f in fixes] == [(12.8, 80.1), (12.9, 80.2)]


def test_status_page_reports_last_fix(receiver):
    base, _, _ = receiver
    with urlopen(f"{base}/gps", timeout=3) as resp:
        assert json.loads(resp.read()) == {"received": 0, "last_fix": None}
    urlopen(f"{base}/gps?lat=1.5&lon=2.5", timeout=3).read()
    with urlopen(f"{base}/gps", timeout=3) as resp:
        status = json.loads(resp.read())
    assert status["received"] == 1 and status["last_fix"]["lat"] == 1.5


def test_unfilled_placeholders_are_unknown_not_failures():
    fix = parse_push_fix({"lat": "12.8", "lon": "80.1", "acc": "", "sat": "%SAT", "spd": ""})
    assert fix is not None
    assert fix.accuracy_m is None and fix.n_sats is None and fix.speed_kmh == 0.0
    # Unknown fields must not trip the quality gate.
    assert gate_fix(fix)[0] is True


def test_network_fix_zero_sats_not_rejected():
    fix = parse_push_fix({"lat": "12.8", "lon": "80.1", "acc": "5", "sat": "0"})
    assert fix.n_sats is None
    assert gate_fix(fix)[0] is True


def test_receive_time_stamp_survives_phone_clock_skew():
    fix = parse_push_fix({"lat": "12.8", "lon": "80.1", "time": "2001-01-01T00:00:00Z"})
    assert abs(fix.ts_utc - time.time()) < 1.0


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"lat": "12.8"},
        {"lat": "0", "lon": "0"},
        {"lat": "%LAT", "lon": "%LON"},
        {"lat": "99", "lon": "1"},
    ],
)
def test_no_position_is_ignored(params):
    assert parse_push_fix(params) is None
