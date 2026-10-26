"""Capture-source tests.

The camera probe exists because a demo that starts, captures nothing, and
reports all zeros is indistinguishable from a broken pipeline. These tests pin
the three ordinary failures: nothing listening, wrong network, and a URL that
returns a web page instead of a stream.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from roadscope.agent.ingest import (
    MjpegSource,
    SyntheticSource,
    VideoFileSource,
    build_source,
    local_ipv4_addresses,
    probe_camera,
)


class _Handler(BaseHTTPRequestHandler):
    content_type = "multipart/x-mixed-replace; boundary=frame"
    status = 200

    def do_GET(self) -> None:
        body = b"--frame\r\nContent-Type: image/jpeg\r\n\r\nxxxx\r\n"
        self.send_response(self.status)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep test output clean
        return


class _HtmlHandler(_Handler):
    content_type = "text/html"
    status = 404


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def html_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _HtmlHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _url(httpd: ThreadingHTTPServer, path: str = "/video") -> str:
    host, port = httpd.server_address[:2]
    return f"http://{host}:{port}{path}"


def test_probe_accepts_a_live_stream(server):
    ok, message = probe_camera(_url(server))
    assert ok, message
    assert "reachable" in message


def test_probe_rejects_html_error_page(html_server):
    ok, message = probe_camera(_url(html_server))
    assert not ok
    assert "404" in message


def test_probe_reports_refused_port():
    # Bind then close, so the port is almost certainly free.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    ok, message = probe_camera(f"http://127.0.0.1:{port}/video", timeout_s=2.0)
    assert not ok
    assert "refused" in message.lower()


def test_probe_rejects_url_without_host():
    ok, message = probe_camera("http:///video")
    assert not ok
    assert "host" in message.lower()


def test_probe_accepts_url_without_scheme(server):
    host, port = server.server_address[:2]
    ok, message = probe_camera(f"{host}:{port}/video")
    assert ok, message


def test_local_addresses_exclude_loopback():
    for ip in local_ipv4_addresses():
        assert not ip.startswith("127.")
        assert ip.count(".") == 3


def test_build_source_prefers_camera_url():
    source = build_source("http://192.0.2.10:4747/video")
    assert isinstance(source, MjpegSource)


def test_build_source_falls_back_to_synthetic():
    assert isinstance(build_source(None), SyntheticSource)


def test_build_source_uses_video_file_when_given(tmp_path):
    # An unreadable path must raise, proving the file branch is taken.
    with pytest.raises(FileNotFoundError):
        build_source(None, fallback_video=str(tmp_path / "missing.mp4"))
    assert VideoFileSource is not None
