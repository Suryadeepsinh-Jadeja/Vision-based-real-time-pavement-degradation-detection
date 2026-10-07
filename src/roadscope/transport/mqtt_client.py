"""MQTT publisher and receiver.

The agent publishes through the outbox, never directly, so a broker outage
cannot lose a detection. This module owns the paho client lifecycle only.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt

from ..logging_setup import get_logger
from .topics import subscribe_filter, topic_for

log = get_logger(__name__)

# paho 2.x requires explicit callback API versioning.
try:
    _API = mqtt.CallbackAPIVersion.VERSION2
except AttributeError:  # paho 1.x
    _API = None


def _make_client(client_id: str) -> mqtt.Client:
    if _API is not None:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    return mqtt.Client(client_id=client_id)


class MQTTPublisher:
    """Thread-safe publishing client with a last-will status message."""

    def __init__(
        self,
        url: str,
        *,
        client_id: str = "roadscope-agent",
        keepalive: int = 30,
        on_connect: Callable[[], None] | None = None,
    ) -> None:
        self.url = url
        self._on_connect = on_connect
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self.client = _make_client(client_id)
        self.client.on_connect = self._handle_connect
        self.client.on_disconnect = self._handle_disconnect

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def _handle_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        code = int(getattr(reason_code, "value", reason_code) or 0)
        if code == 0:
            self._connected.set()
            log.info("MQTT connected: %s", self.url)
            if self._on_connect:
                self._on_connect()
        else:
            self._connected.clear()
            log.error("MQTT connect refused: %s (code %s)", self.url, reason_code)

    def _handle_disconnect(self, client, userdata, *args) -> None:
        self._connected.clear()
        log.warning("MQTT disconnected from %s", self.url)

    def set_last_will(self, topic: str, payload: dict[str, Any], qos: int = 1) -> None:
        """Status must be visible as offline when the process dies abruptly."""
        self.client.will_set(topic, json.dumps(payload), qos=qos, retain=True)

    def connect_async(self) -> None:
        self.client.connect_async(self.url, keepalive=15)
        self.client.loop_start()

    def publish(
        self, topic: str, payload: dict[str, Any], qos: int = 1, retain: bool = False
    ) -> bool:
        """Publish JSON. Returns False when not connected -- the caller should
        leave the message in the outbox and retry."""
        with self._lock:
            if not self._connected.is_set():
                return False
            info = self.client.publish(
                topic, json.dumps(payload, separators=(",", ":")), qos=qos, retain=retain
            )
            return info.rc == mqtt.MQTT_ERR_SUCCESS

    def wait_connected(self, timeout: float = 5.0) -> bool:
        return self._connected.wait(timeout)

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as exc:
            log.debug("MQTT close: %s", exc)


class MQTTSubscriber:
    """Subscribes and dispatches JSON payloads to a callback."""

    def __init__(
        self,
        url: str,
        topic_prefix: str,
        *,
        client_id: str = "roadscope-server",
        on_message: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.url = url
        self.prefix = topic_prefix
        self.on_message = on_message
        self._connected = threading.Event()
        self.client = _make_client(client_id)
        self.client.on_connect = self._handle_connect
        self.client.on_message = self._handle_message
        self.received = 0

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def _handle_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        code = int(getattr(reason_code, "value", reason_code) or 0)
        if code != 0:
            log.error("MQTT subscribe refused: %s", reason_code)
            return
        client.subscribe(subscribe_filter(self.prefix), qos=1)
        self._connected.set()
        log.info("subscribed to %s", subscribe_filter(self.prefix))

    def _handle_message(self, client, userdata, msg) -> None:
        if not self.on_message:
            return
        channel = msg.topic.rsplit("/", 1)[-1]
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("dropping non-JSON payload on %s", msg.topic)
            return
        self.received += 1
        try:
            self.on_message(channel, payload)
        except Exception as exc:
            log.exception("handler failed for %s: %s", msg.topic, exc)

    def connect_async(self) -> None:
        self.client.connect_async(self.url, keepalive=15)
        self.client.loop_start()

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as exc:
            log.debug("MQTT close: %s", exc)


def publish_direct(
    url: str,
    device_id: str,
    prefix: str,
    channel: str,
    payload: dict[str, Any],
    timeout: float = 5.0,
) -> bool:
    """One-shot publish, for scripts and tests. Blocks until acked."""
    client = MQTTPublisher(url, client_id=f"roadscope-once-{device_id}")
    try:
        client.connect_async()
        if not client.wait_connected(timeout):
            return False
        return client.publish(topic_for(prefix, device_id, channel), payload)
    finally:
        client.close()
