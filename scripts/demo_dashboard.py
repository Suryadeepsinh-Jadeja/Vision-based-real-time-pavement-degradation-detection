#!/usr/bin/env python3
"""RoadScope live demo dashboard -- one screen, no menus.

    ROADSCOPE_SERVER_DB=var/demo.sqlite streamlit run scripts/demo_dashboard.py

Shows what the audience needs to see while the phone is pointed at the road:
is the device live, how many defects were found, where they are, and the
actual frame the model flagged. Refreshes itself every few seconds.

Read-only: it opens the server database and the ``frames/`` folder beside it
(written by ``python -m roadscope.server``) and never writes to either. The
full engineering dashboard (PCI, work orders, health) is still
``scripts/dashboard.py``.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REFRESH_S = 3
LIVE_WINDOW_S = 15  # the agent sends a heartbeat every 5 s
LIST_LIMIT = 100

CLASSES = {
    "D40": ("Pothole", "#e5484d"),
    "D20": ("Alligator crack", "#f5a524"),
    "D10": ("Transverse crack", "#3e8ed0"),
    "D00": ("Longitudinal crack", "#8e6fd8"),
}


def db_path() -> Path:
    raw = os.environ.get("ROADSCOPE_SERVER_DB", "var/demo.sqlite")
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def query(sql: str, params: tuple = ()) -> list[dict]:
    path = db_path()
    if not path.exists():
        return []
    # Read-only URI: the dashboard must never lock or modify the store.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def label(code: str) -> str:
    return CLASSES.get(code, (code, ""))[0]


def ago(ts: float | None) -> str:
    if not ts:
        return "—"
    s = max(0, int(time.time() - ts))
    if s < 60:
        return f"{s} s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    return datetime.fromtimestamp(ts).strftime("%d %b %H:%M")


def image_for(row: dict) -> Path | None:
    name = row.get("thumb_path")
    if not name:
        return None
    path = db_path().parent / "frames" / name
    return path if path.exists() else None


def maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


# -- page -----------------------------------------------------------------
st.set_page_config(page_title="RoadScope Live", page_icon="🛣️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-size: 2.2rem;}
      .det-tag {display:inline-block;padding:2px 10px;border-radius:999px;
                color:white;font-weight:600;font-size:0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🛣️ RoadScope — Live Road Defect Detection")
st.caption(
    "Phone camera → YOLOv8 on the laptop → GPS-tagged → MQTT → server database → this screen"
)

trips = query("SELECT trip_id, started_at FROM trips ORDER BY started_at DESC")
scope = st.radio(
    "Show",
    ["Latest session", "All sessions"],
    horizontal=True,
    label_visibility="collapsed",
)
trip_filter = trips[0]["trip_id"] if (trips and scope == "Latest session") else None


@st.fragment(run_every=REFRESH_S)
def live_view() -> None:
    where, params = ("WHERE d.trip_id = ?", (trip_filter,)) if trip_filter else ("", ())
    rows = query(
        "SELECT d.detection_id, d.ts_utc, d.lat, d.lon, d.class_code, d.confidence,"
        " d.gps_accuracy_m, d.frames_seen, f.thumb_path"
        " FROM detections d LEFT JOIN frames f ON f.frame_id = d.frame_id"
        f" {where} ORDER BY d.ts_utc DESC",
        params,
    )
    heartbeat = query("SELECT MAX(ts_utc) AS ts FROM device_status")
    last_beat = heartbeat[0]["ts"] if heartbeat else None
    live = bool(last_beat and time.time() - last_beat < LIVE_WINDOW_S)

    # Pop an alert when a new detection lands.
    newest = rows[0]["detection_id"] if rows else None
    seen = st.session_state.get("newest")
    if newest and seen and newest != seen:
        st.toast(f"🚨 {label(rows[0]['class_code'])} detected!", icon="📍")
    st.session_state["newest"] = newest

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Phone / agent", "🟢 Live" if live else "⚪ Offline")
    c2.metric("Potholes", sum(1 for r in rows if r["class_code"] == "D40"))
    c3.metric("All road defects", len(rows))
    c4.metric("Last detection", ago(rows[0]["ts_utc"]) if rows else "—")

    if not rows:
        st.info(
            "Waiting for the first detection… point the phone at a pothole and hold it "
            "steady for 2-3 seconds."
        )
        return

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Where")
        df = pd.DataFrame(
            {
                "lat": [r["lat"] for r in rows],
                "lon": [r["lon"] for r in rows],
                "color": [CLASSES.get(r["class_code"], ("", "#888888"))[1] for r in rows],
            }
        )
        # Fixed close zoom for one session; let Streamlit fit the bounds across many.
        st.map(
            df,
            latitude="lat",
            longitude="lon",
            color="color",
            size=12,
            zoom=17 if trip_filter else None,
        )
        st.caption(
            "  ".join(
                f"<span class='det-tag' style='background:{c}'>{n}</span>"
                for n, c in CLASSES.values()
            ),
            unsafe_allow_html=True,
        )

    with right:
        latest = rows[0]
        name, color = CLASSES.get(latest["class_code"], (latest["class_code"], "#888"))
        st.subheader("Latest detection")
        img = image_for(latest)
        if img:
            st.image(str(img), width="stretch")
        else:
            st.caption("(no frame stored for this detection)")
        st.markdown(
            f"<span class='det-tag' style='background:{color}'>{name}</span> "
            f"&nbsp;**{latest['confidence']:.0%}** confidence · {ago(latest['ts_utc'])}",
            unsafe_allow_html=True,
        )
        acc = latest.get("gps_accuracy_m")
        st.markdown(
            f"📍 `{latest['lat']:.6f}, {latest['lon']:.6f}`"
            + (f" (±{acc:.0f} m)" if acc else "")
            + f" · [open in Google Maps]({maps_link(latest['lat'], latest['lon'])})"
        )

    st.subheader(f"All detections ({len(rows)}) - newest first")
    # Fixed-height box that scrolls, so a long session does not push the page
    # down. Rows are already ordered newest first by the query.
    with st.container(height=520):
        for n, r in enumerate(rows[:LIST_LIMIT]):
            name, color = CLASSES.get(r["class_code"], (r["class_code"], "#888"))
            pic, info = st.columns([1, 4], vertical_alignment="center")
            with pic:
                img = image_for(r)
                if img:
                    st.image(str(img), width="stretch")
                else:
                    st.caption("no frame")
            with info:
                when = datetime.fromtimestamp(r["ts_utc"]).strftime("%d %b %H:%M:%S")
                acc = r.get("gps_accuracy_m")
                st.markdown(
                    f"{'🆕 ' if n == 0 else ''}"
                    f"<span class='det-tag' style='background:{color}'>{name}</span> "
                    f"&nbsp;**{r['confidence']:.0%}** confidence"
                    f" &nbsp;·&nbsp; {when} ({ago(r['ts_utc'])})  \n"
                    f"📍 [{r['lat']:.6f}, {r['lon']:.6f}]({maps_link(r['lat'], r['lon'])})"
                    + (f" ±{acc:.0f} m" if acc else ""),
                    unsafe_allow_html=True,
                )
            st.divider()
        if len(rows) > LIST_LIMIT:
            st.caption(f"Showing the latest {LIST_LIMIT} of {len(rows)}.")


live_view()
