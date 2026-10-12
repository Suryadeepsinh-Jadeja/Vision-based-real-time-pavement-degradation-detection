"""RoadScope survey dashboard.

Read-only. It never writes to the store and never runs detection; it renders
what the agent and receiver already recorded. That separation is deliberate --
capture must continue whether or not this process is open, and a UI that
competed for the GPU or the database write lock would break that.

Run with::

    streamlit run -m roadscope.ui.dashboard
"""

from __future__ import annotations

from datetime import UTC, datetime

import folium
import streamlit as st
from streamlit_folium import st_folium

from ..civil.pci import RATING_BANDS, rate
from ..config import get_settings
from ..server.survey import score_trip, work_orders
from ..storage.repository import Repository, detection_to_dict
from . import maps

CLASS_TITLES = {
    "D00": "Longitudinal cracking",
    "D10": "Transverse cracking",
    "D20": "Alligator cracking",
    "D40": "Pothole",
}


@st.cache_resource(show_spinner=False)
def get_repo(db_path: str) -> Repository:
    return Repository(db_path, create=False)


@st.cache_data(ttl=5, show_spinner=False)
def load_trip(db_path: str, trip_id: str) -> dict:
    """Everything one trip needs, cached briefly so reruns are cheap."""
    repo = Repository(db_path, create=False)
    result = score_trip(repo, trip_id, persist=True)
    return {
        "pci": result,
        "units": result["sample_units"],
        "geometry": repo.unit_geometry(trip_id),
        "detections": [detection_to_dict(r) for r in repo.detections(trip_id=trip_id)],
        "orders": work_orders(repo, trip_id),
        "track": [dict(r) for r in repo.track(trip_id)],
    }


def main() -> None:
    st.set_page_config(page_title="RoadScope", page_icon="🛣️", layout="wide")
    settings = get_settings()

    st.title("RoadScope")
    st.caption(
        "Pavement condition survey. PCI per ASTM D6433. Read-only view of what the agent recorded."
    )

    try:
        repo = get_repo(str(settings.server_db))
        trips = repo.list_trips(limit=100)
    except Exception as exc:
        st.error(f"Cannot open the survey store at {settings.server_db}: {exc}")
        st.code("python -m roadscope.agent --check   # confirm the database path")
        return

    if not trips:
        st.info(
            "No surveys recorded yet. Run the agent to collect one:\n\n"
            "```\npython -m roadscope.agent\n```"
        )
        return

    trip_ids = [r["trip_id"] for r in trips]
    labels = {
        r["trip_id"]: (
            f"{r['trip_id']}  ·  "
            f"{datetime.fromtimestamp(r['started_at'], UTC):%Y-%m-%d %H:%M}"
            + (f"  ·  {r['distance_m'] / 1000:.2f} km" if r.get("distance_m") else "")
        )
        for r in trips
    }
    chosen = st.sidebar.selectbox("Survey", trip_ids, format_func=labels.get)

    devices = repo.list_devices()
    if devices:
        st.sidebar.markdown("**Devices**")
        for d in devices:
            seen = d["last_seen"]
            when = datetime.fromtimestamp(seen, UTC).strftime("%H:%M:%S") if seen else "never"
            st.sidebar.write(f"`{d['device_id']}` — last seen {when}")

    try:
        data = load_trip(str(settings.server_db), chosen)
    except Exception as exc:
        st.error(f"Could not score trip {chosen}: {exc}")
        return

    render_header(data, chosen)
    tabs = st.tabs(["Map", "Condition", "Defects", "Work orders", "Health"])

    with tabs[0]:
        render_map(data)
    with tabs[1]:
        render_condition(data)
    with tabs[2]:
        render_defects(data)
    with tabs[3]:
        render_work_orders(data)
    with tabs[4]:
        render_health(repo, chosen)


def render_header(data: dict, trip_id: str) -> None:
    result = data["pci"]
    rating, color, action = rate(result["pci"])
    n = len(data["detections"])

    left, mid, right, right2 = st.columns(4)
    left.metric("Pavement Condition Index", f"{result['pci']:.1f}")
    mid.metric("Rating", rating)
    right.metric("Defects recorded", f"{n}")
    right2.metric("Sample units", f"{len(data['units'])}")

    st.markdown(
        f"<span style='color:{color};font-weight:600'>{rating}</span> — {action}",
        unsafe_allow_html=True,
    )

    if result["limitations"]:
        with st.expander(f"Measurement limitations ({len(result['limitations'])})", expanded=True):
            st.warning(
                "This PCI rests on defaulted values where the detector cannot measure "
                "the defect. It is a defect-density index, not a full distress survey."
            )
            for lim in result["limitations"]:
                st.markdown(f"- {lim}")

    if not data["detections"]:
        st.info(
            "No defects on this survey. PCI is reported as 100 because there is "
            "nothing to deduct -- an unrun survey and a perfect road look identical "
            "here. Check device health below."
        )


def render_map(data: dict) -> None:
    segments = maps.condition_polyline(data["geometry"], data["units"])
    markers = maps.detection_markers(data["detections"])
    pts = maps.all_points(segments, markers)

    if not pts:
        st.info("No mapped positions for this survey.")
        return

    south, west, north, east = maps.bounds(pts)
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=15)
    if abs(north - south) < 1e-6 and abs(east - west) < 1e-6:
        m.fit_bounds([[south - 0.002, west - 0.002], [north + 0.002, east + 0.002]])

    if not segments:
        st.warning("No scored sample units, so no condition line. Defects are still shown.")
    else:
        group = folium.FeatureGroup(name="Pavement condition")
        for seg in segments:
            group.add_child(
                folium.PolyLine(
                    seg["points"],
                    color=seg["color"],
                    weight=7,
                    opacity=0.85,
                    tooltip=f"{seg['rating']} ({len(seg['unit_ids'])} units)",
                )
            )
        m.add_child(group)

    by_class: dict[str, list[dict]] = {}
    for mk in markers:
        by_class.setdefault(mk["class_code"], []).append(mk)

    for code, group_markers in sorted(by_class.items()):
        color, label = maps.CLASS_STYLE.get(code, (maps.DEFAULT_COLOR, code))
        layer = folium.FeatureGroup(name=f"{label} ({len(group_markers)})")
        for mk in group_markers:
            layer.add_child(
                folium.CircleMarker(
                    location=[mk["lat"], mk["lon"]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.85,
                    popup=folium.Popup(maps.marker_popup(mk), max_width=280),
                    tooltip=f"{label} {float(mk.get('confidence') or 0):.0%}",
                )
            )
        m.add_child(layer)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=560)

    st.caption(
        f"{len(segments)} condition segment(s) · {len(markers)} defect marker(s). "
        "Line colour is the PCI rating band; marker colour is the distress type."
    )


def render_condition(data: dict) -> None:
    units = data["units"]
    result = data["pci"]
    if not units:
        st.info("No sample units were scored.")
        return

    st.subheader("Rating bands")
    st.dataframe(
        [
            {
                "PCI at or above": threshold,
                "Rating": rating,
                "Colour": color,
                "Recommended action": action,
            }
            for threshold, rating, color, action in RATING_BANDS
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Sample units")
    st.dataframe(
        [
            {
                "Unit": u["unit_id"],
                "Chainage start (m)": u.get("chainage_start_m"),
                "Chainage end (m)": u.get("chainage_end_m"),
                "TDV": u.get("tdv"),
                "CDV": u.get("cdv"),
                "PCI": u.get("pci"),
                "Rating": u.get("rating"),
            }
            for u in units
        ],
        width="stretch",
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total deduct value", f"{result['tdv']:.1f}")
    c2.metric("Corrected deduct value", f"{result['cdv']:.1f}")
    c3.metric("Deducts applied", f"{len(result['deduct_values'])}")

    if result["defect_breakdown"]:
        st.subheader("Distress counts")
        counts = {k: v for k, v in result["defect_breakdown"].items() if v}
        if counts:
            st.bar_chart(counts)
        else:
            st.caption("No distress recorded.")


def render_defects(data: dict) -> None:
    rows = data["detections"]
    if not rows:
        st.info("No defects recorded on this survey.")
        return

    classes = sorted({r.get("class_code", "?") for r in rows})
    picked = st.multiselect(
        "Distress type",
        classes,
        default=classes,
        format_func=lambda c: CLASS_TITLES.get(c, c),
    )
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)

    shown = [
        r
        for r in rows
        if r.get("class_code") in picked and float(r.get("confidence") or 0) >= min_conf
    ]
    st.caption(f"{len(shown)} of {len(rows)} defects shown")

    st.dataframe(
        [
            {
                "Time": datetime.fromtimestamp(r["ts_utc"], UTC).strftime("%H:%M:%S")
                if r.get("ts_utc")
                else None,
                "Type": CLASS_TITLES.get(r.get("class_code", ""), r.get("class_code")),
                "Code": r.get("class_code"),
                "Confidence": r.get("confidence"),
                "Severity": r.get("severity"),
                "Area (m2)": r.get("area_m2"),
                "Length (m)": r.get("length_m"),
                "Width (mm)": r.get("width_mm"),
                "Latitude": r.get("lat"),
                "Longitude": r.get("lon"),
                "GPS accuracy (m)": r.get("gps_accuracy_m"),
                "Frames": r.get("frames_seen"),
                "Model": r.get("model_version"),
            }
            for r in shown
        ],
        width="stretch",
        hide_index=True,
    )

    unmeasured = [
        r for r in shown if not any(r.get(k) for k in ("area_m2", "length_m", "width_mm"))
    ]
    if unmeasured:
        st.caption(
            f"{len(unmeasured)} of these carry no dimensional measurement — the detector "
            "produces bounding boxes only, so severity is defaulted rather than measured."
        )


def render_work_orders(data: dict) -> None:
    orders = data["orders"]
    if not orders:
        st.info(
            "No work orders: every sample unit scored 85 or above (Good). "
            "Routine preventive maintenance only."
        )
        return

    st.caption(
        f"{len(orders)} unit(s) below Good, worst first. Chainage locates each one for a crew."
    )
    for order in orders:
        _rating, color, _ = rate(order["pci"])
        with st.expander(
            f"{order['road_name'] or 'Unnamed road'} · "
            f"chainage {order['chainage_start_m']:.0f}-{order['chainage_end_m']:.0f} m · "
            f"PCI {order['pci']} ({order['rating']})"
        ):
            st.markdown(
                f"<span style='color:{color};font-weight:600'>{order['rating']}</span>",
                unsafe_allow_html=True,
            )
            st.write(f"**Recommended action:** {order['action']}")
            st.write(f"**Sample unit:** {order['unit_id']}  ·  **Length:** {order['length_m']} m")
            if order["centroid"]:
                st.write(f"**Location:** {order['centroid'][0]:.5f}, {order['centroid'][1]:.5f}")

    st.download_button(
        "Download work orders (CSV)",
        _orders_csv(orders),
        file_name=f"work_orders_{orders[0]['unit_id']}.csv",
        mime="text/csv",
    )


def _orders_csv(orders: list[dict]) -> str:
    cols = [
        "unit_id",
        "road_name",
        "chainage_start_m",
        "chainage_end_m",
        "length_m",
        "pci",
        "rating",
        "action",
    ]
    lines = [",".join(cols)]
    for o in orders:
        vals = [str(o.get(c, "")).replace(",", " ") for c in cols]
        lines.append(",".join(vals))
    return "\n".join(lines)


def render_health(repo: Repository, trip_id: str) -> None:
    st.subheader("Device telemetry")
    devices = repo.list_devices()
    if not devices:
        st.info("No devices have reported.")
        return

    device_id = st.selectbox("Device", [d["device_id"] for d in devices])
    history = repo.status_history(device_id, limit=1000)
    if not history:
        st.info(f"{device_id} has not reported status yet.")
        return

    latest = history[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capture rate", f"{latest['fps']:.1f} fps" if latest["fps"] else "—")
    c2.metric(
        "Frames dropped", f"{latest['drop_pct']:.0f}%" if latest["drop_pct"] is not None else "—"
    )
    c3.metric("Inference", f"{latest['infer_ms']:.0f} ms" if latest["infer_ms"] else "—")
    c4.metric(
        "Queue depth", f"{latest['queue_depth']}" if latest["queue_depth"] is not None else "—"
    )

    st.caption(
        "Dropped frames are reported rather than hidden. A high drop rate means the "
        "detector cannot keep up with capture; stale frames are discarded on purpose."
    )

    st.subheader("Over the survey")
    st.line_chart(
        {
            "capture fps": [r["fps"] for r in history],
            "inference ms": [r["infer_ms"] for r in history],
            "drop %": [r["drop_pct"] for r in history],
        },
        height=260,
    )

    latest_detection = repo.latest_detection()
    if latest_detection:
        age = datetime.now(UTC).timestamp() - latest_detection["ts_utc"]
        st.caption(f"Most recent defect {age / 60:.1f} minutes ago.")

    st.markdown("**Model in use**")
    for d in devices:
        st.write(f"`{d['device_id']}` — {d['model_version'] or 'unknown'}")


main()
