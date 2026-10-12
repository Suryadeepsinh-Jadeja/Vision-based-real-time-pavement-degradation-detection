"""Condition map helpers.

Builds the folium layers the dashboard draws. Kept apart from the Streamlit
code so the map geometry can be tested without starting a UI process -- the
agent is forbidden from importing Streamlit, and the same reasoning applies to
keeping the geometry logic importable on its own.

Colours come from :data:`roadscope.civil.pci.RATING_BANDS`, so a segment's
colour always matches the rating text printed beside it. A second colour table
would eventually disagree with the engine, and a map that lies about severity is
worse than no map.
"""

from __future__ import annotations

from typing import Any

from ..civil.pci import RATING_BANDS, rate

#: Distress code -> (marker colour, human name).
CLASS_STYLE: dict[str, tuple[str, str]] = {
    "D00": ("#2563EB", "Longitudinal crack"),
    "D10": ("#7C3AED", "Transverse crack"),
    "D20": ("#0891B2", "Alligator cracking"),
    "D40": ("#DC2626", "Pothole"),
    "RUTTING": ("#CA8A04", "Rutting"),
}

DEFAULT_COLOR = "#6B7280"


def rating_color(pci: float) -> str:
    """Colour for a PCI score, from the engine's own rating table."""
    _rating, color, _action = rate(pci)
    return color


def condition_polyline(
    geometry: dict[str, list[tuple[float, float]]],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One polyline per contiguous run of same-rated sample units.

    Units are visited in chainage order and a new segment opens whenever the
    rating changes. Drawing one line per unit instead would give a stripey map
    that hides where condition actually changes.
    """
    ordered = [u for u in units if u.get("chainage_start_m") is not None]
    ordered.sort(key=lambda u: u["chainage_start_m"])

    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for unit in ordered:
        points = geometry.get(unit["unit_id"], [])
        if len(points) < 2:
            # A single GPS point cannot draw a line; skip rather than emit a
            # zero-length polyline that renders as an artefact.
            continue
        rating = unit.get("rating")
        if current is None or current["rating"] != rating:
            current = {
                "rating": rating,
                "color": rating_color(unit["pci"])
                if unit.get("pci") is not None
                else DEFAULT_COLOR,
                "points": [],
                "unit_ids": [],
            }
            segments.append(current)
        current["points"].extend(points)
        current["unit_ids"].append(unit["unit_id"])

    # Collapse consecutive duplicate points; they render as spikes.
    for seg in segments:
        deduped: list[tuple[float, float]] = []
        for p in seg["points"]:
            if not deduped or deduped[-1] != p:
                deduped.append(p)
        seg["points"] = deduped
    return [s for s in segments if len(s["points"]) >= 2]


def detection_markers(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Marker metadata for each detection, grouped by class for the legend."""
    markers: list[dict[str, Any]] = []
    for row in detections:
        code = row.get("class_code", "")
        color, label = CLASS_STYLE.get(code, (DEFAULT_COLOR, code or "Unknown"))
        lat, lon = row.get("lat"), row.get("lon")
        if lat is None or lon is None:
            continue
        markers.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "class_code": code,
                "label": label,
                "color": color,
                "confidence": row.get("confidence"),
                "severity": row.get("severity"),
                "area_m2": row.get("area_m2"),
                "width_mm": row.get("width_mm"),
                "length_m": row.get("length_m"),
                "detection_id": row.get("detection_id"),
                "ts_utc": row.get("ts_utc"),
            }
        )
    return markers


def marker_popup(marker: dict[str, Any]) -> str:
    """Popup text for a detection marker.

    States what was *measured* separately from what was *assumed*. A
    defaulted severity has to be visible here, not only in the aggregate
    limitations list, or a reader takes it as a measurement.
    """
    parts = [f"<b>{marker['label']}</b>", f"confidence {float(marker.get('confidence') or 0):.0%}"]
    if marker.get("severity"):
        parts.append(f"severity {marker['severity']}")
    measured = [
        ("area", marker.get("area_m2"), "m²"),
        ("length", marker.get("length_m"), "m"),
        ("width", marker.get("width_mm"), "mm"),
    ]
    for name, value, unit in measured:
        if value is not None:
            parts.append(f"{name} {float(value):.2f} {unit}")
    if not any(v is not None for _n, v, _u in measured):
        parts.append("<i>no dimensional measurement available</i>")
    return "<br>".join(parts)


def bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    """``(south, west, north, east)`` for a set of points, or None if empty."""
    if not points:
        return None
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return (min(lats), min(lons), max(lats), max(lons))


def all_points(
    segments: list[dict[str, Any]], markers: list[dict[str, Any]]
) -> list[tuple[float, float]]:
    """Every coordinate worth fitting the map to."""
    pts: list[tuple[float, float]] = []
    for seg in segments:
        pts.extend((p[0], p[1]) for p in seg["points"])
    for m in markers:
        pts.append((m["lat"], m["lon"]))
    return pts


def rating_legend() -> list[dict[str, str]]:
    """The rating table, for display beside the map."""
    return [
        {"rating": rating, "color": color, "pci_from": str(threshold)}
        for threshold, rating, color, _action in RATING_BANDS
    ]
