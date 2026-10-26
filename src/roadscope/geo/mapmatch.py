"""Road-name lookup from OSM, optional dependency.

osmnx is a heavy package and the 'geo' extra is optional. If it is not
installed, or the graph cannot be downloaded, this module returns None for
every position and the dashboard simply shows no road name. That is the
correct degradation: a survey without road names is still a survey, while a
crash on import would make the whole agent unusable on a machine that lacks
the extra.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

try:
    import osmnx as ox  # type: ignore

    _OSMNX_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    ox = None
    _OSMNX_AVAILABLE = False


def osmnx_available() -> bool:
    return _OSMNX_AVAILABLE


def nearest_road_name(lat: float, lon: float, *, dist_m: float = 50.0) -> str | None:
    """Nearest named road within ``dist_m`` metres, or None.

    Cached per (lat, lon) rounded to ~10 m so a stationary vehicle does not
    hammer the graph. Returns None on any failure, including no nearby named
    road -- the caller is expected to treat None as "unknown road".
    """
    if not _OSMNX_AVAILABLE:
        return None
    try:
        graph = _graph_around(lat, lon, dist_m)
    except Exception as exc:
        log.debug("osmnx graph fetch failed: %s", exc)
        return None
    if graph is None:
        return None
    try:
        node = ox.nearest_nodes(graph, lon, lat)
        edge = _nearest_edge(graph, node, lat, lon)
        if edge is None:
            return None
        name = edge.get("name")
        if isinstance(name, list):
            name = name[0] if name else None
        return name if isinstance(name, str) and name.strip() else None
    except Exception as exc:
        log.debug("osmnx nearest-edge lookup failed: %s", exc)
        return None


@lru_cache(maxsize=256)
def _graph_around(lat: float, lon: float, dist_m: float) -> Any:
    """Fetch and cache the OSM graph for a small box around (lat, lon).

    Coordinates are rounded to ~10 m before caching so repeated fixes from a
    stationary vehicle share one graph.
    """
    if not _OSMNX_AVAILABLE:
        return None
    lat_r = round(lat, 4)
    lon_r = round(lon, 4)
    try:
        graph = ox.graph_from_point(
            (lat_r, lon_r), dist=dist_m, network_type="drive", simplify=True
        )
        return graph
    except Exception as exc:
        log.debug("osmnx graph_from_point failed: %s", exc)
        return None


def _nearest_edge(graph: Any, node: int, lat: float, lon: float) -> dict[str, Any] | None:
    """Return the attribute dict of the edge closest to (lat, lon)."""
    best: tuple[float, dict[str, Any]] | None = None
    for u, v, _key, data in graph.edges(keys=True, data=True):
        if "geometry" in data:
            try:
                d = _point_to_linestring_distance(lat, lon, data["geometry"])
            except Exception:
                continue
        else:
            # Fall back to endpoint distance when no geometry is present.
            try:
                du = (graph.nodes[u]["y"] - lat) ** 2 + (graph.nodes[u]["x"] - lon) ** 2
                dv = (graph.nodes[v]["y"] - lat) ** 2 + (graph.nodes[v]["x"] - lon) ** 2
                d = min(du, dv)
            except (KeyError, TypeError):
                continue
        if best is None or d < best[0]:
            best = (d, data)
    return best[1] if best else None


def _point_to_linestring_distance(lat: float, lon: float, geometry: Any) -> float:
    """Squared distance from a point to a linestring, in degree-space."""
    coords = list(geometry.coords)
    best = float("inf")
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        # Project (lon, lat) onto the segment; clamp to endpoints.
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            d = (x1 - lon) ** 2 + (y1 - lat) ** 2
        else:
            t = ((lon - x1) * dx + (lat - y1) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
            d = (px - lon) ** 2 + (py - lat) ** 2
        if d < best:
            best = d
    return best
