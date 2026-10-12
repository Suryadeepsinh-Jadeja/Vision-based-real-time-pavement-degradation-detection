"""Turn stored detections into ASTM D6433 pavement condition.

This is the seam between detection and civil engineering. Detections arrive as
rows in ``detections``; the PCI engine wants :class:`~roadscope.civil.pci.Defect`
objects plus the GPS track; the dashboard wants the result persisted so it does
not recompute on every rerun.

Nothing here invents a number. Where the detector cannot supply a measurement
the engine falls back and records a limitation, and this module surfaces those
limitations rather than dropping them -- a PCI presented without its caveats is
misleading even when the arithmetic is correct.
"""

from __future__ import annotations

import logging
from typing import Any

from ..civil.pci import PCIResult, compute_pci
from ..storage.repository import Repository

log = logging.getLogger(__name__)

#: Nominal sample-unit area. ASTM D6433 permits 250-500 m2; the low end is used
#: so short urban trips still produce more than one unit.
DEFAULT_SAMPLE_AREA_M2 = 250.0


def score_trip(
    repo: Repository,
    trip_id: str,
    *,
    sample_area_m2: float = DEFAULT_SAMPLE_AREA_M2,
    persist: bool = True,
) -> PCIResult:
    """Compute and optionally persist the PCI for one trip.

    Returns the same result whether or not ``persist`` is set, so a caller can
    preview a recomputation without writing.
    """
    rows = repo.detections(trip_id=trip_id, limit=1_000_000)
    defects: list[Any] = [dict(row) for row in rows]
    track = [
        {"latitude": r["lat"], "longitude": r["lon"], "ts_utc": r["ts_utc"]}
        for r in repo.track(trip_id)
    ]

    result = compute_pci(defects, track, sample_area_m2=sample_area_m2)

    if persist and result["sample_units"]:
        repo.replace_sample_units(trip_id, result["sample_units"])
        log.info(
            "trip %s: %d sample units, corridor PCI %.1f (%s)",
            trip_id,
            len(result["sample_units"]),
            result["pci"],
            result["rating"],
        )

    return result


def work_orders(
    repo: Repository,
    trip_id: str,
    *,
    sample_area_m2: float = DEFAULT_SAMPLE_AREA_M2,
) -> list[dict[str, Any]]:
    """Derive maintenance actions, worst first.

    One order per sample unit that falls below the "Good" threshold, named for
    the road it sits on when a road name is known. Chainage is carried through
    so an order can be located by a crew rather than by scrolling a list.
    """
    units = repo.sample_units(trip_id)
    if not units:
        score_trip(repo, trip_id, sample_area_m2=sample_area_m2)
        units = repo.sample_units(trip_id)

    road_name = _dominant_road_name(repo, trip_id)
    geometry = repo.unit_geometry(trip_id)
    # Worst first: the units needing attention should be at the top.
    ordered = sorted(units, key=lambda u: (u["pci"] is None, u["pci"] or 0.0))

    orders: list[dict[str, Any]] = []
    for unit in ordered:
        pci = unit["pci"]
        if pci is None or pci >= 85.0:
            continue
        points = geometry.get(unit["unit_id"], [])
        orders.append(
            {
                "unit_id": unit["unit_id"],
                "road_name": road_name,
                "chainage_start_m": unit["chainage_start_m"],
                "chainage_end_m": unit["chainage_end_m"],
                "length_m": (
                    round(unit["chainage_end_m"] - unit["chainage_start_m"], 1)
                    if unit["chainage_start_m"] is not None and unit["chainage_end_m"] is not None
                    else None
                ),
                "pci": round(pci, 1),
                "rating": unit["rating"],
                "action": _action_for(pci),
                "centroid": _centroid(points),
                "point_count": len(points),
            }
        )
    return orders


def _action_for(pci: float) -> str:
    """The maintenance action for a PCI score.

    Delegates to the same rating table the PCI engine uses, so a work order
    cannot drift from the recommendation printed next to it.
    """
    from ..civil.pci import rate

    return rate(pci)[2]


def _dominant_road_name(repo: Repository, trip_id: str) -> str | None:
    """The most frequently reported road name for a trip.

    A vehicle drives one road, so a trip normally carries a single name. Taking
    the mode rather than the first row avoids a stray value from a single fix
    naming the whole order.
    """
    row = repo._conn.execute(
        "SELECT road_name, COUNT(*) AS n FROM gps_fixes"
        " WHERE trip_id = ? AND road_name IS NOT NULL AND road_name != ''"
        " GROUP BY road_name ORDER BY n DESC LIMIT 1",
        (trip_id,),
    ).fetchone()
    return row["road_name"] if row else None


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )
