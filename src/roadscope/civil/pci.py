"""Pavement Condition Index, ASTM D6433.

Replaces the invented ``a + b * log10(density)`` curves of the deleted
prototype. The method implemented here is the published one:

  1. Divide the corridor into sample units of a fixed area.
  2. For each distress type and severity, total the measured quantity.
  3. Density = (quantity / sample unit area) x 100, with the conversion
     conventions the standard specifies:
       - area distresses  : area in m2 / area m2
       - linear distresses : length x nominal 1 ft width / area, i.e. x 0.3048 m
       - potholes (count)  : each hole counts as 1 ft2, i.e. 0.0929 m2
  4. Read a deduct value per distress type and severity.
  5. Correct the total into a CDV.
  6. PCI = 100 - CDV.

Every sample unit is scored independently and the corridor figure is the
arithmetic mean of the units, which is what the standard prescribes. A single
global score over the whole corridor would let one badly damaged stretch be
averaged away by kilometres of good road.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from ..geo.quality import haversine_m
from .cdv import corrected_deduct_value
from .distress import DISTRESS_TYPES, Unit, distress_for
from .dv_tables import deduct_value
from .severity import Geometry, classify_severity

#: ASTM's nominal conversion for linear distress density (1 ft width).
LINEAR_DENSITY_WIDTH_M = 0.3048
#: ASTM X1.17.1.2: each pothole counts as 1 sq ft of area-equivalent density.
POTHOLE_AREA_EQUIVALENT_M2 = 0.09290304


class PCIResult(TypedDict):
    """Uniform result shape.

    Every key is present on every return, including the empty case. The
    prototype's empty-input path omitted ``deduct_values`` and
    ``severity_breakdown``, which crashed the dashboard with a KeyError.
    """

    pci: float
    rating: str
    color: str
    cdv: float
    tdv: float
    defect_breakdown: dict[str, int]
    severity_breakdown: dict[str, int]
    deduct_values: list[float]
    recommendation: str
    sample_units: list[dict[str, Any]]
    limitations: list[str]


@dataclass(slots=True)
class Defect:
    """One measured defect entering the PCI calculation."""

    class_code: str
    severity: str
    lat: float
    lon: float
    length_m: float | None = None
    width_mm: float | None = None
    area_m2: float | None = None
    equiv_diameter_m: float | None = None
    depth_mm: float | None = None
    detection_id: str | None = None

    @classmethod
    def from_record(cls, row: Any) -> Defect:
        """Build a :class:`Defect` from a stored record.

        Accepts either a mapping (a row out of the server database, or the
        payload the agent published) or a
        :class:`roadscope.fusion.fuser.DefectRecord`.

        This is the seam between Phase 3/5 output and the ASTM D6433 engine.
        It matters that it accepts both: the agent produces dataclass records
        in memory while the server reads rows from SQLite, and both must reach
        the same calculation. An earlier version accepted only mappings, so
        passing a record straight through raised
        ``AttributeError: 'DefectRecord' object has no attribute 'get'`` --
        and the unit tests missed it because they all built
        :class:`Defect` literals directly instead of exercising this path.
        """
        get = row.get if isinstance(row, dict) or hasattr(row, "keys") else _attrgetter(row)

        class_code = get("class_code", None) or "D40"
        area_m2 = get("area_m2", None)
        return cls(
            class_code=class_code,
            severity=get("severity", None) or "M",
            lat=float(get("lat", 0.0)),
            lon=float(get("lon", 0.0)),
            length_m=get("length_m", None),
            width_mm=get("width_mm", None),
            area_m2=area_m2,
            equiv_diameter_m=_diameter_from_area(area_m2),
            depth_mm=get("depth_mm", None),
            detection_id=get("detection_id", None),
        )

    def geometry(self) -> Geometry:
        return Geometry(
            length_m=self.length_m,
            width_mm=self.width_mm,
            area_m2=self.area_m2,
            equiv_diameter_m=self.equiv_diameter_m,
            depth_mm=self.depth_mm,
        )


def _diameter_from_area(area_m2: float | None) -> float | None:
    if not area_m2 or area_m2 <= 0:
        return None
    return 2.0 * math.sqrt(area_m2 / math.pi)


def _attrgetter(obj: Any):
    """Return a ``get(key, default)``-shaped accessor for a plain object."""

    def get(key: str, default: Any = None) -> Any:
        return getattr(obj, key, default)

    return get


@dataclass(slots=True)
class SampleUnit:
    """A 250-500 m2 stretch of pavement, scored independently."""

    unit_id: str
    chainage_start_m: float
    chainage_end_m: float
    area_m2: float
    pci: float = 100.0
    rating: str = "Good"
    tdv: float = 0.0
    cdv: float = 0.0
    defect_count: int = 0
    limitations: list[str] = field(default_factory=list)


RATING_BANDS: tuple[tuple[float, str, str, str], ...] = (
    (85, "Good", "#10B981", "Routine preventive maintenance & crack sealing."),
    (70, "Satisfactory", "#84CC16", "Preventive surface seal, minor crack sealing."),
    (55, "Fair", "#FBBF24", "Cold patching of isolated potholes, slurry seal overlay."),
    (40, "Poor", "#F97316", "Full depth patch repairs and thin bituminous concrete overlay."),
    (25, "Very Poor", "#EF4444", "Extensive mill-and-fill resurfacing required immediately."),
    (10, "Serious", "#B91C1C", "Sub-base reconstruction & asphalt base course required."),
    (0, "Failed", "#7F1D1D", "Total road failure: complete pavement reconstruction mandated."),
)


def rate(pci: float) -> tuple[str, str, str]:
    """Return ``(rating, colour, recommended action)`` for a PCI."""
    for threshold, rating, color, action in RATING_BANDS:
        if pci >= threshold:
            return rating, color, action
    return "Failed", "#7F1D1D", RATING_BANDS[-1][3]


def density_pct(quantity: float, unit: Unit, sample_area_m2: float) -> float:
    """Distress density in percent, using ASTM's conversion conventions.

    ASTM D6433 section 9.2 defines density as quantity over sample area, but the
    quantity must first be converted to an area equivalent:

    * linear distresses are taken at a nominal 1 ft (0.3048 m) width
    * each pothole counts as 1 sq ft (0.0929 m2)
    * area distresses use their real area
    """
    if sample_area_m2 <= 0:
        raise ValueError("sample_area_m2 must be positive")
    if unit is Unit.LINEAR_M:
        quantity = quantity * LINEAR_DENSITY_WIDTH_M
    elif unit is Unit.COUNT:
        quantity = quantity * POTHOLE_AREA_EQUIVALENT_M2
    return max(0.0, min(100.0, quantity / sample_area_m2 * 100.0))


def score_sample_unit(defects: list[Defect], area_m2: float) -> SampleUnit:
    """Score one sample unit per ASTM D6433 section 9."""
    unit = SampleUnit(
        unit_id="",
        chainage_start_m=0.0,
        chainage_end_m=0.0,
        area_m2=area_m2,
        defect_count=len(defects),
    )

    # 9.1 - total the quantity per distress type and severity.
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for defect in defects:
        dtype = distress_for(defect.class_code)
        quantity = _quantity_for(dtype.unit, defect)
        totals[(defect.class_code, defect.severity.upper())] += quantity

    # 9.3 - a deduct value per type/severity pair.
    deducts: list[float] = []
    for (code, severity), quantity in totals.items():
        dtype = distress_for(code)
        density = density_pct(quantity, dtype.unit, area_m2)
        if density <= 0:
            continue
        deducts.append(deduct_value(code, severity, density))

    unit.tdv = round(sum(deducts), 1)
    unit.cdv = round(corrected_deduct_value(deducts), 1)
    unit.pci = max(0.0, round(100.0 - unit.cdv, 1))
    unit.rating = rate(unit.pci)[0]
    return unit


def _quantity_for(unit: Unit, defect: Defect) -> float:
    """The measured quantity in the unit ASTM expects for this distress."""
    if unit is Unit.SQ_M:
        return max(0.0, defect.area_m2 or 0.0)
    if unit is Unit.LINEAR_M:
        return max(0.0, defect.length_m or 0.0)
    return 1.0  # a pothole counts as one


def split_into_sample_units(
    defects: list[Defect], track: list[dict[str, Any]], area_m2: float = 250.0
) -> list[SampleUnit]:
    """Chop the corridor into sample units along chainage and score each.

    ``track`` supplies the ordered GPS fixes used to build the chainage. Each
    unit is the stretch of track covering ``area_m2`` of pavement, assuming a
    nominal 3.5 m lane.
    """
    if not track:
        return [score_sample_unit(defects, area_m2)]

    lane_width_m = 3.5
    units: list[SampleUnit] = []
    bucket: list[Defect] = []
    start_chainage = 0.0
    chainage = 0.0
    cumulative = 0.0

    for i, point in enumerate(track):
        if i:
            cumulative += haversine_m(
                track[i - 1]["latitude"],
                track[i - 1]["longitude"],
                point["latitude"],
                point["longitude"],
            )
        position = (point["latitude"], point["longitude"])
        for defect in defects:
            if haversine_m(defect.lat, defect.lon, *position) <= 20.0:
                bucket.append(defect)

        chainage = cumulative
        if (chainage - start_chainage) * lane_width_m >= area_m2:
            scored = score_sample_unit(bucket, area_m2)
            scored.unit_id = f"unit-{len(units) + 1:03d}"
            scored.chainage_start_m = round(start_chainage, 1)
            scored.chainage_end_m = round(chainage, 1)
            units.append(scored)
            start_chainage = chainage
            bucket = []

    if bucket:
        scored = score_sample_unit(bucket, area_m2)
        scored.unit_id = f"unit-{len(units) + 1:03d}"
        scored.chainage_start_m = round(start_chainage, 1)
        scored.chainage_end_m = round(chainage, 1)
        units.append(scored)
    return units


def compute_pci(
    defects: list[Defect], track: list[dict[str, Any]] | None = None, sample_area_m2: float = 250.0
) -> PCIResult:
    """Corridor PCI: mean of per-sample-unit scores.

    A corridor of uniform quality scores the same either way; a corridor with
    one bad stretch only differs because averaging over units stops kilometres
    of good road from hiding it.

    ``defects`` accepts :class:`Defect`, a ``DefectRecord`` from the fusion
    layer, or a database row, and is normalised here. Requiring the caller to
    convert first was a live bug: passing a fusion record -- the actual output
    of Phases 3 and 5 -- raised ``AttributeError: 'NoneType' object has no
    attribute 'upper'`` on its ``severity`` field, because ``DefectRecord``
    leaves severity unset when the detector cannot measure it.
    """
    defects = [d if isinstance(d, Defect) else Defect.from_record(d) for d in defects]

    empty: PCIResult = {
        "pci": 100.0,
        "rating": "Good",
        "color": "#10B981",
        "cdv": 0.0,
        "tdv": 0.0,
        "defect_breakdown": {},
        "severity_breakdown": {},
        "deduct_values": [],
        "recommendation": RATING_BANDS[0][3],
        "sample_units": [],
        "limitations": [],
    }
    if not defects:
        return empty

    breakdown: dict[str, int] = {code: 0 for code in DISTRESS_TYPES}
    sev_breakdown: dict[str, int] = {"L": 0, "M": 0, "H": 0}
    limitations: set[str] = set()
    deducts: list[float] = []

    for defect in defects:
        breakdown[defect.class_code] = breakdown.get(defect.class_code, 0) + 1
        sev = defect.severity.upper()
        sev_breakdown[sev] = sev_breakdown.get(sev, 0) + 1

        # Re-grade where the measurement supports it, and record the gaps.
        result = classify_severity(defect.class_code, defect.geometry())
        if not result.complete:
            limitations.add(result.basis)
        deducts.append(
            deduct_value(
                defect.class_code,
                defect.severity.upper(),
                density_pct(
                    _quantity_for(distress_for(defect.class_code).unit, defect),
                    distress_for(defect.class_code).unit,
                    sample_area_m2,
                ),
            )
        )

    units = split_into_sample_units(defects, track or [], sample_area_m2)
    pci = round(sum(u.pci for u in units) / len(units), 1) if units else 0.0
    rating, color, action = rate(pci)

    return {
        "pci": pci,
        "rating": rating,
        "color": color,
        "cdv": round(sum(u.cdv for u in units) / len(units), 1) if units else 0.0,
        "tdv": round(sum(u.tdv for u in units), 1),
        "defect_breakdown": breakdown,
        "severity_breakdown": sev_breakdown,
        "deduct_values": [round(d, 1) for d in sorted(deducts, reverse=True)],
        "recommendation": action,
        "sample_units": [asdict(u) for u in units],
        "limitations": sorted(limitations),
    }
