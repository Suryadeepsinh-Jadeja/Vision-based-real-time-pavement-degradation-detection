"""ASTM D6433 severity classification.

Severity thresholds are transcribed from the standard (Appendix X1), not
invented. The definitions differ by distress type, so there is no single
generic rule.

    Alligator (X1.5.1)
      L - fine longitudinal hairline cracks running parallel, no or only a few
          interconnecting cracks
      M - further development; light spalling of pieces
      H - fully developed pattern; pieces loosened, spalling, pumping

    Longitudinal / transverse cracking (X1.14.2)
      L - nonfilled crack width < 10 mm, or filled crack in good condition
      M - nonfilled 10 mm <= width < 75 mm
      H - nonfilled > 75 mm, or surrounded by medium/high random cracking

    Pothole (X1.17.1, Table X1.1)
      Severity is set by depth: L <= 25 mm, M 25-50 mm, H > 50 mm.
      Potholes over 750 mm diameter are converted to an equivalent number of
      holes (area / 0.5 m2) before severity is applied.

**The width limitation.** Crack severity above is defined on crack *width*, and
width cannot be recovered from a bounding box -- a 3 mm crack fills a box tens
of pixels wide. RDD2022 is detection-only, so ``width_mm`` is normally absent.
:class:`WidthUnavailable` records that fact so the PCI report can state the
limitation rather than quietly substituting box width, which would produce a
number that looks right and measures the wrong thing.

With no width, severity falls back to the class default for that distress,
which is flagged in the returned record so downstream reporting can say so.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distress import CRACK_CODES


class WidthUnavailable(RuntimeError):
    """Raised when crack width is required to grade severity but is absent."""


@dataclass(slots=True)
class Geometry:
    """What the vision layer measured.

    ``width_mm`` is None unless a real mask was available (see
    ``scripts/train_cloud.md`` section 4a: RDD2022 is detection-only, so no
    masks, so no width).
    """

    length_m: float | None = None
    width_mm: float | None = None
    area_m2: float | None = None
    equiv_diameter_m: float | None = None
    depth_mm: float | None = None


@dataclass(slots=True)
class SeverityResult:
    """A severity grade plus whether it is fully supported by measurements."""

    severity: str
    basis: str
    complete: bool


#: Fallback severity when the deciding measurement is unavailable.
DEFAULT_SEVERITY = {"D20": "M", "D00": "M", "D10": "M", "D40": "H"}


def classify_severity(class_code: str, geom: Geometry) -> SeverityResult:
    """Grade a defect per ASTM D6433 Appendix X1.

    Returns a :class:`SeverityResult` whose ``complete`` flag is False when a
    measurement the standard requires was unavailable, so the limitation can be
    reported rather than hidden.
    """
    if class_code == "D20":
        return _alligator(geom)
    if class_code in CRACK_CODES:
        return _crack(geom)
    if class_code == "D40":
        return _pothole(geom)
    return SeverityResult(DEFAULT_SEVERITY.get(class_code, "M"), "unknown class", False)


def _alligator(geom: Geometry) -> SeverityResult:
    """X1.5.1 grades alligator cracking by development, not by a measurement.

    Width is only a cross-check (H has crack width > 13 mm), so the grade can
    be assigned without it. A single-keyhole camera cannot see 'interconnecting
    cracks' or 'pumping', so this is a conservative default rather than a
    measurement.
    """
    if geom.width_mm is not None:
        if geom.width_mm > 13:
            return SeverityResult("H", "alligator: width > 13 mm (X1.5.1 H)", True)
        if geom.width_mm > 6:
            return SeverityResult("M", "alligator: width 6-13 mm", True)
        return SeverityResult("L", "alligator: fine cracks", True)
    return SeverityResult(
        DEFAULT_SEVERITY["D20"],
        "alligator: width unavailable, defaulted to M",
        False,
    )


def _crack(geom: Geometry) -> SeverityResult:
    """X1.14.2 grades cracks on width, which needs a real mask."""
    if geom.width_mm is None:
        return SeverityResult(
            DEFAULT_SEVERITY.get("D00", "M"),
            "crack width unavailable (detection-only model); defaulted to M",
            False,
        )
    if geom.width_mm > 75:
        return SeverityResult("H", "crack width > 75 mm (X1.14.2 H)", True)
    if geom.width_mm >= 10:
        return SeverityResult("M", "crack width 10-75 mm (X1.14.2 M)", True)
    return SeverityResult("L", "crack width < 10 mm (X1.14.2 L)", True)


def _pothole(geom: Geometry) -> SeverityResult:
    """X1.17.1 / Table X1.1 grades potholes by depth.

    Depth is not observable from a dashcam frame. Where it is missing, ASTM
    treats a >750 mm hole over 25 mm deep as high severity, and potholes are a
    safety hazard, so the conservative grade is used and flagged incomplete.
    """
    if geom.depth_mm is not None:
        if geom.depth_mm > 50:
            return SeverityResult("H", "pothole depth > 50 mm (Table X1.1 H)", True)
        if geom.depth_mm > 25:
            return SeverityResult("M", "pothole depth 25-50 mm (Table X1.1 M)", True)
        return SeverityResult("L", "pothole depth <= 25 mm (Table X1.1 L)", True)

    if geom.equiv_diameter_m is not None and geom.equiv_diameter_m <= 0.75:
        return SeverityResult(
            DEFAULT_SEVERITY["D40"],
            "pothole depth unobservable from a dashcam; defaulted to H",
            False,
        )
    # >750 mm: ASTM converts area to an equivalent hole count (area / 0.5 m2).
    return SeverityResult(
        DEFAULT_SEVERITY["D40"],
        "pothole depth unobservable; defaulted to H",
        False,
    )


def equivalent_hole_count(area_m2: float) -> float:
    """X1.17.1.2: potholes over 750 mm count as area / 0.5 m2 holes."""
    return area_m2 / 0.5
