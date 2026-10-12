"""Distress schema for the RDD2022 four-class model.

References ASTM D6433 distress types (Appendix X1) and IRC 82-2015 treatment
practice. Only the types this project can detect are represented.

The measurement *unit* matters: ASTM computes density differently depending on
how a distress is measured, and mixing them up silently produces wrong
deduct values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Unit(StrEnum):
    """How ASTM measures a distress, which fixes how density is computed."""

    SQ_M = "sq_m"  # area, e.g. alligator cracking
    LINEAR_M = "linear_m"  # length, e.g. cracks
    COUNT = "count"  # occurrences, e.g. potholes


@dataclass(frozen=True, slots=True)
class DistressType:
    """One ASTM distress type as RoadScope reports it."""

    code: str
    name: str
    unit: Unit
    astm_number: int | None = None
    hazard: str = "medium"

    @property
    def is_area_based(self) -> bool:
        return self.unit is Unit.SQ_M


#: ASTM D6433 distress types mapped to the RDD2022 model classes.
DISTRESS_TYPES: dict[str, DistressType] = {
    "D20": DistressType("D20", "Alligator (Fatigue) Cracking", Unit.SQ_M, 1, "high"),
    "D00": DistressType("D00", "Longitudinal Cracking", Unit.LINEAR_M, 10, "low_medium"),
    "D10": DistressType("D10", "Transverse Cracking", Unit.LINEAR_M, 10, "low_medium"),
    "D40": DistressType("D40", "Pothole", Unit.COUNT, 13, "critical"),
}

#: ASTM groups longitudinal and transverse cracking under one curve family.
CRACK_CODES = ("D00", "D10")

#: ASTM type 15. RoadScope cannot detect rutting, so this type exists only so
#: the published worked example in the standard can be reproduced exactly
#: during conformance testing. It is never emitted by the detector.
RUTTING = DistressType("RUTTING", "Rutting", Unit.SQ_M, 15, "medium")
DISTRESS_TYPES["RUTTING"] = RUTTING


def distress_for(code: str) -> DistressType:
    """Look up a distress type, defaulting to pothole for unknown codes."""
    return DISTRESS_TYPES.get(code, DISTRESS_TYPES["D40"])
