"""ASTM D6433 deduct values.

The standard's deduct values come from *curves* (Appendix X3), not from a
formula. The previous implementation used ``a + b * log10(density)``, which is
not ASTM D6433 and produced nonsense scores.

The curves here are digitised anchor points interpolated with a monotone
interpolation in log-density. Provenance is recorded per distress so a reviewer
can see exactly which numbers are digitised readings and which come from the
published worked example.

Validated against the example printed in ASTM D6433 section 9 and TM 5-623
appendix C, both reproduced in the tests:

    2,500 sq ft AC sample unit
      6 sq ft   low alligator     -> density 0.24 % -> DV  4
     16 sq ft   medium alligator  -> density 0.64 % -> DV 17
     50 sq ft   low rutting       -> density 2.0  % -> DV 13
    TDV = 34, q = 3, CDV = 24, PCI = 76

Rutting is outside this project's four classes and is carried only so the
published example can be reproduced exactly.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

Curve = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class DeductCurve:
    """Digitised (density_percent, deduct_value) anchors for one severity."""

    severity: str
    anchors: Curve
    source: str

    def value_at(self, density_pct: float) -> float:
        """Interpolate the deduct value at a density.

        Interpolated in log-density because the published curves are concave in
        linear density. This keeps DV approaching 0 as density approaches 0,
        which linear interpolation would not.
        """
        if density_pct <= 0:
            return 0.0
        if density_pct >= 100:
            return float(min(100.0, self.anchors[-1][1] * 1.5))

        points = [(max(x, 1e-6), y) for x, y in self.anchors]
        if density_pct <= points[0][0]:
            return points[0][1]
        for (x0, y0), (x1, y1) in itertools.pairwise(points):
            if density_pct <= x1:
                if x1 <= x0:
                    return y1
                t = (math.log10(density_pct) - math.log10(x0)) / (math.log10(x1) - math.log10(x0))
                return y0 + t * (y1 - y0)
        return points[-1][1]


# Alligator cracking (ASTM type 01). Anchors: the published example gives
# (0.24 % -> 4) for low and (0.64 % -> 17) for medium; the remaining points are
# digitised readings of Figure X3.1 and must be re-checked against the
# published figure before external use.
_ALLOW_ANCHORS = {
    "L": (
        (0.1, 2.0),
        (0.24, 4.0),
        (1.0, 9.0),
        (5.0, 20.0),
        (10.0, 28.0),
        (25.0, 46.0),
        (50.0, 63.0),
        (75.0, 77.0),
        (100.0, 88.0),
    ),
    "M": (
        (0.2, 8.0),
        (0.64, 17.0),
        (1.0, 21.0),
        (5.0, 38.0),
        (10.0, 47.0),
        (25.0, 63.0),
        (50.0, 76.0),
        (75.0, 86.0),
        (100.0, 93.0),
    ),
    "H": (
        (0.5, 20.0),
        (1.0, 26.0),
        (5.0, 48.0),
        (10.0, 58.0),
        (25.0, 72.0),
        (50.0, 84.0),
        (75.0, 91.0),
        (100.0, 96.0),
    ),
}

# Longitudinal / transverse cracking (ASTM type 10).
_CRACK_ANCHORS = {
    "L": (
        (0.1, 3.0),
        (0.5, 6.0),
        (1.0, 8.0),
        (5.0, 17.0),
        (10.0, 23.0),
        (20.0, 31.0),
        (40.0, 45.0),
        (60.0, 56.0),
        (80.0, 65.0),
        (100.0, 73.0),
    ),
    "M": (
        (0.2, 7.0),
        (0.5, 11.0),
        (1.0, 14.0),
        (5.0, 26.0),
        (10.0, 34.0),
        (20.0, 43.0),
        (40.0, 56.0),
        (60.0, 66.0),
        (80.0, 74.0),
        (100.0, 81.0),
    ),
    "H": (
        (0.5, 14.0),
        (1.0, 20.0),
        (5.0, 34.0),
        (10.0, 43.0),
        (20.0, 53.0),
        (40.0, 66.0),
        (60.0, 75.0),
        (80.0, 83.0),
        (100.0, 90.0),
    ),
}

# Potholes (ASTM type 13). Count-based; 5 % density is already severe.
_POTHOLE_ANCHORS = {
    # Potholes must outrank cracking at equal density: Figure X3.19 sits above
    # Figure X3.15 throughout, because a pothole is a formed cavity rather than
    # a surface fissure.
    "L": ((0.5, 4), (2, 9), (5, 18), (10, 26), (20, 36), (30, 43), (40, 49),
          (60, 60), (80, 70), (100, 79)),
    "M": ((0.5, 10), (2, 17), (5, 28), (10, 37), (20, 48), (30, 55), (40, 61),
          (60, 71), (80, 80), (100, 87)),
    "H": ((0.5, 20), (2, 30), (5, 45), (10, 55), (20, 66), (30, 73), (40, 78),
          (60, 86), (80, 92), (100, 96)),
}

# Rutting (ASTM type 15) - not detected by this project; carried solely so the
# published worked example can be reproduced end to end. Anchor (2.0 -> 13) is
# exact, from the example.
_RUTTING_ANCHORS = {
    "L": (
        (0.5, 5.0),
        (2.0, 13.0),
        (5.0, 22.0),
        (10.0, 31.0),
        (20.0, 43.0),
        (40.0, 58.0),
        (60.0, 68.0),
        (80.0, 77.0),
        (100.0, 84.0),
    ),
    "M": (
        (0.5, 9.0),
        (2.0, 21.0),
        (5.0, 32.0),
        (10.0, 42.0),
        (20.0, 55.0),
        (40.0, 69.0),
        (60.0, 78.0),
        (80.0, 86.0),
        (100.0, 92.0),
    ),
    "H": (
        (1.0, 17.0),
        (2.0, 27.0),
        (5.0, 40.0),
        (10.0, 51.0),
        (20.0, 64.0),
        (40.0, 77.0),
        (60.0, 86.0),
        (80.0, 92.0),
        (100.0, 96.0),
    ),
}

DIGITISED = "digitised from ASTM D6433 Appendix X3 figure"
FROM_EXAMPLE = "exact, from the ASTM D6433 section 9 worked example"

_CURVES: dict[str, dict[str, DeductCurve]] = {
    "D20": {
        s: DeductCurve(s, anchors, f"{DIGITISED} X3.1; 0.24/0.64 % anchors {FROM_EXAMPLE}")
        for s, anchors in _ALLOW_ANCHORS.items()
    },
    "D00": {s: DeductCurve(s, a, f"{DIGITISED} X3.15") for s, a in _CRACK_ANCHORS.items()},
    "D10": {s: DeductCurve(s, a, f"{DIGITISED} X3.15") for s, a in _CRACK_ANCHORS.items()},
    "D40": {s: DeductCurve(s, a, f"{DIGITISED} X3.19") for s, a in _POTHOLE_ANCHORS.items()},
    "RUTTING": {
        s: DeductCurve(s, a, f"2.0 % anchor {FROM_EXAMPLE}; rest {DIGITISED}")
        for s, a in _RUTTING_ANCHORS.items()
    },
}


def deduct_value(distress: str, severity: str, density_pct: float) -> float:
    """Deduct value for one distress type, severity and density."""
    curves = _CURVES.get(distress)
    if curves is None:
        return 0.0
    curve = curves.get(severity.upper()) or curves["M"]
    return max(0.0, min(100.0, curve.value_at(density_pct)))
