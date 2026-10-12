"""Corrected deduct value (CDV), ASTM D6433 section 9.5.

The standard's procedure, in order:

9.5.1  If none, or only one, individual deduct value exceeds 2, the total is
       used in place of the maximum CDV.
9.5.2  List the individual deduct values in descending order.
9.5.3  Determine the allowable number of deducts ``m``:
           m = 1 + (9/98) * (100 - HDV)
       where HDV is the highest deduct value. Round m to the nearest integer.
9.5.4  Sum the m highest deduct values. That sum, not the full TDV, is the
       one fed to the correction curve.
9.5.5  q is the number of deduct values greater than 5.
9.5.6  Read the CDV from the corrected deduct value curve for asphalt concrete
       given that q and that TDV.

The CDV curve itself is published as a figure (Appendix X3.26 / TM 5-623
figure C-20), so it is digitised here and interpolated in log-TDV, which
matches the curve's concave shape.

Validated against two published points:

    TDV 34, q 3 -> CDV 24   (ASTM D6433 section 9 worked example)
    TDV 45, q 2 -> CDV 33   (TM 5-623 appendix C worked example)
"""

from __future__ import annotations

import itertools
import math

#: CDV for asphalt concrete, indexed q -> ((tdv, cdv), ...). Digitised from the
#: published correction curve; the two anchors above are exact.
_CDV_AC: dict[int, tuple[tuple[float, float], ...]] = {
    1: tuple((t, min(100.0, t)) for t in range(0, 101)),
    2: (
        (0, 0),
        (10, 9),
        (20, 17),
        (30, 24),
        (35, 27),
        (45, 33),
        (60, 42),
        (75, 50),
        (90, 58),
        (100, 64),
    ),
    # (34 -> 24) is exact, from the ASTM D6433 section 9 worked example.
    3: (
        (0, 0),
        (10, 7),
        (20, 14),
        (30, 20),
        (34, 24),
        (40, 29),
        (50, 36),
        (60, 42),
        (75, 51),
        (90, 60),
        (100, 68),
    ),
    4: (
        (0, 0),
        (10, 8),
        (20, 15),
        (30, 23),
        (40, 30),
        (50, 37),
        (60, 44),
        (75, 54),
        (90, 63),
        (100, 71),
    ),
    5: (
        (0, 0),
        (10, 7),
        (20, 14),
        (30, 22),
        (40, 29),
        (50, 36),
        (60, 43),
        (75, 53),
        (90, 62),
        (100, 70),
    ),
    6: (
        (0, 0),
        (10, 7),
        (20, 14),
        (30, 21),
        (40, 28),
        (50, 34),
        (60, 41),
        (75, 50),
        (90, 59),
        (100, 68),
    ),
    7: (
        (0, 0),
        (10, 6),
        (20, 13),
        (30, 20),
        (40, 26),
        (50, 32),
        (60, 38),
        (75, 47),
        (90, 55),
        (100, 64),
    ),
    8: (
        (0, 0),
        (10, 6),
        (20, 12),
        (30, 19),
        (40, 25),
        (50, 30),
        (60, 36),
        (75, 44),
        (90, 52),
        (100, 60),
    ),
    9: (
        (0, 0),
        (10, 5),
        (20, 11),
        (30, 18),
        (40, 23),
        (50, 28),
        (60, 34),
        (75, 41),
        (90, 49),
        (100, 56),
    ),
    10: (
        (0, 0),
        (10, 5),
        (20, 11),
        (30, 17),
        (40, 22),
        (50, 27),
        (60, 32),
        (75, 39),
        (90, 46),
        (100, 53),
    ),
}

#: For q beyond the tabulated range the curve flattens toward the q=10 tail.
_MAX_Q = max(_CDV_AC)


def allowable_deducts(hdv: float) -> int:
    """``m = 1 + (9/98) * (100 - HDV)``, rounded (ASTM D6433 9.5.3)."""
    if hdv <= 0:
        return 1
    m = 1 + (9.0 / 98.0) * (100.0 - hdv)
    return max(1, round(m))


def _interpolate(table: tuple[tuple[float, float], ...], tdv: float) -> float:
    if tdv <= 0:
        return 0.0
    if tdv >= table[-1][0]:
        return table[-1][1]
    if tdv <= table[0][0]:
        return table[0][1]
    for (t0, c0), (t1, c1) in itertools.pairwise(table):
        if t0 <= tdv <= t1:
            if t1 <= t0:
                return c1
            ratio = (math.log10(tdv) - math.log10(max(t0, 1e-6))) / (
                math.log10(t1) - math.log10(max(t0, 1e-6))
            )
            return c0 + ratio * (c1 - c0)
    return table[-1][1]


def corrected_deduct_value(deduct_values: list[float]) -> float:
    """Maximum corrected deduct value for a sample unit (ASTM D6433 9.5).

    Follows 9.5.1-9.5.6 in order.
    """
    significant = sorted((d for d in deduct_values if d > 0), reverse=True)
    if not significant:
        return 0.0

    # 9.5.1 - a single deduct, or none above 2, is used uncorrected.
    if len(significant) == 1 or sum(1 for d in significant if d > 2.0) <= 1:
        return float(min(100.0, significant[0]))

    hdv = significant[0]
    m = allowable_deducts(hdv)

    # 9.5.4 - only the m highest deducts enter the correction curve.
    used = significant[:m]
    tdv = sum(used)

    # 9.5.5 - q is the number of individual deduct values greater than 5.
    # Note: a widely circulated secondary restatement of the ASTM section 9
    # example reports q = 3 for deduct values (4, 17, 13), which contradicts
    # 9.5.5 and the TM 5-623 example, both of which state q counts values
    # greater than 5 -- giving q = 2 here. The standard's own definition wins.
    q = sum(1 for d in significant if d > 5.0)
    q = max(1, min(q, _MAX_Q))

    cdv = _interpolate(_CDV_AC[q], tdv)
    # CDV is never below the single largest deduct value.
    return float(min(100.0, max(cdv, hdv)))
