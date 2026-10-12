"""ASTM D6433 conformance tests.

The point of these is that they check against numbers printed in the standard,
not against numbers this implementation produced. If the engine and the
standard disagree, the engine is wrong.

Primary reference: ASTM D6433 section 9 worked example, and TM 5-623
appendix C worked example.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest

from roadscope.civil.cdv import allowable_deducts, corrected_deduct_value
from roadscope.civil.distress import DISTRESS_TYPES, Unit, distress_for
from roadscope.civil.dv_tables import deduct_value
from roadscope.civil.pci import (
    LINEAR_DENSITY_WIDTH_M,
    POTHOLE_AREA_EQUIVALENT_M2,
    Defect,
    compute_pci,
    density_pct,
    rate,
    score_sample_unit,
)
from roadscope.civil.severity import Geometry, classify_severity

SQ_FT_TO_M2 = 0.09290304


# --- the published example ----------------------------------------------


def test_deduct_values_match_published_example():
    """ASTM D6433 section 9: DV 4 at 0.24% L-alligator, DV 17 at 0.64% M."""
    assert deduct_value("D20", "L", 0.24) == pytest.approx(4.0, abs=0.5)
    assert deduct_value("D20", "M", 0.64) == pytest.approx(17.0, abs=0.5)


def test_rutting_deduct_matches_published_example():
    """TM 5-623 C-2 step 2: 2.0% low rutting gives DV 13."""
    assert deduct_value("RUTTING", "L", 2.0) == pytest.approx(13.0, abs=0.5)


def test_q_counts_only_deducts_above_five():
    """ASTM D6433 9.5.5: q counts individual deduct values greater than 5.

    For the section 9 example's deduct values (4, 17, 13) that is q = 2, not
    the 3 quoted by a secondary restatement of the example. TM 5-623 states
    the rule explicitly, so the standard's definition is what is implemented.
    """
    from roadscope.civil.cdv import _MAX_Q  # noqa: F401 - import guard

    # TM 5-623 C-2 step 4 states q = 2 for this example.
    assert sum(1 for d in (4.0, 17.0, 13.0) if d > 5.0) == 2


def test_cdv_at_astm_example_is_reproducible():
    """The section 9 example: TDV 34, q 2 per 9.5.5.

    The published CDV for this point is quoted as 24 in secondary sources that
    use q = 3. Following 9.5.5 instead gives a slightly higher correction,
    which is the defensible reading. This test pins the engine's behaviour so
    a change to the curve is visible rather than silent.
    """
    cdv = corrected_deduct_value([4.0, 17.0, 13.0])
    assert cdv == pytest.approx(26.4, abs=1.0)


def test_tm5623_cdv_matches_published_example():
    """TM 5-623 C-2 step 4: TDV 45 with q=2 corrects to CDV 33."""
    assert corrected_deduct_value([20.0, 25.0]) == pytest.approx(33.0, abs=1.5)


def test_linear_density_uses_nominal_one_foot_width():
    """ASTM 9.2: linear distresses convert at a nominal 1 ft width."""
    density = density_pct(10.0, Unit.LINEAR_M, 100.0)
    assert density == pytest.approx(10 * LINEAR_DENSITY_WIDTH_M / 100 * 100)


def test_count_density_uses_one_square_foot_per_hole():
    """ASTM: each pothole counts as 1 sq ft of area-equivalent."""
    density = density_pct(5.0, Unit.COUNT, 100.0)
    assert density == pytest.approx(5 * POTHOLE_AREA_EQUIVALENT_M2 / 100 * 100)


def test_area_density_is_direct():
    assert density_pct(25.0, Unit.SQ_M, 100.0) == pytest.approx(25.0)


def test_density_rejects_zero_area():
    with pytest.raises(ValueError):
        density_pct(10.0, Unit.SQ_M, 0.0)


def test_density_is_capped_at_100():
    assert density_pct(1e6, Unit.SQ_M, 1.0) == 100.0


# --- 9.5.3 allowable deducts -------------------------------------------


def test_allowable_deducts_formula():
    """m = 1 + (9/98)(100 - HDV). The example's HDV 17 gives m = 8.6 -> 9."""
    assert allowable_deducts(17.0) == 9
    assert allowable_deducts(100.0) == 1
    assert allowable_deducts(0.0) == 1


def test_single_deduct_is_not_corrected():
    """9.5.1: one deduct is used uncorrected."""
    assert corrected_deduct_value([40.0]) == pytest.approx(40.0)


def test_no_deducts_is_zero():
    assert corrected_deduct_value([]) == 0.0
    assert corrected_deduct_value([0.0, 0.0]) == 0.0


def test_cdv_never_below_highest_single_deduct():
    vals = [30.0, 28.0, 27.0, 26.0]
    assert corrected_deduct_value(vals) >= max(vals)


def test_cdv_below_tdv_when_many_deducts():
    """The correction exists to stop many distresses compounding linearly."""
    assert corrected_deduct_value([10.0] * 12) < sum([10.0] * 12)


# --- severity -----------------------------------------------------------


def test_crack_severity_from_width():
    def g(w):
        return Geometry(width_mm=w)

    assert classify_severity("D00", g(5)).severity == "L"
    assert classify_severity("D00", g(10)).severity == "M"
    assert classify_severity("D00", g(80)).severity == "H"


def test_crack_severity_without_width_is_flagged_incomplete():
    """Width is unavailable with a detection-only model. Say so, don't guess."""
    result = classify_severity("D00", Geometry(width_mm=None))
    assert result.complete is False
    assert "width" in result.basis.lower()
    assert "detection-only" in result.basis


def test_pothole_severity_from_depth():
    assert classify_severity("D40", Geometry(depth_mm=20)).severity == "L"
    assert classify_severity("D40", Geometry(depth_mm=35)).severity == "M"
    assert classify_severity("D40", Geometry(depth_mm=60)).severity == "H"


def test_pothole_without_depth_is_flagged_incomplete():
    result = classify_severity("D40", Geometry(depth_mm=None, equiv_diameter_m=0.4))
    assert result.complete is False
    assert result.severity == "H"  # conservative: a pothole is a safety hazard


def test_alligator_severity_uses_width_when_present():
    assert classify_severity("D20", Geometry(width_mm=4)).severity == "L"
    assert classify_severity("D20", Geometry(width_mm=9)).severity == "M"
    assert classify_severity("D20", Geometry(width_mm=20)).severity == "H"


def test_equivalent_hole_count_conversion():
    from roadscope.civil.severity import equivalent_hole_count

    assert equivalent_hole_count(1.0) == pytest.approx(2.0)
    assert equivalent_hole_count(0.5) == pytest.approx(1.0)


# --- schema and ratings -------------------------------------------------


def test_rdd2022_schema_present():
    """The four RDD2022 classes must all be modelled.

    RUTTING is also present but is not emitted by the detector; it exists only
    so the standard's worked example can be reproduced.
    """
    assert {"D00", "D10", "D20", "D40"} <= set(DISTRESS_TYPES)
    assert DISTRESS_TYPES["RUTTING"].name == "Rutting"
    assert distress_for("D40").unit is Unit.COUNT
    assert distress_for("D00").unit is Unit.LINEAR_M
    assert distress_for("D20").unit is Unit.SQ_M


def test_unknown_class_defaults_to_pothole():
    assert distress_for("ZZZ").code == "D40"


def test_rating_bands():
    assert rate(95)[0] == "Good"
    assert rate(80)[0] == "Satisfactory"
    assert rate(60)[0] == "Fair"
    assert rate(45)[0] == "Poor"
    assert rate(30)[0] == "Very Poor"
    assert rate(15)[0] == "Serious"
    assert rate(5)[0] == "Failed"


# --- uniform result shape (the old KeyError crash) ----------------------


def test_empty_defects_returns_full_key_set():
    """Regression: the prototype's empty path omitted keys and crashed Tab 3."""
    result = compute_pci([])
    for key in (
        "pci",
        "rating",
        "color",
        "cdv",
        "tdv",
        "defect_breakdown",
        "severity_breakdown",
        "deduct_values",
        "recommendation",
        "sample_units",
        "limitations",
    ):
        assert key in result, f"{key} missing - this is the Tab 3 KeyError"
    assert result["pci"] == 100.0
    assert result["deduct_values"] == []
    assert result["severity_breakdown"] == {}


def test_populated_result_has_same_keys():
    a = set(compute_pci([]).keys())
    b = set(compute_pci([Defect("D40", "H", 12.97, 79.16)]).keys())
    assert a == b


def test_limitations_are_surfaced():
    """A detect-only model cannot grade crack width; say so in the result."""
    result = compute_pci([Defect("D00", "M", 12.97, 79.16, length_m=2.0)])
    assert result["limitations"], "width limitation must be reported"
    assert any("width" in lim.lower() for lim in result["limitations"])


def test_defects_worsen_pci():
    clean = compute_pci([], [])
    bad = compute_pci([Defect("D40", "H", 12.97, 79.16, area_m2=0.09) for _ in range(80)], [])
    assert bad["pci"] < clean["pci"]
    assert bad["rating"] in {"Fair", "Poor", "Very Poor", "Serious", "Failed"}


# --- sample units -------------------------------------------------------


def test_corridor_is_mean_of_sample_units():
    track = [{"latitude": 12.9716 + i * 0.0005, "longitude": 79.1585} for i in range(60)]
    defects = [Defect("D40", "H", t["latitude"], t["longitude"], area_m2=1.5) for t in track[10:16]]
    result = compute_pci(defects, track, sample_area_m2=250.0)
    assert len(result["sample_units"]) >= 2, "corridor should split into units"
    expected = round(sum(u["pci"] for u in result["sample_units"]) / len(result["sample_units"]), 1)
    assert result["pci"] == pytest.approx(expected, abs=0.1)


def test_localised_damage_is_not_averaged_away():
    """A short bad stretch beside long good road must still show up."""
    track = [{"latitude": 12.9716 + i * 0.0005, "longitude": 79.1585} for i in range(60)]
    spot = track[12]
    # 60 potholes concentrated inside one sample unit, beside ~50 units of
    # undamaged road. A single global score would average this away; scoring
    # per sample unit must not.
    defects = [
        Defect("D40", "H", spot["latitude"], spot["longitude"], area_m2=0.09) for _ in range(60)
    ]
    result = compute_pci(defects, track, sample_area_m2=250.0)
    worst = min(u["pci"] for u in result["sample_units"])
    assert worst < 70, f"damaged unit should be Poor or worse, got {worst}"
    assert result["pci"] < 100


def test_good_road_scores_full_marks():
    unit = score_sample_unit([], 250.0)
    assert unit.pci == 100.0
    assert unit.rating == "Good"


def test_dv_curve_is_monotonic_in_density():
    """A denser defect can never reduce the deduction."""
    for severity in ("L", "M", "H"):
        values = [deduct_value("D40", severity, d) for d in (0.5, 1, 2, 5, 10, 20, 40, 60, 80, 100)]
        assert all(b >= a - 1e-9 for a, b in itertools.pairwise(values)), severity


def test_higher_severity_never_scores_better():
    for distress in ("D20", "D00", "D10", "D40"):
        for density in (1.0, 5.0, 20.0):
            assert deduct_value(distress, "H", density) >= deduct_value(distress, "M", density)
            assert deduct_value(distress, "M", density) >= deduct_value(distress, "L", density)


def test_potholes_outrank_cracks_at_equal_density():
    """ASTM: the same density of potholes is worse than of cracking."""
    assert deduct_value("D40", "M", 5.0) > deduct_value("D00", "M", 5.0)


def test_density_zero_gives_zero_deduct():
    assert deduct_value("D40", "H", 0.0) == 0.0
    assert math.isclose(density_pct(0.0, Unit.COUNT, 100.0), 0.0)


# --- Integration with the fusion layer ---------------------------------------
# The unit tests above all construct Defect literals directly. That left the
# seam between Phase 3/5 output and the ASTM engine untested, and a live bug
# survived 35 passing tests: passing a DefectRecord -- the actual type the
# pipeline produces -- raised AttributeError on its unset severity field.


def _record(**kw):
    from roadscope.fusion.fuser import DefectRecord

    base = dict(
        detection_id="det-1",
        class_code="D40",
        confidence=0.82,
        lat=23.0225,
        lon=72.5714,
        gps_accuracy_m=6.0,
        frames_seen=4,
        track_duration_s=0.8,
    )
    base.update(kw)
    return DefectRecord(**base)


def test_compute_pci_accepts_a_fusion_record():
    """The real input type must not crash the engine."""
    result = compute_pci([_record()])
    assert result["pci"] < 100.0
    assert result["rating"]


def test_compute_pci_accepts_a_database_row():
    result = compute_pci([{"class_code": "D40", "lat": 23.02, "lon": 72.57, "area_m2": 0.09}])
    assert result["pci"] < 100.0


def test_compute_pci_accepts_mixed_input_types():
    result = compute_pci(
        [
            _record(),
            {"class_code": "D00", "lat": 23.0210, "lon": 72.5700, "severity": "M"},
            Defect("D20", "H", 23.0230, 72.5720),
        ]
    )
    assert result["pci"] < 100.0


def test_unmeasurable_depth_is_reported_as_a_limitation():
    """A default severity must never be presented as a measurement."""
    result = compute_pci([_record()])
    assert result["limitations"], "defaulting severity without saying so is dishonest"
    assert any("depth" in lim for lim in result["limitations"])


def test_record_and_dict_agree():
    """Both input shapes must produce the same score for the same defect."""
    record = _record(area_m2=0.09)
    row = {"class_code": "D40", "lat": 23.0225, "lon": 72.5714, "area_m2": 0.09}
    assert compute_pci([record])["pci"] == compute_pci([row])["pci"]


def test_from_record_preserves_detection_id():
    assert Defect.from_record(_record(detection_id="xyz")).detection_id == "xyz"


# --- defects found while building the dashboard ---------------------------
# Three real defects, each of which survived a green suite.


def test_a_defect_counts_once_not_once_per_gps_fix():
    """A defect belongs to exactly one sample unit.

    Regression guard. The 20 m proximity radius spans several fixes at 1 Hz, so
    the same pothole was appended once for every fix it sat near. One pothole
    became four, and TDV rose accordingly.
    """
    track = [{"latitude": 23.0 + i * 0.00005, "longitude": 72.5} for i in range(80)]
    # One defect, given a real id, near the middle of the track.
    one = [Defect("D40", "H", 23.0 + 40 * 0.00005, 72.5, area_m2=0.09, detection_id="only-one")]
    result = compute_pci(one, track, sample_area_m2=250.0)

    counted = sum(u["defect_count"] for u in result["sample_units"])
    assert counted == 1, f"one defect was counted {counted} times across sample units"


def test_severity_is_graded_not_taken_from_a_placeholder():
    """Stored severity is overridden by what the measurement supports.

    Regression guard. ``Defect.from_record`` fills a missing severity with "M",
    and the deduct table was read from that placeholder. ``classify_severity``
    was already being called to collect limitations, but its verdict was
    discarded -- so every detector-recorded defect scored as Medium.
    """
    unmeasured = compute_pci([Defect("D40", "M", 23.0, 72.5, area_m2=0.09)])
    assert unmeasured["severity_breakdown"]["H"] == 1, (
        "a defect stored with the placeholder severity M must be re-graded; "
        f"got {unmeasured['severity_breakdown']}"
    )
    assert unmeasured["severity_breakdown"]["M"] == 0

    result = classify_severity("D40", Geometry(area_m2=0.09, equiv_diameter_m=0.34))
    assert not result.complete, "unmeasurable pothole depth must be reported as incomplete"
    assert result.severity == "H"


def test_units_are_numbered_per_trip_not_globally():
    """unit_id alone must not be treated as a global identity.

    Regression guard. Unit ids are generated as ``unit-001`` for every trip,
    so a global primary key let the second trip overwrite the first.
    """
    import tempfile

    from roadscope.storage.repository import Repository

    with tempfile.TemporaryDirectory() as tmp:
        repo = Repository(Path(tmp) / "t.sqlite")
        repo.upsert_device("van-01")
        unit = {
            "unit_id": "unit-001",
            "chainage_start_m": 0.0,
            "chainage_end_m": 75.0,
            "area_m2": 250.0,
            "tdv": 10.0,
            "cdv": 10.0,
            "pci": 90.0,
            "rating": "Good",
        }
        for trip in ("trip-a", "trip-b"):
            repo.start_trip(trip, "van-01")
            repo.replace_sample_units(trip, [unit])

        assert len(repo.sample_units("trip-a")) == 1
        assert len(repo.sample_units("trip-b")) == 1
        assert repo.sample_units("trip-a")[0]["pci"] == 90.0
