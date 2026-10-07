# Legacy Reference — domain content carried forward

Content extracted from the pre-rebuild `app/` package before it was deleted. Everything here is
**domain knowledge that the rebuild does not re-derive** — it was hand-written from civil
standards, not generated. Reinstating it into `src/roadscope/` is planned in Phase 1 and Phase 6.

The legacy code itself is recoverable at commit `a4008379` or tag `pre-rebuild-cleanup`.

---

## 1. Repair treatments — IRC 82-2015

From `app/work_order.py:32-53`. Maps defect class + severity to prescribed civil treatment.
Used by `civil/` (Phase 6) when generating work orders.

```python
CIVIL_REPAIR_TREATMENTS = {
    "D40": {
        "H": "Immediate Full-Depth Patching: Saw-cut rectangular boundaries, tack coat, 75mm Hot Mix Asphalt compaction (IRC:82-2015 Cl. 4.3).",
        "M": "Semi-Permanent Patching: Clear debris, emulsion tack coat, cold mix asphalt compaction.",
        "L": "Temporary Throw-and-Roll Patching with cold asphalt mix until scheduled resurfacing.",
    },
    "D20": {
        "H": "Structural Rehabilitation: 50mm cold milling followed by 40mm Bituminous Concrete (BC) overlay.",
        "M": "Stress Absorbing Membrane Interlayer (SAMI) and 25mm corrective asphalt overlay.",
        "L": "Fog Seal / Slurry Seal application to arrest water infiltration.",
    },
    "D10": {
        "H": "Crack Routing and Hot-Poured Polymer Modified Bituminous Sealant (ASTM D6690).",
        "M": "High-pressure air wand cleaning and rubberized asphalt sealant injection.",
        "L": "Routine sand-emulsion slurry squeegee treatment.",
    },
    "D00": {
        "H": "Milling along crack corridor (150mm width) and micro-surfacing infill.",
        "M": "Air cleaning and elastomeric bitumen crack sealing.",
        "L": "Preventive surface seal coat application.",
    },
}
```

## 2. Priority levels and dispatch windows

From `app/work_order.py:55-60`.

```python
PRIORITY_LEVELS = {
    "D40": {"H": "CRITICAL (24-48 hrs)", "M": "HIGH (5 days)",   "L": "MEDIUM (14 days)"},
    "D20": {"H": "HIGH (7 days)",      "M": "MEDIUM (21 days)", "L": "ROUTINE (45 days)"},
    "D10": {"H": "MEDIUM (14 days)",   "M": "ROUTINE (30 days)", "L": "MONITOR (60 days)"},
    "D00": {"H": "MEDIUM (14 days)",   "M": "ROUTINE (30 days)", "L": "MONITOR (60 days)"},
}
```

## 3. RDD2022 distress schema

From `app/pci_engine.py:16-21` and `app/detector.py:19-38, 239-244`.

```python
RDD_DISTRESS_MAP = {
    "D00": {"name": "Longitudinal Crack",     "unit": "linear_m", "hazard": "low_medium"},
    "D10": {"name": "Transverse Crack",       "unit": "linear_m", "hazard": "low_medium"},
    "D20": {"name": "Alligator (Fatigue) Crack","unit": "sq_m",    "hazard": "high"},
    "D40": {"name": "Pothole",                "unit": "count",    "hazard": "critical"},
}

CLASS_CODES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

RDD_CLASS_NAME_MAP = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

CLASS_COLORS = {   # BGR, for OpenCV annotation
    "D00": (59, 130, 246),    # Blue
    "D10": (168, 85, 247),   # Purple
    "D20": (245, 158, 11),   # Amber
    "D40": (239, 68, 68),    # Red (critical)
}
```

> Note: the *class index → code* mapping above is only valid for a model genuinely trained on the
> 4-class RDD2022 schema. The legacy detector misapplied it to a single-class model
> (`detector.py:103`). Phase 1 must use `model.names` verbatim instead.

## 4. PCI rating bands and prescribed actions

From `app/pci_engine.py:138-166`. **The bands are correct and match ASTM D6433.** What was wrong was
the deduct-value maths feeding into them (invented `a + b·log₁₀(d)` curves, now replaced by the
published lookup tables in Phase 3 / §6.4 of the plan).

| PCI | Rating | Colour | Prescribed action |
| --- | --- | --- | --- |
| 85–100 | Good | `#10B981` | Routine preventive maintenance & crack sealing. |
| 70–84 | Satisfactory | `#84CC16` | Preventive surface seal, minor crack sealing. |
| 55–69 | Fair | `#FBBF24` | Cold patching of isolated potholes, slurry seal overlay. |
| 40–54 | Poor | `#F97316` | Full depth patch repairs and thin bituminous concrete overlay. |
| 25–39 | Very Poor | `#EF4444` | Extensive mill-and-fill resurfacing required immediately. |
| 10–24 | Serious | `#B91C1C` | Structural failure: Sub-base reconstruction & asphalt base course. |
| 0–9 | Failed | `#7F1D1D` | Total road failure: Complete pavement reconstruction mandated. |

## 5. Team allocation

From `app/dashboard.py:597-603`.

| Member | Reg. no. | Pillar | Contribution |
| --- | --- | --- | --- |
| Jadeja Suryadeepsinh Bharatsinh | 23BEL1027 | Lead / System Orchestrator | End-to-end pipeline integration, video-to-telemetry timestamp matching, prototype deployment |
| Vishnu Nair | 23BEL1028 | Computer Science / AI | YOLOv8 model inference, RDD2022 dataset preparation, mAP evaluation metrics |
| Udayvardhan Singh Rathore | 23BEL1018 | Civil Engineering | ASTM D6433 formulation, deduct value curves, PCI calculation engine |
| Ainessh Kumar S | 23BEL1039 | Geospatial & GIS | Folium Web-GIS mapping, GPS route interpolation, spatial clustering, heatmaps |
| Shreyan Biswas | 23BEL1052 | Urban Governance / UI | Streamlit municipal dashboard, CSV/HTML work order generator (PWD standards) |

## 6. Work order HTML report styling

The printable PWD report template at `app/work_order.py:117-200` — header, 4-card PCI summary grid,
prioritised repair table, and a signature block for the Assistant Municipal Engineer (PWD).

Worth retaining as the visual target for the Phase 6 work order export; the layout is sound, only
its data source changes. Recreate rather than copy — it needs live road names, measured dimensions
and GPS accuracy added per defect.

## 7. Sample data → test fixtures

Moved to `tests/fixtures/`, per plan §17:

- `sample_gps_track.csv` — 21 fixes, t = 0–40 s at 2 s intervals, ~12.9716 N 79.1585 E
- `sample_defects.json` — 6 defects, mixed D00/D10/D20/D40, severities L/M/H

These are now **test fixtures**, not demo data. The dashboard must read from the database (Phase 1/6).

---

## What was deliberately *not* carried forward

| Dropped | Reason |
| --- | --- |
| OpenCV pavement heuristic detector | Superseded by a real trained model (Phase 3) |
| Heuristic class mapping + crack supplement | Was fabricating D00/D10/D20 from contours |
| Area-based severity thresholds | Bounding-box area has no physical meaning |
| `a + b·log₁₀(d)` deduct curves | Invented; replaced by published ASTM tables |
| Linear `interpolate_gps` | Cuts corners; replaced by Kalman + map matching |
| `HTTPServer` + in-memory telemetry list | Nothing persisted; replaced by outbox + DB |
| Synthetic dashboard canvas | Demo was fake; replaced by live camera ingest |
| `sys.path` manipulation | Symptom of bad layout; replaced by `pyproject.toml` package |