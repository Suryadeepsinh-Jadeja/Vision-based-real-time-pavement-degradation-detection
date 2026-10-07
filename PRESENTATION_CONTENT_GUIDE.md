# Presentation Content Guide: Vision-Based Real-Time Pavement Degradation Detection and GIS System

> ## ⚠️ SUPERSEDED — describes the deleted prototype
>
> **This guide documents the pre-rebuild prototype, which has been deleted from the repository.**
> The run instructions in §Demo Steps, the file paths throughout (`app/dashboard.py`, `run.sh`,
> `requirements.txt`, `data/sample_*.json`), and every demo command are **no longer valid**.
> There is no runnable application in this repository at present.
>
> Retained as source material for structure and argument. The technical claims are superseded by
> [`REBUILD_PLAN.md`](REBUILD_PLAN.md), and salvaged domain content is in
> [`docs/legacy_reference.md`](docs/legacy_reference.md). **Rewrite for the rebuild during Phase 7.**
>
> Historical: recoverable at commit `a4008379`.

---

> **Purpose:** This is the master content file for preparing a project presentation, review, viva, or demonstration. It describes the implemented application, the engineering rationale behind it, speaker notes, demo steps, and the evidence needed for claims.
>
> **Important presentation rule:** Separate what the current repository demonstrably implements from planned production capabilities. The prototype implements a local Streamlit application, an HTTP GPS receiver, YOLO/OpenCV detection logic, PCI-style scoring, Folium mapping, and work-order export. Do not state unverified model accuracy, FPS, field accuracy, or scalability figures as measured results unless you can show the corresponding evaluation data.

---

## 1. Project Identity

| Item | Detail |
| --- | --- |
| Project title | Vision-Based Real-Time Pavement Degradation Detection and GIS System |
| Project type | Smart-city infrastructure monitoring prototype |
| Core objective | Convert road imagery and location data into actionable defect, pavement-condition, GIS, and maintenance information. |
| Primary users | Municipal road departments, survey teams, maintenance planners, and project evaluators. |
| Current interface | Streamlit Web-GIS dashboard running locally at `http://localhost:8501`. |
| Technology stack | Python, Streamlit, Folium, OpenCV, Ultralytics YOLOv8 (optional), Pandas, NumPy, Matplotlib, Jinja2. |
| Civil references named in the code | ASTM D6433-20 for PCI methodology and IRC 82-2015 for bituminous-surface maintenance guidance. |
| Detection taxonomy | RDD2022-style D00, D10, D20, and D40 distress codes. |

### Team roles

| Team member | Registration number | Suggested presentation ownership |
| --- | --- | --- |
| Jadeja Suryadeepsinh Bharatsinh | `23BEL1027` | System integration, pipeline narrative, telemetry synchronization, live prototype. |
| Vishnu Nair | `23BEL1028` | Computer vision, YOLO/OpenCV detection, defect classification. |
| Udayvardhan Singh Rathore | `23BEL1018` | Civil-engineering interpretation, PCI scoring, maintenance logic. |
| Ainessh Kumar S | `23BEL1039` | GIS mapping, coordinates, route and heatmap explanation. |
| Shreyan Biswas | `23BEL1052` | Municipal workflow, work orders, interface, export outputs. |

---

## 2. Executive Summary

Road-maintenance agencies need more than images of damaged roads. They need to know where each defect is, how severe it appears, how it affects pavement condition, and which maintenance response has priority. Manual inspections are time-consuming and can yield inconsistent records. A photograph without coordinates does not directly identify a field location, while a map with no condition metric does not justify a maintenance decision.

This project demonstrates an integrated workflow:

1. A dashcam image or video frame supplies visual road information.
2. A smartphone or sample dataset supplies latitude, longitude, speed, heading, and time.
3. The detection engine classifies pavement distress and estimates severity.
4. The PCI engine converts defect extent and severity into deduct values and a road-condition score.
5. The GIS module locates defects along a route and shows clusters or heat concentration.
6. The municipal module creates a priority-ranked work-order schedule with a prescribed treatment.

The key contribution is the transition from **detection** to **maintenance action**. The prototype does not stop at drawing boxes around road damage; it connects image evidence to location, condition assessment, and an operational response.

---

## 3. Problem Statement

### 3.1 Road-maintenance problem

Municipal road networks face recurring pavement deterioration from traffic loading, water ingress, thermal movement, poor drainage, and ageing. Conventional inspection often relies on field personnel recording defects manually. This creates four practical difficulties:

| Problem | Consequence |
| --- | --- |
| Inspection is manual and periodic | Damage can worsen between inspections. |
| Image evidence lacks location context | Field crews spend time locating reported defects. |
| Defects are not consistently graded | Maintenance decisions may depend on subjective observation. |
| Repair lists are not linked to condition metrics | Agencies tend toward reactive repair instead of planned preventive maintenance. |

### 3.2 Proposed response

The system uses vehicle-mounted imagery and GPS telemetry to generate a geo-referenced distress record. It applies a standardized defect taxonomy, translates severity into a pavement-condition score, visualizes locations, and produces repair priorities. The prototype offers a low-cost software starting point because it can accept ordinary uploaded road media and smartphone-style GPS payloads.

### 3.3 Scope boundary

This repository demonstrates the software workflow. It does not include a deployed cloud backend, authenticated user accounts, production database, calibrated physical camera model, or a documented training/evaluation experiment. State this boundary clearly when asked about deployment readiness.

---

## 4. System Architecture

### 4.1 Architecture narrative

The architecture contains four engineering pillars that meet in one municipal decision workflow.

```text
Road image or video                     GPS telemetry or sample track
        |                                            |
        v                                            v
OpenCV/YOLO detection             HTTP receiver and interpolation
        |                                            |
        +------------------+-------------------------+
                           |
                           v
      Defect record: class, confidence, severity, extent, coordinates, time
                           |
              +------------+-------------+
              |                          |
              v                          v
       PCI-style condition score     GIS markers and heatmap
              |                          |
              +------------+-------------+
                           |
                           v
          Municipal work order with priority and treatment recommendation
```

### 4.2 Module-to-file mapping

| Module | Source file | Main responsibility |
| --- | --- | --- |
| Dashboard | `app/dashboard.py` | Presents five application views and coordinates module output. |
| Detection | `app/detector.py` | Loads optional YOLO weights or uses an OpenCV fallback. |
| Telemetry | `app/telemetry.py` | Receives and retains GPS fixes; interpolates route coordinates. |
| PCI | `app/pci_engine.py` | Calculates density, deduct values, corrected deduct value, PCI, rating, and recommendation. |
| Work orders | `app/work_order.py` | Maps defects to treatment, priority, CSV data, and printable HTML. |
| Demo data | `data/sample_defects.json`, `data/sample_gps_track.csv` | Supplies a reproducible sample corridor. |

### 4.3 Architecture slide speaking notes

Say: “The project has four technical pillars. The vision layer tells us what the road defect looks like. The telemetry layer tells us where and when it occurred. The civil-engineering layer tells us how the recorded distress affects pavement condition. The GIS and work-order layer turns the result into something a municipal team can act on.”

---

## 5. Road-Damage Taxonomy

The detection layer uses RDD2022-style distress identifiers. This common vocabulary makes downstream reporting consistent.

| Code | Name | Visual interpretation | Engineering significance | Prototype hazard level |
| --- | --- | --- | --- | --- |
| `D00` | Longitudinal crack | Crack generally parallel to direction of travel | Can permit moisture ingress and propagate with traffic. | Low to medium |
| `D10` | Transverse crack | Crack generally perpendicular to direction of travel | Often associated with movement, shrinkage, or thermal effects. | Low to medium |
| `D20` | Alligator crack | Interconnected, mesh-like fatigue cracking | Suggests structural fatigue in the pavement system. | High |
| `D40` | Pothole | Localized cavity or depression | Immediate ride-quality, vehicle-damage, and safety concern. | Critical |

### 5.1 Detection record schema

Each detection produced by the code contains fields similar to the following:

```json
{
  "class": "D40",
  "label": "D40 - Pothole",
  "confidence": 0.92,
  "bbox": [x1, y1, x2, y2],
  "severity": "H",
  "area_px": 5420,
  "extent_sq_m": 0.45
}
```

The dashboard can pair the record with GPS and timestamp data before showing it on the map or submitting it to PCI and work-order modules.

---

## 6. Telemetry and Time Synchronization

### 6.1 Engineering problem

Video and GPS have different sampling rates. A camera can generate many frames per second, while a typical phone GPS source may send fixes much less often. Reusing the same location for every intermediate frame makes a moving survey inaccurate.

### 6.2 Implemented solution

`app/telemetry.py` provides:

- A thread-safe `TelemetryStore` holding the latest point and a history of up to 2,000 fixes.
- A local HTTP server started by the Streamlit dashboard on port `5050`.
- `POST /gps` for new GPS payloads and `GET /gps` for the latest point.
- Linear interpolation between two timestamped route points.
- A generated fallback route with 150 points when the sample CSV is unavailable.

### 6.3 Accepted GPS payload

```json
{
  "latitude": 12.9716,
  "longitude": 79.1585,
  "speed_kmh": 28.5,
  "heading": 85.0,
  "accuracy": 2.4,
  "timestamp": 1760000000
}
```

The receiver also accepts `speed` in place of `speed_kmh` and `bearing` in place of `heading`.

### 6.4 Interpolation equation

For a frame at time `t` that lies between GPS points at `t1` and `t2`:

```text
ratio = (t - t1) / (t2 - t1)
latitude(t)  = latitude1  + ratio x (latitude2  - latitude1)
longitude(t) = longitude1 + ratio x (longitude2 - longitude1)
```

The same proportional method applies to speed in the implementation.

### 6.5 HUD output

For the preloaded demo and uploaded media, the dashboard can overlay GPS coordinate, speed, heading, and elapsed survey time on an annotated frame. Explain that this helps a reviewer see the link between visual detection and location data immediately.

### 6.6 Telemetry limitations to state honestly

- The HTTP endpoint uses no authentication and should remain a development/demo interface.
- The history is in memory; it disappears when the app stops.
- The prototype accepts coordinates but does not quantify real-world GPS error or camera-to-road calibration error.
- A real fleet deployment would need secure transport, device identity, retry handling, storage, and monitoring.

---

## 7. Computer Vision Pipeline

### 7.1 Two operating modes

| Mode | Activation | Behavior |
| --- | --- | --- |
| YOLOv8 | Install `requirements-yolo.txt`; `models/pothole_yolov8.pt` loads successfully. | Runs Ultralytics YOLO inference, maps class names to RDD codes, and annotates detections. |
| OpenCV fallback | YOLO dependency or model loading is unavailable. | Uses image-processing heuristics to find dark pothole-like regions and crack-like contours. |

This fallback gives the dashboard a usable demo path even without downloading the larger PyTorch/Ultralytics dependency. It is not a substitute for a calibrated trained model evaluation.

### 7.2 YOLOv8 path

1. The detector checks several candidate weight paths, starting with `models/pothole_yolov8.pt`.
2. It imports `ultralytics.YOLO` and runs inference using the sidebar confidence threshold.
3. The code maps model class names to `D00`, `D10`, `D20`, or `D40`.
4. It estimates bounding-box relative area using `box_area / frame_area`.
5. It assigns severity by relative area:

| Relative bounding-box area | Severity |
| --- | --- |
| Less than `0.015` | Low (`L`) |
| `0.015` to less than `0.045` | Medium (`M`) |
| `0.045` or greater | High (`H`) |

If the loaded YOLO model has one class, the code treats it as pothole detection and runs the OpenCV crack logic to add non-pothole distress candidates.

### 7.3 OpenCV fallback path

The fallback focuses on the lower 60% of each frame, treating it as the pavement region of interest.

**Pothole-like region logic**

1. Convert the road region to grayscale.
2. Apply a `7 x 7` Gaussian blur.
3. Determine a dark-pixel threshold from the blurred-image mean minus 32, with a minimum threshold of 30.
4. Create an inverse binary mask for dark areas.
5. Apply morphological opening and closing with a `9 x 9` elliptical kernel.
6. Keep contours with area greater than 1,200 pixels and less than 15% of the total frame area.
7. Keep contours with aspect ratio from `0.5` to `3.0`.

**Crack-like region logic**

1. Apply Canny edge detection with thresholds 60 and 160.
2. Dilate edges with a `3 x 3` rectangular kernel.
3. Keep contours with area from 600 to 8,000 pixels.
4. Use bounding-box aspect ratio to classify direction:

| Aspect ratio | Prototype class |
| --- | --- |
| Greater than `2.8` | `D10`, transverse crack |
| Less than `0.4` | `D00`, longitudinal crack |
| Otherwise | `D20`, alligator crack |

### 7.4 Severity and extent caveat

`extent_sq_m` in this prototype is derived from frame-relative pixel area multiplied by a fixed scaling factor. It is useful for a demonstration of the data flow, but it is not a physically calibrated area measurement. A production implementation would need camera calibration, camera height, lens parameters, road-plane geometry, and validation against measured defects.

---

## 8. Pavement Condition Index Engine

### 8.1 Purpose of PCI

Counting defects alone cannot communicate overall road condition. PCI creates a normalized 0 to 100 indicator by considering defect type, severity, and density. Higher-severity potholes and fatigue cracking receive larger penalties than minor isolated cracking.

### 8.2 Implemented calculation flow

The implementation in `app/pci_engine.py` uses a default sample area of 500 square metres when called from the dashboard.

```text
Defect extent and severity
        |
        v
Density (%) = extent / sample area x 100
        |
        v
Deduct value based on distress type, severity, and log10(density)
        |
        v
Total deduct value (TDV) = sum of individual deduct values
        |
        v
Corrected deduct value (CDV) = TDV adjusted for multiple significant deduct values
        |
        v
PCI = max(0, 100 - CDV)
```

### 8.3 Density formula

```text
Density (%) = (distress extent / sample area) x 100
```

The code constrains density to a maximum of 100% and falls back to 250 square metres if an invalid sample area is supplied to the lower-level density function.

### 8.4 Deduct-value model in the code

The code uses calibrated logarithmic approximations of the form:

```text
DV = a + b x log10(density)
```

The `a` and `b` coefficients depend on the defect code and severity. In the implementation, high-severity potholes receive the largest baseline penalty, followed by fatigue cracking, then longitudinal/transverse cracking. Values are clipped between 0 and 100.

### 8.5 Corrected deduct value

Only deduct values above 2 are retained. The code counts `q`, the number of deduct values greater than 5.

```text
When q <= 1: CDV = TDV

When q > 1:
q_factor = 1 / (1 + 0.22 x (q - 1))
CDV = max(maximum individual DV, TDV x q_factor)
```

This prevents simple addition of every defect penalty from overstating damage when several defects coexist. It is an approximation in the prototype and should be described as an ASTM D6433-inspired implementation unless the coefficient calibration and standard lookup curves have been formally validated.

### 8.6 PCI ratings used by the dashboard

| PCI range | Rating | Recommended action in the implementation |
| --- | --- | --- |
| 85 to 100 | Good | Routine preventive maintenance and crack sealing. |
| 70 to 84.9 | Satisfactory | Preventive surface seal and minor crack sealing. |
| 55 to 69.9 | Fair | Isolated pothole patching and slurry-seal overlay. |
| 40 to 54.9 | Poor | Full-depth patch repairs and a thin bituminous-concrete overlay. |
| 25 to 39.9 | Very Poor | Extensive mill-and-fill resurfacing. |
| 10 to 24.9 | Serious | Sub-base reconstruction and asphalt base course. |
| 0 to 9.9 | Failed | Complete pavement reconstruction. |

---

## 9. GIS and Map Module

### 9.1 Purpose

The GIS view turns a list of coordinates into a maintenance map. This is useful because field teams need geographic context, planners need to identify concentrations, and reviewers need visual confirmation that detections connect to a route.

### 9.2 Implemented dashboard controls

The map view includes:

- Basemap selection between CartoDB Positron and OpenStreetMap.
- Toggleable vehicle-route polyline.
- Toggleable heatmap.
- Toggleable clustered defect markers.
- A Folium layer control.

### 9.3 Map rendering flow

1. The dashboard computes the center of the GPS track.
2. It creates a Folium map with an initial zoom of 17.
3. It draws the GPS path as a polyline.
4. It converts defect coordinates into weighted heatmap points; high severity has greater weight.
5. It creates clustered markers to reduce marker overlap.
6. Each marker popup presents the defect ID, class, severity, confidence, extent, coordinate, and repair priority.

### 9.4 Why heatmaps matter

A single pothole requires a local repair response. Repeated high-severity defects in one corridor can indicate a drainage, sub-base, utility-cut, or traffic-loading issue. A heatmap gives a planner a way to see where isolated repair may be insufficient and a corridor-level intervention deserves investigation.

### 9.5 GIS limitations

- The current sample route is a small demonstration corridor, not a city-scale road inventory.
- The map relies on external tile providers for basemap imagery.
- The prototype stores no geospatial database or historical layer.
- No coordinate reference system transformation, road-segment matching, or duplicate-detection logic is implemented.

---

## 10. Work-Order Automation

### 10.1 Why it matters

The municipal module connects a technical finding to a maintenance instruction. It answers: Which issue must be repaired first, how soon, and what treatment is suggested?

### 10.2 Generated fields

Each work order contains:

- Work-order ID
- Logged date
- Road section
- Defect code and description
- Severity
- GPS latitude and longitude
- Repair priority
- Prescribed engineering treatment

### 10.3 Priority rules in the implementation

| Defect and severity | Priority |
| --- | --- |
| D40 pothole, high | Critical: 24 to 48 hours |
| D40 pothole, medium | High: 5 days |
| D40 pothole, low | Medium: 14 days |
| D20 alligator crack, high | High: 7 days |
| D20 alligator crack, medium | Medium: 21 days |
| D20 alligator crack, low | Routine: 45 days |
| D00 or D10, high | Medium: 14 days |
| D00 or D10, medium | Routine: 30 days |
| D00 or D10, low | Monitor: 60 days |

### 10.4 Treatment examples

| Defect | Severity | Treatment text generated by the prototype |
| --- | --- | --- |
| D40 pothole | High | Full-depth patching with saw-cut boundaries, tack coat, and 75 mm hot-mix asphalt compaction. |
| D40 pothole | Medium | Semi-permanent patching with debris removal, emulsion tack coat, and cold-mix asphalt compaction. |
| D20 alligator crack | High | Structural rehabilitation with 50 mm cold milling and 40 mm bituminous-concrete overlay. |
| D20 alligator crack | Medium | Stress-absorbing membrane interlayer and 25 mm corrective asphalt overlay. |
| D00 longitudinal crack | Medium | Air cleaning and elastomeric bitumen crack sealing. |
| D10 transverse crack | High | Crack routing and hot-poured polymer-modified bituminous sealant. |

### 10.5 Exports

The dashboard provides:

- CSV download for a tabular maintenance schedule.
- Printable HTML report that contains a municipal-style header, PCI summary, road-health rating, defect count, priority table, and sign-off block.

State that the output is a decision-support artifact. Final treatment selection still requires field inspection, site conditions, budget, traffic management, and approval by the responsible engineering authority.

---

## 11. Sample Dataset and Reproducible Demonstration Results

### 11.1 Included data

`data/sample_defects.json` contains six pre-calibrated defect records. `data/sample_gps_track.csv` contains a 21-point route from 0 to 40 seconds, with latitude, longitude, speed, and heading values.

### 11.2 Sample-defect summary

| Metric | Value |
| --- | --- |
| Total defects | 6 |
| D40 potholes | 3 |
| D20 alligator cracks | 1 |
| D00 longitudinal cracks | 1 |
| D10 transverse cracks | 1 |
| Low severity | 2 |
| Medium severity | 2 |
| High severity | 2 |
| Mean listed confidence | 0.878, or 87.8% |
| Sum of listed estimated extent | 4.00 square metres |

### 11.3 Current calculated PCI result

Running the repository's `compute_pci_score` against the six sample records with a 500 square metre sample area produces:

| Output | Value |
| --- | --- |
| Total deduct value | 108.6 |
| Corrected deduct value | 65.4 |
| PCI | 34.6 / 100 |
| Rating | Very Poor |
| Recommendation | Extensive mill-and-fill resurfacing required immediately. |

### 11.4 Sample operational interpretation

The sample dataset generates two critical high-severity pothole orders with a 24 to 48 hour response target, a high-priority medium pothole order, one medium-priority alligator-crack order, and two monitor-level crack orders. This is a strong live-demo narrative because it shows that the system distinguishes emergency hazards from defects suitable for scheduled maintenance.

---

## 12. Suggested Presentation Structure

Use this as a 16-slide main presentation. Keep speaker notes in presenter view and avoid placing all detail on the slide.

### Slide 1: Title

**On slide**

- Vision-Based Real-Time Pavement Degradation Detection and GIS System
- Smart Cities Infrastructure and Mobility
- Team members, registration numbers, department, guide/faculty name if required

**Speaker notes**

“Our project develops a road-monitoring prototype that starts with pavement imagery and ends with a map-based maintenance action. It integrates computer vision, GPS telemetry, civil-engineering condition analysis, and municipal work-order generation.”

**Visual**

Use a clear road-surface photo with visible pavement damage or a full-width screenshot of the working dashboard. Do not use a generic abstract AI background.

### Slide 2: Problem Context

**On slide**

- Road damage grows between manual inspections
- Image-only reports lack precise field location
- Defect lists do not automatically provide condition scores or repair priority

**Speaker notes**

“A road department needs more than a complaint photo. It needs location, severity, corridor-level condition, and a recommended action. Manual surveys create gaps in each of those steps.”

**Visual**

Show a simple comparison: unstructured road photo on one side and a map-linked defect record on the other.

### Slide 3: Project Objective and Scope

**On slide**

- Detect common pavement distress from road media
- Attach position and time from GPS telemetry
- Calculate a PCI-style health indicator
- Generate map layers and maintenance work orders

**Speaker notes**

“The objective is a connected workflow. We are not presenting a finished city platform; we are demonstrating the workflow and its software modules.”

### Slide 4: End-to-End Architecture

**On slide**

Use the architecture flow in Section 4.1 with six labeled stages: sensing, telemetry, detection, defect record, PCI/GIS, work order.

**Speaker notes**

“This slide is the backbone of the presentation. Every later slide expands one part of this flow.”

### Slide 5: Transdisciplinary Contribution

**On slide**

| Pillar | Project contribution |
| --- | --- |
| Telemetry | Time-tagged GPS fixes and interpolation |
| AI and computer vision | YOLO/OpenCV defect classification and localization |
| Civil engineering | Density, deduct values, PCI, maintenance meaning |
| GIS and governance | Map layers, dispatch priority, work-order export |

**Speaker notes**

“The value comes from integration. A detector alone cannot allocate maintenance resources, and a map alone cannot grade pavement condition.”

### Slide 6: Defect Classes

**On slide**

Present D00, D10, D20, and D40 using the table in Section 5. Keep each definition to one line and add representative road photographs if available.

**Speaker notes**

“D20 and D40 are especially important because they can indicate structural fatigue and an immediate hazard respectively. The class code gives the rest of the application a standardized input.”

### Slide 7: Telemetry Synchronization

**On slide**

- GPS API: `POST /gps` on port `5050`
- Frame timestamp falls between two GPS fixes
- Linear interpolation estimates location for the frame
- Dashboard overlays location, speed, and heading

Include the interpolation equation, not a dense paragraph.

**Speaker notes**

“The camera and GPS operate at different rates. Interpolation avoids assigning stale coordinates to every frame during vehicle motion.”

### Slide 8: Detection Engine

**On slide**

- Optional YOLOv8 model path for learned inference
- OpenCV fallback for the runnable base demo
- Outputs: class, confidence, bounding box, severity, estimated extent
- Confidence threshold is controlled in the sidebar

**Speaker notes**

“The fallback makes the project demonstrable without a heavyweight deep-learning install. However, the strongest accuracy claims require a formal evaluation of the trained model.”

### Slide 9: OpenCV Image-Processing Logic

**On slide**

Show a concise flow: lower road ROI, grayscale/blur, dark-region morphology for potholes, Canny edges for cracks, aspect-ratio rule for crack type.

**Speaker notes**

“This is an interpretable baseline. It relies on visual contrast and geometry, so illumination, shadows, stains, and markings may cause errors. That is why a trained and validated vision model remains important for deployment.”

### Slide 10: PCI Calculation

**On slide**

- Density = extent / sample area x 100
- Defect and severity produce a deduct value
- Multiple significant defects receive corrected aggregation
- PCI = 100 minus corrected deduct value

Include the rating table from Section 8.6.

**Speaker notes**

“PCI lets us discuss road condition on a 0 to 100 scale. The prototype uses a transparent, code-based approximation of the deduct and correction process.”

### Slide 11: GIS Portal

**On slide**

- Route polyline
- Color-coded defect markers with detail popups
- Marker clustering
- Severity-weighted heatmap
- Switchable basemap

**Speaker notes**

“The map helps a crew locate an individual defect and helps a planner identify a corridor with repeated high-severity distress.”

### Slide 12: Municipal Work Orders

**On slide**

Use a cropped screenshot of the work-order table. Highlight: defect code, location, priority, response horizon, prescribed treatment.

**Speaker notes**

“The output converts detection into operational language. A high-severity pothole becomes a critical 24 to 48 hour work item, while a low-severity crack can be monitored or scheduled.”

### Slide 13: Sample-Dataset Result

**On slide**

- 6 sample defects
- 3 potholes, 1 alligator crack, 2 crack records
- PCI: 34.6 / 100
- Rating: Very Poor
- Recommended corridor action: extensive mill-and-fill resurfacing

**Speaker notes**

“These figures are reproducible from the sample JSON currently in the repository. They are not model-accuracy metrics; they demonstrate the decision pipeline.”

### Slide 14: Live Demonstration

**On slide**

1. Vision and defect detection
2. GIS map and heatmap
3. PCI analytics
4. Work-order export

**Speaker notes**

“We will now show one evidence path from road frame to a GIS point, then to a PCI score and maintenance schedule.”

### Slide 15: Limitations and Risk Controls

**On slide**

| Current limitation | Production improvement |
| --- | --- |
| Pixel-derived extent | Camera calibration and surveyed ground truth |
| In-memory local telemetry | Secure API and persistent database |
| Prototype PCI approximation | Standards-validated tables and engineering review |
| No documented model benchmark in repo | Held-out dataset evaluation and error analysis |
| Local map demo | Road-network database and duplicate management |

**Speaker notes**

“Naming limitations increases credibility. We have a working integrated prototype and a clear path from this prototype to a deployable system.”

### Slide 16: Future Work and Closing

**On slide**

- Field-calibrated camera and GPS survey
- YOLO training/evaluation report with test data
- Secure multi-vehicle backend and database
- Temporal PCI tracking and predictive maintenance
- Mobile reporting and municipal-system integration

**Speaker notes**

“The project establishes the workflow. The next phase turns each local prototype component into a validated field and operations component.”

---

## 13. Live Demonstration Script

### 13.1 Pre-demo checklist

1. Confirm Python 3.11 and project dependencies are available.
2. Start the dashboard with `./run.sh` or `python -m streamlit run app/dashboard.py` inside the active virtual environment.
3. Open `http://localhost:8501` and wait for the initial dashboard load.
4. Keep the sample data unchanged so the PCI result remains reproducible.
5. Confirm port `5050` is free if you plan to demonstrate the GPS endpoint.
6. Keep `data/sample_defects.json` and `data/sample_gps_track.csv` in place.
7. Prepare an image or short road-video backup in case a reviewer asks about upload mode.

### 13.2 Recommended sequence

**Step A: Vision module**

- Select “Preloaded Smart City Corridor Footage.”
- Move the sequence slider to a timestamp such as 8, 12, 26, or 38 seconds, which corresponds to a sample defect timestamp.
- Point out the annotated frame, confidence threshold, HUD, and detector-engine status.

Say: “This frame shows the front-end of the analysis. The system associates visual distress with the route timestamp and displays the active inference engine.”

**Step B: Map module**

- Open “Web-GIS Map and Heatmaps.”
- Enable the route, defects, and heatmap.
- Click a marker and point to its class, severity, confidence, coordinates, and prescribed priority.

Say: “The same record has become a field location. A repair team can use the coordinates, while planners can inspect concentrations through the heatmap.”

**Step C: PCI module**

- Open “ASTM D6433 PCI Analytics.”
- Use the default 500 square metre sample area.
- Point to the 34.6 PCI score, Very Poor rating, CDV, and recommended action.

Say: “The project does not simply count six defects. It applies severity and estimated extent to produce an interpretable corridor condition score.”

**Step D: Work-order module**

- Open “Municipal Work Orders.”
- Keep the default road name or enter a demonstration road name.
- Show the critical D40 orders and their 24 to 48 hour response window.
- Download CSV or HTML only if time allows.

Say: “This is the operational handoff. The highest-severity potholes appear first as critical repair work, while low-severity cracks receive monitoring or scheduled maintenance.”

### 13.3 Optional GPS API demo

Use a separate terminal:

```bash
curl -X POST http://localhost:5050/gps \
  -H "Content-Type: application/json" \
  -d '{"latitude":12.9716,"longitude":79.1585,"speed_kmh":28.5,"heading":85.0}'
```

Then show the live-fix values in the sidebar. Explain that this demonstrates the input contract, while a production deployment would add authentication and durable storage.

---

## 14. Viva Questions and Strong Answers

### Q1. Why is GPS interpolation needed?

**Answer:** Camera frames arrive much more frequently than GPS fixes. If every frame used only the last known GPS point, the location would lag while the vehicle moves. The system estimates the coordinate between two timestamped GPS records using a linear interpolation ratio. This gives every analyzed frame a more appropriate position estimate.

### Q2. Why use YOLOv8 and OpenCV together?

**Answer:** YOLOv8 provides a learned object-detection path when the optional model dependency is available. The OpenCV path provides an interpretable fallback based on contrast, morphology, edges, and geometric rules. Combining the paths lets the prototype remain runnable while showing how a trained detector can integrate with the same downstream GIS and PCI workflow.

### Q3. How does the application classify severity?

**Answer:** In the current implementation, severity comes from bounding-box area relative to the frame. Relative area below 1.5% is low, from 1.5% to below 4.5% is medium, and 4.5% or greater is high. This is a prototype heuristic. A production system would calibrate severity using camera geometry and field-measured dimensions.

### Q4. Why not just count potholes?

**Answer:** A count cannot distinguish a minor isolated defect from an extensive high-severity defect or structural fatigue cracking. The PCI module uses distress type, severity, and density to generate a road-condition indicator that supports maintenance planning.

### Q5. What is the difference between TDV and CDV?

**Answer:** TDV is the raw sum of individual deduct values. CDV adjusts the combined effect when multiple significant distresses exist, so the final pavement penalty does not simply grow by naive addition. The final PCI is 100 minus CDV.

### Q6. Why map heatmaps as well as markers?

**Answer:** Markers identify individual repair locations. Heatmaps reveal spatial concentration. A cluster of high-severity distress can justify corridor investigation rather than repeated isolated patches.

### Q7. What does the work-order module add beyond the map?

**Answer:** The map gives spatial awareness. The work-order module assigns a deadline and recommended treatment. This bridges the gap between a technical observation and a maintenance action.

### Q8. What are the main limitations of the prototype?

**Answer:** The current project uses local, in-memory telemetry; pixel-derived extent estimates; a PCI-style approximation; and no documented model evaluation set in the repository. It is an integrated prototype, not a field-validated production deployment.

### Q9. How would you scale this to multiple vehicles?

**Answer:** Each vehicle could perform inference locally and transmit compact defect metadata rather than raw video. A secure backend would authenticate devices, store records in a spatial database, deduplicate repeated observations, and publish a central GIS dashboard. The current local HTTP receiver demonstrates the expected GPS data shape but is not yet that backend.

### Q10. How would you validate accuracy?

**Answer:** Build a labeled held-out image dataset with class and bounding-box ground truth. Measure precision, recall, mAP, confusion matrix, false positives under shadows and road markings, and latency on target hardware. Separately validate defect-size estimates and PCI outputs against field measurements and qualified pavement surveys.

---

## 15. Evidence Checklist for a Strong Presentation

### Evidence already present in the repository

- Working Streamlit dashboard source.
- Optional YOLO model weight file.
- OpenCV fallback implementation.
- Reproducible sample GPS and defect datasets.
- PCI calculation source and sample result.
- GIS and heatmap source.
- Work-order CSV and HTML generation source.
- Existing Review 2 outline and viva guide.

### Evidence to obtain before making stronger claims

| Claim you may want to make | Evidence needed |
| --- | --- |
| mAP, precision, recall | Test dataset, evaluation script, class-wise metrics, date/version of the model. |
| FPS or latency | Hardware specification, input resolution, batch size, warm-up method, repeated timing results. |
| GPS accuracy | Device, environment, reference survey, error distribution. |
| Defect-area accuracy | Calibrated camera setup and ground-truth field measurements. |
| PCI standards compliance | Documented coefficient source, sample-unit procedure, and engineering review. |
| City-scale readiness | Backend architecture, load tests, authentication, database, retention strategy. |

### Do not present without proof

The existing `Review_2_Presentation.md` mentions numerical model metrics and speed figures. The current repository does not contain the dataset split, evaluation notebook, benchmark log, or training configuration that proves those numbers. Until that evidence is added, present model evaluation as a planned validation activity rather than a completed measured result.

---

## 16. Setup Notes for the Presenter

### Base dashboard installation

```bash
git clone https://github.com/Suryadeepsinh-Jadeja/Vision-based-real-time-pavement-degradation-detection.git
cd Vision-based-real-time-pavement-degradation-detection
./run.sh
```

The current launcher expects Python 3.11 when it creates a new virtual environment. The base dependencies are listed in `requirements.txt`.

### Optional YOLO installation

```bash
source venv/bin/activate
python -m pip install -r requirements-yolo.txt
```

The YOLO package brings a larger PyTorch dependency. If the optional package does not load, the dashboard should show the OpenCV inference engine and remain usable for the demo.

### Suggested screenshots to capture before the presentation

1. Vision tab with a road frame, detection annotation, and telemetry HUD.
2. GIS tab showing route, clustered markers, and heatmap.
3. PCI tab with 34.6 score and Very Poor rating from the sample data.
4. Work-order table with the critical D40 entry visible.
5. Downloaded printable HTML work order opened in a browser.

---

## 17. Closing Statement

This project demonstrates a connected pavement-management workflow: visual road observation becomes a classified distress record, a geo-referenced location, a condition score, and a maintenance recommendation. The working prototype proves the integration of those layers. The next engineering phase is validation: calibrated geometry, measured model performance, secure persistence, and real-world fleet deployment.
