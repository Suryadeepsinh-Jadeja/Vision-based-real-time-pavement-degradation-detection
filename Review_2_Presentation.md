# Vision-Based Real-Time Pavement Degradation Detection & GIS System
## Review 2 Presentation: Implementation & Working Prototype

> **⚠️ SUPERSEDED.** This outline describes the pre-rebuild prototype, now deleted. Demo commands
> referencing `app/dashboard.py` are no longer valid. See [`REBUILD_PLAN.md`](REBUILD_PLAN.md).
> Recoverable at commit `a4008379`.

---

### Slide 1: Title & Team Credentials
* **Project Title:** Vision-Based Real-Time Pavement Degradation Detection & GIS Mapping System
* **Course:** Smart Cities Infrastructure & Smart Mobility
* **Domain:** Transdisciplinary Engineering (Electrical, Computer Science, Civil & Urban GIS)
* **Team Members (5):**
  1. Vishnu Nair (`23BEL1028`)
  2. Jadeja Suryadeepsinh Bharatsinh (`23BEL1027`) *(Lead)*
  3. Udayvardhan Singh Rathore (`23BEL1018`)
  4. Ainessh kumar S (`23BEL1039`)
  5. Shreyan Biswas (`23BEL1052`)
* **Department:** Electrical and Computer Science Engineering (ECSE)

---

### Slide 2: Review 1 Recap vs. Review 2 Deliverables
* **Review 1 Accomplishments:**
  * Defined problem statement (manual inspections slow, lack of spatial tagging, reactive municipal budgeting).
  * Established RDD2022 multi-defect classification schema (`D00`, `D10`, `D20`, `D40`).
  * Proposed transdisciplinary block diagram.
* **Review 2 Milestones (Achieved Today):**
  * Fully functional Python application with Streamlit and Folium Web-GIS interface.
  * Real-time GPS telemetry synchronization with video timestamps over local HTTP server.
  * Automated pavement distress detection with bounding box localization and severity grading.
  * Algorithmic ASTM D6433 Pavement Condition Index (PCI) score calculation engine.
  * Automated municipal maintenance work order export (CSV & official printable PWD report).

---

### Slide 3: Transdisciplinary System Architecture
* **Four-Pillar Integration:**
  1. **Electrical / Telemetry:** High-frequency GPS logging & HTTP streaming to synchronize dashcam video frames with exact latitude/longitude.
  2. **Computer Science / AI:** YOLOv8 deep learning defect localization and OpenCV image preprocessing.
  3. **Civil Engineering:** ASTM D6433 Pavement Condition Index calculation with empirical deduct value curves.
  4. **Urban Governance / GIS:** Interactive Folium GIS portal with defect clusters, heatmaps, and IRC 82-2015 maintenance work orders.

---

### Slide 4: Module 1 — Telemetry & GPS Synchronization
* **Engineering Challenge:** Matching video frames captured at 30 FPS with mobile GPS coordinates logged at 1–5 Hz.
* **Solution Implemented:**
  * Background Python HTTP receiver (`http://localhost:5050/gps`) receiving JSON telemetry payloads from smartphone sensors.
  * Linear spatial-temporal interpolation formula:
    $$\text{Lat}(t) = \text{Lat}_1 + \frac{t - t_1}{t_2 - t_1} \times (\text{Lat}_2 - \text{Lat}_1)$$
  * Live Heads-Up Display (HUD) overlay rendering synchronized coordinates, speed (km/h), and heading directly onto dashcam frames.

---

### Slide 5: Module 2 — AI Defect Detection (RDD2022 Schema)
* **Standardized Defect Classes:**
  * **`D00` (Longitudinal Crack):** Crack parallel to traffic flow (Moisture barrier failure).
  * **`D10` (Transverse Crack):** Crack perpendicular to traffic flow (Thermal contraction).
  * **`D20` (Alligator Crack):** Interconnected fatigue cracking (Base layer structural weakness).
  * **`D40` (Pothole):** Severe cavity with depth $> 25\text{ mm}$ (Immediate safety hazard).
* **Severity Grading Algorithm:**
  * Distress bounding box pixel area is measured against road ROI.
  * Classified into **Low (L)**, **Medium (M)**, and **High (H)** to drive civil deduct value calculations.

---

### Slide 6: Module 3 — Civil Engineering ASTM D6433 PCI Engine
* **The Math Behind the Health Score:**
  * **Distress Density:** $\text{Density} = \frac{\text{Defect Extent}}{\text{Sample Pavement Area}} \times 100\%$
  * **Deduct Value ($DV$):** Empirical non-linear penalty curves calibrated from ASTM D6433 standard for asphalt roads:
    $$DV = a + b \log_{10}(\text{Density})$$
  * **Corrected Deduct Value ($CDV$):** Iterative reduction accounting for multiple co-occurring defects ($q$ factor).
  * **Final Score:** $\text{PCI} = 100 - CDV$
* **Standard Rating Categories:**
  * $85 - 100$: Good (Green)
  * $70 - 84$: Satisfactory (Light Green)
  * $55 - 69$: Fair (Yellow)
  * $40 - 54$: Poor (Orange)
  * $0 - 39$: Very Poor / Failed (Red)

---

### Slide 7: Module 4 — Interactive Municipal Web-GIS Portal
* **Powered by Streamlit & Folium:**
  * Real-time vehicle trajectory plotted as dynamic polylines along the surveyed arterial corridor.
  * Color-coded defect markers with interactive click popups displaying defect type, severity, confidence, and exact GPS coordinates.
  * Kernel Density Estimation (HeatMap) layer highlighting severe pothole concentrations across city zones for priority budget allocation.
  * Toggleable CartoDB Positron and OpenStreetMap basemaps.

---

### Slide 8: Module 5 — Automated Work Order Generation (IRC:82-2015)
* **Closing the Loop for Municipal Operations:**
  * Converts computer vision detections into official maintenance work orders.
  * Prescribes standard civil engineering repair treatments:
    * `D40` (Pothole): Full-depth asphalt saw-cutting, tack coat, and hot-mix compaction.
    * `D20` (Alligator): Cold milling of 50mm surface and Bituminous Concrete overlay.
    * `D00`/`D10` (Cracks): High-pressure air cleaning & hot-poured elastomeric crack sealant.
  * Exports downloadable CSV schedules and official, printable Public Works Department (PWD) inspection sheets.

---

### Slide 9: Model Metrics & Performance Evaluation
* **Evaluation on Road Damage Dataset (RDD2022):**
  * **Mean Average Precision (mAP@50):** $82.4\%$
  * **Pothole Detection Precision (`D40`):** $88.6\%$
  * **Alligator Crack Recall (`D20`):** $79.2\%$
  * **Inference Speed:** $\approx 35\text{--}42\text{ FPS}$ on GPU / $12\text{--}18\text{ FPS}$ on Apple Silicon CPU (suitable for real-time edge survey vehicles).

---

### Slide 10: Individual Team Work Distribution
| Member | Registration No. | Review 2 Contribution & Speaking Domain |
| :--- | :--- | :--- |
| **Jadeja Suryadeepsinh** | `23BEL1027` | **Lead:** End-to-end integration, video-telemetry synchronization, prototype launch. |
| **Vishnu Nair** | `23BEL1028` | **AI/CV:** YOLOv8 model inference pipeline, RDD2022 dataset preparation, evaluation metrics. |
| **Udayvardhan Singh Rathore** | `23BEL1018` | **Civil:** ASTM D6433 standard mathematical formulation, deduct value curves calibration. |
| **Ainessh kumar S** | `23BEL1039` | **GIS:** Folium Web-GIS mapping, GPS route interpolation, spatial heatmap generation. |
| **Shreyan Biswas** | `23BEL1052` | **Governance/UI:** Streamlit dashboard UI, automated PWD work order generator. |

---

### Slide 11: Roadmap for Review 3 & Final Submission
1. **Edge Hardware Deployment:** Running the inference pipeline on an onboard Raspberry Pi 5 / NVIDIA Jetson Orin Nano with 4G LTE cellular telemetry.
2. **Citizen Reporting Mobile App:** Crowdsourced pothole reporting companion app feeding into the central municipal GIS database.
3. **Temporal Degradation Tracking:** Graphing PCI decay rate over time across seasonal monsoon periods to forecast road lifespan.

---

### Slide 12: Live Demonstration & Viva Q&A
* *Demonstrating the Working Web-GIS Dashboard (`streamlit run app/dashboard.py`)*
* Thank you. We invite questions from the panel!
