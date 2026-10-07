# AI-Based Pothole Detection and GIS Mapping System
**Transdisciplinary Smart City Infrastructure & Mobility Project**

> **⚠️ SUPERSEDED SPECIFICATION.** This is the original design spec, retained for history and
> viva reference. §6 "How to Run the Working Prototype" no longer applies — the prototype it
> describes has been deleted. The replacement architecture is in
> [`REBUILD_PLAN.md`](REBUILD_PLAN.md). Recoverable at commit `a4008379`.

---

## 1. Executive Summary & Problem Statement

### 1.1 Project Overview
This project presents an open-source, low-cost, AI-driven road maintenance and infrastructure monitoring system. By fusing computer vision, mobile telemetry, and civil engineering standards, the system automates pavement distress detection and visualizes real-time health metrics on an interactive municipal GIS dashboard.

### 1.2 Core Problem Statement
* **Manual & Inefficient Surveys:** Traditional road inspections rely on manual human checks, which are slow, expensive, highly subjective, and miss critical damage until accidents occur.
* **Lack of Spatial Data:** Raw video or photos alone fail to provide precise, real-time GPS locations, making it difficult for field crews to pinpoint exact repair spots.
* **Unorganized Repair Prioritization:** Municipalities lack automated tools to measure actual damage severity (such as Pavement Condition Index scores), leading to reactive maintenance instead of proactive data-driven scheduling.

---

## 2. Transdisciplinary Architecture & System Design

The system integrates four distinct engineering domains:
1. **Electrical / Telemetry:** Real-time smartphone GPS data streaming over local HTTP/Wi-Fi to pair physical coordinates with video frames.
2. **Computer Science / AI:** YOLOv8 deep learning inference and OpenCV image processing pipelines conforming to the international RDD2022 schema.
3. **Civil Engineering:** Standardized Pavement Condition Index (PCI) calculations conforming to **ASTM D6433** guidelines and remediation per **IRC 82-2015**.
4. **Urban Governance / GIS:** Interactive web-based mapping via **Streamlit and Folium**, automated spatial heatmaps, and municipal work order export.

### 2.1 High-Level System Architecture Flow

```
+-----------------------------------------------------------------------------------+
|                            MOBILE SENSING LAYER                                   |
|   +------------------------------------+    +---------------------------------+   |
|   | 1080p/4K Dashcam Video Stream      |    | Smartphone GPS Telemetry        |   |
|   | (FPS: 30, Vehicle Mounted)         |    | (Lat, Lon, Speed, Bearing)      |   |
|   +-----------------+------------------+    +----------------+----------------+   |
+---------------------|----------------------------------------|--------------------+
                      |                                        |
                      v                                        v
+-----------------------------------------------------------------------------------+
|                        SYNCHRONIZATION & PREPROCESSING                            |
|   - Video Frame Extraction (OpenCV at 2-5 FPS sampling)                           |
|   - Sub-second Timestamp Alignment (Linear Spatial-Temporal Interpolation)         |
+---------------------------------------------+-------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                          AI COMPUTER VISION INFERENCE                             |
|   - YOLOv8 Object Detection Pipeline (RDD2022 Schema)                             |
|       * D00: Longitudinal Crack                                                   |
|       * D10: Transverse Crack                                                     |
|       * D20: Alligator Crack (Structural Fatigue)                                 |
|       * D40: Pothole (Severe Cavity)                                              |
|   - Geometric Bounding Box & Pavement Area Estimation ($cm^2$, $m^2$)             |
|   - Severity Classification: Low (L), Medium (M), High (H)                        |
+---------------------------------------------+-------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                      CIVIL ENGINEERING PCI SCORING ENGINE                         |
|   - ASTM D6433-20 Standard Pavement Condition Index Calculation                   |
|   - Defect Density Computation: Density = (Distress Area / Sample Area) * 100     |
|   - Deduct Value (DV) Curve Evaluation                                            |
|   - Iterative Corrected Deduct Value (CDV) Calculation: PCI = 100 - CDV           |
|   - Condition Grading: Good (85-100), Fair (55-84), Poor (<55)                    |
+---------------------------------------------+-------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------+
|                      MUNICIPAL GEOPORTAL & WORK ORDERS                            |
|   - Interactive Folium Web-GIS with CartoDB Positron / OSM Basemaps               |
|   - Defect Spatial Clustering & Kernel Density Heatmaps                           |
|   - Color-Coded Pavement Condition Polylines along Survey Corridor                |
|   - Automated Work Order Generation (CSV & Printable Official HTML/PDF)           |
+-----------------------------------------------------------------------------------+
```

---

## 3. Road Damage Categorization (RDD2022 Schema)

| Defect Code | Distress Classification | Visual Characteristics | Structural Threat Level |
| :--- | :--- | :--- | :--- |
| **`D00`** | Longitudinal Crack | Single line crack running parallel to vehicle path | Low - Medium (Moisture ingress) |
| **`D10`** | Transverse Crack | Crack perpendicular to traffic direction | Low - Medium (Thermal expansion) |
| **`D20`** | Alligator Crack | Interconnected polygonal pattern ("crocodile skin") | High (Sub-base structural fatigue) |
| **`D40`** | Pothole | Bowl-shaped depression, minimum 150mm diameter | Critical (Accident & vehicle damage hazard) |

---

## 4. Civil Engineering Metrics: ASTM D6433 Pavement Condition Index

### 4.1 Methodology
1. **Sample Unit Definition:** Survey corridor is divided into sample units of nominal area $A_s = 250 - 500\text{ m}^2$.
2. **Distress Density Calculation:**
   $$\text{Density} (\%) = \frac{\text{Distress Quantity}}{\text{Sample Unit Area}} \times 100$$
3. **Deduct Value ($DV$):**
   Calculated from empirical ASTM polynomial curves for each defect type and severity:
   $$DV = a + b \log_{10}(\text{Density})$$
4. **Corrected Deduct Value ($CDV$):**
   Diminishes cumulative deduct values when multiple distresses co-occur:
   $$CDV = f(\text{TDV}, q)$$
   where $\text{TDV} = \sum DV_i$ and $q$ is the count of deduct values $> 5.0$.
5. **Final PCI Score:**
   $$\text{PCI} = 100 - CDV$$

### 4.2 Rating Scale & Municipal Actions (ASTM D6433 / IRC 82-2015)
* **$85 - 100$ (Good):** Preventive routine inspection, minor crack sealing.
* **$70 - 84$ (Satisfactory):** Surface seal coat, targeted crack pouring.
* **$55 - 69$ (Fair):** Cold patching of isolated potholes, slurry seal overlay.
* **$40 - 54$ (Poor):** Full-depth patching, corrective 25mm bituminous concrete overlay.
* **$0 - 39$ (Very Poor to Failed):** Immediate mill-and-fill or complete structural reconstruction.

---

## 5. Team Work Breakdown (5 Members)

| Team Member | Registration No. | Engineering Pillar | Review 2 Deliverables & Viva Role |
| :--- | :--- | :--- | :--- |
| **Jadeja Suryadeepsinh Bharatsinh** *(Lead)* | `23BEL1027` | **System Lead / Integration** | Overall pipeline orchestration, video frame synchronization with GPS telemetry, prototype execution. |
| **Vishnu Nair** | `23BEL1028` | **Computer Science / AI** | YOLOv8 model training/inference, RDD2022 dataset preparation, detection metrics (mAP, Precision, Recall). |
| **Udayvardhan Singh Rathore** | `23BEL1018` | **Civil Engineering** | ASTM D6433 standard formulation, deduct value curves calibration, PCI score calculation engine. |
| **Ainessh kumar S** | `23BEL1039` | **Geospatial & Web-GIS** | Folium Web-GIS mapping, GPS route interpolation, spatial clustering and defect heatmaps. |
| **Shreyan Biswas** | `23BEL1052` | **Urban Governance / UI** | Streamlit municipal dashboard UI, CSV/HTML work order generator conforming to PWD standards. |

---

## 6. How to Run the Working Prototype

```bash
# 1. Navigate to the project directory
cd "/Users/suryadeepsinhjadeja/Desktop/Smart Cities"

# 2. Run the automated launcher script
./run.sh

# Alternatively, manual run:
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```