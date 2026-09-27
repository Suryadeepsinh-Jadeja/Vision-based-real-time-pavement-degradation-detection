# Vision-Based Real-Time Pavement Degradation Detection & GIS System
**Transdisciplinary Smart City Infrastructure & Mobility Project**  
*Review 2 Working Prototype & Implementation*

---

## 📌 Project Overview
This repository contains the complete implementation of an AI-driven, transdisciplinary road monitoring platform. The system bridges mobile telemetry, computer vision, civil engineering pavement metrics (ASTM D6433 / IRC 82-2015), and municipal GIS mapping.

### Team Members (5)
* **Vishnu Nair** (`23BEL1028`) — AI / Computer Vision Lead
* **Jadeja Suryadeepsinh Bharatsinh** (`23BEL1027`) — Project Lead & System Orchestrator
* **Udayvardhan Singh Rathore** (`23BEL1018`) — Civil Engineering Standards & PCI Lead
* **Ainessh kumar S** (`23BEL1039`) — Geospatial Analysis & Web-GIS Lead
* **Shreyan Biswas** (`23BEL1052`) — Urban Governance & Municipal Work Order Lead

---

## 🚀 Quick Start Guide

### Option 1: Automated Launch (Recommended)
Open your terminal in this directory and execute:
```bash
./run.sh
```
This script will automatically configure your Python environment, install required libraries, and launch the Web-GIS dashboard at `http://localhost:8501`.

### Option 2: Manual Launch
```bash
# 1. Activate Python virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Municipal Geoportal
streamlit run app/dashboard.py
```

---

## 📂 Repository File Structure

```
├── app/
│   ├── __init__.py
│   ├── dashboard.py         # Streamlit Web-GIS Municipal Dashboard
│   ├── detector.py          # YOLOv8 & OpenCV RDD2022 Defect Detector
│   ├── pci_engine.py        # ASTM D6433 Civil Pavement Condition Index Engine
│   ├── telemetry.py         # Smartphone GPS HTTP Receiver & Interpolator
│   └── work_order.py        # Municipal PWD Maintenance Work Order Generator
├── data/
│   ├── sample_gps_track.csv # Pre-synchronized smart city survey corridor GPS
│   └── sample_defects.json  # Pre-calibrated RDD2022 road defect detections
├── run.sh                   # One-click execution script
├── requirements.txt         # Project dependencies
├── new.md                   # Updated Transdisciplinary System Specification
├── Review_2_Presentation.md # Slide-by-slide presentation deck with talking points
├── VIVA_DEFENSE_GUIDE.md    # Viva defense questions & answers for all 5 team members
└── smart ppt.pdf            # Original Review 1 presentation
```

---

## 🔬 Transdisciplinary Modules

1. **Electrical / Telemetry (`app/telemetry.py`):**
   * Real-time HTTP listener on `http://localhost:5050/gps` receiving smartphone GPS fixes.
   * Linear temporal interpolation matching 30 FPS camera frames with 1–5 Hz GPS sensors.
2. **Computer Science / AI (`app/detector.py`):**
   * Object detection conforming to RDD2022 international schema: `D00` (Longitudinal), `D10` (Transverse), `D20` (Alligator), `D40` (Pothole).
   * Automatic bounding box pixel-to-metric area estimation and severity classification (`Low`, `Medium`, `High`).
3. **Civil Engineering (`app/pci_engine.py`):**
   * Strict adherence to **ASTM D6433-20** Pavement Condition Index formulation.
   * Empirical Deduct Value ($DV$) non-linear curve evaluation and iterative Corrected Deduct Value ($CDV$) reduction.
4. **Urban Governance / GIS (`app/dashboard.py` & `app/work_order.py`):**
   * Interactive Folium map with defect marker clusters, vehicle route polyline, and Kernel Density Estimation heatmaps.
   * Automated Public Works Department (PWD) repair tickets following Indian Road Congress (**IRC:82-2015**) standards.
