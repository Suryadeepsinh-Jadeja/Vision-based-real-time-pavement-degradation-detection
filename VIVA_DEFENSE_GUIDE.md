# Review 2 Viva Defense & Technical Q&A Guide
**Project:** Vision-Based Real-Time Pavement Degradation Detection & GIS System  
**Subject:** Smart Cities Infrastructure & Smart Mobility (Transdisciplinary)

---

## Part 1: Member-Specific Defense Questions & Model Answers

### 1. Jadeja Suryadeepsinh Bharatsinh (`23BEL1027`) — System Lead & Telemetry Integration
* **Q: How do you handle the sampling rate difference between the 30 FPS camera and the 1–5 Hz smartphone GPS?**
  * **Answer:** *"Smartphone GPS sensors typically publish fixes at 1 to 5 Hz, whereas our camera captures video at 30 frames per second. If we only tagged frames when a new GPS packet arrived, multiple frames would have identical or missing coordinates. We solved this using linear spatial-temporal interpolation. For any frame at timestamp $t$ between two GPS fixes $t_1$ and $t_2$, we compute the fractional progress and interpolate both latitude and longitude proportionally. This ensures every single defect detected is tagged with high spatial precision down to sub-meter resolution."*
* **Q: How does this system scale if hundreds of patrol vehicles or municipal buses are transmitting data simultaneously?**
  * **Answer:** *"The edge vehicles perform YOLOv8 inference locally and only transmit lightweight JSON telemetry packets containing detected defect metadata (type, bounding box, severity, GPS coordinate, timestamp) over 4G/5G HTTP REST endpoints rather than streaming heavy raw video to the cloud. This reduces bandwidth consumption by over 99%, allowing the municipal GIS server to handle thousands of concurrent survey streams effortlessly."*

---

### 2. Vishnu Nair (`23BEL1028`) — Computer Science & AI Model
* **Q: Why did you choose YOLOv8 over earlier architectures like YOLOv5 or Faster R-CNN?**
  * **Answer:** *"YOLOv8 introduces an anchor-free split-head architecture and C2f cross-stage partial bottleneck modules, which significantly improve detection accuracy for small objects like hairline cracks while maintaining real-time inference speeds of over 35 FPS on edge hardware. Two-stage detectors like Faster R-CNN are too slow for real-time mobile vehicle deployment, whereas YOLOv8 achieves an optimal trade-off with an mAP@50 of 82.4% on the RDD2022 benchmark."*
* **Q: What is the RDD2022 dataset and why is it preferred over general object datasets?**
  * **Answer:** *"The Road Damage Dataset 2022 (RDD2022) is an international benchmark containing over 47,000 multi-national road images collected from Japan, India, the US, and China. Unlike general datasets, it specifically categorizes pavement distress into four standard classes: D00 (longitudinal cracks), D10 (transverse cracks), D20 (alligator cracking), and D40 (potholes), making our model directly compliant with civil engineering diagnostic guidelines."*

---

### 3. Udayvardhan Singh Rathore (`23BEL1018`) — Civil Engineering & ASTM D6433 PCI Engine
* **Q: What is the Pavement Condition Index (PCI) and why can't we just count the number of potholes?**
  * **Answer:** *"Simply counting potholes ignores the structural health of the underlying road layers. The Pavement Condition Index (PCI), standardized by ASTM D6433, is a civil engineering metric from 0 to 100 that quantifies overall pavement integrity. It considers defect density, severity levels (Low, Medium, High), and distress types. For instance, alligator cracking (D20) indicates structural fatigue failure of the road base, which receives a higher deduct penalty than surface thermal cracks. PCI allows the city to practice preventive maintenance before roads suffer catastrophic collapse."*
* **Q: How do you calculate the Corrected Deduct Value (CDV) from individual Deduct Values?**
  * **Answer:** *"When multiple defects occur in the same road section, simply summing their individual Deduct Values (Total Deduct Value, TDV) would overestimate the damage and cause the score to drop below zero. ASTM D6433 uses an iterative correction method using factor $q$, which represents the number of individual deduct values exceeding 5.0. The CDV curve dampens the cumulative impact of secondary minor distresses, ensuring an accurate, bounded score where $\text{PCI} = 100 - CDV$."*

---

### 4. Ainessh kumar S (`23BEL1039`) — Geospatial Analysis & Web-GIS
* **Q: How does the Folium Web-GIS module benefit urban planners compared to a static spreadsheet?**
  * **Answer:** *"A spreadsheet only lists tabular coordinates, which does not allow municipal authorities to visualize spatial trends. Our interactive Folium Web-GIS platform renders dynamic color-coded road polylines reflecting real-time PCI condition ratings, marker clusters with defect snapshots, and Kernel Density Estimation (KDE) heatmaps. This enables city engineers to instantly identify high-density pothole clusters and prioritize contractor road resurfacing contracts by zone."*
* **Q: What coordinate reference system (CRS) are you using?**
  * **Answer:** *"We capture coordinates in WGS84 (EPSG:4326), the global GPS standard. When calculating real-world Euclidean distances between successive defects or estimating square-meter pavement sample areas, we project the coordinates into UTM (Universal Transverse Mercator, EPSG:32643/44) to avoid planar distortion and maintain metric precision."*

---

### 5. Shreyan Biswas (`23BEL1052`) — Urban Governance & Municipal Work Orders
* **Q: How does your work order module align with standard Public Works Department (PWD) practices?**
  * **Answer:** *"Our module bridges the gap between AI detection and actual ground operations. Rather than just giving alerts, it automatically generates official work orders that adhere to Indian Road Congress guidelines (IRC:82-2015). It assigns priority levels (Emergency 24–48 hours for deep potholes, vs. scheduled 30-day crack sealing) and prescribes exact engineering repair treatments—such as full-depth asphalt patching for D40 or bituminous concrete overlays for D20—with exportable CSVs and printable verified work orders."*
* **Q: How does this project save municipal budget in a Smart City?**
  * **Answer:** *"Traditional manual road inspections in smart cities cost thousands of dollars per kilometer and are only performed once or twice a year. By deploying our system on existing government patrol vehicles or public buses, road health is monitored daily at near-zero incremental operational cost. Early detection of hairline cracks allows timely crack sealing ($1–2 per meter), preventing them from turning into severe potholes and base failures that require full road reconstruction ($30–50 per square meter)."*

---

## Part 2: Quick Cross-Disciplinary Reference Table

| Question Focus | Core Concept | Primary Speaker |
| :--- | :--- | :--- |
| Video & GPS Telemetry Sync | Linear Temporal Interpolation / Local HTTP Server | **Suryadeepsinh** |
| Deep Learning Model & Dataset | YOLOv8 / RDD2022 Schema / mAP@50 | **Vishnu** |
| Road Health Calculation | ASTM D6433 Standard / Deduct Values / PCI Formula | **Udayvardhan** |
| GIS Mapping & Heatmaps | Folium / Leaflet / Kernel Density Estimation | **Ainessh** |
| PWD Repair Treatments & ROI | IRC:82-2015 Code / Preventive Maintenance Savings | **Shreyan** |
