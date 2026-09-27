# Vision-Based Real-Time Pavement Degradation Detection and GIS System

An AI-assisted smart-city road-monitoring platform that brings together pavement-defect detection, GPS telemetry, ASTM D6433 pavement-condition analysis, interactive GIS mapping, and municipal repair work-order generation.

The application includes sample survey data and can run immediately in its OpenCV-based detection mode. An included YOLOv8 weight file can be enabled with one optional dependency install.

## What It Does

| Area | Capability |
| --- | --- |
| Vision | Identifies RDD2022-style longitudinal cracks (D00), transverse cracks (D10), alligator cracks (D20), and potholes (D40). |
| Telemetry | Receives live GPS fixes over HTTP and synchronizes a route with survey timestamps. |
| Civil analytics | Calculates a Pavement Condition Index (PCI) using an ASTM D6433-inspired deduct-value workflow. |
| GIS | Shows survey routes, defect markers, clusters, and heatmaps in an interactive map. |
| Municipal operations | Produces repair-priority work orders based on defect type and severity. |

## Requirements

- Git
- Python 3.11 (the current launcher creates its environment with `python3.11`)
- `pip`, which is installed with standard Python distributions

The project is currently validated with Python 3.11. Newer Python versions may work, but the computer-vision and YOLO dependencies have not yet been tested across them. Check your installed version with:

```bash
python3.11 --version
```

On Windows, use `py -3.11` in place of `python3.11` in the commands below.

## Quick Start

### macOS and Linux

```bash
git clone https://github.com/Suryadeepsinh-Jadeja/Vision-based-real-time-pavement-degradation-detection.git
cd Vision-based-real-time-pavement-degradation-detection
./run.sh
```

The launcher creates `venv/` when needed, installs the base dashboard dependencies, and starts the application. Open [http://localhost:8501](http://localhost:8501) in a browser. Stop the server with `Ctrl+C`.

If the shell reports a permission error, make the launcher executable once:

```bash
chmod +x run.sh
./run.sh
```

### Windows or Manual Setup

```powershell
git clone https://github.com/Suryadeepsinh-Jadeja/Vision-based-real-time-pavement-degradation-detection.git
cd Vision-based-real-time-pavement-degradation-detection
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

If PowerShell blocks activation, run this for the current shell and retry the activation command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

For a Unix-like manual setup, use:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

## Enable YOLOv8 Detection (Optional)

The base installation runs the dashboard using its built-in OpenCV pavement-feature detector. To enable the included `models/pothole_yolov8.pt` weights, install the optional YOLO dependency after the base setup:

```bash
python -m pip install -r requirements-yolo.txt
```

This installs Ultralytics and its PyTorch dependency, which can require a substantial download. On the next application start, the sidebar should report `YOLOv8 Deep Learning (pothole_yolov8.pt)`. If it instead reports the OpenCV engine, the dashboard remains usable; check the terminal output for the model-loading error.

## Try the Demo

1. Start the dashboard and leave **Preloaded Smart City Corridor Footage** selected.
2. Move the sequence playback slider to inspect a synthetic road frame with synchronized sample GPS data and detected distress annotations.
3. Open **Web-GIS Map and Heatmaps** to inspect the sample route and defect locations.
4. Open **ASTM D6433 PCI Analytics** to view the calculated pavement score and its defect breakdown.
5. Open **Municipal Work Orders** to generate repair priorities and export work-order output.

You can also select **Upload Road Video / Image** in the vision module. Supported image formats are JPG and PNG; supported video formats are MP4, MOV, and AVI.

## Live GPS Telemetry

Starting the dashboard also starts a local telemetry server on port `5050`. Send a JSON `POST` request to `http://localhost:5050/gps` from a mobile app, GPS logger, or another local client.

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

`speed` is also accepted in place of `speed_kmh`, and `bearing` is accepted in place of `heading`. A `GET` request to the same endpoint returns the latest received fix.

Example using `curl`:

```bash
curl -X POST http://localhost:5050/gps \
  -H "Content-Type: application/json" \
  -d '{"latitude":12.9716,"longitude":79.1585,"speed_kmh":28.5,"heading":85.0}'
```

The telemetry server binds to all local network interfaces. Treat it as a development/demo endpoint; it does not provide authentication or persistent storage.

## Project Structure

```text
app/
  dashboard.py       Streamlit dashboard and user workflow
  detector.py        YOLOv8/OpenCV pavement-defect detection
  pci_engine.py      Pavement Condition Index calculation
  telemetry.py       GPS receiver, storage, and route interpolation
  work_order.py      Municipal maintenance work-order generation
data/
  sample_defects.json
  sample_gps_track.csv
models/
  pothole_yolov8.pt  Included optional YOLOv8 weights
requirements.txt     Base dashboard dependencies
requirements-yolo.txt Optional YOLOv8 dependency
run.sh               macOS/Linux launcher
```

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `python3.11: command not found` | Install Python 3.11, then rerun the setup command. |
| `Address already in use` | Stop the process using port `8501` or `5050`, then restart the dashboard. |
| Dashboard opens but YOLO is inactive | Run `python -m pip install -r requirements-yolo.txt`, then restart. The OpenCV fallback is expected without this optional package. |
| Map tiles do not load | Confirm the browser has internet access; Folium map tiles are fetched from their external provider. |
| A dependency installation fails | Upgrade pip with `python -m pip install --upgrade pip`, confirm Python 3.11 is active, and rerun the installation command. |

## Team

- Vishnu Nair (`23BEL1028`) - AI and computer vision
- Jadeja Suryadeepsinh Bharatsinh (`23BEL1027`) - project lead and system orchestration
- Udayvardhan Singh Rathore (`23BEL1018`) - civil engineering standards and PCI
- Ainessh Kumar S (`23BEL1039`) - geospatial analysis and Web-GIS
- Shreyan Biswas (`23BEL1052`) - urban governance and municipal work orders

## Supporting Material

- `new.md`: system specification
- `Review_2_Presentation.md`: presentation outline and speaker notes
- `VIVA_DEFENSE_GUIDE.md`: review and viva preparation material
- `smart ppt.pdf`: earlier presentation
