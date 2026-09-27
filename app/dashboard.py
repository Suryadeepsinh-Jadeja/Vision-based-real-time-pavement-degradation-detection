"""
Smart City Pavement Degradation Detection & GIS Mapping Dashboard
================================================================
Transdisciplinary Review 2 Working Prototype
Integrating:
1. Electrical / Telemetry (Smartphone GPS & Video Timestamp Matching)
2. AI / Vision (YOLOv8 & OpenCV Pavement Distress Detection - RDD2022)
3. Civil Engineering (ASTM D6433 Pavement Condition Index Calculation)
4. Urban Governance / GIS (Interactive Folium Spatial Mapping & Municipal Work Orders)
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path regardless of where streamlit is executed from
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Also add the app directory itself to sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import json
import time
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import cv2
from PIL import Image

# Import internal modular engines (with fallback to direct imports)
try:
    from app.pci_engine import compute_pci_score, RDD_DISTRESS_MAP
    from app.detector import RoadDamageDetector, annotate_frame, CLASS_COLORS
    from app.telemetry import (
        GLOBAL_TELEMETRY, 
        start_telemetry_server, 
        interpolate_gps, 
        generate_simulated_route
    )
    from app.work_order import generate_work_orders, generate_html_work_order
except ImportError:
    from pci_engine import compute_pci_score, RDD_DISTRESS_MAP
    from detector import RoadDamageDetector, annotate_frame, CLASS_COLORS
    from telemetry import (
        GLOBAL_TELEMETRY, 
        start_telemetry_server, 
        interpolate_gps, 
        generate_simulated_route
    )
    from work_order import generate_work_orders, generate_html_work_order

# Streamlit Page Configuration
st.set_page_config(
    page_title="Pavement Degradation Detection & GIS Platform",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Start background GPS receiver server once
@st.cache_resource
def init_telemetry_server():
    server = start_telemetry_server(port=5050)
    return server

init_telemetry_server()

# Custom CSS for UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 18px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        font-weight: 600;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        padding-top: 8px;
        padding-bottom: 8px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# App State Initialization
if "survey_defects" not in st.session_state:
    # Load default pre-calibrated defects
    sample_file = "data/sample_defects.json"
    if os.path.exists(sample_file):
        with open(sample_file, "r") as f:
            st.session_state.survey_defects = json.load(f)
    else:
        st.session_state.survey_defects = []

if "gps_track" not in st.session_state:
    sample_track_file = "data/sample_gps_track.csv"
    if os.path.exists(sample_track_file):
        st.session_state.gps_track = pd.read_csv(sample_track_file).to_dict(orient="records")
    else:
        st.session_state.gps_track = generate_simulated_route()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/road.png", width=64)
st.sidebar.title("Survey Configuration")

nav_choice = st.sidebar.radio(
    "Module Navigation",
    [
        "📹 Vision & Defect Detection",
        "🗺️ Web-GIS Map & Heatmaps",
        "📊 ASTM D6433 PCI Analytics",
        "📋 Municipal Work Orders",
        "👥 Team Roles & Review 2 Spec"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("AI Inference Engine")

detector = RoadDamageDetector()
if detector.use_yolo:
    st.sidebar.success(f"🟢 **Deep Learning Active**\n`{detector.engine_name}`")
else:
    st.sidebar.info(f"⚙️ **Computer Vision Active**\n`{detector.engine_name}`")

conf_thresh = st.sidebar.slider("Detection Confidence Threshold", 0.20, 0.90, 0.35, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Physical Telemetry Stream")
st.sidebar.markdown(f"**GPS Server:** `http://localhost:5050/gps`")
latest_t = GLOBAL_TELEMETRY.get_latest()
st.sidebar.markdown(f"**Live Fix:** `{latest_t['latitude']:.5f}, {latest_t['longitude']:.5f}`")
st.sidebar.markdown(f"**Current Speed:** `{latest_t['speed_kmh']} km/h`")

# Header Banner
st.markdown("<div class='main-header'>Vision-Based Real-Time Pavement Degradation Detection & GIS System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Transdisciplinary Smart Infrastructure Monitoring Platform | Review 2 Working Prototype</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 1: VISION & DEFECT DETECTION PIPELINE
# -------------------------------------------------------------
if nav_choice == "📹 Vision & Defect Detection":
    st.subheader("1. AI Computer Vision & Telemetry Ingestion Pipeline")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**Dashcam Video / Frame Ingestion**")
        input_source = st.radio("Input Source", ["Preloaded Smart City Corridor Footage", "Upload Road Video / Image"], horizontal=True)
        
        detector = RoadDamageDetector()
        
        if input_source == "Preloaded Smart City Corridor Footage":
            st.info("Demonstrating pre-synchronized test sequence: Corridor A-1 (Main Arterial Road)")
            
            # Interactive frame step slider
            step = st.slider("Corridor Sequence Playback (Frame / Time Step)", 0, len(st.session_state.gps_track)-1, 4)
            current_gps = st.session_state.gps_track[step]
            
            # Create synthetic realistic dashcam frame with road and defects
            canvas_w, canvas_h = 720, 480
            frame = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 80 # Asphalt dark gray
            
            # Draw perspective road markings
            cv2.line(frame, (int(canvas_w*0.1), canvas_h), (int(canvas_w*0.35), int(canvas_h*0.4)), (200, 200, 200), 4) # left curb
            cv2.line(frame, (int(canvas_w*0.9), canvas_h), (int(canvas_w*0.65), int(canvas_h*0.4)), (200, 200, 200), 4) # right curb
            # Center dashed line
            for dash_y in range(int(canvas_h*0.42), canvas_h, 45):
                cv2.line(frame, (int(canvas_w*0.5), dash_y), (int(canvas_w*0.5), dash_y + 25), (255, 230, 0), 3)

            # Draw defects if any correspond to this timestamp
            matched_defects = [d for d in st.session_state.survey_defects if abs(d.get("time_sec", -10) - current_gps.get("time_sec", 0)) <= 3.0]
            
            for md in matched_defects:
                code = md.get("class", "D40")
                if code == "D40": # Draw pothole depression
                    cv2.ellipse(frame, (int(canvas_w*0.45), int(canvas_h*0.75)), (55, 32), 15, 0, 360, (25, 25, 25), -1)
                    cv2.ellipse(frame, (int(canvas_w*0.45), int(canvas_h*0.75)), (58, 35), 15, 0, 360, (40, 40, 40), 2)
                elif code == "D20": # Alligator crack pattern
                    cx, cy = int(canvas_w*0.62), int(canvas_h*0.68)
                    pts = np.array([[cx, cy], [cx+40, cy-15], [cx+70, cy+10], [cx+30, cy+35], [cx-10, cy+20]], np.int32)
                    cv2.polylines(frame, [pts], True, (30, 30, 30), 2)
                    cv2.line(frame, (cx+15, cy-5), (cx+30, cy+35), (30, 30, 30), 2)
                else: # Crack
                    cv2.line(frame, (int(canvas_w*0.35), int(canvas_h*0.6)), (int(canvas_w*0.42), int(canvas_h*0.85)), (25, 25, 25), 3)

            # Run detection
            detections = detector.detect(frame, conf_threshold=conf_thresh)
            annotated_frame = annotate_frame(frame, detections)
            
            # Overlay Telemetry HUD (Speed, Lat, Lon, Time)
            hud_bg = annotated_frame.copy()
            cv2.rectangle(hud_bg, (10, 10), (360, 95), (15, 23, 42), -1)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.3, hud_bg, 0.7, 0)
            
            cv2.putText(annotated_frame, f"GPS: {current_gps['latitude']:.6f} N, {current_gps['longitude']:.6f} E", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"SPEED: {current_gps['speed_kmh']} km/h | HEADING: {current_gps['heading']}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"SURVEY TIME: +{current_gps.get('time_sec', 0):.1f}s | FIX: RTK-FLOAT", (18, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)
            
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Processed Frame (Step {step}) with Synchronized GPS Telemetry")

        else:
            uploaded_file = st.file_uploader("Upload Image or Video (JPG, PNG, MP4, MOV, AVI)", type=["jpg", "png", "jpeg", "mp4", "mov", "avi"])
            if uploaded_file is not None:
                file_name = uploaded_file.name.lower()
                is_video = file_name.endswith(('.mp4', '.mov', '.avi', '.mkv')) or (uploaded_file.type and 'video' in uploaded_file.type)
                
                if is_video:
                    # Write video to a temporary file
                    suffix = Path(uploaded_file.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                        tfile.write(uploaded_file.read())
                        temp_video_path = tfile.name
                        
                    cap = cv2.VideoCapture(temp_video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    duration_sec = total_frames / max(1.0, fps)
                    
                    st.success(f"🎬 Video Loaded: **{uploaded_file.name}** ({total_frames} frames, {fps:.1f} FPS, {duration_sec:.1f}s)")
                    
                    # Interactive frame scrubber
                    video_step = st.slider("Scrub Video Timeline (Frame Index)", 0, max(0, total_frames - 1), 0, step=max(1, int(fps/2)))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, video_step)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        current_time_sec = video_step / max(1.0, fps)
                        # Interpolate GPS along track or use live telemetry
                        current_gps = interpolate_gps(st.session_state.gps_track, current_time_sec)
                        
                        detections = detector.detect(frame, conf_threshold=conf_thresh)
                        annotated_frame = annotate_frame(frame, detections)
                        
                        # Overlay Telemetry HUD (Speed, Lat, Lon, Time)
                        hud_bg = annotated_frame.copy()
                        cv2.rectangle(hud_bg, (10, 10), (380, 95), (15, 23, 42), -1)
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.3, hud_bg, 0.7, 0)
                        
                        cv2.putText(annotated_frame, f"GPS: {current_gps['latitude']:.6f} N, {current_gps['longitude']:.6f} E", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        cv2.putText(annotated_frame, f"SPEED: {current_gps['speed_kmh']} km/h | VIDEO TIME: +{current_time_sec:.2f}s", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        cv2.putText(annotated_frame, f"FRAME: {video_step}/{total_frames} | DETECTIONS: {len(detections)}", (18, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)
                        
                        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Analyzed Video Frame {video_step} ({current_time_sec:.2f}s)")
                        
                        # Option to add detected defects to municipal map
                        if detections:
                            if st.button("➕ Add Detected Distress to Municipal Map & Work Orders"):
                                for d in detections:
                                    st.session_state.survey_defects.append({
                                        "id": f"DEF-{len(st.session_state.survey_defects)+1:03d}",
                                        "class": d["class"],
                                        "label": d["label"],
                                        "confidence": d["confidence"],
                                        "severity": d["severity"],
                                        "latitude": current_gps["latitude"],
                                        "longitude": current_gps["longitude"],
                                        "area_px": d.get("area_px", 2000),
                                        "extent_sq_m": d.get("extent_sq_m", 0.5),
                                        "time_sec": current_time_sec,
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                    })
                                st.success(f"Added {len(detections)} defect(s) to GIS Survey Ledger!")
                    else:
                        st.warning("Could not read frame at this timestamp.")
                    cap.release()
                    
                else: # It's an image
                    img = Image.open(uploaded_file)
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    current_gps = GLOBAL_TELEMETRY.get_latest()
                    detections = detector.detect(frame, conf_threshold=conf_thresh)
                    annotated_frame = annotate_frame(frame, detections)
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Inference on {uploaded_file.name}")
                    
                    if detections and st.button("➕ Add Image Detections to Municipal Map & Work Orders"):
                        for d in detections:
                            st.session_state.survey_defects.append({
                                "id": f"DEF-{len(st.session_state.survey_defects)+1:03d}",
                                "class": d["class"],
                                "label": d["label"],
                                "confidence": d["confidence"],
                                "severity": d["severity"],
                                "latitude": current_gps["latitude"],
                                "longitude": current_gps["longitude"],
                                "area_px": d.get("area_px", 2000),
                                "extent_sq_m": d.get("extent_sq_m", 0.5),
                                "time_sec": 0.0,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                        st.success(f"Added {len(detections)} defect(s) to GIS Survey Ledger!")
            else:
                st.info("Upload road imagery or dashcam video to perform custom inference.")
                detections = []
                current_gps = GLOBAL_TELEMETRY.get_latest()

    with col2:
        st.markdown("**Real-Time Frame Telemetry & Defect Ledger**")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Synchronized Speed</div>
                <div class='metric-val'>{current_gps['speed_kmh'] if 'current_gps' in locals() else 28.5} km/h</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Detected In Frame</div>
                <div class='metric-val'>{len(detections) if 'detections' in locals() else 0} Defects</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### Detected Distress Features in Current View")
        if 'detections' in locals() and detections:
            for d in detections:
                color_hex = "#EF4444" if d["class"] == "D40" else ("#F59E0B" if d["class"] == "D20" else "#3B82F6")
                st.markdown(f"""
                <div style="border-left: 4px solid {color_hex}; padding: 6px 12px; background: #F8FAFC; margin-bottom: 6px; border-radius: 4px;">
                    <strong>{d['label']}</strong><br>
                    <span style="font-size: 12px; color: #64748B;">
                        Confidence: <strong>{int(d['confidence']*100)}%</strong> | Severity: <strong>{d['severity']}</strong> | Est. Area: <strong>{d['extent_sq_m']} m²</strong>
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Pavement surface clear of major distress in this frame window.")

        st.divider()
        st.markdown("##### Standard RDD2022 Defect Schema")
        st.markdown("""
        * **D00:** Longitudinal Cracking (Parallel to lane)
        * **D10:** Transverse Cracking (Perpendicular to lane)
        * **D20:** Alligator / Fatigue Cracking (Network failure)
        * **D40:** Pothole (Structural cavity & traffic hazard)
        """)

# -------------------------------------------------------------
# TAB 2: INTERACTIVE WEB-GIS MAP & HEATMAPS
# -------------------------------------------------------------
elif nav_choice == "🗺️ Web-GIS Map & Heatmaps":
    st.subheader("2. Interactive Municipal Web-GIS Defect Geoportal")
    
    # Calculate map center
    if st.session_state.gps_track:
        center_lat = st.session_state.gps_track[0]["latitude"]
        center_lon = st.session_state.gps_track[0]["longitude"]
    else:
        center_lat, center_lon = 12.9716, 79.1585

    m_col1, m_col2 = st.columns([4, 1])
    with m_col2:
        st.markdown("##### Map Layers")
        show_heatmap = st.checkbox("Distress Density Heatmap", value=True)
        show_pins = st.checkbox("Defect Pin Markers", value=True)
        show_trajectory = st.checkbox("Vehicle Survey Trajectory", value=True)
        basemap = st.selectbox("Basemap Style", ["CartoDB Positron", "OpenStreetMap"])
        
        st.divider()
        st.markdown("##### Defect Legend")
        st.markdown("🔴 **D40 - Pothole**")
        st.markdown("🟠 **D20 - Alligator Crack**")
        st.markdown("🟣 **D10 - Transverse Crack**")
        st.markdown("🔵 **D00 - Longitudinal Crack**")

    with m_col1:
        tile_source = "cartodbpositron" if basemap == "CartoDB Positron" else "OpenStreetMap"
        m = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles=tile_source)
        
        # 1. Vehicle Trajectory Polyline
        if show_trajectory and st.session_state.gps_track:
            coords = [[pt["latitude"], pt["longitude"]] for pt in st.session_state.gps_track]
            folium.PolyLine(
                coords,
                color="#0284C7",
                weight=4,
                opacity=0.8,
                tooltip="Survey Vehicle Trajectory (Corridor A-1)"
            ).add_to(m)

        # 2. Defect Density Heatmap
        if show_heatmap and st.session_state.survey_defects:
            heat_data = []
            for d in st.session_state.survey_defects:
                weight = 1.0 if d.get("class") == "D40" else 0.6
                heat_data.append([d["latitude"], d["longitude"], weight])
            HeatMap(heat_data, radius=18, blur=14, max_zoom=18).add_to(m)

        # 3. Individual Defect Markers
        if show_pins and st.session_state.survey_defects:
            marker_cluster = MarkerCluster(name="Defects").add_to(m)
            for d in st.session_state.survey_defects:
                code = d.get("class", "D40")
                sev = d.get("severity", "M")
                
                # Pick marker color
                if code == "D40":
                    color = "red"
                    icon = "exclamation-triangle"
                elif code == "D20":
                    color = "orange"
                    icon = "random"
                elif code == "D10":
                    color = "purple"
                    icon = "arrows-h"
                else:
                    color = "blue"
                    icon = "road"
                    
                popup_html = f"""
                <div style="font-family: Arial; width: 220px;">
                    <h4 style="margin: 0 0 6px 0; color: #0F172A;">{d['label']}</h4>
                    <p style="margin: 2px 0; font-size: 12px;"><b>Severity:</b> <span style="color:red; font-weight:bold;">{sev}</span></p>
                    <p style="margin: 2px 0; font-size: 12px;"><b>Confidence:</b> {int(d['confidence']*100)}%</p>
                    <p style="margin: 2px 0; font-size: 12px;"><b>Coords:</b> {d['latitude']:.5f}, {d['longitude']:.5f}</p>
                    <p style="margin: 2px 0; font-size: 12px;"><b>Estimated Area:</b> {d.get('extent_sq_m', 0.5)} m²</p>
                    <p style="margin: 2px 0; font-size: 11px; color:#64748B;">Timestamp: {d.get('timestamp', 'Live')}</p>
                </div>
                """
                
                folium.Marker(
                    location=[d["latitude"], d["longitude"]],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{d['label']} ({sev})",
                    icon=folium.Icon(color=color, icon=icon, prefix="fa")
                ).add_to(marker_cluster)

        st_folium(m, width=950, height=520)

# -------------------------------------------------------------
# TAB 3: ASTM D6433 CIVIL PCI ANALYTICS
# -------------------------------------------------------------
elif nav_choice == "📊 ASTM D6433 PCI Analytics":
    st.subheader("3. Civil Engineering Pavement Condition Index (PCI) Analysis")
    st.markdown("Evaluation conforming to **ASTM D6433-20 Standard** and **IRC:82-2015 Highway Maintenance Code**.")
    
    pci_result = compute_pci_score(st.session_state.survey_defects, sample_area_sq_m=500.0)
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 4px solid {pci_result['color']};'>
            <div class='metric-title'>ASTM D6433 PCI Score</div>
            <div class='metric-val' style='color: {pci_result['color']};'>{pci_result['pci']} / 100</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Civil Condition Rating</div>
            <div class='metric-val'>{pci_result['rating']}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Total Distress Count</div>
            <div class='metric-val'>{len(st.session_state.survey_defects)} Defects</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Critical Potholes (D40)</div>
            <div class='metric-val' style='color: #EF4444;'>{pci_result['defect_breakdown'].get('D40', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 12px 18px; border-radius: 4px; margin: 16px 0;">
        <strong style="color: #92400E;">Prescribed Civil Engineering Remediation:</strong><br>
        <span style="color: #78350F; font-size: 14px;">{pci_result['recommendation']}</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Defect Class Breakdown (RDD2022 Schema)")
        dist_df = pd.DataFrame([
            {"Distress Class": f"{k} ({RDD_DISTRESS_MAP.get(k, {}).get('name', 'Distress')})", "Count": v}
            for k, v in pci_result['defect_breakdown'].items()
        ])
        st.bar_chart(dist_df.set_index("Distress Class"))

    with c2:
        st.markdown("##### Deduct Value Calculation Formula & Variables")
        st.latex(r"PCI = 100 - CDV(\sum Deduct\ Values, q)")
        st.markdown(f"""
        * **Total Deduct Value (TDV):** `{pci_result['tdv']}`
        * **Corrected Deduct Value (CDV):** `{pci_result['cdv']}`
        * **Top Individual Deduct Values ($DV_i$):** `{pci_result['deduct_values'][:4]}`
        * **Survey Unit Pavement Area:** `500.0 m²`
        """)

# -------------------------------------------------------------
# TAB 4: MUNICIPAL WORK ORDERS
# -------------------------------------------------------------
elif nav_choice == "📋 Municipal Work Orders":
    st.subheader("4. Automated Municipal Maintenance Work Order Dispatcher")
    st.markdown("Converts detected defects into official, field-ready repair tickets for municipal road crews.")

    road_name = st.text_input("Survey Corridor Identifier", "Smart City Arterial Corridor A-1 (Zone 3)")
    work_orders_df = generate_work_orders(st.session_state.survey_defects, road_name=road_name)
    pci_result = compute_pci_score(st.session_state.survey_defects)

    st.dataframe(work_orders_df, use_container_width=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        csv_data = work_orders_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Work Orders (CSV)",
            data=csv_data,
            file_name=f"work_orders_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with btn_col2:
        html_report = generate_html_work_order(
            work_orders_df, 
            pci_result, 
            road_name=road_name,
            surveyor_name="Jadeja Suryadeepsinh Bharatsinh (23BEL1027)"
        )
        st.download_button(
            label="📄 Export Official PWD Work Order (Printable HTML / PDF)",
            data=html_report.encode('utf-8'),
            file_name=f"PWD_Work_Order_{time.strftime('%Y%m%d')}.html",
            mime="text/html",
            use_container_width=True
        )

# -------------------------------------------------------------
# TAB 5: TEAM ROLES & REVIEW 2 SPECS
# -------------------------------------------------------------
elif nav_choice == "👥 Team Roles & Review 2 Spec":
    st.subheader("5. Transdisciplinary System Architecture & Team Allocation")
    
    st.markdown("""
    ### 4 Core Transdisciplinary Pillars
    1. **Electrical / Telemetry:** High-frequency GPS logging, HTTP telemetry streaming, time-synchronization between mobile sensors and camera stream.
    2. **Computer Science / AI:** YOLOv8 deep learning defect localization, confidence scoring, OpenCV image preprocessing.
    3. **Civil Engineering:** ASTM D6433 Pavement Condition Index algorithm, severity deduction curves, IRC 82-2015 remediation guidelines.
    4. **Urban Governance / GIS:** Interactive Web-GIS spatial heatmapping, automated municipal maintenance work order dispatching.
    """)
    
    st.divider()
    st.markdown("### Team Work Breakdown (5 Members)")
    
    team_data = [
        {"Member": "Jadeja Suryadeepsinh (23BEL1027)", "Pillar": "Lead / System Orchestrator", "Core Contribution": "Overall end-to-end pipeline integration, video-to-telemetry timestamp matching, prototype deployment."},
        {"Member": "Vishnu Nair (23BEL1028)", "Pillar": "Computer Science / AI", "Core Contribution": "YOLOv8 model inference optimization, RDD2022 dataset preparation, mAP evaluation metrics."},
        {"Member": "Udayvardhan Singh Rathore (23BEL1018)", "Pillar": "Civil Engineering", "Core Contribution": "ASTM D6433 standard formulation, deduct value curves calibration, PCI score calculation engine."},
        {"Member": "Ainessh kumar S (23BEL1039)", "Pillar": "Geospatial & GIS", "Core Contribution": "Folium Web-GIS mapping, GPS route interpolation, spatial clustering and defect heatmaps."},
        {"Member": "Shreyan Biswas (23BEL1052)", "Pillar": "Urban Governance / UI", "Core Contribution": "Streamlit municipal dashboard UI, CSV/HTML work order generator conforming to PWD standards."}
    ]
    st.table(pd.DataFrame(team_data))

