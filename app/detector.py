"""
AI Defect Detection Pipeline
============================
Integrates YOLOv8 deep learning and OpenCV computer vision pipelines
conforming to the international Road Damage Dataset (RDD2022) schema.

Distress Classes:
- D00: Longitudinal Crack
- D10: Transverse Crack
- D20: Alligator Crack (Structural fatigue)
- D40: Pothole (Severe structural cavity)
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

CLASS_NAMES = {
    0: "D00 - Longitudinal Crack",
    1: "D10 - Transverse Crack",
    2: "D20 - Alligator Crack",
    3: "D40 - Pothole"
}

CLASS_CODES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40"
}

CLASS_COLORS = {
    "D00": (59, 130, 246),   # Blue
    "D10": (168, 85, 247),  # Purple
    "D20": (245, 158, 11),  # Amber / Yellow
    "D40": (239, 68, 68)    # Red (Critical)
}

class RoadDamageDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.use_yolo = False
        self.engine_name = "OpenCV Pavement Contrast & Geometric Feature Extractor"
        self.weights_file = None
        
        # Check specified path or standard model locations
        candidate_paths = [
            model_path,
            "models/pothole_yolov8.pt",
            "models/best.pt",
            "models/rdd2022_yolov8.pt",
            "models/yolov8n.pt",
            "pothole_yolov8.pt",
            "best.pt"
        ]
        
        for p in candidate_paths:
            if p and os.path.exists(p):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(p)
                    self.use_yolo = True
                    self.weights_file = p
                    self.engine_name = f"YOLOv8 Deep Learning ({os.path.basename(p)})"
                    print(f"[Detector] Loaded YOLOv8 weights from {p}")
                    break
                except Exception as e:
                    print(f"[Detector] Could not load weights from {p}: {e}")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """
        Runs defect detection on input BGR frame.
        Returns list of detected distress bounding boxes with severity and class codes.
        """
        h, w = frame.shape[:2]
        detections = []

        if self.use_yolo and self.model is not None:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            model_names = self.model.names  # e.g. {0: '0'} or {0: 'pothole', 1: 'crack', ...}
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    # Smart class mapping:
                    # If model has RDD2022-style class names, map directly
                    # If model is single-class (pothole-only), all detections = D40
                    raw_name = str(model_names.get(cls_id, "")).lower()
                    if "longitudinal" in raw_name or raw_name in ["d00", "0_long"]:
                        code = "D00"
                    elif "transverse" in raw_name or raw_name in ["d10"]:
                        code = "D10"
                    elif "alligator" in raw_name or "fatigue" in raw_name or raw_name in ["d20"]:
                        code = "D20"
                    else:
                        # Default: potholes / single-class model
                        code = CLASS_CODES.get(cls_id, "D40")
                        if len(model_names) == 1:
                            code = "D40"  # Single-class pothole model

                    box_w = x2 - x1
                    box_h = y2 - y1
                    area_px = box_w * box_h
                    rel_area = area_px / max(1, w * h)

                    # Severity from bounding box area relative to frame
                    if rel_area < 0.015:
                        severity = "L"
                    elif rel_area < 0.045:
                        severity = "M"
                    else:
                        severity = "H"

                    detections.append({
                        "class": code,
                        "label": RDD_CLASS_NAME_MAP.get(code, code),
                        "confidence": round(conf, 2),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "severity": severity,
                        "area_px": int(area_px),
                        "extent_sq_m": round(rel_area * 15.0, 2)
                    })

            # If YOLO model is single-class (pothole only), supplement with
            # OpenCV crack detection to give full RDD2022 multi-class coverage
            if len(model_names) == 1:
                cv_detections = self._detect_opencv_heuristics(frame, conf_threshold)
                # Only add crack types (not D40, which YOLO already covers)
                crack_types = [d for d in cv_detections if d["class"] != "D40"]
                detections.extend(crack_types)

            return detections

        # Fallback: Built-in OpenCV Pavement Distress Feature Extractor
        return self._detect_opencv_heuristics(frame, conf_threshold)

    def _detect_opencv_heuristics(self, frame: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
        """
        Computer-vision heuristic detector for pavement distress:
        Analyzes lower 60% of frame (road region of interest), uses adaptive thresholding,
        dark-spot morphological segmentation for potholes, and gradient filters for cracks.
        """
        h, w = frame.shape[:2]
        roi_top = int(h * 0.40) # focus on pavement
        roi = frame[roi_top:h, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        detections = []
        
        # 1. Detect Potholes (Dark structural depressions with high shadow contrast)
        blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
        # Potholes exhibit distinctly low intensity compared to surrounding asphalt
        mean_val = np.mean(blurred)
        thresh_val = max(30, int(mean_val - 32))
        _, dark_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1200 and area < (w * h * 0.15):
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / max(1, bh)
                
                # Pothole geometry: roughly elliptical/convex depression
                if 0.5 <= aspect_ratio <= 3.0:
                    abs_y1 = roi_top + y
                    abs_y2 = abs_y1 + bh
                    abs_x1 = x
                    abs_x2 = x + bw
                    
                    # Severity classification
                    rel_area = area / (w * h)
                    sev = "H" if rel_area > 0.04 else ("M" if rel_area > 0.015 else "L")
                    conf = min(0.96, max(conf_threshold, 0.72 + (area / 15000.0) * 0.2))
                    
                    detections.append({
                        "class": "D40",
                        "label": "D40 - Pothole",
                        "confidence": round(conf, 2),
                        "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                        "severity": sev,
                        "area_px": int(area),
                        "extent_sq_m": round(rel_area * 14.0, 2)
                    })

        # 2. Detect Cracks (Longitudinal / Transverse / Alligator)
        # Using Canny edge detector and Hough / contour orientation
        edges = cv2.Canny(gray_roi, 60, 160)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, edge_kernel, iterations=1)
        crack_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in crack_contours:
            c_area = cv2.contourArea(cnt)
            if 600 < c_area < 8000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / max(1, bh)
                abs_y1 = roi_top + y
                abs_y2 = abs_y1 + bh
                abs_x1 = x
                abs_x2 = x + bw
                
                # Distinguish Longitudinal vs Transverse vs Alligator
                if aspect > 2.8:
                    code = "D10" # Transverse Crack
                elif aspect < 0.4:
                    code = "D00" # Longitudinal Crack
                else:
                    code = "D20" # Alligator / Mesh Crack
                    
                rel_area = c_area / (w * h)
                sev = "H" if c_area > 4000 else ("M" if c_area > 1800 else "L")
                conf = min(0.91, max(conf_threshold, 0.65 + (c_area / 8000.0) * 0.2))
                
                detections.append({
                    "class": code,
                    "label": f"{code} - {RDD_CLASS_NAME_MAP.get(code, 'Crack')}",
                    "confidence": round(conf, 2),
                    "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                    "severity": sev,
                    "area_px": int(c_area),
                    "extent_sq_m": round(rel_area * 12.0, 2)
                })

        return detections

RDD_CLASS_NAME_MAP = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole"
}

def annotate_frame(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draws professional HUD bounding boxes and badges on video frames."""
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        code = d["class"]
        conf = d["confidence"]
        sev = d["severity"]
        
        color = CLASS_COLORS.get(code, (0, 255, 0))
        # Draw bounding rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label string: e.g. "D40 - Pothole | Sev: H | 88%"
        label_text = f"{code} ({sev}) {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw badge background
        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
        cv2.putText(annotated, label_text, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    return annotated
