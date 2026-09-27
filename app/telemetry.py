"""
Mobile Telemetry & GPS Synchronization Module
=============================================
Manages real-time physical GPS coordinates received from smartphone sensors
or pre-recorded drive logs, synchronized with dashcam video timestamps.

Transdisciplinary Link: Electrical / Telemetry Engineering
"""

import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple
import math

class TelemetryStore:
    """Thread-safe storage for incoming real-time GPS telemetry packets."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_point = {
            "latitude": 12.9716, # Default urban coords (e.g. Bangalore / Chennai Smart City)
            "longitude": 79.1585,
            "speed_kmh": 28.5,
            "heading": 85.0,
            "accuracy_m": 2.4,
            "timestamp": time.time()
        }
        self.history: List[Dict] = [self.latest_point.copy()]

    def update(self, data: Dict):
        with self.lock:
            self.latest_point = {
                "latitude": float(data.get("latitude", self.latest_point["latitude"])),
                "longitude": float(data.get("longitude", self.latest_point["longitude"])),
                "speed_kmh": float(data.get("speed", data.get("speed_kmh", 25.0))),
                "heading": float(data.get("heading", data.get("bearing", 0.0))),
                "accuracy_m": float(data.get("accuracy", 3.0)),
                "timestamp": float(data.get("timestamp", time.time()))
            }
            self.history.append(self.latest_point.copy())
            if len(self.history) > 2000:
                self.history.pop(0)

    def get_latest(self) -> Dict:
        with self.lock:
            return self.latest_point.copy()

    def get_history(self) -> List[Dict]:
        with self.lock:
            return list(self.history)

# Global Telemetry Instance
GLOBAL_TELEMETRY = TelemetryStore()

class TelemetryHTTPHandler(BaseHTTPRequestHandler):
    """Receives JSON POST telemetry from Android/iOS GPS logger or Phyphox."""
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode('utf-8'))
            GLOBAL_TELEMETRY.update(payload)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(GLOBAL_TELEMETRY.get_latest()).encode('utf-8'))

    def log_message(self, format, *args):
        return # Suppress noisy console logs

def start_telemetry_server(port: int = 5050) -> Optional[HTTPServer]:
    """Spawns background HTTP server for smartphone telemetry streaming."""
    try:
        server = HTTPServer(('0.0.0.0', port), TelemetryHTTPHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
    except Exception as e:
        print(f"[TelemetryServer] Port {port} unavailable: {e}")
        return None

def interpolate_gps(track_points: List[Dict], target_time_sec: float) -> Dict:
    """
    Interpolates GPS latitude & longitude for a given video elapsed time.
    """
    if not track_points:
        return GLOBAL_TELEMETRY.get_latest()
    
    if len(track_points) == 1:
        return track_points[0]
        
    start_time = track_points[0].get("time_sec", 0.0)
    end_time = track_points[-1].get("time_sec", track_points[-1].get("timestamp", 0.0) - track_points[0].get("timestamp", 0.0))
    
    if target_time_sec <= start_time:
        return track_points[0]
    if target_time_sec >= end_time:
        return track_points[-1]
        
    for i in range(len(track_points) - 1):
        t1 = track_points[i].get("time_sec", i)
        t2 = track_points[i+1].get("time_sec", i+1)
        if t1 <= target_time_sec <= t2:
            ratio = (target_time_sec - t1) / max(0.0001, (t2 - t1))
            lat = track_points[i]["latitude"] + ratio * (track_points[i+1]["latitude"] - track_points[i]["latitude"])
            lon = track_points[i]["longitude"] + ratio * (track_points[i+1]["longitude"] - track_points[i]["longitude"])
            speed = track_points[i].get("speed_kmh", 30.0) + ratio * (track_points[i+1].get("speed_kmh", 30.0) - track_points[i].get("speed_kmh", 30.0))
            return {
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(speed, 1),
                "timestamp": time.time()
            }
            
    return track_points[-1]

def generate_simulated_route(center_lat: float = 12.9716, center_lon: float = 79.1585, num_points: int = 150) -> List[Dict]:
    """Generates realistic vehicle trajectory through urban smart city road segments."""
    route = []
    curr_lat, curr_lon = center_lat, center_lon
    heading = 45.0
    for i in range(num_points):
        # Slight curved driving trajectory
        heading += (math.sin(i / 10.0) * 8.0)
        rad = math.radians(heading)
        curr_lat += math.cos(rad) * 0.00015
        curr_lon += math.sin(rad) * 0.00015
        speed = 28.0 + 8.0 * math.sin(i / 5.0)
        route.append({
            "step": i,
            "time_sec": i * 0.5,
            "latitude": round(curr_lat, 6),
            "longitude": round(curr_lon, 6),
            "speed_kmh": round(speed, 1),
            "heading": round(heading % 360, 1)
        })
    return route
