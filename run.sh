#!/usr/bin/env bash
# ==============================================================================
# Vision-Based Pavement Degradation Detection & GIS System
# Automated Execution Script for Review 2 Demonstration
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "====================================================================="
echo " Smart City Pavement Degradation Detection & GIS Mapping Platform"
echo " Review 2 Working Prototype Launcher"
echo "====================================================================="

# Check Python environment
if [ -d "venv" ]; then
    echo "[1/3] Activating Python virtual environment..."
    source venv/bin/activate
else
    echo "[1/3] Initializing new Python 3.11 virtual environment..."
    python3.11 -m venv venv
    source venv/bin/activate
fi

# Check if core dependencies are already satisfied
if python -c "import streamlit, folium, streamlit_folium, pandas, cv2" 2>/dev/null; then
    echo "[2/3] All dependencies are already installed and verified!"
else
    echo "[2/3] Installing/updating lightweight dashboard requirements..."
    pip install -r requirements.txt
fi

# Launch Streamlit Dashboard
echo "[3/3] Launching Web-GIS Municipal Dashboard..."
echo "---------------------------------------------------------------------"
echo "Dashboard URL: http://localhost:8501"
echo "Press Ctrl+C in this terminal to stop the server."
echo "---------------------------------------------------------------------"
streamlit run app/dashboard.py
