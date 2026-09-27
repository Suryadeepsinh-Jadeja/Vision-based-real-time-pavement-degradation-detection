"""
ASTM D6433 Pavement Condition Index (PCI) Calculation Engine
============================================================
Implements civil engineering pavement assessment standards to grade road health
from automated computer-vision defect detections.

Standards reference:
- ASTM D6433-20: Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys
- IRC 82-2015: Code of Practice for Maintenance of Bituminous Surfaces of Highways (India)
"""

import math
from typing import Dict, List, Any, Tuple

# RDD2022 Schema to ASTM Standard Mapping
RDD_DISTRESS_MAP = {
    "D00": {"name": "Longitudinal Crack", "unit": "linear_m", "hazard": "low_medium"},
    "D10": {"name": "Transverse Crack", "unit": "linear_m", "hazard": "low_medium"},
    "D20": {"name": "Alligator (Fatigue) Crack", "unit": "sq_m", "hazard": "high"},
    "D40": {"name": "Pothole", "unit": "count", "hazard": "critical"}
}

def calculate_density(distress_count_or_area: float, sample_unit_area_sq_m: float = 250.0) -> float:
    """Calculates distress density as a percentage of total pavement sample area."""
    if sample_unit_area_sq_m <= 0:
        sample_unit_area_sq_m = 250.0
    return min(100.0, (distress_count_or_area / sample_unit_area_sq_m) * 100.0)

def get_deduct_value(distress_type: str, severity: str, density: float) -> float:
    """
    Computes ASTM D6433 Deduct Value (DV) using calibrated empirical curves.
    severity: 'L' (Low), 'M' (Medium), 'H' (High)
    """
    severity = severity.upper()
    if severity not in ['L', 'M', 'H']:
        severity = 'M'
    
    # Safe minimum density check
    if density <= 0.001:
        return 0.0

    log_d = math.log10(max(0.01, density))

    # Calibrated ASTM D6433 Deduct Value curves for asphalt pavements
    if distress_type == "D40":  # Potholes (Highest penalty)
        if severity == 'L':
            dv = 20.0 + 18.0 * log_d
        elif severity == 'M':
            dv = 42.0 + 24.0 * log_d
        else: # High
            dv = 68.0 + 26.0 * log_d
            
    elif distress_type == "D20":  # Alligator Cracking (Structural Fatigue)
        if severity == 'L':
            dv = 12.0 + 15.0 * log_d
        elif severity == 'M':
            dv = 25.0 + 22.0 * log_d
        else: # High
            dv = 45.0 + 28.0 * log_d
            
    elif distress_type in ["D00", "D10"]:  # Longitudinal / Transverse Cracks
        if severity == 'L':
            dv = 4.0 + 8.0 * log_d
        elif severity == 'M':
            dv = 10.0 + 14.0 * log_d
        else: # High
            dv = 20.0 + 18.0 * log_d
            
    else:  # General unclassified defect
        dv = 8.0 + 10.0 * log_d

    return max(0.0, min(100.0, dv))

def compute_cdv(deduct_values: List[float]) -> float:
    """
    Computes Corrected Deduct Value (CDV) from list of individual deduct values.
    ASTM D6433 iterative correction methodology.
    """
    valid_dvs = sorted([d for d in deduct_values if d > 2.0], reverse=True)
    if not valid_dvs:
        return 0.0
    
    tdv = sum(valid_dvs)
    q = len([d for d in valid_dvs if d > 5.0])
    
    if q <= 1:
        return min(100.0, tdv)
    
    # ASTM D6433 CDV curve approximation
    # CDV diminishes the cumulative effect of multiple deducts
    q_factor = 1.0 / (1.0 + 0.22 * (q - 1))
    cdv = tdv * q_factor
    
    # Cap CDV so it is never less than the maximum single deduct value
    max_dv = valid_dvs[0]
    return min(100.0, max(max_dv, cdv))

def compute_pci_score(defects: List[Dict[str, Any]], sample_area_sq_m: float = 500.0) -> Dict[str, Any]:
    """
    Computes full ASTM D6433 PCI metrics for a given road section.
    
    Each defect dict should have:
    - 'class': 'D00' | 'D10' | 'D20' | 'D40'
    - 'severity': 'L' | 'M' | 'H'
    - 'estimated_area_sq_m' or 'count'
    """
    if not defects:
        return {
            "pci": 100.0,
            "rating": "Good",
            "color": "#10B981", # Emerald green
            "cdv": 0.0,
            "tdv": 0.0,
            "defect_breakdown": {},
            "recommendation": "Road is in excellent condition. Routine visual monitoring only."
        }

    deduct_values = []
    defect_breakdown = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
    severity_breakdown = {"L": 0, "M": 0, "H": 0}

    for d in defects:
        code = d.get("class", "D40")
        sev = d.get("severity", "M").upper()
        defect_breakdown[code] = defect_breakdown.get(code, 0) + 1
        severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
        
        # Calculate equivalent extent
        extent = d.get("extent_sq_m", 0.5 if code == "D40" else 1.0)
        density = calculate_density(extent, sample_area_sq_m)
        dv = get_deduct_value(code, sev, density)
        deduct_values.append(dv)

    tdv = sum(deduct_values)
    cdv = compute_cdv(deduct_values)
    pci = max(0.0, round(100.0 - cdv, 1))

    # ASTM D6433 Condition Categories
    if pci >= 85:
        rating = "Good"
        color = "#10B981" # Green
        action = "Routine preventive maintenance & crack sealing."
    elif pci >= 70:
        rating = "Satisfactory"
        color = "#84CC16" # Lime
        action = "Preventive surface seal, minor crack sealing."
    elif pci >= 55:
        rating = "Fair"
        color = "#FBBF24" # Yellow
        action = "Cold patching of isolated potholes, slurry seal overlay."
    elif pci >= 40:
        rating = "Poor"
        color = "#F97316" # Orange
        action = "Full depth patch repairs and thin bituminous concrete overlay."
    elif pci >= 25:
        rating = "Very Poor"
        color = "#EF4444" # Red
        action = "Extensive mill-and-fill resurfacing required immediately."
    elif pci >= 10:
        rating = "Serious"
        color = "#B91C1C" # Deep Red
        action = "Structural failure: Sub-base reconstruction & asphalt base course."
    else:
        rating = "Failed"
        color = "#7F1D1D" # Dark Crimson
        action = "Total road failure: Complete pavement reconstruction mandated."

    return {
        "pci": pci,
        "rating": rating,
        "color": color,
        "cdv": round(cdv, 1),
        "tdv": round(tdv, 1),
        "defect_breakdown": defect_breakdown,
        "severity_breakdown": severity_breakdown,
        "deduct_values": [round(x, 1) for x in sorted(deduct_values, reverse=True)],
        "recommendation": action
    }
