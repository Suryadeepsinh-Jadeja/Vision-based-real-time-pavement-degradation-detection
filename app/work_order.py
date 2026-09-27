"""
Municipal Maintenance Work Order Generator
==========================================
Transforms automated AI pavement distress detections and PCI civil ratings
into standardized municipal road repair work orders conforming to IRC 82-2015 guidelines.

Transdisciplinary Link: Urban Governance & Smart Infrastructure Management
"""

import datetime
from typing import List, Dict, Any

try:
    import pandas as pd
except ImportError:
    class DummyDataFrame:
        def __init__(self, data):
            self.data = data
        def to_csv(self, index=False):
            if not self.data:
                return ""
            header = ",".join(f'"{k}"' for k in self.data[0].keys())
            rows = [",".join(f'"{v}"' for v in d.values()) for d in self.data]
            return header + "\n" + "\n".join(rows)
        def iterrows(self):
            for i, d in enumerate(self.data):
                yield i, d
        def __len__(self):
            return len(self.data)
    pd = type('pd', (), {'DataFrame': DummyDataFrame})

CIVIL_REPAIR_TREATMENTS = {
    "D40": {
        "H": "Immediate Full-Depth Patching: Saw-cut rectangular boundaries, tack coat, 75mm Hot Mix Asphalt compaction (IRC:82-2015 Cl. 4.3).",
        "M": "Semi-Permanent Patching: Clear debris, emulsion tack coat, cold mix asphalt compaction.",
        "L": "Temporary Throw-and-Roll Patching with cold asphalt mix until scheduled resurfacing."
    },
    "D20": {
        "H": "Structural Rehabilitation: 50mm cold milling followed by 40mm Bituminous Concrete (BC) overlay.",
        "M": "Stress Absorbing Membrane Interlayer (SAMI) and 25mm corrective asphalt overlay.",
        "L": "Fog Seal / Slurry Seal application to arrest water infiltration."
    },
    "D10": {
        "H": "Crack Routing and Hot-Poured Polymer Modified Bituminous Sealant (ASTM D6690).",
        "M": "High-pressure air wand cleaning and rubberized asphalt sealant injection.",
        "L": "Routine sand-emulsion slurry squeegee treatment."
    },
    "D00": {
        "H": "Milling along crack corridor (150mm width) and micro-surfacing infill.",
        "M": "Air cleaning and elastomeric bitumen crack sealing.",
        "L": "Preventive surface seal coat application."
    }
}

PRIORITY_LEVELS = {
    "D40": {"H": "CRITICAL (24-48 hrs)", "M": "HIGH (5 days)", "L": "MEDIUM (14 days)"},
    "D20": {"H": "HIGH (7 days)", "M": "MEDIUM (21 days)", "L": "ROUTINE (45 days)"},
    "D10": {"H": "MEDIUM (14 days)", "M": "ROUTINE (30 days)", "L": "MONITOR (60 days)"},
    "D00": {"H": "MEDIUM (14 days)", "M": "ROUTINE (30 days)", "L": "MONITOR (60 days)"}
}

def generate_work_orders(defects: List[Dict[str, Any]], road_name: str = "Smart City Corridor A-1") -> pd.DataFrame:
    """Creates tabular municipal repair schedule from detected defects."""
    records = []
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    for i, d in enumerate(defects, start=1):
        code = d.get("class", "D40")
        sev = d.get("severity", "M")
        lat = d.get("latitude", 12.9716)
        lon = d.get("longitude", 79.1585)
        
        treatment = CIVIL_REPAIR_TREATMENTS.get(code, {}).get(sev, "General Pavement Maintenance")
        priority = PRIORITY_LEVELS.get(code, {}).get(sev, "SCHEDULED")
        
        records.append({
            "Work Order ID": f"WO-{datetime.datetime.now().strftime('%y%m%d')}-{i:03d}",
            "Date Logged": today,
            "Road Section": road_name,
            "Defect Code": code,
            "Distress Description": d.get("label", code),
            "Severity": sev,
            "GPS Latitude": lat,
            "GPS Longitude": lon,
            "Repair Priority": priority,
            "Prescribed Engineering Treatment": treatment
        })
        
    return pd.DataFrame(records)

def generate_html_work_order(
    work_orders_df: pd.DataFrame, 
    pci_data: Dict[str, Any], 
    road_name: str = "Smart City Arterial Corridor",
    surveyor_name: str = "Jadeja Suryadeepsinh Bharatsinh (23BEL1027)"
) -> str:
    """Generates printable, executive-ready Municipal Works Department official report."""
    now_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    
    table_rows = ""
    for _, row in work_orders_df.iterrows():
        p_class = "p-crit" if "CRITICAL" in row["Repair Priority"] else ("p-high" if "HIGH" in row["Repair Priority"] else "p-norm")
        table_rows += f"""
        <tr>
            <td><strong>{row['Work Order ID']}</strong></td>
            <td><span class="badge {p_class}">{row['Repair Priority']}</span></td>
            <td>{row['Defect Code']} ({row['Severity']})</td>
            <td>{row['GPS Latitude']:.5f}, {row['GPS Longitude']:.5f}</td>
            <td style="font-size: 13px;">{row['Prescribed Engineering Treatment']}</td>
        </tr>
        """
        
    pci_score = pci_data.get("pci", 75.0)
    pci_rating = pci_data.get("rating", "Fair")
    pci_color = pci_data.get("color", "#FBBF24")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Municipal Work Order - {road_name}</title>
<style>
    body {{ font-family: 'Segoe UI', Helvetica, Arial, sans-serif; margin: 30px; color: #1e293b; background: #fff; }}
    .header {{ border-bottom: 3px solid #0f172a; padding-bottom: 12px; margin-bottom: 24px; }}
    .header h1 {{ margin: 0; font-size: 24px; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px; }}
    .header p {{ margin: 4px 0 0 0; color: #64748b; font-size: 14px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }}
    .card-title {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 4px; }}
    .card-value {{ font-size: 24px; font-weight: 700; color: #0f172a; }}
    .pci-badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: bold; color: #fff; background: {pci_color}; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
    th {{ background: #0f172a; color: #fff; text-align: left; padding: 10px; font-weight: 600; font-size: 13px; }}
    td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
    .p-crit {{ background: #fee2e2; color: #991b1b; }}
    .p-high {{ background: #ffedd5; color: #9a3412; }}
    .p-norm {{ background: #e0f2fe; color: #075985; }}
    .footer {{ margin-top: 40px; border-top: 1px dashed #cbd5e1; padding-top: 20px; display: flex; justify-content: space-between; font-size: 13px; color: #64748b; }}
    .sign-block {{ width: 220px; border-top: 1px solid #0f172a; text-align: center; padding-top: 8px; font-weight: 600; color: #0f172a; margin-top: 50px; }}
    @media print {{ body {{ margin: 10px; }} }}
</style>
</head>
<body>
    <div class="header">
        <h1>Smart City Municipal Corporation — Public Works Dept.</h1>
        <p>Road Asset Maintenance & Automated Distress Remediation Work Order | Generated on {now_str}</p>
    </div>

    <div class="summary-grid">
        <div class="card">
            <div class="card-title">Surveyed Corridor</div>
            <div class="card-value" style="font-size: 18px;">{road_name}</div>
        </div>
        <div class="card">
            <div class="card-title">ASTM D6433 PCI Score</div>
            <div class="card-value"><span class="pci-badge">{pci_score} / 100</span></div>
        </div>
        <div class="card">
            <div class="card-title">Road Health Rating</div>
            <div class="card-value" style="font-size: 20px;">{pci_rating}</div>
        </div>
        <div class="card">
            <div class="card-title">Identified Defects</div>
            <div class="card-value">{len(work_orders_df)} Spots</div>
        </div>
    </div>

    <h3>Prescribed Repair Actions & Priority Dispatch Schedule</h3>
    <table>
        <thead>
            <tr>
                <th style="width: 140px;">Work Order ID</th>
                <th style="width: 150px;">Priority Level</th>
                <th style="width: 120px;">Distress Type</th>
                <th style="width: 180px;">GPS Coordinates</th>
                <th>Civil Treatment Protocol (IRC:82-2015 Standard)</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <div class="footer">
        <div>
            <p><strong>System:</strong> AI Pothole Detection & Web-GIS Pavement Degradation Platform</p>
            <p><strong>Inspection Lead:</strong> {surveyor_name}</p>
        </div>
        <div>
            <div class="sign-block">
                Assistant Municipal Engineer (PWD)<br>
                Verification & Approval Signature
            </div>
        </div>
    </div>
</body>
</html>
"""
    return html
