import os
from pathlib import Path
import pandas as pd

_SRC     = Path(__file__).parent
_DATA    = _SRC.parent / "Data"

IN_FILE  = _DATA / "results" / "priority_scored.csv"
OUT_FILE = _DATA / "results" / "routed_complaints.csv"

os.makedirs(str(OUT_FILE.parent), exist_ok=True)

# ── Routing table: category → department ──────────────────────────────────
routing_table = {
    "road_infrastructure": "Department of Roads and Public Works",
    "waste_management":    "Sanitation Department",
    "water_utilities":     "Water Utility Department",
    "traffic_management":  "Traffic Operations Department",
    "street_lighting":     "Street Lighting Maintenance Division",
    "noise":               "Environmental Control / Noise Regulation Unit",
    "public_safety":       "Public Safety and Emergency Response Department",
    "environment":         "Environmental Protection Department",
    "other":               "General Municipal Services",
}


def route_department(category: str) -> str:
    return routing_table.get(str(category).strip(), "General Municipal Services")


# ── Load, route, and save ──────────────────────────────────────────────────
df = pd.read_csv(IN_FILE)

df["assigned_department"] = df["glm52_pred"].apply(route_department)

df.to_csv(str(OUT_FILE), index=False)

print(df[["text", "glm52_pred", "priority_level", "assigned_department"]].head(10).to_string())
print(f"\nDepartment distribution:\n{df['assigned_department'].value_counts().to_string()}")
print(f"\nSaved routed complaints: {OUT_FILE}")