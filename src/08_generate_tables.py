import os
from pathlib import Path
import pandas as pd

_SRC     = Path(__file__).parent
_DATA    = _SRC.parent / "Data"

IN_FILE  = _DATA / "results" / "routed_complaints.csv"
OUT_DIR  = _DATA / "results"

os.makedirs(str(OUT_DIR), exist_ok=True)

df = pd.read_csv(IN_FILE)

# ── Table 1: Dataset category distribution ────────────────────────────────
table1 = df["category"].value_counts().reset_index()
table1.columns = ["Complaint Category", "Number of Samples"]
path1 = OUT_DIR / "table1_dataset_distribution.csv"
table1.to_csv(str(path1), index=False)
print(f"Table 1 saved: {path1}")

# ── Table 2: Priority level distribution ──────────────────────────────────
table2 = df["priority_level"].value_counts().reset_index()
table2.columns = ["Priority Level", "Number of Complaints"]
path2 = OUT_DIR / "table2_priority_distribution.csv"
table2.to_csv(str(path2), index=False)
print(f"Table 2 saved: {path2}")

# ── Table 3: Department routing distribution ──────────────────────────────
table3 = df["assigned_department"].value_counts().reset_index()
table3.columns = ["Assigned Department", "Number of Complaints"]
path3 = OUT_DIR / "table3_department_routing.csv"
table3.to_csv(str(path3), index=False)
print(f"Table 3 saved: {path3}")

# ── Table 4: Source dataset breakdown ─────────────────────────────────────
if "source_dataset" in df.columns:
    table4 = df["source_dataset"].value_counts().reset_index()
    table4.columns = ["Source Dataset", "Number of Complaints"]
    path4 = OUT_DIR / "table4_source_breakdown.csv"
    table4.to_csv(str(path4), index=False)
    print(f"Table 4 saved: {path4}")

print("\nAll paper tables saved successfully.")