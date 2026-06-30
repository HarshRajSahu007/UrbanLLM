import os
from pathlib import Path
import pandas as pd

# ── Resolve paths relative to this file (works from any CWD) ──────────────
_ROOT        = Path(__file__).parent          # .../UrbanLLM/src/
_WORKSPACE   = _ROOT.parent.parent            # .../UrbanLLM/  (datasets live here)

NYC311_CLEAN = _WORKSPACE / "NYC311_cleaned_final.csv"
INDIA_FILE   = _WORKSPACE / "indian civic complaints research 8k+.csv"
OUT_FILE     = _ROOT.parent / "Data" / "processed" / "complaints_raw.csv"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── Load NYC311 cleaned dataset ────────────────────────────────────────────
# Columns present: unique_key, complaint_type, descriptor, descriptor_2,
#   agency, agency_name, borough, city, status, resolution_description, ...
nyc = pd.read_csv(NYC311_CLEAN, low_memory=False)

nyc_out = pd.DataFrame({
    "complaint_type":  nyc["complaint_type"].fillna("Unknown"),
    "description":     nyc["descriptor"].fillna("Unknown"),
    "agency":          nyc["agency_name"].fillna("Unknown"),
    "location":        nyc["borough"].fillna("Unknown"),
    "status":          nyc.get("status", pd.Series(["Unknown"] * len(nyc))).fillna("Unknown"),
    "source_dataset":  "NYC311",
})

print(f"NYC311 loaded: {len(nyc_out)} rows")

# ── Load Indian civic complaints dataset ───────────────────────────────────
# Columns present: complaint_id, clean_title, description, city, state,
#   search_keyword, source, published_date, ...
india = pd.read_csv(INDIA_FILE, low_memory=False)

india_out = pd.DataFrame({
    "complaint_type":  india["search_keyword"].fillna("Unknown"),
    "description":     (
        india["clean_title"].fillna("") + ". " + india["description"].fillna("")
    ).str.strip(". "),
    "agency":          "Unknown",
    "location":        india["city"].fillna("Unknown") + ", " + india["state"].fillna("Unknown"),
    "status":          "Unknown",
    "source_dataset":  "India",
})

print(f"India dataset loaded: {len(india_out)} rows")

# ── Merge and save ─────────────────────────────────────────────────────────
df = pd.concat([nyc_out, india_out], ignore_index=True)
df.to_csv(OUT_FILE, index=False)

print(f"\nCombined dataset: {len(df)} rows")
print(f"Source breakdown:\n{df['source_dataset'].value_counts().to_string()}")
print(f"\nSaved {len(df)} rows to {OUT_FILE}")
print(f"Columns: {list(df.columns)}")