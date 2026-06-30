import os
import re
from pathlib import Path
import pandas as pd

_SRC     = Path(__file__).parent
_DATA    = _SRC.parent / "Data"

IN_FILE  = _DATA / "processed" / "complaints_raw.csv"
OUT_FILE = _DATA / "processed" / "complaints_clean.csv"

os.makedirs(str(OUT_FILE.parent), exist_ok=True)

# ── Taxonomy mapping ───────────────────────────────────────────────────────
taxonomy = {
    "road_infrastructure": [
        "pothole", "street condition", "sidewalk", "road", "pavement",
        "curb", "asphalt", "bridge", "highway"
    ],
    "waste_management": [
        "dirty", "garbage", "sanitation", "missed collection", "litter",
        "recycling", "graffiti", "dead animal", "bulk item"
    ],
    "water_utilities": [
        "water leak", "sewer", "drain", "hydrant", "flooding", "water main",
        "sewage", "water pressure", "water supply"
    ],
    "traffic_management": [
        "traffic signal", "parking", "blocked driveway", "traffic light",
        "illegal parking", "double parking", "traffic"
    ],
    "street_lighting": [
        "street light", "lighting", "lamp", "light out", "dark street"
    ],
    "noise": [
        "noise", "loud music", "construction noise", "loud",
        "noise complaint", "barking"
    ],
    "public_safety": [
        "illegal activity", "homeless", "animal", "danger", "unsafe",
        "hazard", "emergency", "fire", "safety", "assault"
    ],
    "environment": [
        "air", "pollution", "tree", "dead tree", "fallen tree",
        "environmental", "air quality", "toxic", "pesticide"
    ],
}


def clean_text(text: str) -> str:
    """Lowercase, normalise whitespace, remove non-ASCII punctuation."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", "", text)
    return text.strip()


def map_category(row) -> str:
    """Rule-based taxonomy mapping using complaint_type + description."""
    combined = (
        str(row["complaint_type"]).lower()
        + " "
        + str(row["description"]).lower()
    )
    for category, keywords in taxonomy.items():
        if any(k in combined for k in keywords):
            return category
    return "other"


# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_FILE, low_memory=False)

# ── Build unified text field ───────────────────────────────────────────────
df["text"] = (
    df["complaint_type"].fillna("").astype(str)
    + ". "
    + df["description"].fillna("").astype(str)
)

df["text"] = df["text"].apply(clean_text)

# ── Assign rule-based category ─────────────────────────────────────────────
df["category"] = df.apply(map_category, axis=1)

# ── Basic quality filtering ────────────────────────────────────────────────
df = df[df["text"].str.len() > 10]
df = df.drop_duplicates(subset=["text"])

# ── Save ───────────────────────────────────────────────────────────────────
df.to_csv(str(OUT_FILE), index=False)

print(df["category"].value_counts().to_string())
print(f"\nSource breakdown:\n{df['source_dataset'].value_counts().to_string()}")
print(f"\nSaved clean dataset ({len(df)} rows): {OUT_FILE}")