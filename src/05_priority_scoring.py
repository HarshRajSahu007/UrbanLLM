import os
from pathlib import Path
import pandas as pd

_SRC     = Path(__file__).parent
_DATA    = _SRC.parent / "Data"

IN_FILE  = _DATA / "results" / "glm52_predictions.csv"
OUT_FILE = _DATA / "results" / "priority_scored.csv"

os.makedirs(str(OUT_FILE.parent), exist_ok=True)

# ── Priority base scores by category ──────────────────────────────────────
priority_rules = {
    "public_safety":      90,
    "traffic_management": 80,
    "water_utilities":    75,
    "road_infrastructure":65,
    "street_lighting":    55,
    "waste_management":   50,
    "environment":        45,
    "noise":              30,
    "other":              20,
}

# ── Keywords that escalate priority ───────────────────────────────────────
critical_keywords = [
    "danger", "injury", "accident", "fire", "collapse", "flood",
    "blocked road", "traffic signal out", "water main break",
    "emergency", "hazard", "unsafe", "explosion", "gas leak",
]

sensitive_locations = ["school", "hospital", "playground", "daycare", "clinic"]


def compute_priority(row) -> tuple:
    category = row.get("glm52_pred", "other")
    text = str(row.get("text", "")).lower()

    score = priority_rules.get(category, 20)

    # Escalate for critical keywords
    if any(k in text for k in critical_keywords):
        score += 15

    # Escalate near sensitive locations
    if any(loc in text for loc in sensitive_locations):
        score += 10

    score = min(score, 100)

    if score >= 85:
        level = "critical"
    elif score >= 65:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "low"

    return score, level


# ── Load and score ─────────────────────────────────────────────────────────
df = pd.read_csv(IN_FILE)

df[["priority_score", "priority_level"]] = df.apply(
    lambda row: pd.Series(compute_priority(row)),
    axis=1,
)

df.to_csv(str(OUT_FILE), index=False)

print(df[["glm52_pred", "priority_score", "priority_level"]].head(10).to_string())
print(f"\nPriority level distribution:\n{df['priority_level'].value_counts().to_string()}")
print(f"\nSaved priority results: {OUT_FILE}")