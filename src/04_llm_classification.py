import os
import json
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

_SRC      = Path(__file__).parent
_WORKSPACE = _SRC.parent.parent          # root where .env lives

load_dotenv(dotenv_path=str(_WORKSPACE / ".env"))

_DATA     = _SRC.parent / "Data"
DATA_FILE = _DATA / "processed" / "complaints_clean.csv"
OUT_FILE  = _DATA / "results" / "glm52_predictions.csv"

os.makedirs(str(OUT_FILE.parent), exist_ok=True)

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

MODEL_NAME = os.getenv("GLM_MODEL", "glm-5.2")

LABELS = [
    "road_infrastructure",
    "waste_management",
    "water_utilities",
    "traffic_management",
    "street_lighting",
    "noise",
    "public_safety",
    "environment",
    "other",
]

SYSTEM_PROMPT = """
You are an urban complaint classification assistant for a smart city platform.
Classify each citizen complaint into exactly one category.

Allowed categories:
road_infrastructure, waste_management, water_utilities, traffic_management,
street_lighting, noise, public_safety, environment, other.

Return only valid JSON:
{
  "category": "...",
  "confidence": 0.0,
  "reason": "short reason"
}
"""


def classify_complaint(text: str) -> dict:
    user_prompt = f"""
Citizen complaint:
{text}

Classify the complaint into one allowed category.
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {
            "category":   "other",
            "confidence": 0.0,
            "reason":     "JSON parsing failed",
        }

    if parsed.get("category") not in LABELS:
        parsed["category"] = "other"

    return parsed


# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

# Evaluate on sample for deadline; increase for full run
df_sample = df.sample(min(1000, len(df)), random_state=42).copy()

predictions = []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    try:
        result = classify_complaint(row["text"])
    except Exception as e:
        result = {
            "category":   "other",
            "confidence": 0.0,
            "reason":     str(e),
        }
    predictions.append(result)
    time.sleep(0.2)  # Rate limit buffer

df_sample["glm52_pred"]       = [p["category"]   for p in predictions]
df_sample["glm52_confidence"] = [p["confidence"] for p in predictions]
df_sample["glm52_reason"]     = [p["reason"]     for p in predictions]

df_sample.to_csv(str(OUT_FILE), index=False)

print(f"Saved GLM-5.2 predictions ({len(df_sample)} rows) to {OUT_FILE}")
print(df_sample["glm52_pred"].value_counts().to_string())