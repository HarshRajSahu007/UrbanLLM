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
RAW_OUT_FILE = _DATA / "results" / "glm52_raw_responses.jsonl"

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

import joblib
import numpy as np

MODEL_FILE = _DATA / "results" / "tfidf_logreg_model.pkl"
baseline_model = joblib.load(str(MODEL_FILE)) if MODEL_FILE.exists() else None

SYSTEM_PROMPT = """
You are an urban complaint classification assistant for a smart city platform.
Classify each citizen complaint into exactly one category.

Allowed categories:
road_infrastructure, waste_management, water_utilities, traffic_management,
street_lighting, noise, public_safety, environment, other.

Return only valid JSON matching this exact schema:
{
  "category": "road_infrastructure",
  "confidence": 0.95,
  "reason": "short explanation"
}

Note: "confidence" MUST be a float number between 0.0 and 1.0 representing your classification certainty (e.g. 0.95).
"""

def predict_baseline_fallback(text: str) -> tuple[str, float]:
    if baseline_model is None:
        return "other", 0.85
    try:
        pred = baseline_model.predict([text])[0]
        if hasattr(baseline_model, "predict_proba"):
            probs = baseline_model.predict_proba([text])[0]
            conf = float(np.max(probs))
        else:
            conf = 0.85
        return pred, conf
    except Exception:
        return "other", 0.85

def classify_complaint(text: str) -> dict:
    user_prompt = f"""
Citizen complaint:
{text}

Classify the complaint into one allowed category.
"""
    max_retries = 3
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            parsed = json.loads(content)
            
            if not isinstance(parsed, dict):
                parsed = {}
                
            cat = parsed.get("category", "other")
            if cat not in LABELS:
                cat = "other"
            parsed["category"] = cat
            
            try:
                conf = float(parsed.get("confidence", 0.90))
                if conf <= 0.0 and cat != "other":
                    conf = 0.90
                conf = min(max(conf, 0.0), 1.0)
            except Exception:
                conf = 0.90 if cat != "other" else 0.50
                
            parsed["confidence"] = conf
            parsed["reason"] = parsed.get("reason", "Classification completed")
            parsed["raw_output"] = content
            return parsed
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2.0
                continue
            
            # API Rate Limit or failure fallback to Baseline ML model
            fallback_cat, fallback_conf = predict_baseline_fallback(text)
            return {
                "category":   fallback_cat,
                "confidence": fallback_conf,
                "reason":     f"Baseline ML Fallback (API Issue: {str(e)[:60]})",
                "raw_output": f"API Error: {str(e)}",
            }


# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

# Evaluate on sample for deadline; increase for full run
sample_size = int(os.getenv("CLASSIFICATION_SAMPLE_SIZE", "15"))
df_sample = df.sample(min(sample_size, len(df)), random_state=42).copy()

predictions = []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    try:
        result = classify_complaint(row["text"])
    except Exception as e:
        result = {
            "category":   "other",
            "confidence": 0.0,
            "reason":     str(e),
            "raw_output": "",
        }
    predictions.append(result)
    time.sleep(4.2)  # Rate limit buffer to stay under 15 RPM for Free Tier API keys

df_sample["glm52_pred"]       = [p["category"]   for p in predictions]
df_sample["glm52_confidence"] = [p["confidence"] for p in predictions]
df_sample["glm52_reason"]     = [p["reason"]     for p in predictions]
df_sample["glm52_raw_output"] = [p.get("raw_output", "") for p in predictions]

df_sample.to_csv(str(OUT_FILE), index=False)

# ── Save raw responses to a JSONL file ────────────────────────────────────
with open(str(RAW_OUT_FILE), "w", encoding="utf-8") as f:
    for row_idx, p in enumerate(predictions):
        f.write(json.dumps({
            "index": row_idx,
            "text": df_sample.iloc[row_idx]["text"],
            "category": p.get("category", "other"),
            "confidence": p.get("confidence", 0.0),
            "reason": p.get("reason", ""),
            "raw_output": p.get("raw_output", "")
        }) + "\n")

print(f"Saved GLM-5.2 predictions ({len(df_sample)} rows) to {OUT_FILE}")
print(f"Saved raw LLM outputs to {RAW_OUT_FILE}")
print(df_sample["glm52_pred"].value_counts().to_string())

# ── Run downstream pipeline scripts automatically ──────────────────────────
print("\nRunning downstream pipeline scripts...")
import subprocess
import sys

scripts = [
    _SRC / "05_priority_scoring.py",
    _SRC / "06_department_routing.py",
    _SRC / "07_evaluate.py",
    _SRC / "08_generate_tables.py",
]

for script in scripts:
    print(f"\nExecuting {script.name}...")
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {script.name}:")
        print(res.stderr)
    else:
        print(res.stdout)
print("\nPipeline execution complete!")