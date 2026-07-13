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
    max_retries = 5
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
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # Ensure parsed output is a dict and has all required fields
            if not isinstance(parsed, dict):
                parsed = {
                    "category":   "other",
                    "confidence": 0.0,
                    "reason":     "Response was not a JSON object",
                }
            
            if "category" not in parsed:
                parsed["category"] = "other"
            if "confidence" not in parsed:
                parsed["confidence"] = 0.0
            if "reason" not in parsed:
                parsed["reason"] = "No reason provided by LLM"
                
            if parsed.get("category") not in LABELS:
                parsed["category"] = "other"
                
            parsed["raw_output"] = content
            return parsed
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"API call failed after {max_retries} attempts: {e}")
                raise e
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff:.1f} seconds...")
            time.sleep(backoff)
            backoff *= 2.0


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