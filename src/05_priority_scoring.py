import pandas as pd

IN_FILE="data/results/glm52_predictions.csv"
OUT_FILE="data/results/priority_scored.csv"

priority_rules= {
"public_safety":90,
"traffic_management":80,
"water_utilities":75,
"road_infrastructure":65,
"street_lighting":55,
"waste_management":50,
"environment":45,
"noise":30,
"other":20
}

critical_keywords= [
"danger","injury","accident","fire","collapse","flood",
"blocked road","traffic signal out","water main break",
"emergency","hazard"
]

def compute_priority(row):
    category = row["glm52_pred"]
    text = str(row["text"]).lower()

    score = priority_rules.get(category, 20)

    if any(k in text for k in critical_keywords):
        score += 15

    if "school" in text or "hospital" in text:
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

df=pd.read_csv(IN_FILE)

df[["priority_score","priority_level"]]=df.apply(
lambda row: pd.Series(compute_priority(row)),
axis=1
)

df.to_csv(OUT_FILE,index=False)

print(df[["glm52_pred","priority_score","priority_level"]].head())
print(f"Saved priority results: {OUT_FILE}")