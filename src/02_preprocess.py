import pandas as pd
importre

IN_FILE="data/processed/complaints_raw.csv"
OUT_FILE="data/processed/complaints_clean.csv"

taxonomy= {
"road_infrastructure": ["pothole","street condition","sidewalk","road"],
"waste_management": ["dirty","garbage","sanitation","missed collection"],
"water_utilities": ["water leak","sewer","drain","hydrant"],
"traffic_management": ["traffic signal","parking","blocked driveway"],
"street_lighting": ["street light","lighting","lamp"],
"noise": ["noise","loud music","construction noise"],
"public_safety": ["illegal activity","homeless","animal","danger"],
"environment": ["air","pollution","tree","dead animal"]
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", "", text)
    return text.strip()

def map_category(row):
    text = f"{row.get('complaint_type','')}{row.get('description','')}".lower()
    for category, keywords in taxonomy.items():
        if any(k in text for k in keywords):
            return category
    return "other"

df=pd.read_csv(IN_FILE)

df["text"]= (
df.get("complaint_type","").fillna("").astype(str)
+". "
+df.get("description","").fillna("").astype(str)
)

df["text"]=df["text"].apply(clean_text)
df["category"]=df.apply(map_category,axis=1)

df=df[df["text"].str.len()>10]
df=df.drop_duplicates(subset=["text"])

df.to_csv(OUT_FILE,index=False)

print(df["category"].value_counts())
print(f"Saved clean dataset: {OUT_FILE}")