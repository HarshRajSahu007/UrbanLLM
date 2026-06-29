import pandas as pd

IN_FILE="data/results/priority_scored.csv"
OUT_FILE="data/results/routed_complaints.csv"

routing_table= {
"road_infrastructure":"Department of Roads and Public Works",
"waste_management":"Sanitation Department",
"water_utilities":"Water Utility Department",
"traffic_management":"Traffic Operations Department",
"street_lighting":"Street Lighting Maintenance Division",
"noise":"Environmental Control / Noise Regulation Unit",
"public_safety":"Public Safety and Emergency Response Department",
"environment":"Environmental Protection Department",
"other":"General Municipal Services"
}

def route_department(category):
    return routing_table.get(category,"General Municipal Services")

df=pd.read_csv(IN_FILE)

df["assigned_department"]=df["glm52_pred"].apply(route_department)

df.to_csv(OUT_FILE,index=False)

print(df[["text","glm52_pred","priority_level","assigned_department"]].head())
print(f"Saved routed complaints:{OUT_FILE}")