import pandas as pd

RAW_FILE="data/raw/311_complaints.csv"
OUT_FILE="data/processed/complaints_raw.csv"

df=pd.read_csv(RAW_FILE)

print(df.columns)
print(df.head())

# Adjust column names depending on dataset
df=df.rename(columns={
"Complaint Type":"complaint_type",
"Descriptor":"description",
"Created Date":"created_date",
"Agency":"agency",
"Borough":"location"
})

keep_cols= ["complaint_type","description","agency","location"]

df=df[[c for c in keep_cols if c in df.columns]]
df.to_csv(OUT_FILE,index=False)

print(f"Saved {len(df)} rows to {OUT_FILE}")