import os
import glob
from pathlib import Path
import pandas as pd

# ── Set up dynamic paths ──────────────
_ROOT = Path(__file__).parent          
RAW_DIR = _ROOT.parent / "Data" / "raw" 
OUT_FILE = _ROOT.parent / "Data" / "processed" / "complaints_raw.csv"

os.makedirs(OUT_FILE.parent, exist_ok=True)

# ── Find all CSVs in the raw folder ──────────────
csv_files = glob.glob(str(RAW_DIR / "*.csv"))
df_list = []

print(f"Scanning directory: {RAW_DIR}")

# ── Loop, load, and merge ──────────────
for file in csv_files:
    try:
        df = pd.read_csv(file, low_memory=False)
        # Ensure it has a source dataset column based on the file name
        if "source_dataset" not in df.columns:
            df["source_dataset"] = Path(file).stem
        
        df_list.append(df)
        print(f"SUCCESS: Loaded {Path(file).name} - {len(df)} rows")
    except Exception as e:
        print(f"ERROR loading {file}: {e}")

# ── Save merged output ──────────────
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(OUT_FILE, index=False)
    print(f"\nSuccessfully merged {len(csv_files)} files.")
    print(f"Total Combined dataset: {len(merged_df)} rows")
    print(f"Saved merged data to {OUT_FILE}")
else:
    print(f"WARNING: No CSV files found in {RAW_DIR}.")
    # Create empty dummy file so pipeline doesn't crash
    pd.DataFrame().to_csv(OUT_FILE, index=False)