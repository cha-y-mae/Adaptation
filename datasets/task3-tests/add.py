import pandas as pd

INPUT_CSV = "clinician-data.csv"
OUTPUT_CSV = "clinician-data-final.csv"

df = pd.read_csv(INPUT_CSV)
df.insert(0, "Case ID", range(1, len(df) + 1))
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Done! {len(df)} rows saved to {OUTPUT_CSV}")
print(df.head())