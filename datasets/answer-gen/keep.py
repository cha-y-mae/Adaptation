import pandas as pd

# Load CSV
input_file = "task2.csv"   # change to your file name
df = pd.read_csv(input_file)

# Filter rows where keep == "yes" (case-insensitive, handles spaces too)
filtered_df = df[df["keep"].astype(str).str.strip().str.lower() == "yes"]

# Save to JSON
output_file = "filtered_data.json"
filtered_df.to_json(output_file, orient="records", indent=4, force_ascii=False)

print(f"Saved {len(filtered_df)} rows to {output_file}")