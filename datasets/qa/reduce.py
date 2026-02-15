import json

INPUT_PATH = "medarabench.json"
OUTPUT_PATH = "test.json"

# Load full JSON
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Take first 5 samples
first_five = data[:5]

# Save to new file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(first_five, f, indent=2, ensure_ascii=False)

print(f"Saved {len(first_five)} samples to {OUTPUT_PATH}")
