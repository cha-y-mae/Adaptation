import json
import csv

input_file = "mmlu-arabic.json"
output_file = "mmlu-arabic.csv"

# load json
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# get all keys (in case some entries miss fields)
fieldnames = set()
for item in data:
    fieldnames.update(item.keys())

fieldnames = sorted(fieldnames)

# write csv
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"Converted {len(data)} records to {output_file}")