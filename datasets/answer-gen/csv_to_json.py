import json
import csv

input_file = "medarabenchv2.csv"
output_file = "medarabenchv2.json"

with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    data = list(reader)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Converted {len(data)} records to {output_file}")