import json

input_file = "filtered.json"          # change this to your input JSON file
output_file = "test.json"

# Map answer letters to option field names
answer_map = {
    "A": "opa",
    "B": "opb",
    "C": "opc",
    "D": "opd",
    "E": "ope",
    "F": "opf"
}

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# If the JSON is a single object, turn it into a list temporarily
if isinstance(data, dict):
    data = [data]

# Add answer_text field
for row in data:
    answer_letter = str(row.get("answer", "")).strip().upper()
    option_field = answer_map.get(answer_letter)
    
    if option_field:
        row["answer_text"] = row.get(option_field)
    else:
        row["answer_text"] = None

# Save updated JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Saved updated JSON with answer_text to {output_file}")