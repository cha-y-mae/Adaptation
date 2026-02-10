import csv
import json

INPUT_FILE = "test.csv"
OUTPUT_FILE = "mmlu-arabic.json"

NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

def normalize_answer(x: str) -> str:
    """
    Accepts:
      - "A"/"B"/... (returns same)
      - 1..5 -> A..E
    """
    if x is None:
        return ""
    s = str(x).strip()
    if s in ["A", "B", "C", "D", "E", "F"]:
        return s
    return NUM_TO_LETTER.get(s, s)  # fallback

out_data = []

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    # clean header whitespace
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    for row in reader:
        subject = str(row.get("Subject", "")).strip()
        if subject.lower() != "biology":
            continue

        item = {
            "id": str(row.get("ID", "")).strip(),
            "question": str(row.get("Question", "")).strip(),
            "answer": normalize_answer(row.get("Answer Key", "")),
        }

        # options 1..5 -> opa..ope
        opt_map = [("Option 1", "opa"), ("Option 2", "opb"), ("Option 3", "opc"),
                   ("Option 4", "opd"), ("Option 5", "ope")]
        for src, dst in opt_map:
            val = str(row.get(src, "")).strip()
            if val:
                item[dst] = val

        # metadata (optional)
        for src, dst in [
            ("Context", "context"),
            ("Level", "level"),
            ("Group", "group"),
            ("Source", "source"),
            ("Country", "country"),
            ("Subject", "subject"),
            ("is_few_shot", "is_few_shot"),
        ]:
            val = row.get(src, "")
            if val is not None:
                val = str(val).strip()
            if val != "":
                item[dst] = val

        out_data.append(item)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(out_data)} Biology samples to {OUTPUT_FILE}")
