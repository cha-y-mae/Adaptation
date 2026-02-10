import csv
import json
import re
from pathlib import Path

# Arabic option → English letter mapping
AR_TO_EN = {
    "أ": "A",
    "ب": "B",
    "ج": "C",
    "د": "D",
    "هـ": "E",
    "ه": "E",
    "و": "F",
}

OPTION_REGEX = re.compile(r"([أبجدهـو])\.\s*(.+)")

def parse_question(text):
    """
    Splits question stem and options.
    Returns: stem, dict {A: text, B: text, ...}
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    stem = lines[0]

    options = {}
    for line in lines[1:]:
        match = OPTION_REGEX.match(line)
        if match:
            ar_letter, option_text = match.groups()
            en_letter = AR_TO_EN.get(ar_letter)
            if en_letter:
                options[en_letter] = option_text.strip()

    return stem, options


def extract_answer(answer_text):
    """
    Extracts Arabic option letter from answer field and maps it to English.
    """
    match = OPTION_REGEX.match(answer_text.strip())
    if not match:
        raise ValueError(f"Could not parse answer: {answer_text}")

    ar_letter = match.group(1)
    return AR_TO_EN[ar_letter]


def csv_to_json(csv_path, json_path):
    output = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print("Detected headers:", reader.fieldnames)
        for idx, row in enumerate(reader, start=1):
            stem, options = parse_question(row["Question"])
            answer = extract_answer(row["Answer"])

            item = {
                "id": f"{row['Category'].lower()}_{idx:04d}",
                "question": stem,
                "category": row["Category"],
                "answer": answer,
            }

            # add options as opa, opb, opc...
            for letter, text in options.items():
                item[f"op{letter.lower()}"] = text

            output.append(item)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    csv_to_json(
        csv_path="medarabiq.csv",
        json_path="medarabiq.json"
    )
