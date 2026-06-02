"""
Arabic Medical Dialogue Generator
===================================
Reads clinical case templates from a CSV and uses the Anthropic API (Claude)
to generate realistic Arabic patient–doctor assistant dialogues.

Input CSV columns:
    Clinician Name | Organ System | Problem Type | Patient Profile |
    Symptoms & History | Red Flag Symptoms | Primary Reasoning Objective

Outputs:
    dialogues.json
        Full structured JSON: metadata + all dialogue turns (including final patient turn).

    dialogues.csv
        One row per case. Contains all case metadata, the full dialogue as formatted
        text (EXCLUDING the last assistant turn — left blank for clinicians to fill in),
        and an empty 'ideal_response' column for clinicians to complete.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-..."
    python generate_arabic_dialogues.py
"""

import anthropic
import csv
import json
import time
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_CSV   = "yousra.csv"
OUTPUT_JSON = "yousra-dialogue.json"
OUTPUT_CSV  = "yousra-dialogue.csv"
MODEL       = "claude-sonnet-4-20250514"
MAX_TOKENS  = 1500
DELAY_SECS  = 1.0   # seconds between API calls (rate limit buffer)

# ── Prompt Templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are generating a realistic medical conversation in Arabic between 
a patient and a medical assistant chatbot. Use ONLY the information provided in the 
case template. Do not introduce new medical history, vital signs, or test results 
unless they logically emerge through dialogue clarification.

Requirements:
- Write in natural Modern Standard Arabic (فصحى).
- Speakers: patient (مريض) and medical assistant (مساعد طبي).
- Total turns: between 3 and 8.
- The LAST turn MUST be from the patient.
- The assistant may ask clarifying questions when needed.
- Do not escalate to emergency unless clearly justified by the dialogue.
- Do not add numeric thresholds or clinical data not in the template.

Output format — output ONLY valid JSON, no markdown, no preamble:
{
  "turns": [
    {"speaker": "مريض",       "text": "..."},
    {"speaker": "مساعد طبي",  "text": "..."},
    ...
  ]
}"""


def build_user_prompt(row: dict) -> str:
    return f"""Case Template:
- Clinician: {row.get('Clinician Name', 'N/A')}
- Organ System: {row.get('Organ System', 'N/A')}
- Problem Type: {row.get('Problem Type', 'N/A')}
- Patient Profile: {row.get('Patient Profile', 'N/A')}
- Symptoms & History: {row.get('Symptoms & History', 'N/A')}
- Red Flag Symptoms: {row.get('Red Flag Symptoms', 'N/A')}
- Primary Reasoning Objective: {row.get('Primary Reasoning Objective', 'N/A')}

Generate the Arabic dialogue now."""


# ── API Call ───────────────────────────────────────────────────────────────────

def generate_dialogue(client: anthropic.Anthropic, row: dict) -> list:
    """Call Claude and return a list of turn dicts: [{speaker, text}, ...]"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(row)}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    return parsed.get("turns", [])


# ── Dialogue formatter ─────────────────────────────────────────────────────────

def format_dialogue_text(turns: list, omit_last: bool = False) -> str:
    """
    Convert turns list into a readable plain-text block.
    If omit_last=True, the final turn (must be patient) is included but
    the assistant response slot is left blank — that is what the clinician fills in.

    Format:
        مريض: ...
        مساعد طبي: ...
        مريض: ...        ← last patient turn, always included
    """
    display_turns = turns  # all turns including final patient turn
    lines = [f"{t['speaker']}: {t['text']}" for t in display_turns]
    return "\n".join(lines)


# ── Build JSON record ──────────────────────────────────────────────────────────

def build_json_record(case_id: int, row: dict, turns: list) -> dict:
    """One richly-labelled JSON object per case. Stores ALL turns."""
    return {
        "case_id": case_id,
        "metadata": {
            "clinician_name":              row.get("Clinician Name", ""),
            "organ_system":                row.get("Organ System", ""),
            "problem_type":                row.get("Problem Type", ""),
            "patient_profile":             row.get("Patient Profile", ""),
            "symptoms_and_history":        row.get("Symptoms & History", ""),
            "red_flag_symptoms":           row.get("Red Flag Symptoms", ""),
            "primary_reasoning_objective": row.get("Primary Reasoning Objective", ""),
        },
        "dialogue": {
            "language":   "Arabic (Modern Standard)",
            "turn_count": len(turns),
            "turns": [
                {
                    "turn_index":   i + 1,
                    "speaker":      t.get("speaker", ""),
                    "speaker_role": (
                        "patient" if "مريض" in t.get("speaker", "")
                        else "assistant"
                    ),
                    "text": t.get("text", ""),
                }
                for i, t in enumerate(turns)
            ],
        },
    }


# ── Build CSV row ──────────────────────────────────────────────────────────────

# CSV columns:
#   Case metadata (7 cols) | dialogue text (all turns up to + including last
#   patient turn, NO final assistant turn) | ideal_response (blank for clinician)

CSV_FIELDNAMES = [
    "case_id",
    "clinician_name",
    "organ_system",
    "problem_type",
    "patient_profile",
    "symptoms_and_history",
    "red_flag_symptoms",
    "primary_reasoning_objective",
    "dialogue",        # full conversation text; last turn is patient; no assistant reply
    "ideal_response",  # LEFT BLANK — clinician fills in the ideal final assistant turn
]


def build_csv_row(record: dict) -> dict:
    """
    One CSV row per case.
    'dialogue' contains all turns INCLUDING the final patient message.
    'ideal_response' is blank — clinicians write the best possible assistant reply.
    """
    meta  = record["metadata"]
    turns = record["dialogue"]["turns"]

    # The generated dialogue already ends with a patient turn (enforced by prompt).
    # We show the full dialogue so clinicians have complete context,
    # then they write the ideal final assistant response in 'ideal_response'.
    dialogue_text = format_dialogue_text(turns)

    return {
        "case_id":                     record["case_id"],
        "clinician_name":              meta["clinician_name"],
        "organ_system":                meta["organ_system"],
        "problem_type":                meta["problem_type"],
        "patient_profile":             meta["patient_profile"],
        "symptoms_and_history":        meta["symptoms_and_history"],
        "red_flag_symptoms":           meta["red_flag_symptoms"],
        "primary_reasoning_objective": meta["primary_reasoning_objective"],
        "dialogue":                    dialogue_text,
        "ideal_response":              "",   # blank for clinician annotation
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"ERROR: Input file '{INPUT_CSV}' not found.")
        sys.exit(1)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    all_json_records = []
    all_csv_rows     = []

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total = len(reader)
    print(f"Found {total} case(s) in '{INPUT_CSV}'. Starting generation...\n")

    for idx, row in enumerate(reader, start=66):
        case_id = idx
        print(f"[{idx}/{total}] Generating: "
              f"{row.get('Organ System', '?')} — {row.get('Problem Type', '?')}")

        try:
            turns  = generate_dialogue(client, row)
            record = build_json_record(case_id, row, turns)
            all_json_records.append(record)
            all_csv_rows.append(build_csv_row(record))
            print(f"         ✓ {len(turns)} turns generated.")

        except json.JSONDecodeError as e:
            print(f"         ✗ JSON parse error for case {case_id}: {e}")
            all_json_records.append({
                "case_id":  case_id,
                "error":    "JSON parse error",
                "metadata": dict(row),
            })

        except Exception as e:
            print(f"         ✗ API error for case {case_id}: {e}")
            all_json_records.append({
                "case_id":  case_id,
                "error":    str(e),
                "metadata": dict(row),
            })

        if idx < total:
            time.sleep(DELAY_SECS)

    # ── Write JSON ──────────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_info": {
                    "description": "Arabic medical patient–assistant dialogues",
                    "language":    "Modern Standard Arabic",
                    "model_used":  MODEL,
                    "total_cases": total,
                },
                "cases": all_json_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n✅ JSON saved  → '{OUTPUT_JSON}'")

    # ── Write CSV ───────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        # utf-8-sig adds BOM so Excel opens Arabic text correctly
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"✅ CSV saved   → '{OUTPUT_CSV}'")
    print(f"\nDone! {len(all_json_records)} case(s) processed.")


if __name__ == "__main__":
    main()