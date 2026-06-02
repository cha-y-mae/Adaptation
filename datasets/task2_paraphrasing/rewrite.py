"""
rewrite_questions.py

Pipeline:
  1. Drop rows where keep == "no"
  2. Rewrite question stems via Claude
  3. Add answer_text column (resolve answer letter → option text)
  4. Output CSV + JSON (strips review metadata col: keep)

Input CSV columns:
  id, level, question, specialty, umbrella_specialty,
  opa, opb, opc, opd, ope, opf, answer, group, keep

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python rewrite_questions.py \
        --input  path/to/input.csv \
        --output path/to/output \
        --prompt path/to/rewrite_prompt.txt
"""

import argparse
import csv
import json
import os
import re
import time
import sys
import anthropic


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL      = "claude-sonnet-4-20250514"
MAX_TOKENS = 200
DELAY_SECS = 0.3

ANSWER_MAP = {
    "A": "opa",
    "B": "opb",
    "C": "opc",
    "D": "opd",
    "E": "ope",
    "F": "opf",
}

COLS_TO_DROP = {"keep"}

# ── Rewriter ───────────────────────────────────────────────────────────────────

def rewrite_question(client: anthropic.Anthropic, system_prompt: str, question_stem: str) -> str:
    """Send a question stem to Claude and return the rewritten version."""
    clean = question_stem.strip().lstrip(".")
    clean = re.sub(r"\(\s*\d{4}[^)]*\)", "", clean).strip()

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": clean}],
    )
    return response.content[0].text.strip()


# ── answer_text resolver ───────────────────────────────────────────────────────

def resolve_answer_text(row: dict) -> str | None:
    """Map answer letter (A-F) to the corresponding option text."""
    letter = str(row.get("answer", "")).strip().upper()
    option_field = ANSWER_MAP.get(letter)
    if option_field:
        return row.get(option_field) or None
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rewrite Arabic MCQ stems and resolve answer text.")
    parser.add_argument("--input",  required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output base path (no extension)")
    parser.add_argument("--prompt", required=True, help="Path to system prompt .txt file")
    args = parser.parse_args()

    output_base = re.sub(r"\.(csv|json)$", "", args.output, flags=re.IGNORECASE)
    output_csv  = output_base + ".csv"
    output_json = output_base + ".json"

    # Load system prompt
    if not os.path.exists(args.prompt):
        print(f"ERROR: Prompt file not found: {args.prompt}")
        sys.exit(1)
    with open(args.prompt, encoding="utf-8") as f:
        system_prompt = f.read().strip()
    print(f"Loaded system prompt ({len(system_prompt)} chars)")

    # Init Anthropic client
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    # Load input CSV
    if not os.path.exists(args.input):
        print(f"ERROR: Input CSV not found: {args.input}")
        sys.exit(1)
    with open(args.input, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {args.input}")

    # ── Step 1: Split by keep value ────────────────────────────────────────────
    to_keep = []
    to_drop = []
    for row in rows:
        keep_val = (row.get("keep") or "").strip().lower()
        if keep_val == "no":
            to_drop.append(row)
        else:
            to_keep.append(row)

    print(f"  Keeping:  {len(to_keep)} rows")
    print(f"  Dropping: {len(to_drop)} rows (keep=no)")

    # ── Step 2: Rewrite + resolve answer_text ─────────────────────────────────
    output_rows = []
    failed_ids  = []

    for i, row in enumerate(to_keep, 1):
        row_id   = row.get("id", i)
        question = (row.get("question") or "").strip()

        print(f"[{i}/{len(to_keep)}] id={row_id} | rewriting...")

        try:
            rewritten = rewrite_question(client, system_prompt, question)
        except Exception as e:
            print(f"  ✗ API error: {e}")
            rewritten = ""

        if not rewritten:
            print(f"  [WARN] Empty rewrite for id={row_id}, keeping original")
            rewritten = question
            failed_ids.append(row_id)

        print(f"  Original : {question[:80]}")
        print(f"  Rewritten: {rewritten[:80]}")

        out_row = {k: v for k, v in row.items() if k not in COLS_TO_DROP}
        out_row["question_old"] = out_row.pop("question")  # rename original to question_old
        out_row["question"]     = rewritten
        out_row["answer_text"]  = resolve_answer_text(row)
        output_rows.append(out_row)

        time.sleep(DELAY_SECS)

    # ── Step 3: Write outputs ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # CSV
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\n✅ CSV saved  → {output_csv}")

    # JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON saved → {output_json}")

    # Dropped rows log
    if to_drop:
        dropped_path = output_base + "_dropped.csv"
        with open(dropped_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(to_drop[0].keys()))
            writer.writeheader()
            writer.writerows(to_drop)
        print(f"✅ Dropped   → {dropped_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\nDone.")
    print(f"  Input rows:    {len(rows)}")
    print(f"  Output rows:   {len(output_rows)}")
    print(f"  Dropped:       {len(to_drop)}")
    if failed_ids:
        print(f"  Rewrite failures (kept original): {len(failed_ids)} → ids: {failed_ids}")


if __name__ == "__main__":
    main()