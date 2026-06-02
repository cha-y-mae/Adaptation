"""
contamination_check_gpt.py

Contamination probe for GPT-4.5 on MedAraBench.

Method:
    - Sample 200 random questions from MedAraBench JSON
    - Feed only the question stem to GPT, ask it to generate the answer options
    - Measure BERTScore (F1) between generated options and original options
      using CAMeLBERT as the backbone
    - Flag questions above a similarity threshold as potentially contaminated
    - Output per-question CSV + summary stats

Usage:
    pip install openai bert-score transformers
    export OPENAI_API_KEY="sk-..."
    python contamination_check_gpt.py \
        --input  medarabench.json \
        --output contamination_gpt.csv \
        [--sample 200] \
        [--threshold 0.85] \
        [--model gpt-4.5-preview] \
        [--seed 42]
"""

import argparse
import json
import csv
import os
import random
import time
from pathlib import Path

from openai import OpenAI
from bert_score import score as bert_score


# ── Config ─────────────────────────────────────────────────────────────────────

# CAMeLBERT MSA — Arabic-native BERT, best backbone for Arabic BERTScore
BERTSCORE_MODEL = "xlm-roberta-large"
SLEEP_SECS      = 0.5   # pause between API calls


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """أنت مساعد طبي. سيُعطى لك سؤال طبي بدون خيارات الإجابة.
مهمتك هي توليد أربعة خيارات إجابة محتملة (أ، ب، ج، د) للسؤال.
أخرج الخيارات فقط بالتنسيق التالي، بدون أي مقدمة أو شرح:
أ) ...
ب) ...
ج) ...
د) ..."""


def build_prompt(question: str) -> str:
    return f"السؤال: {question}\n\nأكمل الخيارات الأربعة:"


# ── Load data ──────────────────────────────────────────────────────────────────

def load_and_sample(json_path: str, n: int, seed: int) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict-of-list formats
    if isinstance(data, dict):
        items = list(data.values())
        # Flatten if values are lists
        flat = []
        for v in items:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        items = flat
    else:
        items = data

    # Filter out items missing required fields
    valid = [
        item for item in items
        if item.get("question") and item.get("opa") and item.get("opb")
        and item.get("opc") and item.get("opd")
    ]

    random.seed(seed)
    sample = random.sample(valid, min(n, len(valid)))
    print(f"Loaded {len(items)} items, sampled {len(sample)} valid questions.")
    return sample


# ── Format original options into a single string for BERTScore ─────────────────

def format_original_options(item: dict) -> str:
    return (
        f"أ) {item['opa']}\n"
        f"ب) {item['opb']}\n"
        f"ج) {item['opc']}\n"
        f"د) {item['opd']}"
    )


# ── GPT call ───────────────────────────────────────────────────────────────────

def generate_options(client: OpenAI, model: str, question: str) -> str:
    """Ask GPT to complete the answer options given only the question stem."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(question)},
            ],
            temperature=0.0,
            max_completion_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return ""


# ── BERTScore ──────────────────────────────────────────────────────────────────

def compute_bertscore(generated_list: list, reference_list: list) -> list:
    """
    Compute BERTScore F1 for each (generated, reference) pair.
    Returns a list of float F1 scores.
    """
    print(f"\nComputing BERTScore for {len(generated_list)} pairs "
          f"using {BERTSCORE_MODEL}...")
    P, R, F1 = bert_score(
        generated_list,
        reference_list,
        model_type=BERTSCORE_MODEL,
        lang="ar",
        verbose=False,
    )
    return F1.tolist()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,  help="Path to MedAraBench JSON")
    parser.add_argument("--output",    required=True,  help="Output CSV path")
    parser.add_argument("--sample",    type=int,   default=200)
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="BERTScore F1 above which a question is flagged")
    parser.add_argument("--model",     default="gpt-4.5-preview",
                        help="OpenAI model name")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    # Init
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    # Load & sample
    sample = load_and_sample(args.input, args.sample, args.seed)

    # Generate completions
    results        = []
    generated_list = []
    reference_list = []

    for i, item in enumerate(sample):
        q_id      = item.get("id", i)
        question  = item["question"]
        original  = format_original_options(item)
        answer    = item.get("answer", "")
        specialty = item.get("specialty", "")
        level     = item.get("level", "")

        print(f"[{i+1}/{len(sample)}] id={q_id} | {specialty}")
        generated = generate_options(client, args.model, question)

        if not generated:
            print(f"  [WARN] Empty generation, skipping.")
            continue

        generated_list.append(generated)
        reference_list.append(original)
        results.append({
            "id":               q_id,
            "specialty":        specialty,
            "level":            level,
            "answer":           answer,
            "question":         question,
            "original_options": original,
            "generated_options":generated,
            "bertscore_f1":     None,   # filled in after batch scoring
            "flagged":          None,
        })

        time.sleep(SLEEP_SECS)

    # Compute BERTScore in one batch (much faster than one-by-one)
    f1_scores = compute_bertscore(generated_list, reference_list)

    flagged_count = 0
    for row, f1 in zip(results, f1_scores):
        row["bertscore_f1"] = round(f1, 4)
        row["flagged"]      = f1 >= args.threshold
        if row["flagged"]:
            flagged_count += 1

    # Write CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "specialty", "level", "answer",
        "question", "original_options", "generated_options",
        "bertscore_f1", "flagged",
    ]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total     = len(results)
    avg_f1    = sum(r["bertscore_f1"] for r in results) / total if total else 0
    contam_pct = (flagged_count / total * 100) if total else 0

    print(f"\n{'='*50}")
    print(f"Model:               {args.model}")
    print(f"Dataset:             MedAraBench")
    print(f"Questions probed:    {total}")
    print(f"Avg BERTScore F1:    {avg_f1:.4f}")
    print(f"Flagged (≥{args.threshold}):     {flagged_count} / {total} ({contam_pct:.1f}%)")
    print(f"Output:              {args.output}")
    print(f"{'='*50}")

    # Also print flagged questions for quick inspection
    flagged = [r for r in results if r["flagged"]]
    if flagged:
        print(f"\nTop flagged questions (by F1):")
        for r in sorted(flagged, key=lambda x: x["bertscore_f1"], reverse=True)[:5]:
            print(f"  id={r['id']} | F1={r['bertscore_f1']} | {r['question'][:60]}...")


if __name__ == "__main__":
    main()