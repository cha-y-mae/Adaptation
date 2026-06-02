"""
Standalone LLM-as-judge for Task 2 / Task 3 predictions.
Reads an existing predictions CSV, adds judge labels, updates metrics JSON.

Usage:
    export OPENAI_API_KEY="sk-..."
    python run_judge_standalone.py \
        --predictions_csv results/lora_task2_ar.csv \
        --metrics_json   results/lora_task2_ar_metrics.json \
        [--judge_prompt_file prompts/judge_task2.txt]
"""

import os
import re
import json
import time
import logging
import argparse

import pandas as pd
from openai import OpenAI

# ── Constants ────────────────────────────────────────────────────────────────
MODEL        = "gpt-4o"
MAX_TOKENS   = 50
TEMPERATURE  = 0
DELAY_SECS   = 0.5
MAX_RETRIES  = 3
RETRY_DELAY  = 5.0
VALID_LABELS = {"Correct", "Incorrect"}

DEFAULT_SYSTEM = (
    "You are an expert medical evaluator. You will be given a medical question, "
    "a reference answer, and a generated answer. Evaluate whether the generated "
    "answer is correct and respond only with the label in brackets []."
)
DEFAULT_USER = (
    "Question: {question_stem}\n"
    "Reference answer: {reference_answer}\n"
    "Generated answer: {generated_answer}\n\n"
    "Is the generated answer correct?\n"
    "Respond with exactly one of: [Correct] or [Incorrect]."
)


def load_prompt(prompt_path=None):
    if not prompt_path:
        return DEFAULT_SYSTEM, DEFAULT_USER
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "---" in content:
        system_part, user_part = content.split("---", 1)
        return system_part.strip(), user_part.strip()
    return DEFAULT_SYSTEM, content.strip()


def parse_label(raw: str) -> str:
    m = re.search(r"\[([^\]]+)\]", raw)
    if m:
        candidate = m.group(1).strip()
        for label in VALID_LABELS:
            if candidate.lower() == label.lower():
                return label
    for label in VALID_LABELS:
        if label.lower() in raw.lower():
            return label
    logging.warning(f"[judge] could not parse label from: {repr(raw)}")
    return ""


def call_judge(client, system_prompt, user_template, question_stem, reference_answer, generated_answer):
    user_prompt = user_template.format(
        question_stem=question_stem,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return parse_label(response.choices[0].message.content.strip())
        except Exception as e:
            logging.warning(f"[judge] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    logging.error("[judge] all retries exhausted")
    return ""


def compute_judge_metrics(labels):
    total  = len(labels)
    valid  = [l for l in labels if l in VALID_LABELS]
    counts = {label: valid.count(label) for label in VALID_LABELS}
    pcts   = {
        f"{label.lower()}_pct": round(counts[label] / len(valid) * 100, 2) if valid else 0.0
        for label in VALID_LABELS
    }
    return {
        "judge_model":     MODEL,
        "judge_total":     total,
        "judge_valid":     len(valid),
        "judge_empty":     total - len(valid),
        "judge_correct":   counts["Correct"],
        "judge_incorrect": counts["Incorrect"],
        **pcts,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv",  type=str, required=True,
                        help="CSV produced by inference script (needs id, input, prediction, ground_truth cols)")
    parser.add_argument("--metrics_json",     type=str, required=True,
                        help="Metrics JSON to update in-place with judge results")
    parser.add_argument("--judge_prompt_file", type=str, default=None,
                        help="Optional prompt txt file (--- separator). Uses built-in default if omitted.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    system_prompt, user_template = load_prompt(args.judge_prompt_file)

    df    = pd.read_csv(args.predictions_csv)
    total = len(df)
    logging.info(f"Loaded {total} rows from {args.predictions_csv}")

    labels = []
    for i, row in df.iterrows():
        logging.info(f"[{i+1}/{total}] judging id={row.get('id', i)}")
        label = call_judge(
            client=client,
            system_prompt=system_prompt,
            user_template=user_template,
            question_stem=str(row["input"]).strip(),
            reference_answer=str(row["ground_truth"]).strip(),
            generated_answer=str(row["prediction"]).strip(),
        )
        labels.append(label)
        logging.info(f"  → {label!r}")
        if i < total - 1:
            time.sleep(DELAY_SECS)

    df["judge_label"] = labels
    df.to_csv(args.predictions_csv, index=False, encoding="utf-8")
    logging.info(f"Saved predictions with judge labels to {args.predictions_csv}")

    judge_metrics = compute_judge_metrics(labels)
    logging.info(f"Judge metrics:\n{json.dumps(judge_metrics, indent=2, ensure_ascii=False)}")

    # merge into existing metrics JSON
    existing = {}
    if os.path.exists(args.metrics_json):
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(judge_metrics)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)
    logging.info(f"Updated metrics JSON at {args.metrics_json}")


if __name__ == "__main__":
    main()