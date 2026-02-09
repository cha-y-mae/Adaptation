'''

This script implements metrics for mcq and answer_generation evaluation. It provides bleu/rouge/bertscore metrics

'''

import re
from typing import List, Optional, Dict, Any
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

try:
    import bert_score
except Exception:
    bert_score = None

try:
    from bleurt import score as bleurt_score
except Exception:
    bleurt_score = None

#------------------------------------------Task 1 ----------------------------------------------#

def extract_letter(text):
    if not text:
        return None
    t = str(text).strip()

    m = re.search(r"(?i)\banswer(?:\s*is)?\s*[:：]?\s*([A-F])\b", t)
    if m:
        return m.group(1).upper()

    first = next((ln for ln in t.splitlines() if ln.strip()), "")
    m = re.match(r"^\s*([A-Fa-f])\s*[\.\)]?\s*$", first)
    if m:
        return m.group(1).upper()

    m = re.match(r"^\s*([A-Fa-f])\s*$", t)
    if m:
        return m.group(1).upper()

    return None


def calculate_accuracy(predictions, ground_truths):
    correct = 0
    total = 0

    for p, g in zip(predictions, ground_truths):
        total += 1
        p_letter = extract_letter(p) if p is not None else None
        g_letter = extract_letter(g)

        if p_letter and g_letter and p_letter == g_letter:
            correct += 1

    return correct / total if total > 0 else 0.0


#------------------------------------------Task 2 ----------------------------------------------#


def calculate_bleu(predictions: List[str], references: List[str], lang: str = "en") -> float:
    smoothie = SmoothingFunction().method1
    scores = []

    for hyp, ref in zip(predictions, references):
        hyp = "" if hyp is None else str(hyp)
        ref = "" if ref is None else str(ref)

        ref_tokens = word_tokenize(ref, language=lang) if ref else []
        hyp_tokens = word_tokenize(hyp, language=lang) if hyp else []

        if not ref_tokens and not hyp_tokens:
            scores.append(1.0)
            continue
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue

        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        scores.append(float(bleu))

    return float(sum(scores) / len(scores)) if scores else 0.0


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1 = 0.0
    r2 = 0.0
    rl = 0.0
    n = 0

    for hyp, ref in zip(predictions, references):
        hyp = "" if hyp is None else str(hyp)
        ref = "" if ref is None else str(ref)

        if not hyp and not ref:
            r1 += 1.0
            r2 += 1.0
            rl += 1.0
            n += 1
            continue

        scores = scorer.score(ref, hyp)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure
        n += 1

    n = n or 1
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rl / n}


def calculate_bert_score(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    device: str = "cpu",
    rescale_with_baseline: bool = True,
) -> Optional[Dict[str, float]]:
    if bert_score is None:
        return None

    hyps = ["" if p is None else str(p) for p in predictions]
    refs = ["" if r is None else str(r) for r in references]

    if not hyps or not refs:
        return None

    try:
        P, R, F1 = bert_score.score(
            hyps,
            refs,
            lang=lang,
            model_type=model_type,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
        f1 = F1.detach().cpu().numpy()
        p = P.detach().cpu().numpy()
        r = R.detach().cpu().numpy()

        f1 = np.clip(f1, 0.0, 1.0)
        p = np.clip(p, 0.0, 1.0)
        r = np.clip(r, 0.0, 1.0)

        return {
            "bert_precision": float(p.mean()),
            "bert_recall": float(r.mean()),
            "bert_f1": float(f1.mean()),
        }
    except Exception:
        return None


