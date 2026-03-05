import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# CONFIG
# =========================
HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_CACHE, "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# pick one that exists
TRAIN_FILE = os.path.join(REPO_ROOT, "datasets", "train", "train.json")   # or train.jsonl
EVAL_FILE  = None

# Path to your training data (JSONL recommended)
# Each line: {"id":..., "question":..., "opa":..., ... , "answer":"A", ...}
TRAIN_JSONL = "datasets/train/train.json"
EVAL_JSONL  = None  # optional: "datasets/qa/dev.jsonl"

OUTPUT_DIR = "checkpoints/mistral_small_lora_mcq"

# System instruction used in eval (keep identical!)
SYSTEM_PROMPT = "You are a medical assistant. Answer the multiple-choice question."

# =========================
# Helpers
# =========================
def build_mcq_text(sample: Dict) -> str:
    stem = (sample.get("question") or "").strip()
    if not stem:
        return ""

    option_map = {
        "A": sample.get("opa"),
        "B": sample.get("opb"),
        "C": sample.get("opc"),
        "D": sample.get("opd"),
        "E": sample.get("ope"),
        "F": sample.get("opf"),
    }

    lines = []
    for letter in ["A", "B", "C", "D", "E", "F"]:
        txt = option_map.get(letter)
        if txt is None:
            continue
        txt = str(txt).strip()
        if txt:
            lines.append(f"{letter}) {txt}")

    return stem + ("\n\n" + "\n".join(lines) if lines else "")

def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    ans = str(ans).strip().upper()
    m = re.search(r"\b([A-F])\b", ans)
    return m.group(1) if m else ""

def format_training_text(sample: Dict) -> Optional[str]:
    q = build_mcq_text(sample)
    if not q:
        return None
    gold = normalize_answer(sample.get("answer") or sample.get("cop") or "")
    if gold not in {"A","B","C","D","E","F"}:
        return None

    # Keep your eval anchor exactly
    return (
        f"{SYSTEM_PROMPT}\n"
        "Return only one letter (A-F) in the format: Answer: X\n\n"
        f"QUESTION:\n{q}\n\n"
        f"Answer: {gold}"
    )

# =========================
# Main
# =========================
def main():
    use_qlora = bool(int(os.getenv("USE_QLORA", "1")))  # default ON (safer for 24B)
    print(f"[train] USE_QLORA={use_qlora}")

    # ---- Load dataset
    data_files = {"train": TRAIN_FILE}
    if EVAL_FILE:
        data_files["eval"] = EVAL_FILE

    ds = load_dataset("json", data_files=data_files)

    def map_fn(example):
        text = format_training_text(example)
        return {"text": text if text is not None else ""}

    ds = ds.map(map_fn, remove_columns=ds["train"].column_names)

    # Filter out empty rows
    ds["train"] = ds["train"].filter(lambda x: len(x["text"]) > 0)
    if "eval" in ds:
        ds["eval"] = ds["eval"].filter(lambda x: len(x["text"]) > 0)

    # ---- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE,
        local_files_only=bool(int(os.getenv("HF_HUB_OFFLINE", "0"))),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Quant config (for QLoRA)
    bnb_config = None
    model_kwargs = dict(
        cache_dir=HF_CACHE,
        local_files_only=bool(int(os.getenv("HF_HUB_OFFLINE", "0"))),
    )

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs.update(dict(
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=None,  # training uses DDP typically
        ))
    else:
        model_kwargs.update(dict(
            torch_dtype=torch.bfloat16,
            device_map=None,
        ))

    # ---- Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    # QLoRA prep
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # ---- LoRA config
    # Mistral-like targets:
    # attention: q_proj, k_proj, v_proj, o_proj
    # mlp: gate_proj, up_proj, down_proj
    lora = LoraConfig(
        r=int(os.getenv("LORA_R", "16")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
        lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- Tokenization
    max_len = int(os.getenv("MAX_LEN", "1024"))

    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        return out

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Data collator for causal LM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- Training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "16")),
        learning_rate=float(os.getenv("LR", "2e-4")),
        num_train_epochs=float(os.getenv("EPOCHS", "1")),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        evaluation_strategy="no" if "eval" not in tokenized else "steps",
        eval_steps=200 if "eval" in tokenized else None,
        report_to=[],
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("eval"),
        data_collator=collator,
    )

    trainer.train()

    # Save adapters + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[train] Done. Saved LoRA adapters to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()