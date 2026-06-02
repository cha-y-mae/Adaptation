import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from transformers import (
    Trainer,
    TrainingArguments,
    Mistral3ForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LETTER_SET = {"A", "B", "C", "D", "E", "F"}

# ---------------------------------------------------------------------------
# Layer index conversion:
#   paper notation L_n = output of transformer block (n-1) in 0-indexed terms.
#   e.g. L32 = block 31,  L24 = block 23,  L40 = block 39.
#
# Defaults replicate Exp 1 (L32-L40 = blocks 31-39).
# For Exp 1b (L24-L40) pass:  --layer_start 23 --layer_end 39
# ---------------------------------------------------------------------------


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def normalize_letter(x: Any) -> str:
    x = str(x).strip().upper()
    if x not in LETTER_SET:
        raise ValueError(f"Invalid answer letter: {x}")
    return x


def build_mcq_text(sample: dict) -> str:
    stem = (sample.get("question") or "").strip()
    if not stem:
        return ""

    option_fields = {
        "A": sample.get("opa"),
        "B": sample.get("opb"),
        "C": sample.get("opc"),
        "D": sample.get("opd"),
        "E": sample.get("ope"),
        "F": sample.get("opf"),
    }

    lines = []
    for letter in ["A", "B", "C", "D", "E", "F"]:
        val = option_fields.get(letter)
        if val is None:
            continue
        val = str(val).strip()
        if val:
            lines.append(f"{letter}) {val}")

    if not lines:
        return stem
    return stem + "\n\n" + "\n".join(lines)


def load_tokenizer(model_name: str) -> MistralTokenizer:
    if os.path.isdir(model_name):
        tok_path = os.path.join(model_name, "tekken.json")
        return MistralTokenizer.from_file(tok_path)
    return MistralTokenizer.from_hf_hub(model_name)


def tokenize_target_text(tokenizer: MistralTokenizer, text: str) -> List[int]:
    text = text.strip()
    if hasattr(tokenizer, "instruct_tokenizer") and hasattr(tokenizer.instruct_tokenizer, "tokenizer"):
        return tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=True)
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer.encode(text, bos=False, eos=True)
    raise AttributeError("Could not find a raw tokenizer encode path on MistralTokenizer.")


def encode_example(
    tokenizer: MistralTokenizer,
    system_prompt: str,
    user_text: str,
    gold_letter: str,
) -> Dict[str, List[int]]:
    req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
    )

    prompt_tok = tokenizer.encode_chat_completion(req)
    prompt_ids = list(prompt_tok.tokens)
    target_ids = tokenize_target_text(tokenizer, gold_letter)

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    return {"input_ids": input_ids, "labels": labels}


class MCQDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        tokenizer: MistralTokenizer,
        system_prompt: str,
        label_key: str = "answer",
        max_length: int = 2048,
    ):
        self.samples = []
        dropped = 0

        for row in rows:
            try:
                if label_key not in row:
                    raise KeyError(f"Missing label key '{label_key}'. Available keys: {list(row.keys())}")

                user_text = build_mcq_text(row)
                if not user_text:
                    raise ValueError("Empty question/options after formatting")

                gold = normalize_letter(row[label_key])
                enc = encode_example(
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    user_text=user_text,
                    gold_letter=gold,
                )

                if len(enc["input_ids"]) > max_length:
                    raise ValueError(f"Sequence too long: {len(enc['input_ids'])} > {max_length}")

                self.samples.append(enc)

            except Exception as e:
                print(f"[WARN] dropping sample id={row.get('id', 'N/A')}: {type(e).__name__}: {e}")
                dropped += 1

        print(f"[Dataset] kept={len(self.samples)} dropped={dropped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_model(model_name: str, use_qlora: bool):
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_CACHE,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_CACHE,
        )
    return model


def apply_lora(
    model,
    targeted_layers: List[int],
    paper_label: str,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    targeted: bool = True,
):
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    if targeted:
        print(f"[LoRA] Targeted mode: blocks {targeted_layers[0]}-{targeted_layers[-1]} "
              f"(paper notation {paper_label})")
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            layers_to_transform=targeted_layers,
            layers_pattern=["layers"],
        )
    else:
        print("[LoRA] Full mode: applying adapters to all layers")
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--label_key", type=str, default="answer")
    parser.add_argument("--max_length", type=int, default=2048)

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # --targeted applies LoRA only to the specified block range.
    # Omit to replicate original full-layer LoRA behaviour.
    parser.add_argument("--targeted", action="store_true",
                        help="Restrict LoRA to --layer_start..--layer_end blocks.")

    # Block indices (0-indexed). Paper notation: L_n = block (n-1).
    #   Exp 1  (L32-L40): --layer_start 31 --layer_end 39  [default]
    #   Exp 1b (L24-L40): --layer_start 23 --layer_end 39
    parser.add_argument("--layer_start", type=int, default=31,
                        help="First transformer block to adapt (0-indexed). "
                             "Default 31 = paper L32.")
    parser.add_argument("--layer_end", type=int, default=39,
                        help="Last transformer block to adapt (0-indexed, inclusive). "
                             "Default 39 = paper L40.")

    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build layer list and human-readable label for logging / metadata
    targeted_layers = list(range(args.layer_start, args.layer_end + 1))
    paper_label = f"L{args.layer_start + 1}–L{args.layer_end + 1}"

    print(f"[INFO] LoRA window: blocks {targeted_layers[0]}-{targeted_layers[-1]} "
          f"({paper_label} in paper notation)")

    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    system_prompt = (
        "You are a medical expert answering multiple-choice exam questions. "
        "You will receive exactly ONE question followed by answer options labeled: A), B), C), D), E), and sometimes F). "
        "You must output exactly ONE line in this format: ANSWER: <LETTER>"
        "Rules: - Output ONLY that line. - Do NOT repeat or paraphrase the question. - Do NOT translate anything. - Do NOT explain your reasoning. - Do NOT list the options."
    )

    print("[INFO] Loading data...")
    train_rows = load_json(args.train_file)
    val_rows = load_json(args.val_file) if args.val_file else None

    print("[INFO] Building datasets...")
    train_dataset = MCQDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        label_key=args.label_key,
        max_length=args.max_length,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after preprocessing.")

    eval_dataset = None
    if val_rows is not None:
        eval_dataset = MCQDataset(
            rows=val_rows,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            label_key=args.label_key,
            max_length=args.max_length,
        )
        if len(eval_dataset) == 0:
            print("[WARN] Validation dataset is empty after preprocessing; disabling eval.")
            eval_dataset = None

    print("[INFO] Loading model...")
    model = load_model(args.model_name, use_qlora=args.use_qlora)

    print("[INFO] Applying LoRA...")
    model = apply_lora(
        model,
        targeted_layers=targeted_layers,
        paper_label=paper_label,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targeted=args.targeted,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalLM(pad_token_id=0),
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving adapter...")
    trainer.model.save_pretrained(args.output_dir)

    meta = {
        "base_model": args.model_name,
        "system_prompt": system_prompt,
        "label_key": args.label_key,
        "max_length": args.max_length,
        "use_qlora": args.use_qlora,
        "targeted_lora": args.targeted,
        "targeted_layers_blocks": targeted_layers if args.targeted else "all",
        "targeted_layers_paper": paper_label if args.targeted else "all",
    }
    with open(os.path.join(args.output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()