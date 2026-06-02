"""
train_lora_logitlens.py
-----------------------
Exp 2: Layer-targeted LoRA + logit lens supervision loss.

Builds on train_lora_targeted.py. Two changes only:
  1. LogitLensTrainer overrides compute_loss to add auxiliary CE
     at probe layers (default L38, L40 in paper notation = blocks 37, 39).
  2. Default LoRA target is L24-L40 (blocks 23-39), matching Exp 1b.

The auxiliary loss applies the model's own RMSNorm + unembedding (logit lens)
to intermediate hidden states at the final token position, then penalises
low probability assigned to the correct answer letter at those layers:

    L_total = L_CE + lambda * (1/|P|) * sum_{l in P} L_CE(logitlens(h_l), y*)

where P = probe layers, h_l = hidden state at layer l, y* = answer token.

Usage:
  python train_lora_logitlens.py \
      --train_file  ../datasets/train/train.json \
      --val_file    ../datasets/train/val.json \
      --output_dir  ./lora_logitlens \
      --num_train_epochs 2.0 \
      --learning_rate 2e-4 \
      --aux_loss_weight 0.5 \
      --probe_layers 38 40
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
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
# Data utilities  (identical to train_lora_targeted.py)
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
        "A": sample.get("opa"), "B": sample.get("opb"),
        "C": sample.get("opc"), "D": sample.get("opd"),
        "E": sample.get("ope"), "F": sample.get("opf"),
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
                    raise KeyError(f"Missing label key '{label_key}'.")
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


# ---------------------------------------------------------------------------
# Model loading + LoRA  (identical to train_lora_targeted.py)
# ---------------------------------------------------------------------------

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
):
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Logit lens helpers
# ---------------------------------------------------------------------------

def get_norm_and_lmhead(peft_model):
    """
    Retrieve the final RMSNorm and lm_head from underneath the PEFT wrapper.

    Mistral3ForConditionalGeneration is a multimodal model. The actual language
    model backbone sits at base.model.language_model (a MistralForCausalLM),
    which in turn has .model.norm and .lm_head.

    Tries paths in order and prints which one resolved.
    If all fail, prints the actual module tree to make the fix obvious.
    """
    base = peft_model.base_model.model          # Mistral3ForConditionalGeneration

    # Path 1: base.language_model.{model.norm, lm_head}
    #         (some versions expose language_model directly on the top-level class)
    if hasattr(base, "language_model"):
        lm = base.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "norm") and hasattr(lm, "lm_head"):
            print("[LogitLens] resolved: base.language_model.model.norm + base.language_model.lm_head")
            return lm.model.norm, lm.lm_head
        if hasattr(lm, "norm") and hasattr(lm, "lm_head"):
            print("[LogitLens] resolved: base.language_model.norm + base.language_model.lm_head")
            return lm.norm, lm.lm_head

    # Path 2: base.model.language_model.{model.norm, lm_head}
    #         Mistral3Model wraps a MistralForCausalLM at .language_model
    if hasattr(base, "model") and hasattr(base.model, "language_model"):
        lm = base.model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "norm") and hasattr(lm, "lm_head"):
            print("[LogitLens] resolved: base.model.language_model.model.norm + base.model.language_model.lm_head")
            return lm.model.norm, lm.lm_head
        if hasattr(lm, "norm") and hasattr(lm, "lm_head"):
            print("[LogitLens] resolved: base.model.language_model.norm + base.model.language_model.lm_head")
            return lm.norm, lm.lm_head

    # Path 3: base.model.norm + base.lm_head  (standard CausalLM)
    if hasattr(base, "model") and hasattr(base.model, "norm") and hasattr(base, "lm_head"):
        print("[LogitLens] resolved: base.model.norm + base.lm_head")
        return base.model.norm, base.lm_head

    # Path 4: Mistral3ForConditionalGeneration — confirmed structure:
    #   base.model (Mistral3Model)
    #     └── language_model (MistralModel)
    #           └── norm (MistralRMSNorm)
    #   base.lm_head (Linear)
    if (hasattr(base, "model") and hasattr(base.model, "language_model")
            and hasattr(base.model.language_model, "norm") and hasattr(base, "lm_head")):
        print("[LogitLens] resolved: base.model.language_model.norm + base.lm_head")
        return base.model.language_model.norm, base.lm_head

    # Nothing matched — print the actual tree so we can fix it in one edit
    def _tree(mod, prefix="", depth=3):
        if depth == 0:
            return
        for name, child in mod._modules.items():
            print(f"  {prefix}{name}: {type(child).__name__}")
            _tree(child, prefix + "  ", depth - 1)

    print("[LogitLens] ERROR — could not resolve norm/lm_head. Model tree (depth 3):")
    _tree(base)
    raise AttributeError(
        "Cannot locate norm/lm_head. See tree printed above and add the correct "
        "path to get_norm_and_lmhead() in train_lora_logitlens.py."
    )


def get_answer_positions(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each sequence in the batch, find:
      - pred_pos : the token position whose logit should predict the answer letter
      - target_id: the answer letter token id

    Labels layout: [-100, ..., -100, letter_id, eos_id, (-100 padding)]
    The first non-(-100) position is the answer letter.
    The logit at (answer_pos - 1) predicts the answer letter.
    """
    batch_size, seq_len = labels.shape
    # First non-(-100) index per row
    mask = (labels != -100)                              # (B, T) bool
    # argmax returns first True; if all False returns 0 (guard below)
    first_ans = mask.long().argmax(dim=1)                # (B,)
    pred_pos  = (first_ans - 1).clamp(min=0)             # (B,)
    target_id = labels[torch.arange(batch_size), first_ans]  # (B,)
    return pred_pos, target_id


# ---------------------------------------------------------------------------
# Custom Trainer with logit lens auxiliary loss
# ---------------------------------------------------------------------------

class LogitLensTrainer(Trainer):
    """
    Extends HuggingFace Trainer with an auxiliary cross-entropy loss computed
    via the logit lens at specified probe layers.

    probe_layers     : list of paper-notation layer indices (e.g. [38, 40])
                       converted internally to hidden_states tuple indices.
    aux_loss_weight  : lambda in L_total = L_CE + lambda * L_aux
    """

    def __init__(self, *args, probe_layers=(38, 40), aux_loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_hs_indices = list(probe_layers)
        self.aux_loss_weight  = aux_loss_weight
        self._norm    = None   # resolved lazily on first forward pass
        self._lm_head = None
        print(f"[LogitLens] Probe layers (paper): {list(probe_layers)}  "
              f"→ hidden_states indices: {self.probe_hs_indices}")
        print(f"[LogitLens] Auxiliary loss weight λ = {aux_loss_weight}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels         = inputs.get("labels")
        input_ids      = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        # ── Forward pass requesting all hidden states ─────────────────────
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        main_loss     = outputs.loss
        hidden_states = outputs.hidden_states  # tuple: embed + one per layer

        # ── Logit lens projection components (resolved once, cached) ──────
        if self._norm is None:
            self._norm, self._lm_head = get_norm_and_lmhead(model)
        norm, lm_head = self._norm, self._lm_head

        # ── Answer token positions ─────────────────────────────────────────
        pred_pos, target_ids = get_answer_positions(labels)
        batch_idx = torch.arange(labels.shape[0], device=labels.device)

        # ── Auxiliary loss at each probe layer ────────────────────────────
        aux_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)

        for hs_idx in self.probe_hs_indices:
            if hs_idx >= len(hidden_states):
                print(f"[WARN] probe index {hs_idx} out of range "
                      f"(model has {len(hidden_states)} hidden states). Skipping.")
                continue

            h = hidden_states[hs_idx]                       # (B, T, D)
            h_final = h[batch_idx, pred_pos, :]             # (B, D) — final-token hidden state
            h_final = h_final.to(lm_head.weight.dtype)      # match lm_head dtype (bfloat16)

            # Logit lens: RMSNorm then unembedding
            logits_probe = lm_head(norm(h_final))            # (B, vocab)

            layer_loss = F.cross_entropy(logits_probe, target_ids)
            aux_loss   = aux_loss + layer_loss

        aux_loss = aux_loss / max(len(self.probe_hs_indices), 1)
        total_loss = main_loss + self.aux_loss_weight * aux_loss

        # ── Logging ───────────────────────────────────────────────────────
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss_main": main_loss.item(),
                "loss_aux":  aux_loss.item(),
                "loss_total": total_loss.item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  type=str, required=True)
    parser.add_argument("--val_file",    type=str, default=None)
    parser.add_argument("--model_name",  type=str,
                        default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    parser.add_argument("--output_dir",  type=str, required=True)

    parser.add_argument("--label_key",   type=str,  default="answer")
    parser.add_argument("--max_length",  type=int,  default=2048)
    parser.add_argument("--use_qlora",   action="store_true")
    parser.add_argument("--lora_r",      type=int,  default=16)
    parser.add_argument("--lora_alpha",  type=int,  default=32)
    parser.add_argument("--lora_dropout",type=float,default=0.05)

    # LoRA window — default L24-L40 (blocks 23-39), our best Exp 1 target
    parser.add_argument("--layer_start", type=int, default=23,
                        help="First block to adapt (0-indexed). Default 23 = paper L24.")
    parser.add_argument("--layer_end",   type=int, default=39,
                        help="Last block to adapt (0-indexed, inclusive). Default 39 = paper L40.")

    # Logit lens supervision
    parser.add_argument("--probe_layers", type=int, nargs="+", default=[38, 40],
                        help="Paper-notation layer indices for auxiliary loss. "
                             "Default: 38 40 (L38 and L40 where logit lens shows collapse).")
    parser.add_argument("--aux_loss_weight", type=float, default=0.5,
                        help="Lambda: weight of auxiliary logit lens loss. "
                             "Total loss = CE + lambda * aux. Default 0.5.")

    parser.add_argument("--num_train_epochs",            type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int,   default=1)
    parser.add_argument("--per_device_eval_batch_size",  type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=16)
    parser.add_argument("--learning_rate",               type=float, default=2e-4)
    parser.add_argument("--warmup_steps",                type=int,   default=100)
    parser.add_argument("--weight_decay",                type=float, default=0.0)
    parser.add_argument("--logging_steps",               type=int,   default=10)
    parser.add_argument("--save_steps",                  type=int,   default=200)
    parser.add_argument("--eval_steps",                  type=int,   default=200)
    parser.add_argument("--save_total_limit",            type=int,   default=2)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    targeted_layers = list(range(args.layer_start, args.layer_end + 1))
    paper_label     = f"L{args.layer_start + 1}–L{args.layer_end + 1}"

    print(f"[INFO] LoRA window : {paper_label} (blocks {targeted_layers[0]}-{targeted_layers[-1]})")
    print(f"[INFO] Probe layers: {args.probe_layers}  λ={args.aux_loss_weight}")

    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    system_prompt = (
        "You are a medical expert answering multiple-choice exam questions. "
        "You will receive exactly ONE question followed by answer options labeled: A), B), C), D), E), and sometimes F). "
        "You must output exactly ONE line in this format: ANSWER: <LETTER>"
        "Rules: - Output ONLY that line. - Do NOT repeat or paraphrase the question. "
        "- Do NOT translate anything. - Do NOT explain your reasoning. - Do NOT list the options."
    )

    print("[INFO] Loading data...")
    train_rows = load_json(args.train_file)
    val_rows   = load_json(args.val_file) if args.val_file else None

    print("[INFO] Building datasets...")
    train_dataset = MCQDataset(
        rows=train_rows, tokenizer=tokenizer, system_prompt=system_prompt,
        label_key=args.label_key, max_length=args.max_length,
    )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after preprocessing.")

    eval_dataset = None
    if val_rows is not None:
        eval_dataset = MCQDataset(
            rows=val_rows, tokenizer=tokenizer, system_prompt=system_prompt,
            label_key=args.label_key, max_length=args.max_length,
        )
        if len(eval_dataset) == 0:
            print("[WARN] Validation dataset empty; disabling eval.")
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

    trainer = LogitLensTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalLM(pad_token_id=0),
        probe_layers=args.probe_layers,
        aux_loss_weight=args.aux_loss_weight,
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
        "lora_window_blocks": targeted_layers,
        "lora_window_paper": paper_label,
        "probe_layers_paper": args.probe_layers,
        "aux_loss_weight": args.aux_loss_weight,
    }
    with open(os.path.join(args.output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()