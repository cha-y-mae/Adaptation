import gc
import os
import re
from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LETTER_SET = {"A", "B", "C", "D", "E", "F"}


def extract_letter_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    s = str(text).strip().upper()

    patterns = [
        r"\bANSWER\s*[:=]\s*([A-F])\b",
        r"\bCORRECT\s+ANSWER\s*[:=]?\s*([A-F])\b",
        r"\bOPTION\s*([A-F])\b",
        r"^\s*([A-F])\s*$",
        r"\b([A-F])\b",
    ]

    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)

    return None


class BiMediXMCQHandler:
    def __init__(
        self,
        model_name: str = "BiMediX/BiMediX-Bi",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        use_plain_prompt: bool = False,
        max_new_tokens_cap: int = 12,
        temperature: float = 0.0,
        use_cache: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = bool(offline)
        self.use_plain_prompt = bool(use_plain_prompt)
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        self.default_temperature = float(temperature)
        self.use_cache = bool(use_cache)

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = self.offline
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_gpus = torch.cuda.device_count()
        print(f"[BiMediX] GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available.")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        print("[BiMediX] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=False,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print("[BiMediX] Loading model...")
        max_memory = {i: "30GiB" for i in range(num_gpus)}
        max_memory["cpu"] = "160GiB"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

        try:
            self.model.config.use_cache = self.use_cache
        except Exception:
            pass

        try:
            self.model.config.output_router_logits = False
            print("[BiMediX] Disabled output_router_logits for inference.")
        except Exception as e:
            print(f"[BiMediX] Could not disable output_router_logits: {e}")

        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        print(f"[BiMediX] chat_template? {self.has_chat_template}")

        if hasattr(self.model, "hf_device_map"):
            print("[BiMediX] hf_device_map:", self.model.hf_device_map)

        self.first_param_device = next(self.model.parameters()).device
        print(f"[BiMediX] first_param_device: {self.first_param_device}")

        self.stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.append(self.tokenizer.eos_token_id)

    @staticmethod
    def _build_mcq_text(sample: Dict[str, Any]) -> str:
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
            val = option_map.get(letter)
            if val is None:
                continue
            val = str(val).strip()
            if val:
                lines.append(f"{letter}) {val}")

        return stem + ("\n\n" + "\n".join(lines) if lines else "")

    @staticmethod
    def _strict_system_prompt(extra_instruction: str = "") -> str:
        base = (
            "You are a bilingual medical multiple-choice assistant.\n"
            "Return exactly ONE uppercase letter from A, B, C, D, E, or F.\n"
            "Do not explain.\n"
            "Do not repeat the question.\n"
            "Do not translate.\n"
            "Do not output any words other than the single answer letter.\n"
        )
        extra_instruction = (extra_instruction or "").strip()
        return base + (f"\nAdditional instruction:\n{extra_instruction}" if extra_instruction else "")

    def _build_plain_prompt(self, system_prompt: str, user_text: str) -> str:
        return (
            f"{system_prompt}\n\n"
            "Question:\n"
            f"{user_text}\n\n"
            "Final answer (one letter only):"
        )

    def _prepare_inputs(self, system_prompt: str, user_text: str):
        """
        Prepare model inputs.

        Strategy:
        1. Try chat_template if available.
        2. If the template rejects the role structure, automatically fall back to plain prompting.
        """

        if (not self.use_plain_prompt) and self.has_chat_template:
            try:
                merged_user = f"{system_prompt}\n\n{user_text}"

                messages = [
                    {"role": "user", "content": merged_user},
                ]

                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

            except Exception as e:
                print(f"[BiMediX] chat_template failed → falling back to plain prompt ({e})")

        prompt_str = self._build_plain_prompt(system_prompt, user_text)

        return self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=True,
        )

    def _generate_once(
        self,
        sample: Dict[str, Any],
        instruction: str,
        max_tokens: int,
        temperature: float,
        plain_retry: bool = False,
    ):
        user_text = self._build_mcq_text(sample)
        if not user_text:
            return None, ""

        system_prompt = self._strict_system_prompt(instruction)

        original_plain = self.use_plain_prompt
        if plain_retry:
            self.use_plain_prompt = True

        try:
            inputs = self._prepare_inputs(system_prompt, user_text)
            inputs = {k: v.to(self.first_param_device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=min(int(max_tokens), self.max_new_tokens_cap),
                do_sample=bool(temperature and temperature > 0),
                top_p=1.0,
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            if temperature and temperature > 0:
                gen_kwargs["temperature"] = float(temperature)

            with torch.inference_mode():
                output = self.model.generate(**gen_kwargs)

            gen_ids = output[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = extract_letter_from_text(raw_text)
            return pred, raw_text

        finally:
            self.use_plain_prompt = original_plain

    def prompt(
        self,
        sample: Dict[str, Any],
        instruction: str,
        max_tokens: int = 12,
        temperature: float = 0.0,
        task_type=None,
        **kwargs,
    ):
        sample_id = sample.get("id", "N/A")
        temp = temperature if temperature is not None else self.default_temperature

        pred, raw = self._generate_once(
            sample=sample,
            instruction=instruction,
            max_tokens=max_tokens,
            temperature=temp,
            plain_retry=False,
        )

        print(f"\n[BiMediX] id={sample_id} raw={raw!r} pred={pred}", flush=True)

        if pred:
            return pred

        pred2, raw2 = self._generate_once(
            sample=sample,
            instruction=instruction,
            max_tokens=max_tokens,
            temperature=temp,
            plain_retry=True,
        )

        print(f"[BiMediX][retry-plain] id={sample_id} raw={raw2!r} pred={pred2}", flush=True)
        return pred2