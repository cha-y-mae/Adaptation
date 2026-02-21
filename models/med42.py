import os
import re
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Med42MCQHandler:
    """
    MCQ-only handler for m42-health/Llama3-Med42-70B.

    Interface matches your reference scripts (MedGemma/Meditron):
      - Input: one dict sample with keys: question, opa/opb/opc/opd/(ope/opf optional)
      - Output: single letter A–F (or None)

    Keeps Med42-specific details:
      - 4-bit bitsandbytes quantization (nf4 + double quant)
      - tokenizer left padding (safe for decoder-only; not essential for single-sample prompt())
      - stop tokens logic retained
    """

    def __init__(
        self,
        model_name: str = "m42-health/Llama3-Med42-70B",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        use_plain_prompt: bool = False,  # False => chat_template like MedGemma; True => plain "Answer:" anchor like Meditron
    ):
        print(f"[Med42_70B_MCQ] Handler file: {__file__}")
        print(f"[Med42_70B_MCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[Med42_70B_MCQ] cache_dir={cache_dir}")
        print(f"[Med42_70B_MCQ] offline={offline}")
        print(f"[Med42_70B_MCQ] use_plain_prompt={use_plain_prompt}")
        print(f"[Med42_70B_MCQ] model_name={model_name}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = bool(offline)
        self.use_plain_prompt = bool(use_plain_prompt)

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        # -------------------------
        # Inspect GPUs (debug)
        # -------------------------
        num_gpus = torch.cuda.device_count()
        print(f"[Med42_70B_MCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for Med42_70B_MCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        # -------------------------
        # Load tokenizer
        # -------------------------
        print("[Med42_70B_MCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
        )

        # Decoder-only nicety (not strictly required for single-sample prompt()).
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Stop token ids (retain your previous logic)
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != self.tokenizer.eos_token_id:
                self.stop_token_ids.append(eot_id)
        except Exception:
            pass

        # --------------------------
        # Load model (Med42-specific: 4-bit)
        # --------------------------
        print("[Med42_70B_MCQ] Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print("[Med42_70B_MCQ] Loading model in 4-bit...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        try:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        except Exception:
            pass

        print("[Med42_70B_MCQ] Model loaded.")
        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        print(f"[Med42_70B_MCQ] tokenizer.chat_template available? {self.has_chat_template}")

        if hasattr(self.model, "hf_device_map"):
            print("[Med42_70B_MCQ] Model device distribution:")
            for layer, dev in self.model.hf_device_map.items():
                print(f"  {layer}: {dev}")

        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved")

    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
        """
        Build a clean MCQ block from your JSON schema.

        Expected:
          sample["question"]
          sample["opa"], sample["opb"], sample["opc"], sample["opd"]
          optional: sample["ope"], sample["opf"]
        """
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
            if txt == "":
                continue
            lines.append(f"{letter}) {txt}")

        return stem + "\n\n" + "\n".join(lines)

    def _build_plain_prompt(self, instruction: str, user_text: str) -> str:
        # Same style as your Meditron reference (kept as an option)
        instruction = (instruction or "").strip()
        return (
            "You are a medical assistant. Answer the multiple-choice question.\n"
            f"{'INSTRUCTION: ' + instruction + chr(10) if instruction else ''}"
            "Return only one letter (A-F) in the format: Answer: X\n\n"
            f"QUESTION:\n{user_text}\n\n"
            "Answer:"
        )

    def prompt(self, sample: dict, instruction: str, max_tokens: int = 12):
        """
        MCQ-only interface:
        - sample is ONE JSON record (dict).
        - returns: 'A'..'F' or None
        """
        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[Med42_70B_MCQ] Empty stem/options; cannot build prompt.")
            return None

        system_prompt = (instruction or "").strip()

        print("[Med42_70B_MCQ] MAX TOKENS:", max_tokens)

        try:
            torch.cuda.empty_cache()

            if (not self.use_plain_prompt) and self.has_chat_template:
                # Match MedGemma-style: chat template
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                # Plain anchor prompt (for strict comparability across non-chat models)
                prompt_str = self._build_plain_prompt(system_prompt, user_text)
                inputs = self.tokenizer(
                    prompt_str,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

            # Put inputs on the main model device (works with device_map="auto" too)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.stop_token_ids[0],
                )

            gen_ids = generation[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print("[Med42_70B_MCQ] Error during generation:", e)
            return None
        finally:
            torch.cuda.empty_cache()

        if not raw_text:
            return None

        print(f"[Med42_70B_MCQ] MCQ raw generated: {repr(raw_text)}")

        upper = raw_text.upper()

        # Prefer strict format: ANSWER: X
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        # Fallback: first standalone A-F
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        print("[Med42_70B_MCQ] Could not extract a clean letter.")
        return None
