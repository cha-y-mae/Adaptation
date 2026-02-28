import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Med42MCQHandler:
    def __init__(
        self,
        model_name: str = "m42-health/Llama3-Med42-70B",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        use_plain_prompt: bool = False,
        max_new_tokens_cap: int = 16,  # hard cap to prevent accidental large generations
        use_cache: bool = False,       # memory saver for generation
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = bool(offline)
        self.use_plain_prompt = bool(use_plain_prompt)
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        self.use_cache = bool(use_cache)

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        num_gpus = torch.cuda.device_count()
        print(f"[Med42] GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available.")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        print("[Med42] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stop_token_ids = [self.tokenizer.eos_token_id]
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != self.tokenizer.eos_token_id:
                self.stop_token_ids.append(eot_id)
        except Exception:
            pass

        print("[Med42] Setting up 4-bit quantization...")
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print("[Med42] Loading model (4-bit) with balanced sharding...")
        max_memory = {i: "36GiB" for i in range(num_gpus)}  # leave headroom on each A100-40GB
        max_memory["cpu"] = "120GiB"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="balanced",      # ✅ spreads across GPUs better than "auto"
            max_memory=max_memory,      # ✅ prevents GPU0 from being overfilled
            low_cpu_mem_usage=True,     # ✅ reduces peak memory during load
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        # keep generation memory stable
        try:
            self.model.config.use_cache = self.use_cache
        except Exception:
            pass

        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        print(f"[Med42] chat_template? {self.has_chat_template}")

        if hasattr(self.model, "hf_device_map"):
            print("[Med42] hf_device_map summary:")
            from collections import Counter
            c = Counter(self.model.hf_device_map.values())
            print(dict(c))

        # device to place inputs on (first shard device)
        self.first_param_device = next(self.model.parameters()).device
        print(f"[Med42] first_param_device: {self.first_param_device}")

    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
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

    def _build_plain_prompt(self, instruction: str, user_text: str) -> str:
        instruction = (instruction or "").strip()
        return (
            "You are a medical assistant. Answer the multiple-choice question.\n"
            f"{'INSTRUCTION: ' + instruction + chr(10) if instruction else ''}"
            "Return only one letter (A-F) in the format: Answer: X\n\n"
            f"QUESTION:\n{user_text}\n\n"
            "Answer:"
        )

    def prompt(self, sample: dict, instruction: str, max_tokens: int = 12):
        user_text = self._build_mcq_text(sample)
        if not user_text:
            return None

        system_prompt = (instruction or "").strip()
        max_new = min(int(max_tokens), self.max_new_tokens_cap)

        if (not self.use_plain_prompt) and self.has_chat_template:
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
            prompt_str = self._build_plain_prompt(system_prompt, user_text)
            inputs = self.tokenizer(
                prompt_str,
                return_tensors="pt",
                add_special_tokens=True,
            )

        # ✅ do NOT force cuda:0; move to the first shard device
        inputs = {k: v.to(self.first_param_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.stop_token_ids[0],
            )

        gen_ids = generation[0][input_len:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not raw_text:
            return None

        upper = raw_text.upper()
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)
        m = re.search(r"\b([A-F])\b", upper)
        return m.group(1) if m else None