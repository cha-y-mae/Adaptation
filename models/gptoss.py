import os
import re
import torch
from transformers import pipeline

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class GPTOSS20BHandler:
    """
    MCQ-only handler.
    Uses HF pipeline("text-generation") so Harmony formatting is handled automatically.

    Input: dict with keys question, opa/opb/opc/opd/(ope/opf optional)
    Output: single letter A–F (or None)
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
    ):
        print(f"[GPTOSS20BMCQ] Handler file: {__file__}")
        print(f"[GPTOSS20BMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[GPTOSS20BMCQ] cache_dir={cache_dir}")
        print(f"[GPTOSS20BMCQ] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = offline

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        # Debug GPUs
        num_gpus = torch.cuda.device_count()
        print(f"[GPTOSS20BMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for GPTOSS20BMCQ.")

        # Create pipeline (this is the “HF card” way)
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype="auto",
            device_map="auto",
            # cache_dir is supported by pipeline via model loading under the hood in most versions,
            # but not always. Keeping HF_HOME set usually suffices on HPC.
        )

        print("[GPTOSS20BMCQ] Pipeline ready.")

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
            if not txt:
                continue
            lines.append(f"{letter}) {txt}")

        return stem + "\n\n" + "\n".join(lines)

    @staticmethod
    def _extract_assistant_text(generated_text):
        """
        HF pipeline returns either:
        - a string
        - OR a list of {'role','content'} messages (chat mode)
        We want the assistant's content.
        """
        if isinstance(generated_text, list):
            # find last assistant message, fall back to last item
            for msg in reversed(generated_text):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return (msg.get("content") or "").strip()
            last = generated_text[-1]
            return (last.get("content") if isinstance(last, dict) else str(last)).strip()
        return str(generated_text).strip()

    @staticmethod
    def _clean_harmony(text: str) -> str:
        """
        gpt-oss may emit 'analysis...' without separators.
        We'll strip common leading tags and keep the tail.
        """
        t = text.strip()

        # remove leading 'analysis' token if stuck to text
        # examples: 'analysisWe need to ...' or 'analysis\nWe need to ...'
        t = re.sub(r"^\s*analysis\s*", "", t, flags=re.IGNORECASE)

        # also strip leading 'assistant' if appears
        t = re.sub(r"^\s*assistant\s*[:\-]?\s*", "", t, flags=re.IGNORECASE)

        return t.strip()

    @staticmethod
    def _extract_letter(text: str):
        upper = text.upper()

        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        # sometimes it outputs just 'C' or '(C)'
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        return None

    def prompt(self, sample: dict, instruction: str, max_tokens: int = 12):
        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[GPTOSS20BMCQ] Empty stem/options; cannot build prompt.")
            return None
    
        # Reduce reasoning verbosity (Harmony supports this)
        # Put this at the TOP of the system message.
        system_prompt = "Reasoning: low\n" + instruction.strip()
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
    
        try:
            torch.cuda.empty_cache()
    
            # IMPORTANT: return_dict=True to avoid your previous shape issue
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
    
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max(64, int(max_tokens)),  # give room
                    do_sample=False,
                )
    
            completion_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            raw = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
    
        except Exception as e:
            print("[GPTOSS20BMCQ] Error during generation:", e)
            return None
        finally:
            torch.cuda.empty_cache()
    
        if not raw.strip():
            return None
    
        print(f"[GPTOSS20BMCQ] RAW COMPLETION: {repr(raw)[:300]}")
    
        # ---- Parse Harmony "final" ----
        # Many gpt-oss outputs contain markers like "assistantfinal".
        # Keep everything after assistantfinal if present.
        upper = raw.upper()
    
        # 1) If "ASSISTANTFINAL" exists, use text after it
        if "ASSISTANTFINAL" in upper:
            raw_final = raw.split("assistantfinal", 1)[-1]
        else:
            # 2) Otherwise, try to strip any leading analysis tag
            raw_final = re.sub(r"^\s*analysis\s*", "", raw, flags=re.IGNORECASE)
    
        raw_final = raw_final.strip()
        print(f"[GPTOSS20BMCQ] PARSED FINAL: {repr(raw_final)[:200]}")
    
        # Extract letter
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", raw_final.upper())
        if m:
            return m.group(1)
    
        m = re.search(r"\b([A-F])\b", raw_final.upper())
        if m:
            return m.group(1)
    
        print("[GPTOSS20BMCQ] Could not extract a clean letter.")
        return None
