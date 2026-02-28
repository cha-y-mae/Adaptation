# openai_handler.py
from openai import OpenAI
import re
import time
from typing import Optional, Dict, Any


def extract_letter_from_text_en(text: str) -> Optional[str]:
    """
    Extract a single MCQ letter A–F from model output.
    Returns 'A'..'F' or None.
    """
    if not text:
        return None
    t = str(text).strip().upper()

    # Prefer strict "ANSWER: X"
    m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", t)
    if m:
        return m.group(1)

    # Then any standalone letter
    m = re.search(r"\b([A-F])\b", t)
    if m:
        return m.group(1)

    # Then letter followed by punctuation/end
    m = re.search(r"\b([A-F])(?=[\.\)\]:;\s]|$)", t)
    return m.group(1) if m else None


class OpenAIHandler:
    """
    MCQ-only handler for OpenAI models, aligned with the Meditron eval setup:
      - input is sample: dict with keys question, opa/opb/opc/opd/(ope/opf optional)
      - prompt() returns a single letter A–F (or None if extraction fails)

    Routing:
      - gpt-5*, o1*, o3* → Responses API
      - gpt-4*, gpt-3.5* → Chat Completions API
    """

    def __init__(self, api_key: str, model: str, max_retries: int = 3):
        print("[OpenAIHandler] model:", model)
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    # ----------- routing helpers -----------

    def _is_responses_model(self) -> bool:
        ml = (self.model or "").lower()
        return ml.startswith("gpt-5") or ml.startswith("o1") or ml.startswith("o3")

    # ----------- build MCQ text (matches Meditron) -----------

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
            txt = option_map.get(letter)
            if txt is None:
                continue
            txt = str(txt).strip()
            if txt:
                lines.append(f"{letter}) {txt}")

        return stem + "\n\n" + "\n".join(lines) if lines else stem

    # ----------- prompt style (keep pipeline, align with Meditron anchors) -----------

    def _build_user_block(self, instruction: str, user_text: str) -> str:
        """
        Keep your pipeline style (instruction + question), but add the same strong
        anchor Meditron uses ("Answer:") to reduce verbose outputs.
        """
        instruction = (instruction or "").strip()

        # This block is what we send as the "user" content (for both APIs)
        # We keep it simple + deterministic + anchored.
        base = []
        if instruction:
            base.append(instruction)
        base.append("Return only one letter (A-F) in the format: Answer: X")
        base.append("")
        base.append("QUESTION:")
        base.append(user_text)
        base.append("")
        base.append("Answer:")
        return "\n".join(base).strip()

    # ----------- parsing helpers -----------

    def _parse_responses_text(self, resp) -> str:
        """
        Extract aggregated text from Responses API result.
        Prefer resp.output_text; if empty, walk the output array.
        """
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        parts.append(getattr(c, "text", ""))
        return "".join(parts).strip()

    # ----------- retry wrapper -----------

    def _with_retries(self, fn, *args, **kwargs):
        last_err = None
        for i in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                # simple backoff
                time.sleep(min(2 ** i, 8))
        raise last_err

    # ----------- main entrypoint -----------

    def prompt(
        self,
        sample: Dict[str, Any],
        instruction: str,
        task_type: str = "mcq",
        max_tokens: int = 12,   # match Meditron default-ish; you can override from yaml
        temperature: float = 0.0,
        top_p: float = 1.0,
        reasoning_effort: str = "low",
        **kwargs,
    ) -> Optional[str]:
        """
        Returns extracted letter A–F, or None if it can't be extracted.
        """

        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[OpenAIHandler] Empty stem/options; cannot build prompt.")
            return None

        user_block = self._build_user_block(instruction, user_text)

        # -------- Chat Completions path --------
        if not self._is_responses_model():
            resp = self._with_retries(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "Follow instructions strictly."},
                    {"role": "user", "content": user_block},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw = (resp.choices[0].message.content or "").strip()
            letter = extract_letter_from_text_en(raw)
            if letter is None:
                print(f"[OpenAIHandler] Could not extract a clean letter. Raw: {repr(raw)}")
            return letter

        # -------- Responses API path --------
        payload = dict(
            model=self.model,
            instructions="Follow instructions strictly.",
            input=user_block,
            text={"format": {"type": "text"}},
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_tokens,   # respects yaml, no longer forced to 16
            temperature=temperature,
            top_p=top_p,
        )

        resp = self._with_retries(self.client.responses.create, **payload)
        raw = self._parse_responses_text(resp)

        # tiny fallback if empty
        if not raw:
            payload["input"] = user_block + "\n\nAnswer: "
            resp = self._with_retries(self.client.responses.create, **payload)
            raw = self._parse_responses_text(resp)

        raw = (raw or "").strip()
        letter = extract_letter_from_text_en(raw)
        if letter is None:
            print(f"[OpenAIHandler] Could not extract a clean letter. Raw: {repr(raw)}")
        return letter