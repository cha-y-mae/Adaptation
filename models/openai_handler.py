# openai_handler.py
from openai import OpenAI
import re
import time

def extract_letter_from_text_en(text: str):
    """
    Extract a single MCQ letter A–F from model output.
    Returns 'A'..'F' or None.
    """
    if not text:
        return None
    t = str(text).upper()
    m = re.search(r'\b([A-F])\b', t)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-F])(?=[\.\)\]:;\s]|$)', t)
    return m.group(1) if m else None


class OpenAIHandler:
    """
    Unified handler for OpenAI models:
      - gpt-5*, o1*, o3* → Responses API
      - gpt-4*, gpt-3.5* → Chat Completions API
    """

    def __init__(self, api_key: str, model: str): #gpt-5-2025-08-07
        print('model', model)
        self.client = OpenAI(api_key=api_key)
        self.model = model

    # ----------- routing helpers -----------

    def _is_responses_model(self) -> bool:
        ml = self.model.lower()
        return ml.startswith("gpt-5") or ml.startswith("o1") or ml.startswith("o3")

    # ----------- parsing helpers -----------

    def _parse_responses_text(self, resp) -> str:
        """
        Extract aggregated text from Responses API result.
        Prefer resp.output_text; if empty, walk the output array.
        """
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        # Manual aggregation from output items
        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        parts.append(getattr(c, "text", ""))
        return "".join(parts).strip()

    # ----------- API wrappers -----------

    def _responses_create_with_cap(self, **kwargs):
        """
        Call responses.create with a generous output cap.
        Tries max_output_tokens, falls back to max_completion_tokens, then none.
        """
        try:
            return self.client.responses.create(**kwargs, max_output_tokens=600)
        except TypeError:
            try:
                return self.client.responses.create(**kwargs, max_completion_tokens=600)
            except TypeError:
                return self.client.responses.create(**kwargs)

    # ----------- main entrypoint -----------

    def prompt(
        self,
        question: str,
        instruction: str,
        task_type: str = " ",
        max_tokens: int = 16,   # keep for compatibility
        **kwargs,
    ) -> str:
    
        # Force Responses API minimum
        max_tokens = 16
    
        base_prompt = (
            f"{instruction or ''}\n\n{question}\n\n"
            "Return ONLY one letter."
        ).strip()
    
        # -------- Chat Completions path --------
        if not self._is_responses_model():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": instruction or "Follow instructions strictly."},
                    {"role": "user", "content": base_prompt},
                ],
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return extract_letter_from_text_en(raw) or raw
    
        # -------- Responses API path --------
        payload = dict(
            model=self.model,
            instructions=instruction or "Follow instructions strictly.",
            input=base_prompt,
            text={"format": {"type": "text"}},
            reasoning={"effort": "low"},
            max_output_tokens=16,
        )
    
        resp = self.client.responses.create(**payload)
        raw = self._parse_responses_text(resp)
    
        if not raw:
            payload["input"] += "\n\nFinal answer (single letter A–F) only:"
            resp = self.client.responses.create(**payload)
            raw = self._parse_responses_text(resp)
    
        return extract_letter_from_text_en(raw) or raw
