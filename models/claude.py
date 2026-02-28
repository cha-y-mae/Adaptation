# models/claude_opus45_mcq.py
import os
import re
from typing import Optional, Dict, Any

import anthropic


def extract_letter_from_text(text: str) -> Optional[str]:
    """Extract a single option letter A-F from model output."""
    if not text:
        return None
    s = str(text).strip().upper()

    # Prefer strict format first
    m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", s)
    if m:
        return m.group(1)

    # Fallback: any standalone A-F
    m = re.search(r"\b([A-F])\b", s) or re.search(r"\b([A-F])(?=[\.\)\]:;\s]|$)", s)
    return m.group(1) if m else None


class ClaudeOpus45MCQHandler:
    """
    MCQ-only handler for Claude Opus 4.5 that matches your eval setup:
      - Input: ONE dict with keys: question, opa/opb/opc/opd/(ope/opf optional)
      - instruction: passed as system (verbatim from your instruction file)
      - Output: 'A'..'F' or None
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None,
        **kwargs,  # <-- swallow any extra config fields
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing (export ANTHROPIC_API_KEY=...).")
    
        chosen_model = model or model_name or "claude-opus-4-5-20250901"
    
        # Anthropic SDK supports max_retries on the client in many versions.
        # If your installed SDK doesn't, we'll fall back gracefully.
        try:
            if max_retries is not None:
                self.client = anthropic.Anthropic(api_key=api_key, max_retries=int(max_retries))
            else:
                self.client = anthropic.Anthropic(api_key=api_key)
        except TypeError:
            # Older SDK: no max_retries parameter
            self.client = anthropic.Anthropic(api_key=api_key)
    
        self.model = chosen_model
    

    @staticmethod
    def _build_mcq_text(sample: Dict[str, Any]) -> str:
        """
        Build the question+options block from your JSON sample.
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
            if not txt:
                continue
            lines.append(f"{letter}) {txt}")

        return stem + "\n\n" + "\n".join(lines)

    def prompt(
        self,
        sample: Dict[str, Any],
        instruction: str,
        max_tokens: int = 8,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """
        - sample: dict record
        - instruction: your shared MCQ instruction text (verbatim)
        - returns: 'A'..'F' or None
        """
        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[ClaudeOpus45MCQ] Empty stem/options; cannot build prompt.")
            return None

        system_prompt = (instruction or "").strip()

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_text},
                ],
            )

            # Anthropic returns a list of content blocks; collect text blocks
            raw = ""
            for block in getattr(resp, "content", []) or []:
                # SDK may return objects or dicts depending on version
                if isinstance(block, dict) and block.get("type") == "text":
                    raw += block.get("text", "") or ""
                elif getattr(block, "type", None) == "text":
                    raw += getattr(block, "text", "") or ""

            raw = raw.strip()
            if not raw:
                return None

            print(f"[ClaudeOpus45MCQ] MCQ raw generated: {repr(raw)}")
            return extract_letter_from_text(raw)

        except Exception as e:
            print(f"[ClaudeOpus45MCQ] Error during generation: {e}")
            return None