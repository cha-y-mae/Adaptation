import os
import time
from typing import Optional, Dict, Any

from google import genai
from google.genai import types

LETTER_SET = {"A", "B", "C", "D", "E", "F"}


class Gemini3ProHandler:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        min_request_interval: float = 2.6,
        max_retries: int = 6,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.min_request_interval = float(min_request_interval)
        self.max_retries = int(max_retries)
        self._last_call_time = 0.0

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

        return (
            "Choose the single best answer.\n"
            "Return exactly one uppercase letter from this set only: A, B, C, D, E, F.\n"
            "Do not explain.\n"
            "Do not output JSON.\n"
            "Do not output any words.\n\n"
            f"{stem}\n\n" + "\n".join(lines)
        )

    def _respect_rate_limit(self):
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    @staticmethod
    def _extract_label(resp) -> Optional[str]:
        # Fast path
        text = getattr(resp, "text", None)
        if text:
            t = str(text).strip().upper().strip('"').strip()
            if t in LETTER_SET:
                return t

        # Candidate/parts fallback
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            for part in parts or []:
                txt = getattr(part, "text", None)
                if txt:
                    t = str(txt).strip().upper().strip('"').strip()
                    if t in LETTER_SET:
                        return t
        return None

    @staticmethod
    def _debug_response(resp):
        try:
            print("[GeminiMCQ][DEBUG] prompt_feedback:", getattr(resp, "prompt_feedback", None), flush=True)
            candidates = getattr(resp, "candidates", None) or []
            print(f"[GeminiMCQ][DEBUG] num_candidates={len(candidates)}", flush=True)
            for i, cand in enumerate(candidates):
                print(f"[GeminiMCQ][DEBUG] candidate[{i}].finish_reason={getattr(cand, 'finish_reason', None)}", flush=True)
                print(f"[GeminiMCQ][DEBUG] candidate[{i}].safety_ratings={getattr(cand, 'safety_ratings', None)}", flush=True)
            print(f"[GeminiMCQ][DEBUG] raw_text={getattr(resp, 'text', None)!r}", flush=True)
            print(f"[GeminiMCQ][DEBUG] usage_metadata={getattr(resp, 'usage_metadata', None)}", flush=True)
        except Exception as e:
            print(f"[GeminiMCQ][DEBUG] response inspection failed: {e}", flush=True)

    def prompt(
        self,
        sample,
        instruction=None,
        max_tokens=4,
        temperature=0.0,
        task_type=None,
        **kwargs,
    ):
        user_text = self._build_mcq_text(sample)
        if not user_text:
            return None

        sample_id = sample.get("id", "N/A")
        print(f"\n[GeminiMCQ] Processing sample id={sample_id}", flush=True)

        system_prompt = (
            (instruction or "").strip()
            or "You answer medical multiple-choice questions. Output exactly one uppercase letter only."
        )

        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=float(temperature),
            max_output_tokens=int(max_tokens),

            # Disable thinking for 2.5 Flash
            thinking_config=types.ThinkingConfig(thinking_budget=0),

            # Make output plain text, not JSON
            response_mime_type="text/plain",

            # Disable AFC noise/pathways
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            ),
        )

        last_err = None
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limit()

                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_text,
                    config=cfg,
                )
                self._last_call_time = time.time()

                print("\n==============================", flush=True)
                print("[GeminiMCQ] RAW RESPONSE:", flush=True)
                print(getattr(resp, "text", None), flush=True)
                print("==============================", flush=True)

                pred = self._extract_label(resp)
                print(f"[GeminiMCQ] PARSED PREDICTION: {pred}", flush=True)

                if pred:
                    return pred

                self._debug_response(resp)

                # One retry with a bit more room if still truncated
                candidates = getattr(resp, "candidates", None) or []
                finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
                if str(finish_reason) == "MAX_TOKENS" and attempt < self.max_retries - 1:
                    cfg.max_output_tokens = 8
                    continue

                return None

            except Exception as e:
                last_err = e
                msg = str(e)
                print(f"[GeminiMCQ] Error: {msg}", flush=True)

                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    sleep_s = max(7.0, self.min_request_interval * (attempt + 2))
                    print(f"[GeminiMCQ] Sleeping {sleep_s:.1f}s before retry...", flush=True)
                    time.sleep(sleep_s)
                    continue

                return None

        print(f"[GeminiMCQ] Failed after {self.max_retries} retries: {last_err}", flush=True)
        return None