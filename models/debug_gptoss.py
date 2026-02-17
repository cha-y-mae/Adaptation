import re
import torch
from transformers import pipeline

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

instruction = (
    "You are a medical assistant specialized in solving multiple-choice exam questions.\n"
    "Return ONLY one line exactly in this format:\n"
    "ANSWER: <LETTER>\n"
    "Where <LETTER> is one of A, B, C, D, E, F.\n"
    "No explanation."
)

mcq = (
    "A 45-year-old man presents with crushing chest pain radiating to the left arm.\n"
    "Which of the following is the most likely diagnosis?\n\n"
    "A) Gastroesophageal reflux\n"
    "B) Panic attack\n"
    "C) Myocardial infarction\n"
    "D) Pneumonia\n"
)

messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": mcq},
]

out = pipe(
    messages,
    max_new_tokens=32,
    do_sample=False,  # greedy like your eval
)

# HF pipeline returns a list of dicts. For chat input, generated_text is usually a list of messages.
gen = out[0]["generated_text"]

print("RAW generated_text type:", type(gen))
print("RAW generated_text:", gen)

# Extract the final assistant content robustly
if isinstance(gen, list):
    # e.g. [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    final_msg = gen[-1]
    text = final_msg.get("content", "") if isinstance(final_msg, dict) else str(final_msg)
else:
    text = str(gen)

text = text.strip()
print("\nFINAL TEXT:", repr(text))

upper = text.upper()
m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
if not m:
    m = re.search(r"\b([A-F])\b", upper)

print("EXTRACTED:", m.group(1) if m else None)
