import os, json, torch
os.environ["HF_HOME"]        = "/scratch/ca2627/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration
from peft import PeftModel

SNAPSHOT = "/scratch/ca2627/huggingface/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/cb17b97769b0305ddc717ede4a4ef6fd54ef8371"
TOK_SNAP = "/scratch/ca2627/huggingface/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/95a6d26c4bfb886c58daf9d3f7332c857cb27b43"
ADAPTER  = "/scratch/ca2627/clinicalAI/Adaptation/models/lora_logitlens"

data   = json.load(open("./datasets/task3/msa.json"))
item   = data[0]
prompt = open("./prompts/task3-MSA.txt").read().strip()

tok   = MistralTokenizer.from_file(f"{TOK_SNAP}/tekken.json")
base  = Mistral3ForConditionalGeneration.from_pretrained(
    SNAPSHOT, torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

req       = ChatCompletionRequest(messages=[
    {"role": "system", "content": prompt},
    {"role": "user",   "content": [{"type": "text", "text": item["Dialogue"]}]},
])
tokenized = tok.encode_chat_completion(req)
input_ids = torch.tensor([tokenized.tokens], dtype=torch.long, device=model.device)

with torch.inference_mode():
    out = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=False)

gen_ids = out[0][len(tokenized.tokens):]
print("RAW OUTPUT:", tok.decode(gen_ids))