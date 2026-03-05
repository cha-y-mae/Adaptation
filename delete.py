import os
from huggingface_hub import snapshot_download

repo_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
local_dir = "/scratch/ca2627/hf_models/mistral_small_3_2_24b_2506"

os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    # download the important files (weights + config/tokenizer)
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "tokenizer*",
        "*.json",
        "*.safetensors",
        "*.index.json",
        "*.model",
    ],
    # if your environment needs auth, set HF_TOKEN in env and uncomment:
    # token=os.environ.get("HF_TOKEN"),
)

print("Downloaded to:", local_dir)