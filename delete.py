import os
from huggingface_hub import snapshot_download

HF_HOME = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_HOME

print("Downloading SILMA-9B-Instruct-v1.0 into shared HF cache...")

snapshot_download(
    repo_id="silma-ai/SILMA-9B-Instruct-v1.0",
    cache_dir=HF_HOME,
    local_files_only=False,   # MUST be False for first download
    resume_download=True,
)

print("✅ Done. Cached under:", HF_HOME)