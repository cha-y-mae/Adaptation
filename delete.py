import os
from huggingface_hub import snapshot_download

HF_HOME = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_HOME

# pick ONE of these two approaches:

# Approach 1: download into the standard HF cache layout (so repo id works offline)
snapshot_download(
    repo_id="QCRI/Fanar-1-9B",
    cache_dir=HF_HOME,
    local_files_only=False,
)

print("Done. Cached under:", HF_HOME)