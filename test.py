import json

# Read both snapshots and print everything useful
paths = [
    "/scratch/ca2627/huggingface/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/95a6d26c4bfb886c58daf9d3f7332c857cb27b43/config.json",
    "/scratch/ca2627/huggingface/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/cb17b97769b0305ddc717ede4a4ef6fd54ef8371/config.json",
]

for path in paths:
    print(f"\n=== {path} ===")
    try:
        with open(path) as f:
            cfg = json.load(f)
        # Print full config so we can see the structure
        print(json.dumps(cfg, indent=2))
    except Exception as e:
        print(f"Error: {e}")