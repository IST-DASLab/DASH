from huggingface_hub import list_repo_files, hf_hub_download
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_dir", type=str, default="/tmp/ISTA-DASLab/datasets/c4", required=True)
parser.add_argument("--repo_id", type=str, default="ISTA-DASLab/C4-tokenized-llama2", required=True)
args = parser.parse_args()

repo_id = args.repo_id
local_dir = args.local_dir
os.makedirs(local_dir, exist_ok=True)

# List all files in the repo
files = list_repo_files(repo_id, repo_type="dataset")

# Filter for chunk files
chunk_files = [f for f in files if f.startswith("chunk_") or f == "val.bin"]

# Download all chunks
for f in chunk_files:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=f,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded {f} -> {local_path}")

