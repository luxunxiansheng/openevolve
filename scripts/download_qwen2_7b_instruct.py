from huggingface_hub import snapshot_download
import os

# Download Qwen/Qwen2-7B-Instruct to /workspaces/openevolve/models/Qwen2-7B-Instruct
output_dir = "/workspaces/openevolve/models/Qwen2-7B-Instruct"
os.makedirs(output_dir, exist_ok=True)
snapshot_download(
    repo_id="Qwen/Qwen2-7B-Instruct", local_dir=output_dir, local_dir_use_symlinks=False
)
print(f"Model downloaded to {output_dir}")
