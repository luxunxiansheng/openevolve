from huggingface_hub import snapshot_download

# The ID of the model on Hugging Face Hub
repo_id = "unsloth/gemma-3-270m-it"

# The local directory where you want to download the model
local_folder = "./models/gemma-3-270m-itF"

# Download all files from the repository to the specified folder
snapshot_download(repo_id, local_dir=local_folder)

print(f"Model successfully downloaded to: {local_folder}")