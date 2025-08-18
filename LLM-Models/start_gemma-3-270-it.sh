python -m vllm.entrypoints.openai.api_server \
  --model /workspaces/opencontext/LLM-Models/models/gemma-3-270m \
  --host 0.0.0.0 \
  --port 8006 \
  --served-model-name gemma-3-270m-it \
  --gpu-memory-utilization 0.1



