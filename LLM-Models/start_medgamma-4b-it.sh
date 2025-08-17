python -m vllm.entrypoints.openai.api_server \
  --model /workspaces/LLM-Models/models/medgemma-4b-it \
  --host 0.0.0.0 \
  --port 8006 \
  --served-model-name medgemma-4b-it \
  --quantization fp8 \
  --gpu-memory-utilization 0.60