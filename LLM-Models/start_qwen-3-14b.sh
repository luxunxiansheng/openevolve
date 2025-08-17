python -m vllm.entrypoints.openai.api_server \
  --model "$(dirname $(dirname $(realpath $0)))/LLM-Models/models/Qwen3-14B-AWQ" \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 8010 \
  --served-model-name Qwen3-14B-AWQ\
  --gpu-memory-utilization 0.8 \
  --max-model-len 40960 \
  --uvicorn-log-level debug





