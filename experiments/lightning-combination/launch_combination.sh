#!/usr/bin/env bash
# Original fast serving recipe plus the trained rank-16 LoRA.
set -euo pipefail
: "${MODEL_DIR:?Set original Lightning NVFP4 checkpoint}"
: "${ADAPTER_DIR:?Set trained LoRA checkpoint}"
: "${RUN_DIR:?Set isolated writable runtime directory}"
: "${GPU_UUID:?Set the already allocated GPU UUID}"
name=${CONTAINER_NAME:-lightning-nvfp4-trained}
port=${SERVER_PORT:-8300}
mkdir -p "$RUN_DIR/cache"
docker run -d --name "$name" --gpus "device=$GPU_UUID" \
  --network host --shm-size 16g --user "$(id -u):$(id -g)" \
  -v "$MODEL_DIR:/model:ro" -v "$ADAPTER_DIR:/adapter:ro" -v "$RUN_DIR:/work" \
  -e HOME=/work -e HF_HOME=/work/cache -e HF_HUB_OFFLINE=1 \
  -e VLLM_CACHE_ROOT=/work/cache/vllm-lora -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor-lora \
  vllm/vllm-openai:nightly-a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b \
  /model --served-model-name lightning --host 127.0.0.1 --port "$port" \
  --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching --async-scheduling \
  --moe-backend marlin --mamba-backend flashinfer --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 --max-logprobs 20 --kv-cache-dtype bfloat16 \
  --enable-lora --max-lora-rank 16 --lora-modules trained=/adapter
