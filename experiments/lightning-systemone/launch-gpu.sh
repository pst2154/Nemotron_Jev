#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_DIR:?Set the existing Lightning checkpoint directory}"
: "${RUN_DIR:?Set an isolated writable runtime directory}"
image=${VLLM_IMAGE:-vllm/vllm-openai:nightly-a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b}
name=${CONTAINER_NAME:-lightning-systemone}
mapfile -t gpu_ids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#gpu_ids[@]} == 1 ]] || { echo 'Expected one allocated GPU'; exit 1; }
mkdir -p "$RUN_DIR/cache"
docker run -d --name "$name" --gpus "device=${gpu_ids[0]}" \
  --network host --shm-size 16g --user "$(id -u):$(id -g)" \
  -v "$MODEL_DIR:/model:ro" -v "$RUN_DIR:/work" \
  -e HOME=/work -e HF_HOME=/work/cache -e HF_HUB_OFFLINE=1 \
  -e VLLM_CACHE_ROOT=/work/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor \
  "$image" /model --served-model-name lightning --host 127.0.0.1 --port 8300 \
  --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching --async-scheduling \
  --moe-backend marlin --mamba-backend flashinfer --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 --max-logprobs 20 --kv-cache-dtype bfloat16
