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
extra=()
entry=()
if [[ ${SYSTEMONE_ENABLED:-0} == 1 ]]; then
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  : "${SYSTEMONE_PORT:=8790}"
  : "${SYSTEMONE_HOST:=127.0.0.1}"
  if [[ $SYSTEMONE_PORT == "$port" ]]; then
    echo 'Model and System One ports must differ' >&2; exit 1
  fi
  if [[ $SYSTEMONE_HOST != 127.0.0.1 && $SYSTEMONE_HOST != localhost && $SYSTEMONE_HOST != ::1 && -z ${SYSTEMONE_API_KEY:-} ]]; then
    echo 'SYSTEMONE_API_KEY is required for non-loopback access' >&2; exit 1
  fi
  extra=(--entrypoint python3
    -v "$script_dir/serve_systemone.py:/app/serve_systemone.py:ro"
    -v "$script_dir/../lightning-systemone/adapter.py:/app/adapter.py:ro"
    -e "SERVER_PORT=$port" -e "SYSTEMONE_PORT=$SYSTEMONE_PORT"
    -e "SYSTEMONE_HOST=$SYSTEMONE_HOST" -e SYSTEMONE_API_KEY)
  entry=(/app/serve_systemone.py)
fi
docker run -d --name "$name" --gpus "device=$GPU_UUID" \
  --network host --shm-size 16g --user "$(id -u):$(id -g)" \
  -v "$MODEL_DIR:/model:ro" -v "$ADAPTER_DIR:/adapter:ro" -v "$RUN_DIR:/work" \
  -e HOME=/work -e HF_HOME=/work/cache -e HF_HUB_OFFLINE=1 \
  -e VLLM_CACHE_ROOT=/work/cache/vllm-lora -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor-lora \
  ${extra[@]+"${extra[@]}"} \
  vllm/vllm-openai:nightly-a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b \
  ${entry[@]+"${entry[@]}"} \
  /model --served-model-name lightning --host 127.0.0.1 --port "$port" \
  --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching --async-scheduling \
  --moe-backend marlin --mamba-backend flashinfer --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 --max-logprobs 20 --kv-cache-dtype bfloat16 \
  --enable-lora --max-lora-rank 16 --lora-modules trained=/adapter
