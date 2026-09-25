#!/bin/sh
set -eu
# Keep runtime remote-code modules writable with a read-only root/checkpoint.
export HF_HOME="${HF_HOME:-/tmp/hf}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/hf/modules}"
mkdir -p "$HF_MODULES_CACHE"
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    python3 /app/download.py
fi
exec python3 -u /app/server.py
