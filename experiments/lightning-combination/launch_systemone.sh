#!/usr/bin/env bash
# Start both the trained model and the System One API in the pinned vLLM image.
set -euo pipefail
export SYSTEMONE_ENABLED=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch_combination.sh" "$@"
