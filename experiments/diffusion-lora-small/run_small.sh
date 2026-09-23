#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON:?}" "${TRIAL:?}" "${CHECKPOINT:?}" "${SCORER:?}" "${HARNESS:?}" "${DATA:?}" "${TRAIN_OUTPUT:?}"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "SMALL_RECIPE_FAILED exit=$rc"; fi' EXIT
"$PYTHON" "$TRIAL/train.py" --checkpoint "$CHECKPOINT" --app "$SCORER" \
  --harness "$HARNESS" --data "$DATA" --output "$TRAIN_OUTPUT" \
  --recipe "$TRIAL/recipe-small.json"
"$PYTHON" "$TRIAL/benchmark_adapter.py" --checkpoint "$CHECKPOINT" \
  --adapter "$TRAIN_OUTPUT/best-adapter" --app "$SCORER" --harness "$HARNESS" \
  --output "$TRAIN_OUTPUT/paired-latency.jsonl"
"$PYTHON" "$TRIAL/summarize_small.py" --run "$TRAIN_OUTPUT"
echo SMALL_RECIPE_COMPLETE
