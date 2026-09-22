#!/usr/bin/env bash
# Run upstream JevBench unchanged; preserve private evidence outside repositories.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo 'JEVBENCH_REPO=/path/to/pinned/jevbench JEVBENCH_ENDPOINT=http://127.0.0.1:8770 bash run_public.sh'
  echo 'Requires Python 3.10+, pytest, git, and a running frozen candidate with ISOLATE_REQUEST_CACHE=1.'
  exit 0
fi

: "${JEVBENCH_REPO:?Set JEVBENCH_REPO to the upstream JevBench checkout}"
export JEVBENCH_ENDPOINT="${JEVBENCH_ENDPOINT:-http://127.0.0.1:8770}"
revision=51a8d73fa798aa337bb1b26abd10995c0ab847e9
cd "$JEVBENCH_REPO"
[[ "$(git rev-parse HEAD)" == "$revision" ]] || {
  echo "Expected JevBench revision $revision" >&2; exit 1;
}
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo 'Use a clean upstream checkout; benchmark modifications are not permitted.' >&2; exit 1;
}

python3 - <<'PY'
import json
import os
import sys
import urllib.request

if sys.version_info < (3, 10):
    raise SystemExit('Python 3.10+ required')
url = os.environ['JEVBENCH_ENDPOINT'].rstrip('/') + '/health'
with urllib.request.urlopen(url, timeout=15) as response:
    health = json.load(response)
expected = {'ready': True, 'model': 'Nemotron-Labs-Diffusion-14B',
            'backend': 'vllm', 'cache_policy': 'isolated_request'}
for key, value in expected.items():
    if health.get(key) != value:
        raise SystemExit(f'Health preflight failed: expected {key}={value!r}')
print('Serving health preflight passed; image identity must also be verified with Docker.')
PY

python3 -m pytest -q tests
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/nemotron-jevbench.XXXXXX")
echo "Private evidence directory: $run_dir"
tasks=datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl

# Keep partial runs and still summarize them when upstream stops early.
run_status=0
python3 -m jevbench.cli run \
  --tasks "$tasks" --adapter typesafe \
  --endpoint "$JEVBENCH_ENDPOINT" --model Nemotron-Labs-Diffusion-14B \
  --key-env '' --cost-basis self_hosted_no_published_tariff \
  --reserve-usd 0 --cap-usd 1 --delay-s 0.2 \
  --results "$run_dir/results.jsonl" --raw-dir "$run_dir/raw" \
  --ledger "$run_dir/ledger.jsonl" --manifest "$run_dir/manifest.json" || run_status=$?

if [[ -f "$run_dir/results.jsonl" ]]; then
  python3 -m jevbench.cli summarize \
    --tasks "$tasks" --results "$run_dir/results.jsonl" \
    --public-export "$run_dir/public-summary.json"
fi
echo 'Do not publish the raw manifest: it contains the serving address. Review exports before sharing.'
exit "$run_status"
