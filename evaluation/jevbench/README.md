# JevBench submission preparation

Target: [Benchmark Heaven Jev models](https://benchmarkheaven.com/jev-models),
using [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench).
This is not the 324-example Nimble holdout or our 72-case regression suite.

## Status

- Submission candidate: Nemotron-Labs-Diffusion-14B, optimized vLLM decision readout.
- Public container and H100 deployment instructions are available in the root README.
- The unchanged upstream `typesafe` adapter matches our `/v1/systemone` contract.
- Upstream harness commit `51a8d73fa798aa337bb1b26abd10995c0ab847e9` passes its 22 tests.
- **No JevBench model evaluation has been run or submitted yet.**
- The earlier H100 allocation expired. A fresh serving GPU is needed for the self-test.
- Original application code is MIT licensed; vLLM is Apache-2.0 and model weights
  retain the NVIDIA Nemotron Open Model License. These are separate grants.

The pinned harness supplies 231 public items: 48 easy, 72 original/standard, and
111 hard. The full leaderboard evaluates 534 decisions. A public self-test is not
a complete leaderboard score or rank; maintainers must perform the remaining
evaluation on their own infrastructure. Do not request their hidden labels.

## Frozen candidate

| Component | Value |
| --- | --- |
| Application source | `66b19e885e2f330711f69983ab9f5d182630ca26` |
| vLLM source | `2c83d10caa71e3a9eac8fdd9f1288dac8c65596b` |
| Container | `ghcr.io/pst2154/nemotron-jev@sha256:b8afc221e1ef9c8e74847e1a3d304a77123ac5114f1ef9f102048866fb286aa9` |
| Base checkpoint | `nvidia/Nemotron-Labs-Diffusion-14B` |
| Checkpoint revision | `f8c3e2c078e193599b8882d965b1001c456ba738` |
| Precision / hardware | BF16 / one H100 80 GB; prebuilt kernels target Hopper |
| Training | No additional training, adapter, or benchmark-specific calibration |
| Interface | Native Choice, Noul, Score; one masked position per question |
| Limits | 16,384 prompt tokens, 62 options, 512 questions per request |

Use the root deployment guide, adding `-e ISOLATE_REQUEST_CACHE=1` to `docker run`.
Retain all other image defaults. The model may already be loaded, but record that
condition. Do not silently crop long inputs, change option order, tune prompts
after seeing labels, repair malformed distributions, or retry selected failures.
Use one serial request per item, not the earlier 100-question batching benchmark.

## Public self-test

After checking out the pinned upstream revision and starting the container, the
[runner](run_public.sh) performs the health/configuration preflight, upstream
tests, serial run, and summary without modifying upstream code:

```bash
JEVBENCH_REPO=/path/to/jevbench \
JEVBENCH_ENDPOINT=http://127.0.0.1:8770 \
bash evaluation/jevbench/run_public.sh
```

It requires a clean, pinned harness checkout and isolated request caches. Verify
the container digest separately; health metadata alone does not prove image
identity. Outputs are placed in a fresh private temporary directory and retained
even when the harness stops early. The expanded commands below are equivalent.

Python 3.10 or newer is required. Run from the upstream checkout. Replace the
example service URL with your local serving address. The fresh output directory
is outside both repositories and must remain private: the upstream manifest
records the endpoint. Credentials, when needed, stay in environment variables.

```bash
git clone https://github.com/fstandhartinger/jevbench.git
cd jevbench
git checkout 51a8d73fa798aa337bb1b26abd10995c0ab847e9
python3 -m unittest discover -s tests -v

export JEVBENCH_ENDPOINT=http://127.0.0.1:8770
JEVBENCH_RUN=$(mktemp -d)
TASKS=datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl

curl --fail "$JEVBENCH_ENDPOINT/health"
python3 -m jevbench.cli run \
  --tasks "$TASKS" --adapter typesafe \
  --endpoint "$JEVBENCH_ENDPOINT" --model Nemotron-Labs-Diffusion-14B \
  --key-env '' --cost-basis self_hosted_no_published_tariff \
  --reserve-usd 0 --cap-usd 1 --delay-s 0.2 \
  --results "$JEVBENCH_RUN/results.jsonl" --raw-dir "$JEVBENCH_RUN/raw" \
  --ledger "$JEVBENCH_RUN/ledger.jsonl" --manifest "$JEVBENCH_RUN/manifest.json"

python3 -m jevbench.cli summarize \
  --tasks "$TASKS" --results "$JEVBENCH_RUN/results.jsonl" \
  --public-export "$JEVBENCH_RUN/public-summary.json"
```

The budget reservation is zero because this calls your own server, not a billed
API. It does **not** mean GPU inference is free: cost remains unknown/null, with
no invented tariff or composite score. Do not substitute TypeSafe's price.

Preserve all attempted results, including failures. Report per-tier counts,
schema validity, Brier/ECE, first-request latency, and serial p50/p95 with the
client/network path disclosed generically. Our local/network timings do not
replace the maintainer's measurements or their published latency adjustments.
Keep inference settings frozen for the run. Record limitations and any later
changes as a separate candidate and run, not a replacement for unfavorable data.

Before publishing, inspect the public export and all supporting files for
credentials, internal addresses, and personal filesystem paths. Publish only
sanitized reproducibility metadata and public-item evidence; retain originals
privately. Do not publish the upstream manifest unchanged.

## Submission checklist

- [x] Application code license documented (MIT for original code).
- [ ] All 231 public items attempted with the frozen container.
- [ ] Public summary and failure counts recorded without selecting favorable runs.
- [ ] Hardware, cache policy, network conditions, and unknown cost disclosed.
- [x] Immutable image, model revision, and serving instructions provided.
- [ ] Draft updated with actual results and reviewed before posting an issue.

The [issue draft](SUBMISSION_DRAFT.md) is preparation only, not a posted request.
The maintainers can pull the public image and run their own task lists through
their existing `typesafe` adapter without access to our infrastructure.
