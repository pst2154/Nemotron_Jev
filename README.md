# Nemotron Diffusion Decision Lab

Ask typed questions about text or JSON and inspect model-derived probability distributions in a browser. One container runs the model, the original Decision Lab explorer, and a TypeSafe-shaped API.

This is an experimental adapter for the **dense Nemotron-Labs-Diffusion-14B model**, not a mixture-of-experts model, Jev, an official TypeSafe service, or a calibrated replacement for Jev. The repository name describes its interface, not its model identity.

## What works

| Question | Output |
| --- | --- |
| Choice | Selected label, probability per option, maximum-probability confidence |
| Noul | Probability of yes |
| Score | Probability per ordered level, probability-weighted 0-based score, confidence |

The explorer includes samples, State and Questions editors, answer bars, a Noul marker, raw JSON, local history, and share links. Shared URLs contain the entered state and questions; do not share sensitive inputs.

## Optimized container

The H100-validated image starts **both the original explorer UI and the
classification-only vLLM API**. The GHCR package is public; anonymous access
to this image's manifest and configuration has been verified.

Image: `ghcr.io/pst2154/nemotron-jev:14b-vllm-ordered-20260922`.
For an immutable deployment, use
`ghcr.io/pst2154/nemotron-jev@sha256:4dd795850f342183b9e54eb25b350682e63506043f8fd33dde2521d7ce6ae560`.

Answer ordering and code reassignment are enabled by default. This method scored
174/231 on the public JevBench dataset; see [method and measurements](POSITION_ORDERING.md).

The following commands require Docker with NVIDIA Container Toolkit, a CUDA-13-compatible driver,
and an H100/H200-class Hopper GPU. H100 80 GB was tested; H200 was not tested.
The prebuilt image targets compute capability 9.0, not B200/B300 or L40S.

```bash
docker pull ghcr.io/pst2154/nemotron-jev:14b-vllm-ordered-20260922
docker volume create nemotron-models
docker run -d --name nemotron-jev --gpus all --shm-size=8g \
  -p 127.0.0.1:8770:8770 -v nemotron-models:/models \
  ghcr.io/pst2154/nemotron-jev:14b-vllm-ordered-20260922
docker logs -f nemotron-jev
```

The vLLM source remains `2c83d10caa71e3a9eac8fdd9f1288dac8c65596b`.
The original benchmark image, `14b-vllm-20260922`, remains available at
`sha256:b8afc221e1ef9c8e74847e1a3d304a77123ac5114f1ef9f102048866fb286aa9`.

The image contains the serving software, not model weights. First launch fetches
the pinned model into the persistent volume. To reuse an existing checkpoint,
mount its directory read-only at `/models/checkpoint` and set `SKIP_DOWNLOAD=1`.
Keep a writable cache mount at `/models/hf-cache` when using a non-root user.
The matched HTTP benchmark additionally sets `ISOLATE_REQUEST_CACHE=1`.

The older `14b-v2` image is the **original sequential native scorer**, not the
optimized implementation. Its unchanged digest is
`sha256:a1bf099bb919461659339f5bb8c1e962b3d3367ffe40c74fcb92c4c8c2b170ea`.

## Build the optimized version from source

This branch requires the modified vLLM source, not an unmodified vLLM image.
Use Docker with NVIDIA Container Toolkit and a CUDA-13-compatible driver.
The commands below build Hopper kernels for H100/H200. Allow at least 30 GB
for model data and substantial additional disk space for the source build.
GPU validation completed on one H100 80 GB. See [measured latency, accuracy,
and validation results](OPTIMIZATION_REPORT.md).

```bash
git clone --branch feat/nemotron-labs-diffusion https://github.com/pst2154/vllm.git
cd vllm
docker build -f docker/Dockerfile --target vllm-openai \
  --build-arg torch_cuda_arch_list=9.0 \
  --build-arg max_jobs=16 --build-arg nvcc_threads=2 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg GIT_REPO_CHECK=0 -t nemotron-masked-vllm:dev .
cd ..
git clone --branch perf/nemotron-diffusion-optimized https://github.com/pst2154/Nemotron_Jev.git
cd Nemotron_Jev
docker build --build-arg VLLM_IMAGE=nemotron-masked-vllm:dev \
  -t nemotron-jev:14b-vllm .
docker volume create nemotron-models
docker run -d --name nemotron-jev --gpus all --shm-size=8g \
  -p 127.0.0.1:8770:8770 \
  -v nemotron-models:/models \
  nemotron-jev:14b-vllm
docker logs -f nemotron-jev
```

`RUN_WHEEL_CHECK=false` skips the upstream 500 MB PyPI upload-size limit for
this container-only build. The Hopper wheel exceeded that limit; this flag does
not skip compilation or model correctness checks.

The first start downloads the pinned checkpoint; subsequent starts reuse the volume. The same entrypoint starts **both UI and API**. Open `http://localhost:8770/` after the health endpoint responds:

```bash
curl --fail http://localhost:8770/health
```

For access from another machine, use an SSH tunnel or an authenticated TLS reverse proxy. This experimental server has **no built-in authentication, TLS, or rate limiting**; do not expose it directly to the public Internet. No TypeSafe API key is needed.

## API example

```bash
curl --fail-with-body http://localhost:8770/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "Please refund the duplicate invoice charge.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoices, payouts, or refunds",
          "technical": "Software crashes or infrastructure outages"
        }
      },
      "refund_requested": {"type":"noul", "instructions":"Is a refund requested?"},
      "urgency": {
        "type":"score", "instructions":"How urgent is the request?",
        "criteria":["No urgency stated", "Time sensitive", "Immediate business-blocking emergency"]
      }
    }
  }'
```

`jev-latest` is accepted as a compatibility identifier; the returned model is always `Nemotron-Labs-Diffusion-14B`. It never routes to TypeSafe. Structured instructions and criteria are serialized as JSON. Question IDs only identify response entries and are not used in model inference.

Also available: `/explorer` and `/health`. This branch exposes **classification
only**. It does not provide chat completions, AR generation, multi-token
diffusion generation, or self-speculation. The explorer uses scoring, not
generated JSON.

## How scoring works

For each question, options are assigned tokenizer-verified single-token codes. The model causally encodes the prompt, then scores one masked answer position using its diffusion head. A softmax over only the allowed candidate logits gives the reported distribution. No confidence value is invented or requested as generated text.

Choice selects the highest probability; Noul returns the yes probability; Score computes `sum(index * probability)`. Confidence is `max(probabilities)`, matching the previous DiffusionGemma adapter's convention—not necessarily TypeSafe's proprietary confidence statistic. These probabilities are **not empirically calibrated**.

Questions are independent sequences batched by vLLM. Each sequence ends in one
actual mask token, preserving the native scorer's causal-prefix /
one-token-bidirectional attention pattern. Numerical differences remain between
the native and vLLM execution paths; the report records the failed strict parity
check alongside measured accuracy. A single internal
sampling step returns candidate logprobs; its sampled token is discarded and
never fed back into the model. There is no autoregressive answer rollout.

Prefix caching reuses common state blocks. Optional prefix priming evaluates
the first question before scheduling the remaining questions, making those
blocks available to the first batch of suffixes. Optional candidate-only
projection reduces the final vocabulary projection while retaining all
candidate probabilities. These switches and batched tokenization are enabled
by default after H100 measurements. These are execution changes, not fine-tuning.

Requests currently share one inference lock, while questions *within* a request
are batched. Cross-request continuous batching is not implemented. Health and
UI reads remain available during inference.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8770` | UI and API port (keep default for built-in healthcheck) |
| `MODEL_REVISION` | `f8c3e2c078e193599b8882d965b1001c456ba738` | Hugging Face revision |
| `CHECKPOINT_DIR` | `/models/checkpoint` | Container model-cache location |
| `HF_HOME` | `/models/hf-cache` | Container Hugging Face cache |
| `SKIP_DOWNLOAD` | `0` | Set to `1` only with a complete existing checkpoint |
| `OMP_NUM_THREADS` | `1` | Avoid Torch CPU spin-wait contention |
| `RAYON_NUM_THREADS` | `8` | Tokenizer worker limit; match the available CPU allocation |
| `MAX_NUM_SEQS` | `128` | Maximum concurrently scheduled question sequences, not total questions per request |
| `MAX_BATCHED_TOKENS` | `8192` | Scheduler token budget per step |
| `GPU_MEMORY_UTILIZATION` | `0.85` | vLLM device-memory fraction |
| `ENFORCE_EAGER` | `0` | Set to `1` for eager-mode comparisons |
| `PRIME_SHARED_PREFIX` | `1` | Evaluate the first question before the remaining batch |
| `CANDIDATE_ONLY` | `1` | Project only the serving code vocabulary; requires unquantized TP=1 |
| `QUANTIZATION` | `none` | Optional online `fp8`, `fp8_per_channel`, or `fp8_per_block`; requires `CANDIDATE_ONLY=0`. Changes numerical precision and may change answers. |
| `DIRECT_LOGITS` | `1` | Return candidate logits directly and skip the redundant full-vocabulary softmax |
| `BATCH_TOKENIZE` | `1` | Batch prompt tokenization; exact token identity was checked on the benchmark inputs |
| `POSITION_ORDERING` | `1` | Order options and reassign internal codes; map responses back to original labels. `0` restores input order. |
| `ISOLATE_REQUEST_CACHE` | `0` | Set to `1` to reset the prefix cache before each request while retaining within-request reuse |

Limits: 512 questions, 16,384 prompt tokens per question, 2 MB request body, and
a tokenizer-dependent maximum number of candidate codes (**62 for the pinned
checkpoint**). `/health` reports these limits. Exceeding supported
limits returns an error; inputs are not silently truncated. A request with 129
questions is queued through the scheduler rather than rejected for exceeding
the 128-sequence concurrency setting.

### Optional FP8 inference

To enable per-channel FP8, add these options to the container command:

```bash
-e QUANTIZATION=fp8_per_channel -e CANDIDATE_ONLY=0
```

The server quantizes the existing BF16 checkpoint during loading; the saved
weights are not modified. Requests, option ordering, and response types stay the
same, but numerical probabilities and some predictions can change. BF16 remains
the default. `/health` reports the active quantization and output-head settings.
For non-root containers, mount a writable cache or set `HF_HOME=/tmp/hf-cache`.
See [H100 performance measurements](PERFORMANCE.md) for the tested latency and
accuracy tradeoffs.

To build a separate image with FP8 enabled by default:

```bash
docker build --build-arg VLLM_IMAGE=nemotron-masked-vllm:dev \
  --build-arg DEFAULT_QUANTIZATION=fp8_per_channel \
  --build-arg DEFAULT_CANDIDATE_ONLY=0 \
  -t nemotron-jev:14b-vllm-fp8-channel-20260923 .
```

This image starts the same UI and System One API on port 8770. The normal build
still defaults to BF16. The published FP8 image can be launched directly:

```bash
docker run --rm --gpus all --ipc=host -p 8770:8770 \
  -v nemotron-models:/models \
  ghcr.io/pst2154/nemotron-jev:14b-vllm-fp8-channel-20260923
```

This FP8 tag already sets both precision options. The first launch downloads
the checkpoint into the model volume. For reproducible deployments, use
`ghcr.io/pst2154/nemotron-jev@sha256:03e09c01d813d39418be9e05bc20ab8f39e5945794700360816027cbb46127b6`.

## Tests and results

For Benchmark Heaven, see the [JevBench submission preparation](evaluation/jevbench/README.md).
It pins the upstream harness and public-suite runner; no leaderboard score is claimed.

```bash
TEST_URL=http://localhost:8770 node tests/live_scoring.mjs
python3 evaluation/run.py --url http://localhost:8770 --output evaluation/results
```

The Node test checks all three output types, probability normalization,
weighted-score arithmetic, contrasting states, and a 16-question batch. It also
checks the UI/health routes, absence of chat generation, a 129-question response,
maximum-choice coverage, and rejection of oversized requests. The latter two large
valid requests test the response contract, not semantic accuracy. The
Python evaluator uses frozen cases and deterministic labels, not an LLM judge.
See the [optimization report](OPTIMIZATION_REPORT.md) for current latency and
accuracy measurements. The original native baseline's
[raw results](evaluation/results/results.jsonl) are retained for reference.

`evaluation/compare_backends.py` runs both implementations separately on the
same GPU with identical prompts. It records accuracy, candidate distributions,
state lengths, cache settings, and end-to-end latency. The default matrix uses
1/4/8/16/32/64/100 questions at 1,000/4,000/12,000 state tokens.
`evaluation/compare_results.py` checks matched payloads, option coverage,
argmax agreement, and probability drift; missing cases fail the comparison.

## Troubleshooting

- **Connection refused:** inspect `docker logs nemotron-jev`; download and weight loading finish before the server binds its port.
- **GPU unavailable:** verify `nvidia-smi`, Docker GPU access, and NVIDIA Container Toolkit installation.
- **Out of memory:** remove other workloads from the selected GPU, shorten state, or reduce simultaneous deployments. Quantization is not implemented.
- **Download failure:** check disk space, outbound Hugging Face connectivity, and model access/license requirements. Restart with the same volume to reuse downloaded files.
- **Permission denied on a bind-mounted cache:** run with `--user "$(id -u):$(id -g)"` and ensure that user can write the cache. Named volumes avoid typical root-squashed network-filesystem issues. Compiler caches default to writable temporary storage.
- **Old UI or missing bars:** refresh the browser and run a new query. Old history entries may predate the scoring backend.
- **Slow large request:** check the scheduler token budget and compare prefix priming on/off. Do not assume larger batching limits are faster; measure on the intended GPU and state lengths.

## Scope and licensing

The 72-case benchmark is a small synthetic regression set, not proof of general reasoning quality, calibration, adversarial safety, or production readiness. Accuracy can depend on wording, option order, and evidence coverage. No tools execute commands contained in the supplied state.

Original application code in this repository is available under the [MIT License](LICENSE).
This grant does not relicense model weights, vLLM, upstream benchmark datasets,
or other third-party components. The separate vLLM fork remains under its
upstream Apache-2.0 license.

Model weights are downloaded at runtime, not embedded in the image. Review the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) and the [model card](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B). The CUDA/FlashInfer base and Python dependencies retain their respective licenses. This project is not an official NVIDIA or TypeSafe product.
