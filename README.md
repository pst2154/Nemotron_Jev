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

## Prebuilt container

Image: `ghcr.io/pst2154/nemotron-jev:14b-v2`

Published digest: `sha256:a1bf099bb919461659339f5bb8c1e962b3d3367ffe40c74fcb92c4c8c2b170ea`.

The GHCR package is public. Anonymous access to the published manifest and digest has been verified; no GitHub login is required.

```bash
docker pull ghcr.io/pst2154/nemotron-jev:14b-v2
docker volume create nemotron-models
docker run -d --name nemotron-jev --gpus all --shm-size=8g \
  -p 127.0.0.1:8770:8770 -v nemotron-models:/models \
  ghcr.io/pst2154/nemotron-jev:14b-v2
```

For immutable deployments, replace the tag with `ghcr.io/pst2154/nemotron-jev@sha256:a1bf099bb919461659339f5bb8c1e962b3d3367ffe40c74fcb92c4c8c2b170ea`.

## Deploy from source

Tested on one H100 80 GB with Docker and NVIDIA Container Toolkit. Use a CUDA-13-compatible driver. Other GPUs and smaller memory configurations have not been verified. Allow at least 30 GB for model data plus substantial space for the CUDA development image.

```bash
git clone https://github.com/pst2154/Nemotron_Jev.git
cd Nemotron_Jev
docker build -t nemotron-jev:14b-v2 .
docker volume create nemotron-models
docker run -d --name nemotron-jev --gpus all --shm-size=8g \
  -p 127.0.0.1:8770:8770 \
  -v nemotron-models:/models \
  nemotron-jev:14b-v2
docker logs -f nemotron-jev
```

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

Also available: `/explorer`, `/health`, and `/v1/chat/completions` with `messages`, `max_tokens`, and `mode` (`dlm`, `ar`, or `linear_spec`). Chat is non-streaming. The explorer uses the scoring API, **not generated JSON**.

## How scoring works

For each question, options are assigned tokenizer-verified single-token codes. The model causally encodes the prompt, then scores one masked answer position using its diffusion head. A softmax over only the allowed candidate logits gives the reported distribution. No confidence value is invented or requested as generated text.

Choice selects the highest probability; Noul returns the yes probability; Score computes `sum(index * probability)`. Confidence is `max(probabilities)`, matching the previous DiffusionGemma adapter's convention—not necessarily TypeSafe's proprietary confidence statistic. These probabilities are **not empirically calibrated**.

Questions are independent and processed sequentially under one inference lock. Multiple questions can be submitted in one request, but this implementation does not reproduce DiffusionGemma's joint-slot throughput. Larger requests take longer and block other inference requests; health and UI reads remain available.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8770` | UI and API port (keep default for built-in healthcheck) |
| `MODEL_REVISION` | `f8c3e2c078e193599b8882d965b1001c456ba738` | Hugging Face revision |
| `CHECKPOINT_DIR` | `/models/checkpoint` | Container model-cache location |
| `HF_HOME` | `/models/hf-cache` | Container Hugging Face cache |
| `SKIP_DOWNLOAD` | `0` | Set to `1` only with a complete existing checkpoint |
| `OMP_NUM_THREADS` | `8` | CPU worker limit matching the original deployment |

Limits: 512 questions, 16,384 prompt tokens per question, 2 MB request body, and tokenizer-dependent maximum number of candidate codes. Exceeding supported limits returns an error; inputs are not silently truncated. Chat supports up to 1,024 output tokens, with diffusion requests rounded up to a 32-token block boundary.

## Tests and results

```bash
TEST_URL=http://localhost:8770 node tests/live_scoring.mjs
python3 evaluation/run.py --url http://localhost:8770 --output evaluation/results
```

The Node test checks all three output types, probability normalization, weighted-score arithmetic, contrasting states, and a 16-question batch. The Python evaluator uses frozen cases and deterministic labels, not an LLM judge. See [accuracy report](ACCURACY_REPORT.md) and [raw results](evaluation/results/results.jsonl).

## Troubleshooting

- **Connection refused:** inspect `docker logs nemotron-jev`; download and weight loading finish before the server binds its port.
- **GPU unavailable:** verify `nvidia-smi`, Docker GPU access, and NVIDIA Container Toolkit installation.
- **Out of memory:** remove other workloads from the selected GPU, shorten state, or reduce simultaneous deployments. Quantization is not implemented.
- **Download failure:** check disk space, outbound Hugging Face connectivity, and model access/license requirements. Restart with the same volume to reuse downloaded files.
- **Permission denied on a bind-mounted cache:** run with `--user "$(id -u):$(id -g)"` and ensure that user can write the cache. Named volumes avoid typical root-squashed network-filesystem issues. Compiler caches default to writable temporary storage.
- **Old UI or missing bars:** refresh the browser and run a new query. Old history entries may predate the scoring backend.
- **Slow large request:** questions run sequentially. Split interactive work into smaller requests; this does not increase total GPU throughput.

## Scope and licensing

The 72-case benchmark is a small synthetic regression set, not proof of general reasoning quality, calibration, adversarial safety, or production readiness. Accuracy can depend on wording, option order, and evidence coverage. No tools execute commands contained in the supplied state.

Model weights are downloaded at runtime, not embedded in the image. Review the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) and the [model card](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B). The CUDA/FlashInfer base and Python dependencies retain their respective licenses. This project is not an official NVIDIA or TypeSafe product.
