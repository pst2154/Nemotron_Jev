# Optimized Nemotron Diffusion 8B serving

This branch serves `nvidia/Nemotron-Labs-Diffusion-8B` through the existing
SystemOne API and Decision Lab UI. It does not change the published 14B images.

## Deploy

Requires an H100-class Hopper GPU, Docker with NVIDIA Container Toolkit, and
a CUDA-13-compatible driver. Tested on H100 80GB. Other architectures are not
validated by this image.

```bash
git clone --branch perf/nemotron-diffusion-8b https://github.com/pst2154/Nemotron_Jev.git
cd Nemotron_Jev
docker build -t nemotron-jev:8b-optimized .
docker volume create nemotron-8b-models
docker run -d --name nemotron-8b --gpus all --shm-size=8g \
  -p 127.0.0.1:8770:8770 -v nemotron-8b-models:/models \
  nemotron-jev:8b-optimized
docker logs -f nemotron-8b
```

Open `http://localhost:8770` after `/health` reports ready. Startup downloads
the pinned 8B checkpoint and compiles/captures execution graphs. The Dockerfile
reuses our public, digest-pinned modified vLLM runtime; it does not rebuild vLLM.
The `8b-optimized` tag above is built locally, not a published GHCR tag.

```bash
curl http://localhost:8770/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"jev-latest","state":"The customer requests a refund.","questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Software errors"}}}}'
```

The response identifies `Nemotron-Labs-Diffusion-8B`; `jev-latest` is only a
compatibility alias. There is no TypeSafe upstream call. Keep the port private:
the application does not implement authentication.

## Optimizations

- Modified vLLM masked-decision model, compiled execution and CUDA graphs.
- Candidate-only output projection in BF16 rather than the full vocabulary.
- Direct candidate logits, avoiding a full-vocabulary probability calculation.
- Prefix reuse and batched tokenization for multi-question requests.
- Existing frozen option ordering and code reassignment.

The model does not generate an autoregressive answer sequence. API requests are
serialized by the existing server lock; questions within a request are batched.
This branch does not claim cross-request continuous batching.

BF16 remains the default. Per-channel FP8 can be selected with
`-e QUANTIZATION=fp8_per_channel -e CANDIDATE_ONLY=0`; it uses the full output
head and can change answer probabilities.

## Model and reproducibility

- Checkpoint revision: `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc`.
- Underlying vLLM runtime image digest:
  `sha256:03e09c01d813d39418be9e05bc20ab8f39e5945794700360816027cbb46127b6`.
- `MODEL_ID` controls download and response identity. If mounting an existing
  checkpoint with `SKIP_DOWNLOAD=1`, set it to that checkpoint's actual ID.
- Use a separate volume from 14B to avoid mixing checkpoint files.
- For old 14B operation, set `MODEL_ID=nvidia/Nemotron-Labs-Diffusion-14B`;
  its known revision is selected automatically unless explicitly overridden.

## Validation

On the frozen 3,000-question teacher-agreement benchmark:

| Configuration | Teacher matches / 2,778 | Agreement | Median ms | p95 ms |
|---|---:|---:|---:|---:|
| Native eager BF16 (earlier run) | 1,453 | 52.30% | 60.9 | 62.1 |
| **Optimized vLLM BF16, candidate-only** | **1,453** | **52.30%** | **13.9** | **20.0** |
| vLLM per-channel FP8, full output head | 1,435 | 51.66% | 14.9 | 24.5 |

BF16 is the recommended default: about **4.4× lower median latency** than the
earlier native scorer, with the same aggregate agreement. Equal totals do not
mean identical predictions: 61 of 3,000 top labels changed versus the native
scorer. FP8 did not improve latency in this test and lost
18 teacher matches; it remains optional rather than the default.

All 6,000 API predictions completed and passed the common scorer. Benchmark
requests use `ISOLATE_REQUEST_CACHE=1` to clear cached prefix blocks between
requests. Normal serving defaults to shared prefix caching. Tests used one H100
80GB HBM3, concurrency one, three warmups, CUDA graphs, 128 maximum sequences,
and an 8,192-token batching budget. Native timings are in-process research
scoring; optimized timings include the local HTTP round trip. No truncation.

This measures agreement with teacher distributions, not independently verified
accuracy. The 222 tied teacher maxima are excluded from top-answer agreement;
all 3,000 rows enter the probability metrics. Questions were frozen before this
work: 1,000 Noul, 1,000 choice, and 1,000 score, sampled from
`SargeDev/jev-distill-corpus-v3` revision
`fc99c6357a9f89f7512c4a987314352addead049`, `test_set_30k.jsonl`.
Frozen questions SHA256:
`f9229a63d2418a3cd8b537791d6e9bacaf1dc045aafd8b6feeb66bb91e510877`.

- [BF16 predictions](evaluation/8b-serving-results/bf16/predictions.jsonl)
  and [metrics](evaluation/8b-serving-results/bf16/metrics.json).
- [FP8 predictions](evaluation/8b-serving-results/fp8/predictions.jsonl)
  and [metrics](evaluation/8b-serving-results/fp8/metrics.json).
- [HTTP evaluation runner](evaluation/benchmark_frozen_http.py).

The source dataset and frozen input file are not embedded in the container.

The UI returned successfully and a separate synthetic API smoke test passed
with 1, 10, 30, and 100 questions per request. Its median latencies were
13.2, 34.4, 43.9, and 91.1 ms respectively. These were repeated identical short
questions, not representative 100-question benchmark throughput; one of the
three measured 100-question calls took 341.3 ms. See [raw smoke timings](evaluation/8b-serving-results/multiquestion.json).
