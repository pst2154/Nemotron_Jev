# Nemotron Diffusion 8B: public JevBench self-test

Measured September 24, 2026. This is a public self-test, not an official
leaderboard score. No private or sealed items were evaluated.

| Tier | Correct, both passes | Accuracy |
|---|---:|---:|
| Easy | 48/48 | 100.00% |
| Standard | 61/72 | 84.72% |
| Hard | 63/111 | 56.76% |
| Overall | 172/231 | 74.46% |

| Local serial HTTP latency | Pass 1 | Pass 2 |
|---|---:|---:|
| Median | 15.95 ms | 15.74 ms |
| p95 | 79.01 ms | 76.67 ms |

All 231 responses per pass were strictly valid, with zero HTTP failures or
probability renormalizations. Predictions and probabilities were identical
across passes. Overall Brier: 0.352322; top-label ECE: 0.094007.
Cost is unknown, not free. Timings include same-node HTTP, exclude the
intentional 0.2-second pacing, and are not Internet or adjusted leaderboard latency.

## Frozen candidate

- Serving source: `cdc36d6013a6401bf1dfa7f4048cdb69e7138d69`.
- Checkpoint: `nvidia/Nemotron-Labs-Diffusion-8B`, revision
  `16c67f0560b912e93e0c4f5c3086d95fc`.
- One H100 80GB HBM3; BF16, candidate-only projection, modified vLLM,
  CUDA graphs, 128 maximum sequences, 8,192-token batching budget.
- No LoRA, extra training, quantization, generated probability text, or
  autoregressive answer rollout. Native candidate logits become distributions.
- Ordering: `length_overlap_rotate2_recode_v1`. This inherited ordering rule
  was selected using public benchmark examples in earlier 14B experiments.
  These public results are therefore not independent held-out evidence.
- Request prefix caches were isolated with `ISOLATE_REQUEST_CACHE=1`.
  The process was already loaded and had served earlier tests before pass 1.

## Reproduce

Requires Docker, NVIDIA Container Toolkit, an H100 and a CUDA-13-compatible
driver. The base runtime is public and digest-pinned in the Dockerfile.
There is no separately published 8B image tag; build this pinned source:

```bash
git clone https://github.com/pst2154/Nemotron_Jev.git
cd Nemotron_Jev
git checkout cdc36d6013a6401bf1dfa7f4048cdb69e7138d69
docker build -t nemotron-jev:8b-submission .
docker volume create nemotron-8b-models
docker run -d --name nemotron-8b --gpus all --shm-size=8g \
  -p 127.0.0.1:8770:8770 -v nemotron-8b-models:/models \
  -e ISOLATE_REQUEST_CACHE=1 nemotron-jev:8b-submission
docker logs -f nemotron-8b
```

After startup, `/health` should report ready, `Nemotron-Labs-Diffusion-8B`,
`candidate_only_head: true`, `quantization: null`, and
`cache_policy: isolated_request`. The same container starts the explorer UI.
The server has no authentication; keep the loopback binding or use an
authenticated proxy. Do not reuse a 14B checkpoint volume.

Our unchanged public harness was pinned to
`51a8d73fa798aa337bb1b26abd10995c0ab847e9`; all 90 upstream tests passed.
Use the existing `typesafe` adapter with endpoint `http://127.0.0.1:8770`,
model `Nemotron-Labs-Diffusion-8B`, and `--key-env ''`.
The endpoint is the base URL, not the `/v1/systemone` path.
Tasks, in order: `datasets/public/easy.jsonl`,
`datasets/public/original.jsonl`, `datasets/public/hard.jsonl`.
Run serially with `--delay-s 0.2`, fresh result directories, and
`--cost-basis self_hosted_no_published_tariff --reserve-usd 0 --cap-usd 1`.
Maintainers should use their current harness and private tasks for an official entry.

An initial runner invocation appended the API path twice and stopped after
three HTTP 404s without model predictions. Fixing the base URL produced the
two complete passes reported here; no benchmark content or serving code changed.

## Evidence

- [Pass 1 predictions](results/pass1.jsonl), [summary](results/pass1-summary.json).
- [Pass 2 predictions](results/pass2.jsonl), [summary](results/pass2-summary.json).

The existing 14B submission and its evidence are unchanged. Application code is
MIT; the vLLM runtime retains Apache-2.0; model weights retain their upstream license.
