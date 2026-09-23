# H100 inference optimization

Tested September 23, 2026, with Nemotron Labs Diffusion 14B on one H100 PCIe
80 GB. Per-channel FP8 delivered a measured 1.11–1.34× speedup through the
System One HTTP API. It is opt-in; BF16 remains the default.

## Enable

Add these environment settings to the existing container command:

```bash
-e QUANTIZATION=fp8_per_channel -e CANDIDATE_ONLY=0
```

The checkpoint is quantized during loading, without modifying its saved weights.
The current backend requires the full output head for quantized inference.
Prompts, option ordering, typed responses and model architecture are unchanged.
Numerical precision changes, so probabilities and predictions can change.

## Matched HTTP measurements

Both runs used the same built image and normal entrypoint, with 128 maximum
sequences, an 8,192-token scheduling budget, and GPU memory utilization 0.65.
Prefix caching and shared-prefix priming were enabled within each request;
`ISOLATE_REQUEST_CACHE=1` cleared the cache between requests.

Each case used one warmup followed by three timed calls. These are median
client-observed latencies from the same node, including HTTP, prompt preparation,
inference and answer assembly, but excluding startup and remote network travel.
State lengths are exact tokenizer counts; the complete model prompt is longer.
Repeated-word states and simple choices measure latency, not semantic accuracy.

| State tokens | Questions | BF16 ms | Per-channel FP8 ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 26.0 | 21.2 | 1.23× |
| 128 | 8 | 74.9 | 63.4 | 1.18× |
| 128 | 32 | 197.9 | 152.1 | 1.30× |
| 128 | 100 | 541.1 | 418.8 | 1.29× |
| 1,000 | 1 | 74.1 | 64.2 | 1.15× |
| 1,000 | 8 | 129.4 | 96.8 | 1.34× |
| 1,000 | 32 | 243.9 | 197.7 | 1.23× |
| 1,000 | 100 | 594.2 | 489.9 | 1.21× |
| 4,096 | 1 | 277.6 | 220.5 | 1.26× |
| 4,096 | 8 | 339.9 | 280.2 | 1.21× |
| 4,096 | 32 | 518.3 | 440.5 | 1.18× |
| 4,096 | 100 | 1,015.8 | 919.1 | 1.11× |
| 8,192 | 1 | 590.2 | 460.7 | 1.28× |
| 8,192 | 8 | 655.6 | 522.7 | 1.25× |
| 8,192 | 32 | 914.4 | 738.3 | 1.24× |
| 8,192 | 100 | 1,688.2 | 1,424.2 | 1.19× |

These are serial requests, not a cross-client concurrency benchmark. The current
HTTP server serializes requests; vLLM batches questions within a request.

## Accuracy and verification

On the public JevBench easy/original/hard suite (231 questions), the matched
HTTP runs scored 173 correct with BF16 and 175 with per-channel FP8. Three labels
changed: two incorrect answers became correct, and one changed between incorrect
labels. No correct BF16 answer regressed in this particular comparison. This
small public set does not establish general accuracy improvement or calibration.

An earlier offline-engine comparison scored 174 versus 175, with four changed
labels including one regression. Do not conflate those baselines or claim that
FP8 preserves all predictions. The final container reproduced all 231 predictions
from the separate FP8 HTTP run.

Additional checks passed:

- 24 comparisons of individual questions against mixed batches of four and
  sixteen questions had identical per-channel FP8 probabilities.
- Choice, Noul and Score HTTP smoke tests, complete answer maps in all 16 latency
  cases, UI HTML serving and the built-in container healthcheck.
- All 26 local tests, including validation of unsupported precision settings.
- The running image's server source hash matched the tested source.

The benchmark harness was pinned at
`51a8d73fa798aa337bb1b26abd10995c0ab847e9`; the vLLM backend was
`0.1.dev2+g2c83d10ca`, with PyTorch `2.13.0+cu130`.

## Other experiments

The serving profile attributed roughly 84–88% of summed GPU kernel duration to
matrix multiplication. Attention accounted for approximately 3–7%, making GEMM
precision a larger opportunity than replacing attention.

Expanded CUDA graphs and BF16 GEMM autotuning preserved benchmark predictions
but did not materially improve longer-input latency. They were not added to
serving defaults. Block-scaled FP8 improved some long-input timings but regressed
at 256 tokens and changed more labels. Per-channel FP8 offered a more consistent
tradeoff across the tested shapes.

## Published container and evidence

The FP8 image is published separately as
`ghcr.io/pst2154/nemotron-jev:14b-vllm-fp8-channel-20260923`.
Its digest is
`sha256:03e09c01d813d39418be9e05bc20ab8f39e5945794700360816027cbb46127b6`.
It defaults to per-channel FP8 and starts both the UI and System One API on
port 8770. Existing BF16 tags, production endpoints and leaderboard submissions
are unchanged. See the [deployment instructions](README.md#optional-fp8-inference).

Raw measurements and question-level results:

- [BF16 timings](evaluation/h100-fp8/bf16/summary.json) and
  [accuracy](evaluation/h100-fp8/bf16/accuracy.jsonl).
- [FP8 timings](evaluation/h100-fp8/fp8/summary.json) and
  [accuracy](evaluation/h100-fp8/fp8/accuracy.jsonl).
- [HTTP benchmark script](evaluation/h100-fp8/benchmark.py).
