# Optimized answer ordering

The optimized container orders answer options and assigns their internal answer
codes in the new order. API responses retain the original labels, boolean
meaning, and numeric score levels. No model weights are changed and no extra
model inference is required.

`POSITION_ORDERING=1` enables this behavior by default. Set
`POSITION_ORDERING=0` to preserve the input option order. Health and scoring
metadata identify the enabled method as `length_overlap_rotate2_recode_v1`.

## Method

For each option description, compute its tokenizer length divided by the longest
description length, minus twice the fraction of its content words found in the
state or question instructions. Sort descending with stable ties, then rotate
left by two positions. Assign A/B/C and subsequent available single-token codes
to the resulting positions. Decode probabilities back to the original labels.
Structured descriptions are serialized as JSON for ranking.

## Measurements

On 231 public JevBench questions with Nemotron-Labs-Diffusion-14B on one H100:

| Method | Correct | Accuracy | Median latency |
|---|---:|---:|---:|
| Input option order | 161/231 | 69.70% | 20.22 ms |
| Optimized ordering and code assignment | 174/231 | 75.32% | 21.19–21.30 ms |

Both methods reproduced identical probabilities across two runs. Latency was
measured using serial requests to a warm model, not concurrent load. The public
dataset includes examples used to select the ordering rule; these measurements
are not an independent held-out evaluation or an official leaderboard score.

## Container

Use `ghcr.io/pst2154/nemotron-jev:14b-vllm-ordered-20260922` with the Docker command
in the README. It starts both the explorer UI and System One API on port 8770.

To build the application layer without recompiling vLLM:

```bash
docker build \
  --build-arg VLLM_IMAGE=ghcr.io/pst2154/nemotron-jev@sha256:b8afc221e1ef9c8e74847e1a3d304a77123ac5114f1ef9f102048866fb286aa9 \
  -t nemotron-jev:ordered .
```

The earlier `14b-vllm-ordering-9d7f698` image only moves option positions; use the
image above for the measured 174/231 method. The original `14b-vllm-20260922`
benchmark image remains unchanged.
