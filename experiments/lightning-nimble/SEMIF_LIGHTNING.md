# Shared-prefix decision scoring on Nemotron Lightning

## What changed

This implements the shared-prefix execution idea from
[SemIf](https://github.com/TheoLeeCJ/SemIf), while retaining Nimble's
schema/field-selector prompt. It does not change model weights or generate
answer text. One forward processes the common prefix, native attention KV and
Mamba convolution/recurrent states are forked, and suffixes are processed in
bounded batches. Only the allowed answer-token rows of the output head are
projected in FP32.
The resulting option distributions are conditional scores, not calibrated
decision confidence.

Equal-length suffix buckets avoid padding-dependent recurrent-state behavior.
The optimized path skips cache splitting for one question, consumes the
request-local cache directly for the last batch, and groups compatible answer
projections. It never reuses a cache across different requests.

## Verified execution results

H100 80GB, BF16 checkpoint, Transformers 5.17.0, PyTorch 2.13.0+cu130,
grouped-matrix expert implementation. The checkpoint is the existing local
Lightning checkpoint; byte identity with a public release is not established.

The initial experiment uses the existing 24-case authored diagnostic, grouped
into requests of 1, 2, 4, or 8 questions. Every independent/shared pair uses
identical complete prompt tokens. Three timed repetitions follow warm-up for
each shape. Numbers are pooled median model-side milliseconds, excluding
tokenization, network, and HTTP serving. Different question counts change the
schema and prompt length; they are not a fixed-total-token experiment.
Here, independent scoring means **serial full-prompt forwards**, not a
production vLLM server with continuous batching and prefix caching.

| Questions | Independent, optimized kernels | Shared, optimized kernels, batch cap 8 | Shared, fallback kernels |
| --- | ---: | ---: | ---: |
| 1 | 53.5 ms | 53.4 ms | 461.9 ms |
| 2 | 113.4 ms | 111.3 ms | 621.7 ms |
| 4 | 283.7 ms | 117.6 ms | 870.1 ms |
| 8 | 703.6 ms | 152.8 ms | 1,347.4 ms |

Each kernel environment produced **288/288 matching shared-versus-independent
decisions** across the tested batch sizes. Scores were not bit-identical:
maximum absolute logit differences were 1.01 with fallback kernels and 1.11
with optimized kernels. Matching argmax here is not proof that every future
near-tie will remain unchanged.

Kernel changes also change floating-point arithmetic. Across the 24 diagnostic
cases, independent correctness was 12/24, 16/24, 16/24, and 17/24 with fallback
kernels, versus 11/24, 14/24, 16/24, and 17/24 with optimized kernels for the
four respective group sizes. Shared execution matched its corresponding
independent reference. Group-size accuracy differences are prompt changes,
not evidence that cache reuse improves reasoning.

The frozen 324-row Nimble holdout gives 194/324 for the optimized base model.
The original training evaluation gave 195/324 under its different kernel and
autocast environment. These are not identical numerical configurations.

Fallback timings are preliminary: kernel compilation ran on the host during
part of that run, and the GPU reached 87°C. The optimized independent/shared
comparison ran after compilation, on the same GPU with the same kernels.
The fallback allocator recovered from a long-prompt allocation retry; no
question was dropped.

## Optimization and verification

The largest improvement was replacing PyTorch Mamba reference operations with
`mamba-ssm==2.3.2.post1` and `causal-conv1d==1.7.0`. These were installed in an
isolated inference overlay without altering the running training environment.
The optimized eight-question profile attributes roughly 38% of measured GPU
time to grouped expert matrix multiplication. Mamba scan accounts for about
4%; it is no longer the dominant GPU operation.

CPU and optimized-GPU tiny Nemotron-H tests cover attention plus Mamba layers,
mixed suffix lengths, mixed candidate sets, repeated calls, branch ordering,
batch caps 1/2/8, the single-question fast path, and unchanged weight versions.
Full-model label agreement is tested separately rather than inferred from
tiny-model checks.

```bash
python3 test_shared_native.py
python3 test_shared_native.py --device cuda
python3 benchmark_shared_native.py --model "$MODEL_CHECKPOINT" \
  --cases semantic-cases.json --output native-results.json --optimized --profile
python3 evaluate_native.py --model "$MODEL_CHECKPOINT" --output heldout-results.json
```

Build the pinned optional kernels once in the inference environment, not at
every service startup. Installing against a nightly Torch version may compile
CUDA extensions. Preserve the tested Torch/CUDA environment rather than
allowing dependency resolution to replace it.

## Upstream fixtures and fine-tuned checkpoint

The separately run SemIf fixtures use source revision
`1f2dea3e25379f9dfc98cb83c324f00ab5deda37`. Their data is adapted to the same
Nimble prompt; this is **not** a reproduction of SemIf's published prompt or
model results. The 777-row systems fixture has no gold labels: it measures
execution agreement and speed, not accuracy. Its 21-question requests use a
4,096-token limit, beyond the training run's 2,048-token cap, with no truncation.

Base-model fixture results:

| Fixture | Independent | Shared | Shared/independent agreement |
| --- | ---: | ---: | ---: |
| Authored144, labeled | 98/144 correct | 99/144 correct | 142/144 |
| Shape777, no gold labels | 3,525 ms median per 21-question group | 630 ms median | 762/777 |

Both fixtures completed with no missing rows or length errors. Fixture timings
warm the model on the first group, not every later shape, so they may include
shape-specific initialization. Treat them separately from the per-shape-warmed
1/2/4/8-question experiment above.

An additional split-point experiment aligned the shared prefix to Mamba's
128-token chunk boundary without changing any full prompt tokens. Agreement
fell to 750/777, and total shared runtime increased from 21.8 to 33.1 seconds.
It is retained as a negative experiment, **not enabled by default**.

The LoRA run completed all 335 updates. In its original evaluation environment,
the base scored 195/324, the adapter 272/324, and the merged BF16 model 269/324.
Eight predictions changed when merging. The original base and unmerged adapter
are preserved. The saved model was then reloaded and evaluated under the
optimized kernels, without autocast:

| Checkpoint | Frozen holdout correct | Accuracy |
| --- | ---: | ---: |
| Base | 194/324 | 59.9% |
| Unmerged trained adapter | 273/324 | 84.3% |
| Reloaded merged BF16 | 270/324 | 83.3% |

The runtime change and merge are reported separately. The adapter has the
highest measured holdout count; merging removes the adapter execution but is
not numerically lossless.

The reloaded merged model also completed the per-shape-warmed diagnostic:

| Questions per request | Independent ms | Shared ms, batch cap 8 | Shared correct | Shared/independent agreement |
| --- | ---: | ---: | ---: | ---: |
| 1 | 53.6 | 53.6 | 20/24 | 24/24 |
| 2 | 119.5 | 111.7 | 18/24 | 23/24 |
| 4 | 283.5 | 116.7 | 21/24 | 24/24 |
| 8 | 710.9 | 141.9 | 17/24 | 23/24 |

Across all batch caps, merged-model shared scoring matched 284/288 independent
decisions. Training improved several diagnostic group sizes, but it did not
uniformly solve the multi-question task. Question grouping changes the prompt;
field batching changes execution only. These are different effects.

A subsequent fresh-process comparison also measured ordinary full-prompt
batching, so prefix reuse is not compared only with a serial reference:

| Questions | Serial full prompts | Batched full prompts | Shared prefix |
| --- | ---: | ---: | ---: |
| 1 | 53.5 ms | 53.5 ms | 53.5 ms |
| 2 | 119.4 ms | 87.7 ms | 114.0 ms |
| 4 | 285.8 ms | 185.3 ms | 121.2 ms |
| 8 | 715.7 ms | 531.7 ms | 143.6 ms |

For these lengths, ordinary batching wins at two questions; shared-prefix
execution wins at four and eight. No automatic two-question dispatch rule is
enabled: the crossover also depends on context length, and batch execution
can change borderline answers.

This fresh process had different serial accuracy counts (21/24, 18/24, 20/24,
22/24) from the earlier process. Therefore not every observed flip can be
attributed specifically to cache reuse. A follow-up repeated both serial and
shared scoring ten times on each of three eight-question groups: every score
was bit-identical within that loaded process. The cross-process source was
not isolated. Raw per-run counts and logits are retained rather than averaged
into a claim of deterministic equivalence.

The merged checkpoint completed both upstream-data fixtures without omitted
rows or input-length errors:

| Fixture | Independent | Shared | Shared/independent agreement |
| --- | ---: | ---: | ---: |
| Authored144, labeled | 114/144 correct | 113/144 correct | 143/144 |
| Shape777, no gold labels | 3,684 ms median per 21-question group | 653 ms median | 759/777 |

The 144 labeled fixture states have zero exact serialized-state matches with
the training data. This is not a broader contamination or semantic-overlap
audit. These fixture results do not establish calibrated probabilities or
production accuracy.

No production endpoint has been changed by this experiment.

## Restoring the tested environment

Both compiled optional-kernel wheels were preserved on scratch, with verified
SHA-256 hashes in `kernel-wheels.json`, avoiding another CUDA compilation when
using the same tested runtime. They are Python 3.12/Linux x86-64 artifacts built
against the pinned CUDA 13 / Torch container, not universal wheels for arbitrary
Torch installations.

```bash
python3 -m pip install --no-deps "$MAMBA_WHEEL" "$CAUSAL_CONV_WHEEL" peft==0.21.0
python3 benchmark_shared_native.py --model "$MODEL_CHECKPOINT" \
  --cases semantic-cases.json --output batched-comparison.json \
  --optimized --batched-comparison
python3 benchmark_semif_fixture.py --model "$MODEL_CHECKPOINT" \
  --input "$SEMIF_AUTHORED_FIXTURE" --output authored-fixture.json
```

The final CPU and optimized-GPU tests passed, including full-prompt batching.
The adapter, new merged checkpoint, row-level results, profiler output, scripts,
and reusable kernel wheels are saved. The original checkpoint is unchanged.
