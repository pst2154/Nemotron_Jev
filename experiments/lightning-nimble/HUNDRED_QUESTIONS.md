# 100 questions in one logical request: trained Lightning

Measured September 21, 2026, on one H100 80GB with optimized native kernels.
Both trained LoRA and merged BF16 completed ten requests containing **100
distinct questions each**, returning every answer with finite candidate scores.

| Checkpoint | Correct at 100 questions | Median model time | Median prompt preparation |
| --- | ---: | ---: | ---: |
| Trained LoRA | 969/1,000 (96.9%) | 1,327 ms | about 2,500 ms |
| Merged BF16 | 972/1,000 (97.2%) | 1,078 ms | about 2,500 ms |

Model times exclude prompt preparation and HTTP/network overhead. The current
research implementation therefore takes roughly **3.8 seconds LoRA / 3.6
seconds merged** for preparation plus scoring, not 1.3/1.1 seconds end-to-end.
Prompt preparation repeatedly tokenizes full prompts and verifies candidate
boundaries. Its cost is measured separately and has not been optimized here.

## Scaling

Medians after one warm-up per shape; three timed trials per smaller count,
ten at 100. Correctness counts include every measured trial.

| Questions | LoRA model ms | Merged model ms | LoRA correct | Merged correct |
| --- | ---: | ---: | ---: | ---: |
| 1 | 71.7 | 58.9 | 3/3 | 3/3 |
| 8 | 153.3 | 126.2 | 24/24 | 24/24 |
| 20 | 400.2 | 326.4 | 60/60 | 60/60 |
| 30 | 495.9 | 402.4 | 89/90 | 90/90 |
| 50 | 690.4 | 558.1 | 150/150 | 150/150 |
| 100 | 1,327.0 | 1,078.0 | 969/1,000 | 972/1,000 |

Both models scored **96/100** when the same questions were asked individually
with trial-zero state. For that same state, each scored **97/100** in the
100-question request. This does not support a claim that all residual lookup
errors arise only when grouping questions. Different group sizes change the
schema and full prompt, so they are not identical-token execution comparisons.

## Workload and execution

- Exactly 1,000 state tokens under the Lightning tokenizer. The state contains
  records 0 through 99, alternating on/off flags, followed by neutral padding.
- One distinct lookup per record, with enabled/disabled choices. This is a
  simple synthetic lookup diagnostic, **not the 324-item Nimble semantic
  holdout**, nor 1,000 independent semantic examples.
- Trial reference text varies; underlying facts and expected labels remain
  unchanged. All labels come from the fixture, not a teacher model.
- At 100 questions: 8,075 shared-prefix tokens and a longest full prompt of
  8,087 tokens. No truncation. This exceeds the 2,048-token training cap.
- One logical request, client concurrency one, suffix batch cap eight. The
  prefix is processed once; native attention and Mamba caches are forked into
  bounded suffix batches. It is **not all 100 branches resident at once**.
- Candidate-token logits are read directly: no autoregressive answer
  generation, no persistent cache across requests, no endpoint modifications.
- Peak allocated CUDA memory: about 60.72 GiB LoRA / 60.69 GiB merged.
- These are native model-side timings, not directly comparable to older
  HTTP measurements or a different model's hosted endpoint latency.

Raw results: [LoRA](semif-100-adapter.json), [merged](semif-100-merged.json).
Runner: [benchmark_100_native.py](benchmark_100_native.py).
For semantic accuracy and training details, see [the main report](SEMIF_LIGHTNING.md).
