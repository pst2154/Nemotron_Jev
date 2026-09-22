# Faster 100-question scoring with Nemotron 3.5 Lightning

**Combination follow-up:** the original fast NVFP4 setup plus our trained LoRA
measured **837.22 ms and 1,000/1,000 correct**, versus **778.66 ms and 803/1,000**
with the adapter off on the same fresh server. This is the selected configuration
for further testing. See the [combination experiment](../lightning-combination/REPORT.md)
for the unchanged prompts, recipe, raw results, and comparison limits.

Measured September 21, 2026. Historical Lightning HTTP experiment. The old diffusion application has since been removed from this branch; comparison measurements below are retained as historical evidence. See the newer [trained native scorer benchmark](../lightning-nimble/HUNDRED_QUESTIONS.md).

## Result

The final full endpoint sweep returned 100 independent answers in **839.53 ms median**, compared with **5,200.16 ms** for the original Lightning configuration: an observed **6.19× latency improvement**. All ten requests completed. Correctness was **798/1,000**, versus **820/1,000** originally. This is a substantial speed improvement, **not an accuracy improvement**.

The main change was explicitly selecting a **BF16 attention KV cache** instead of the checkpoint-selected FP8 cache. In this tested vLLM build, that reduced the hybrid attention/Mamba logical cache block from **2,128 to 1,072 tokens**. The original shared prompt could then reuse a completed prefix within the same 100-question request. No question was answered by deterministic lookup, no state was discarded, and no model layers were removed.

The final configuration also uses persistent upstream HTTP connections. It does not require artificial prompt padding, a second-pass cascade, a custom engine patch, or manually serializing the first question.

![100-question comparison](hundred-questions.svg)

## Full endpoint curve

![Latency versus question count](latency-vs-questions.svg)

| Questions | Original Lightning ms | Optimized Lightning ms | DiffusionGemma ms | Original correct | Optimized correct | DiffusionGemma correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 155.56 | 189.91 | 483.83 | 10/10 | 10/10 | 10/10 |
| 2 | 223.83 | 230.32 | 465.44 | 20/20 | 20/20 | 20/20 |
| 4 | 265.20 | 250.56 | 468.82 | 40/40 | 40/40 | 40/40 |
| 8 | 450.40 | 242.08 | 547.71 | 75/80 | 74/80 | 80/80 |
| 12 | 561.70 | 261.91 | 560.78 | 114/120 | 112/120 | 120/120 |
| 24 | 1,036.85 | 367.80 | 759.27 | 230/240 | 226/240 | 240/240 |
| 50 | 2,287.61 | 490.97 | 1,148.38 | 453/500 | 462/500 | 500/500 |
| 75 | 3,877.90 | 719.01 | 1,679.70 | 647/750 | 635/750 | 750/750 |
| 100 | 5,200.16 | 839.53 | 2,022.90 | 820/1,000 | 798/1,000 | 1,000/1,000 |

Values are medians. At 100 questions, original / optimized / DiffusionGemma p95 was **5,409.04 / 961.04 / 2,366.87 ms**. With ten samples, nearest-rank p95 is simply the maximum and is a coarse tail estimate. Intermediate points are measured, not interpolated estimates; connecting lines are visual guides only.

Optimized Lightning was 2.41× faster than the tested DiffusionGemma deployment at 100 questions, but DiffusionGemma answered all those lookups correctly. This is a latency/quality tradeoff, not evidence that one model family is universally more efficient.

## TypeSafe accuracy reference

The same endpoint fixture was also submitted to TypeSafe using `jev-latest`, which resolved to **`jev-1.13.0`**. TypeSafe answered **2,760/2,760** correctly across all nine question counts. At 100 questions it returned **1,000/1,000** correct, with **336.44 ms median / 414.61 ms p95**. Its hosting hardware is unknown; this is an API comparison, not an equal-GPU comparison. DiffusionGemma's compatibility alias `jev-latest` does not make it the TypeSafe model.

| Questions | TypeSafe median ms | Correct |
| ---: | ---: | ---: |
| 1 | 249.67 | 10/10 |
| 2 | 265.09 | 20/20 |
| 4 | 265.95 | 40/40 |
| 8 | 286.14 | 80/80 |
| 12 | 262.88 | 120/120 |
| 24 | 281.53 | 240/240 |
| 50 | 279.56 | 500/500 |
| 75 | 341.73 | 750/750 |
| 100 | 336.44 | 1,000/1,000 |

At 100 questions, agreement with TypeSafe was **820/1,000 original Lightning**, **798/1,000 optimized Lightning**, and **1,000/1,000 DiffusionGemma**. Here agreement equals known-label accuracy because TypeSafe matched every fixture label; TypeSafe is a comparison reference, not automatically ground truth.

On the separate ten-case randomized holdout, TypeSafe scored **1,000/1,000**, DiffusionGemma **999/1,000**, and simultaneous BF16 Lightning **828/1,000**. Lightning agreed with TypeSafe on **828/1,000** answers and with DiffusionGemma on **827/1,000**. These reference calls use exactly the saved holdout states and questions; they are not additional Lightning trials. See `reference-typesafe.json` and `reference-diffusiongemma.json` for individual responses.

These results do not isolate model strength from inference method. Model weights, training, prompting and output extraction differ. Lightning normal autoregressive generation on the same fixture was not benchmarked; therefore the accuracy gap cannot be attributed solely to diffusion versus one-token scoring.

## Workload and fairness details

- Identical text and choice questions across the endpoint curves. The fixture contains 100 numbered records with explicit alternating on/off values and irrelevant padding. Each question independently asks about one record, with two options. This is synthetic lookup, not a general reasoning benchmark.
- State text is exactly **1,000 tokens under Lightning's tokenizer**; question framing and chat-template tokens are additional. DiffusionGemma receives the same text, not necessarily the same tokenizer count.
- One System One API request contains all questions. Lightning expands it to independent one-token sequences; DiffusionGemma receives the entire multi-question payload as one request.
- Ten measured outer requests per count, plus one excluded warmup. Outer requests are sequential. Each measured fixture varies a prefix reference. Optimized Lightning additionally uses a unique per-request cache salt shared by its question branches, preventing cache reuse from previous outer requests.
- The first question's prefill and all answers are included in latency. Internal cache reuse within that request is intentional, not a discarded precomputation.
- Client-observed measurements use the same client. Lightning is reached through an SSH tunnel; DiffusionGemma through its remote gateway. They differ in hardware, precision, network and serving stack. The existing NVCF deployment was configured for L40S, but actual placement was not re-verified during this test.
- DiffusionGemma's first 100-question warmup took 11.75 seconds. Its ten measured requests then took 1.95–2.37 seconds. It was not reconfigured or restarted during this work.
- The runs were successive, not simultaneous equal-hardware experiments. H100 software thermal throttling was observed. Neither clocks nor cooling controls were changed. Treat the results as deployment measurements, not peak H100 capabilities or energy/cost measurements.

## Why the change helps

The tested server aligns attention cache pages with larger Mamba recurrent-state pages. FP8 attention storage held many more tokens in the same physical page, resulting in a 2,128-token logical block. Our original prompt was only about 1,218 tokens, so it could not reuse a complete block.

BF16 attention storage changed the alignment to 1,072 tokens. The existing scheduler could reuse that shared prefix and evaluate each question-specific suffix. Server counters measured **106,128 cached input tokens** per 100-question invocation: **1,072 × 99**. This directly supports the shared-prefill explanation. Larger dtype storage improved this particular short shared-prefix workload; it does not imply BF16 caches are always faster or smaller.

Lightning is a hybrid attention/Mamba model. Reuse must include both attention KV and recurrent state. Naively expanding writable tensors across branches would alias their state. This experiment uses the engine's existing state management, not custom `tensor.expand()` mutation or masked-token inference. JSONPath flattening was discussed but not implemented; it is not the source of this measured gain.

## Controlled checks and rejected approaches

| Experiment | Measured outcome |
| --- | --- |
| Selected A/B logprobs without grammar | 100 questions: 5,073 ms vs 5,055 ms paired baseline; no meaningful gain. Both still compute one output token. |
| True zero-output scoring | Installed vLLM returned 404 for `/v1/score` and rejected chat `max_tokens=0`. SGLang zero-generation scoring was not deployed or benchmarked. |
| Shorter state-first prompt | Screening about 3.50 s vs 3.52 s; no material speed gain alone. |
| Question-first prompt | 300/300 correct on alternating screening, but about 4.43 s and no shared question-independent prefix. Not broadly validated. |
| Artificial padding + first-answer cache priming | Alternating: 825 ms, 942/1,000 correct vs 4,901 ms, 884/1,000. Randomized: 921 ms, 714/1,000 vs 5,131 ms, 802/1,000. Rejected as default due to randomized regression. |
| Move padding before state | 836 ms, 213/300 correct vs baseline 243/300. Regression remained. |
| Shared catalog of all questions | 2,304 ms, 217/300 correct vs baseline 242/300. Not retained. |
| Padded scoring plus uncertainty fallback (threshold 0.8) | Randomized: 2,777 ms, 860/1,000 vs 5,129 ms, 807/1,000. Alternating: 2,992 ms, 999/1,000 vs 5,203 ms, 876/1,000. Useful quality/speed tradeoff, but slower than final BF16 path. |
| BF16 + explicitly answer first question before remaining 99 | Slower than simultaneous submission: about 756–758 ms vs 647–668 ms on-node. Engine already reused the prefix; explicit staging was unnecessary. |
| BF16 worker-count sweep | Same-node 16/32/64 workers: 1,070/818/776 ms median. Retained 100 workers; separate 100-worker validation was 647–668 ms. These were successive runs with thermal variation. |

A custom 512-token logical-block wrapper was explored but not validated. It was abandoned in favor of the supported BF16 configuration and is not part of the recommended launch.

## Additional validation

With BF16, unchanged prompts and separate cache namespaces, ten on-node trials per dataset produced:

| Dataset | Simultaneous median / p95 | Correct | First-then-rest median | Correct |
| --- | ---: | ---: | ---: | ---: |
| Random flags, shuffled records | 646.86 / 761.37 ms | 828/1,000 | 756.11 ms | 836/1,000 |
| Alternating records | 667.52 / 758.39 ms | 865/1,000 | 757.92 ms | 867/1,000 |

These use new references and, for randomized data, new seeds. Do not equate their correctness directly with the endpoint fixture's different examples. Every state retains all 100 records. Raw answers, expected labels and GPU temperature/clock snapshots are preserved.

The separate 18-case semantic smoke suite passed **18/18 with zero request errors**, covering Choice, Noul, Score, option reversal, 20 options, negation, prompt injection as data, relationships and access policies. Ten adapter unit tests passed. These small checks do not override the 100-question errors.

A shorter persistent-connection 100-question run measured 729.51 ms and 804/1,000 correct; the final full-sweep value is 839.53 ms and 798/1,000. Both are retained rather than selecting only the fastest result. BF16 without persistent upstream connections measured 860.55 ms in its full sweep. Differences between these successive runs also include scheduling and thermal variability.

## Tested configuration and reproduction

- One H100; driver 595.58.03; device reported 97,871 MiB memory.
- vLLM `0.29.1rc1.dev397+ga8d1aa9c9`, image `vllm/vllm-openai:nightly-a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b`.
- Nemotron 3.5 Lightning 30B-A3B NVFP4 checkpoint, TP=1, Marlin MoE and FlashInfer Mamba. H100 uses weight-only FP4 compression rather than native FP4 arithmetic.
- Cached checkpoint has a documented MTP-weight replacement. MTP/speculative decoding is disabled. It is not claimed byte-identical to the public checkpoint.
- Maximum context 16,384; scheduled-token budget 16,384; 128 sequences; GPU memory utilization 0.85; prefix caching and asynchronous scheduling enabled.
- Explicit **`--kv-cache-dtype bfloat16`**; Mamba state FP16 with stochastic rounding and five Philox rounds. Temperature zero does not imply bitwise-repeatable scores under this configuration.
- Adapter: `STATE_FIRST=1`, `UPSTREAM_CONCURRENCY=100`, `PERSISTENT_UPSTREAM=1`, **`PRIME_SHARED_PREFIX=0`**. `ISOLATE_REQUEST_CACHE=1` is used for cold-cross-request benchmarking; it prevents sharing between outer requests, not within them.

Use `launch-gpu.sh` with your existing checkpoint and writable runtime location set through `MODEL_DIR` and `RUN_DIR`. The script requires Docker/NVIDIA GPU support and one visible GPU. It creates a separate container and does not stop any existing deployment. The adapter's persistent mode requires `httpx`; the tested vLLM image supplies it. Set the upstream URL and model through `UPSTREAM_BASE_URL` and `UPSTREAM_MODEL`, then run `python3 adapter.py`. For network exposure, configure the required adapter key and a TLS proxy; the default is loopback-only.

Run `python3 -m unittest discover -p 'test_*.py'` for unit checks. `quality.py` takes a base URL, while `compare_endpoints.py` takes the complete System One URL through `BENCH_URL` and an optional `BENCH_API_KEY`. Set `BENCH_COUNTS=1,2,4,8,12,24,50,75,100` and a unique `BENCH_SUFFIX` to avoid overwriting saved measurements. The bundled fixture makes the endpoint test reproducible without a model download on the client. `make_charts.py` regenerates both figures directly from raw results.

The original Lightning adapter remains available through its default settings. Optional explicit priming is included for reproducibility but is off by default and not selected by the final recipe. No running deployment was changed by these experiments.

## Scope of conclusions

The fastest configuration validated here substantially reduces 100-question latency without new model training or engine modifications. It still makes lookup errors. Normalized candidate scores are not calibrated probabilities of correctness, and this task does not establish production quality, power savings, general reasoning ability, or the absolute theoretical speed limit. Further architecture-level work should compare exact prompt/state semantics and preserve independent writable Mamba states, rather than assuming a mask token or tensor expansion is sufficient.
