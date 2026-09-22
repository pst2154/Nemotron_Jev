# H100 baseline measurements

These are measured baselines, not results for the new vLLM diffusion implementation.
The completed optimized diffusion comparison is in the
[optimization report](../../OPTIMIZATION_REPORT.md).

All three runs used the same H100 80 GB, identical request payloads, 72 frozen
regression cases, and a synthetic item/color retrieval fixture. Each run completed
135 requests. The latency fixture uses exactly 1,000, 4,000, or 12,000 state tokens
under both tokenizers and question counts 1, 4, 8, 16, 32, 64, and 100, with three
repetitions. Repeated and overlapping questions are not independent accuracy cases.

| Backend | Frozen regression | 100 questions, 1k state | 100 questions, 4k state | 100 questions, 12k state |
| --- | --- | --- | --- | --- |
| Native diffusion BF16 | 72/72 | 8,405.8 ms | 25,355.4 ms | 75,773.7 ms |
| Lightning NVFP4, base | 70/72 | 711.2 ms | 2,468.1 ms | 2,188.8 ms |
| Lightning NVFP4 + trained LoRA | 72/72 | 825.5 ms | 2,788.8 ms | 2,217.7 ms |

Times are medians of three requests. Correct answers out of 100 in each repetition:

| Backend | 1k state | 4k state | 12k state |
| --- | --- | --- | --- |
| Native diffusion BF16 | 89, 89, 89 | 83, 83, 83 | 33, 33, 33 |
| Lightning NVFP4, base | 98, 97, 95 | 90, 90, 89 | 94, 94, 95 |
| Lightning NVFP4 + trained LoRA | 99, 99, 99 | 99, 99, 100 | 100, 100, 100 |

Native diffusion runs questions sequentially through the original masked scorer;
it does not generate a text answer. Lightning uses batched one-token classification
with within-request prefix reuse and isolated request caches. Its timing includes
HTTP and gateway overhead; native diffusion timing is in-process. Lightning uses
NVFP4 weights, whereas diffusion uses BF16, so this is a serving-configuration
comparison, not an isolated comparison of model architectures. Lightning's serving
configuration enables stochastic rounding; repeated answers can differ.

The adapter improves accuracy on these fixtures. The non-monotonic Lightning
latencies at 4k versus 12k are measured values, not an interpolation or a claim
that longer inputs generally cost less. The optimization report includes the
matched-cache-policy diffusion HTTP comparison and final recommendation.

Raw measurements: `native-full.jsonl`, `lightning-base-http.jsonl`, and
`lightning-http.jsonl`. Payload hashes in those files permit exact request matching.
The original native file labels repetitions after the first as `warm`; that means
only a repeated request, not cache reuse. Native prefix caching is disabled for
every repetition. The benchmark now explicitly labels such runs `disabled`;
original measurement files are retained unchanged.
