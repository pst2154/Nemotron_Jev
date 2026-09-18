# Nemotron structured-decision evaluation

Date: 2026-09-18. Model: NVIDIA Nemotron-Labs-Diffusion-14B, revision `f8c3e2c078e193599b8882d965b1001c456ba738`, BF16, one H100 80 GB. Scoring uses masked-token logits normalized over candidate codes; no generated answer parsing or model-based judge.

## Frozen benchmark

The unchanged set contains 24 base cases, each with short, irrelevant-context, and prompt-injection variants: 72 cases total. Tasks are support classification, a two-condition eligibility rule, and exact affected-region counting. Choice uses argmax, Noul uses a 0.5 threshold, and Score accuracy uses the modal level—not a rounded weighted score.

| System | Correct | Choice | Boolean/Noul | Score | Request errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nemotron adapter, new run | 72/72 (100%) | 24/24 | 24/24 | 24/24 | 0 |
| DiffusionGemma, historical run | 71/72 (98.6%) | 24/24 | 24/24 | 23/24 | 0 |
| NanoJev, historical run | 55/72 (76.4%) | 18/24 | 18/24 | 19/24 | 0 |

Nemotron passed 24/24 in each input variant. Boolean Brier score: 0.0035102. Mean absolute weighted-score error: 0.0118403. Client-observed median latency: 328.45 ms per single-question request.

The packaged container was independently rerun on the same 72 cases: **72/72**, zero errors, Boolean Brier 0.0035102, weighted-score MAE 0.0118480. This first run with fresh compiler caches had a 1,177.25 ms median; it is not a warmed-throughput measurement. The tiny score-MAE change did not change any predicted label. See `evaluation/container-results/` for the actual deployed-image results.

Historical results are supplied in `evaluation/historical-baseline.json`; they were not rerun concurrently. Hardware occupancy, serving paths, and timing differ, so latency is not a controlled speed comparison. The historical medians were 246.69 ms for DiffusionGemma and 289.18 ms for NanoJev.

## Reproduction and limitations

Run the evaluator documented in the README. All cases and per-case probabilities, predictions, errors, and timings are included. The exact published case-file SHA-256 is `3f4edf51dd9a71dc12fb1172af89cc1a7ce53ae05a83ba34fc138a64d99ff36d`. The historical summary hashes its JSON serialization without a trailing newline; compare parsed cases when checking identity.

These are simple, correlated synthetic cases, already used for earlier model comparisons—not a blinded general-capability benchmark. Passing all 72 does not establish superiority, calibrated confidence, robust handling of unknown evidence, or adversarial resistance beyond these particular injections. The adapter's initial development smoke test misclassified a failed payout when category definitions overlapped; making the billing policy explicit changed the result. That development case was not added to or removed from this frozen evaluation.

The original explorer's six samples returned valid distributions and rendered answer cards in a scripted DOM harness. That is a functional check, not an accuracy judgment on every sample. Large-question requests and container validation are separate from this 72-case accuracy result.
