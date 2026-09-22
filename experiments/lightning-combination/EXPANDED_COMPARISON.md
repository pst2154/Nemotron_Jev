# Lightning + LoRA versus DiffusionGemma: larger endpoint comparison

Measured September 21, 2026. This extends **Experiment 3**, using the selected
NVFP4 Lightning + rank-16 LoRA configuration without changing either deployment.

**The winner depends on the task.** Lightning + LoRA was faster on these
endpoints and scored higher on the 324-case Nimble semantic set. DiffusionGemma
was more accurate on randomized multi-record lookup, especially in long inputs.
The earlier 1,000/1,000 alternating-record result did not generalize to perfect
accuracy on randomized records.

These are **H100 versus L40S endpoint measurements**, not a hardware-normalized
comparison of autoregressive and diffusion architectures. The fine-tuned
Lightning also has an in-domain advantage on Nimble-style questions.

## What was tested

- **552 matched requests per endpoint, 4,596 scored decisions per endpoint.**
  Repeated workload variations are not 4,596 independent semantic examples.
- 108 randomized lookup requests: 3,450 decisions, 1/10/30/100 questions,
  with 256/1,000/4,000/8,000/12,000 state tokens. The 256-token case uses only
  1 and 10 questions so the evidence fits without truncation.
- All 324 saved Nimble semantic cases, one question each, covering Choice,
  Noul, and Score. These references are provisional, marked not human-reviewed,
  and were already used for evaluation during adapter development.
- The same deterministic 24-case balanced semantic subset at both 4,000 and
  12,000 tokens, eight cases per output type.
- 72 grouped semantic requests: 774 decisions from 24 authored intent,
  sentiment, and evidence cases; 1/6/12/24 questions at 1,000/4,000/12,000 tokens.

Every pair receives identical state, question instructions, and choices.
References never enter the model request. Saved payload hashes verify pairing.
Relevant evidence is placed at the beginning, middle, or end. Long padding is
repeated neutral office-inventory prose: this tests length and evidence
position, not realistic long-document comprehension.

## Semantic accuracy

| Output type | Lightning + LoRA | DiffusionGemma |
| --- | ---: | ---: |
| Choice | 116/146 (79.5%) | 106/146 (72.6%) |
| Noul | 107/114 (93.9%) | **109/114 (95.6%)** |
| Score, highest-probability level | **57/64 (89.1%)** | 29/64 (45.3%) |
| **Total** | **280/324 (86.4%)** | **244/324 (75.3%)** |
| Median single-question HTTP latency | **166 ms** | **252 ms** |

The largest gap is Score classification, not Noul. Score accuracy uses the
most-probable level, not rounding the expected-value `score` field. Returned
Score probability distributions were checked to sum to one.

This is endpoint accuracy on the saved reference labels, not a claim that the
fine-tuned model is universally stronger. Native BF16 scoring, the original
untrained endpoint, and this selected NVFP4+LoRA endpoint are different methods;
their measurements must not be substituted for each other.

## Randomized lookup: 100 questions in one API request

Six trials per cell; median latency includes the full HTTP round trip.

| State tokens | Lightning median | Lightning correct | DiffusionGemma median | DiffusionGemma correct |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | **819 ms** | 585/600 (97.5%) | 1,806 ms | **597/600 (99.5%)** |
| 4,000 | **1,011 ms** | 549/600 (91.5%) | 6,427 ms | **599/600 (99.8%)** |
| 8,000 | **3,531 ms** | 559/600 (93.2%) | 13,947 ms | **595/600 (99.2%)** |
| 12,000 | **3,149 ms** | 540/600 (90.0%) | 22,308 ms* | **494/500 (98.8%)*** |

*At 12,000 tokens/100 questions, one DiffusionGemma request timed out, leaving
100 unanswered decisions. Its accuracy and median above use the five completed
requests. Across all 600 requested decisions, it delivered 494 correct answers
(82.3% end-to-end successful correctness), versus Lightning's 540/600 with no
timeout. The timeout is retained in raw results, not silently retried or
reclassified as a wrong prediction. Its observed client duration was 142.5 s;
the HTTP client had a 120-second per-operation timeout, not a total deadline.

Latency is not strictly monotonic with input length or question count in these
six-sample runs. The measured values are retained as observed; no smoothing or
interpolation is used to assert an unmeasured result.

## One-question lookup latency

Both endpoints answered all six single-question lookup trials correctly at each
length. This is a simple retrieval question, not the semantic workload above.

| State tokens | Lightning + LoRA | DiffusionGemma |
| ---: | ---: | ---: |
| 256 | 165 ms | 268 ms |
| 1,000 | 194 ms | 289 ms |
| 4,000 | 281 ms | 549 ms |
| 8,000 | 388 ms | 1,053 ms |
| 12,000 | 503 ms | 1,605 ms |

## Matched semantic cases in longer inputs

The exact same 24 cases are compared here; this is separate from randomized
lookup. The short baseline is extracted from the 324-case run.

| Input | Lightning correct | DiffusionGemma correct | Lightning median | DiffusionGemma median |
| --- | ---: | ---: | ---: | ---: |
| Original short states | 17/24 | 18/24 | — | — |
| 4,000 tokens | 16/24 | 16/24 | 237 ms | 582 ms |
| 12,000 tokens | 17/24 | 16/24 | 502 ms | 1,624 ms |

This small subset does not establish a broad long-context accuracy advantage
for either model.

## Grouped intent, sentiment, and evidence decisions

These requests ask all 24 authored questions together, repeated six times with
shuffled question order and balanced evidence placement. They are distinct
from the in-domain Nimble set above.

| State tokens | Lightning correct | DiffusionGemma correct | Lightning median | DiffusionGemma median |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 112/144 (77.8%) | **144/144 (100%)** | **334 ms** | 683 ms |
| 4,000 | 112/144 (77.8%) | **141/144 (97.9%)** | **1,404 ms** | 1,880 ms |
| 12,000 | 111/144 (77.1%) | **130/144 (90.3%)** | **1,185 ms** | 5,999 ms |

DiffusionGemma is more accurate here, while Lightning is faster. Together with
the Nimble result, this demonstrates why neither endpoint should be called
universally more accurate based on one benchmark. All intermediate question
counts, per-position lookup accuracy, and p95 values appear in the full tables.

## Serving configurations and interpretation

| Setting | Lightning + LoRA | DiffusionGemma |
| --- | --- | --- |
| GPU | One H100 80GB HBM3 | One L40S, verified from active function metadata |
| Weights | Existing Lightning NVFP4 + rank-16 LoRA | DiffusionGemma 26B-A4B FP8-dynamic |
| Maximum model length | 16,384 | 16,384 |
| Maximum sequences | 128 | 8 |
| Scheduled-token budget | 16,384 | 4,096 |
| Method | Original short prompt per question; constrained one-token scores; state-prefix sharing | Existing structured diffusion gateway, one diffusion sample |

The client, fixtures, and scoring are matched. Hardware, transport, precision,
training, and serving budgets are not. These results support choosing between
the **current deployments**, not a claim about equal-GPU efficiency. Neither
deployment was retuned, restarted, or replaced for this comparison.

Only one outer request per backend was issued at a time. Each matched pair ran
concurrently on its separate server. No caches were flushed; Lightning retains
its existing per-request cache isolation and within-request prefix sharing.
One warmup per endpoint per fixture file is excluded. The latency timer includes
HTTP transport, service work, parsing, and the small client-side scoring step.
With six samples, nearest-rank p95 equals the maximum successful latency.

Token lengths count **state text only with the Lightning tokenizer**, not the
question schema or server prompt. Identical text may tokenize differently in
DiffusionGemma. No input was silently shortened to fit a context window.

## Evidence and reproduction

See [full tables](EXPANDED_TABLES.md) and [machine-readable summary](expanded-summary.json).
The fixtures and raw responses are retained in `expanded-fixtures.json`,
`expanded-results.jsonl`, `grouped-fixtures.json`, and `grouped-results.jsonl`.
`expanded-metadata.json` records runtime settings, provenance, and hashes.
The saved semantic references originate from
[Bespoke Nimble](https://github.com/bespokelabsai/nimble/tree/f136b3f75721fda4ea961f73993cc50b08488835).
Their original reference provenance and the training experiment's evaluation
limitations still apply; no human re-labeling was performed for this comparison.

Use environment variables for your endpoint URLs and credentials. The runner
does not store them in its evidence files:

```bash
export LIGHTNING_URL="$YOUR_LIGHTNING_SYSTEMONE_URL"
export DIFFUSION_URL="$YOUR_DIFFUSION_SYSTEMONE_URL"
export DIFFUSION_API_KEY="$YOUR_DIFFUSION_KEY"
python3 expanded_comparison.py run --fixtures expanded-fixtures.json --output my-expanded-results.jsonl
python3 expanded_comparison.py run --fixtures grouped-fixtures.json --output my-grouped-results.jsonl
```

Use fresh output names for a new measurement; existing rows are treated as a
resume checkpoint. The supplied summarizer verifies matching fixture hashes,
complete paired coverage, and recomputed correctness before producing tables.

The exact trained adapter is also saved in a **private Hugging Face repository**:
[nemotron-3.5-lightning-systemone-lora-r16](https://huggingface.co/einsteiner1983/nemotron-3.5-lightning-systemone-lora-r16).
Its uploaded weights hash matches the serving adapter. Access requires the
repository owner's authorization; base-model weights are not included.
