# JevBench public self-test — Nemotron Diffusion Decision Lab

Measured September 22, 2026. **Public self-test only, not an official leaderboard
score or rank.** The maintainers' non-public items were not evaluated.

## Results

Both passes attempted all 231 public items, with zero HTTP failures, zero schema
failures, and no probability renormalization by the harness.

| Public tier | Correct, pass 1 | Correct, pass 2 | Accuracy |
| --- | ---: | ---: | ---: |
| Easy | 48/48 | 48/48 | 100.00% |
| Original / standard | 55/72 | 55/72 | 76.39% |
| Hard | 58/111 | 58/111 | 52.25% |
| Overall | 161/231 | 161/231 | 69.70% |

| Serial HTTP latency | Pass 1 (primary) | Pass 2 (repeatability) |
| --- | ---: | ---: |
| First item | 134.29 ms | 46.76 ms |
| Median, all 231 items | 21.45 ms | 21.22 ms |
| p95, all 231 items | 122.69 ms | 119.98 ms |

The returned probability distributions were identical for every item across
the two passes. Overall multiclass Brier was **0.433044** and top-label 10-bin
ECE was **0.136736** in both. These are whole-public-set metrics, not the
leaderboard's hard-only calibration axis. Cost is **unknown/null**, not free.

The second pass did not materially improve overall latency. We retain pass 1
as primary and publish both; no best-run selection or averaging of answers.
This model is fast on these inputs, but its hard-tier accuracy is limited.
These results do not establish a rank against full-suite leaderboard entries.

## Protocol and artifacts

- Harness: `fstandhartinger/jevbench`, revision
  `51a8d73fa798aa337bb1b26abd10995c0ab847e9`; unchanged `typesafe` adapter/scorer.
- Tasks, in order: `easy.jsonl`, `original.jsonl`, `hard.jsonl` from the pinned
  public datasets (48 + 72 + 111). Combined upstream dataset hash:
  `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
- One H100 80 GB HBM3, one BF16 server, no other running GPU workload observed.
- Python 3.12.3 and pytest 9.1.1 for the harness; all **90 upstream tests passed**.
- Frozen serving image:
  `ghcr.io/pst2154/nemotron-jev@sha256:b8afc221e1ef9c8e74847e1a3d304a77123ac5114f1ef9f102048866fb286aa9`.
- Unchanged checkpoint and runtime configuration listed in the
  [reproduction guide](README.md#frozen-candidate).
- The model was loaded and vLLM startup compilation completed before pass 1.
  This is not container cold-start latency. The first actual decision is retained.
- `ISOLATE_REQUEST_CACHE=1` throughout both passes. Common prompts are not reused
  across requests or passes. Pass 2 uses the same already-running process.
- One HTTP request per decision, serial, with 0.2-second inter-request pacing.
  The harness's request latency excludes that intentional delay.
- Client and server run on the same node over loopback. These are raw local HTTP
  timings, not Internet timings, not fixed-1k-token inputs, and not the
  leaderboard's adjusted latency or Speed score.
- No retries, cropping, label-order changes, prompt tuning, fine-tuning, or
  calibration fitting. All outcomes are retained. Initial test preflight lacked
  pytest; it stopped before creating a model run. Installing the test dependency
  resolved it without changing inference or benchmark code.

Evidence:

- [Pass 1 per-item results](results/pass1.jsonl) and [upstream summary](results/pass1-summary.json).
- [Pass 2 per-item results](results/pass2.jsonl) and [upstream summary](results/pass2-summary.json).

Only public-item IDs, predictions, distributions, timing and aggregate evidence
are published. Original request/response receipts and endpoint-bearing manifests
are retained privately. There is no official composite score, hosted tariff, or
claim about the 303 items absent from this public checkout.
