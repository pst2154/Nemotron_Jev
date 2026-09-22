# Nemotron Lightning typed decisions

Three separate experiments adapt Nemotron 3.5 Lightning to bounded classification.

| Experiment | Model and execution | Entry point |
| --- | --- | --- |
| 1. Original fast baseline | NVFP4, original short prompts, vLLM, one constrained token/question | [Baseline](experiments/lightning-systemone/README.md) |
| 2. Nimble training / native scoring | BF16 base, trained LoRA and merged checkpoints; native candidate projection and shared-prefix tests | [Native experiment](experiments/lightning-nimble/README.md) |
| 3. Combination — selected | Original NVFP4 + trained LoRA; original prompts and fast vLLM adapter | [Combination](experiments/lightning-combination/README.md) |

**Selected combination experiment for further testing:** original NVFP4 Lightning
plus the trained rank-16 LoRA, original short prompts, and the fast vLLM adapter.
It measured **837 ms median, 1,000/1,000 correct** on the original 100-question
lookup workload. See the [configuration and controlled comparison](experiments/lightning-combination/REPORT.md).
This selected vLLM path emits one constrained answer token per question; it is
distinct from the native zero-generation experiment below.

Each experiment keeps its own report and results. Experiment 3 deliberately
reuses Experiment 1's immutable adapter and input fixture; Experiment 2's
Nimble prompt format is **not** substituted into Experiment 3.

In Experiment 2, the trained adapter scored **273/324 (84.26%)** on the frozen Nimble holdout;
the merged BF16 checkpoint scored **270/324 (83.33%)**. Candidate probabilities
are not calibrated confidence. These are research measurements, not a claim
of equivalence to TypeSafe Jev or Bespoke-Nimble-9B.

This branch no longer contains the earlier diffusion application, its container
recipe, or its application-specific evaluations. They remain recoverable in Git
history and on the original branch. Historical comparative benchmark results are
retained; no running deployment was changed. No model weights or compiled kernel
wheels are distributed here.
