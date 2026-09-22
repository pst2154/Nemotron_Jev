# Nemotron Lightning typed decisions

Experiments adapting Nemotron 3.5 Lightning to bounded classification: candidate-token
scoring, Nimble-style LoRA training, and SemIf-style shared-prefix inference.
The native scorer returns decisions without autoregressively generating answer text.

- [100-question trained-model benchmark](experiments/lightning-nimble/HUNDRED_QUESTIONS.md)
- [Training, holdout accuracy, and SemIf workloads](experiments/lightning-nimble/SEMIF_LIGHTNING.md)
- [Reproduction instructions](experiments/lightning-nimble/README.md)
- [Earlier HTTP serving experiments](experiments/lightning-systemone/REPORT.md)

The trained adapter scored **273/324 (84.26%)** on the frozen Nimble holdout;
the merged BF16 checkpoint scored **270/324 (83.33%)**. Candidate probabilities
are not calibrated confidence. These are research measurements, not a claim
of equivalence to TypeSafe Jev or Bespoke-Nimble-9B.

This branch no longer contains the earlier diffusion application, its container
recipe, or its application-specific evaluations. They remain recoverable in Git
history and on the original branch. Historical comparative benchmark results are
retained; no running deployment was changed. No model weights or compiled kernel
wheels are distributed here.
