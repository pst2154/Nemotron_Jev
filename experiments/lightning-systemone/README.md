# Experiment 1: original fast Lightning baseline

Untrained NVFP4 Lightning, original short per-question prompts, and vLLM prefix
caching. Historical headline: **839.53 ms, 798/1,000 correct** on the original
100-question lookup fixture. This experiment does not apply the trained LoRA.

- [Report, configuration, reproduction, and limits](REPORT.md)
- [Original launch recipe](launch-gpu.sh)
- [System One adapter](adapter.py)
- [Saved inputs](comparison-inputs.json) and [HTTP benchmark](compare_endpoints.py)
- [Historical chart](hundred-questions.svg)

The original scripts and saved measurements are retained for reproducibility,
including rejected optimization screens and historical reference-model results.
Those reference results are not additional top-level experiments or deployments.
Use a fresh `BENCH_SUFFIX` when rerunning the original benchmark to avoid replacing
historical files.

The trained/native work is [Experiment 2](../lightning-nimble/README.md).
The selected fast setup with the fine-tune added is
[Experiment 3](../lightning-combination/README.md), which owns its new measurements.
