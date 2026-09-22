# Experiment 3: fast NVFP4 Lightning + trained LoRA

**Selected configuration for further testing.** Combine Experiment 1's serving
method with Experiment 2's rank-16 adapter, without changing the original prompts.
Direct LoRA loading works; no new FP4 quantization was needed.

| Same-server control | 100-question median | Correct |
| --- | ---: | ---: |
| Adapter off | 778.66 ms | 803/1,000 |
| Adapter on | **837.22 ms** | **1,000/1,000** |

These are ten requests on a synthetic lookup fixture, not general semantic
accuracy. See [the report](REPORT.md) for full configuration and limitations.

- [Launch recipe](launch_combination.sh)
- [Isolated benchmark runner](run_benchmark.py)
- [Base-control results](comparison-nvfp4-base-lora-engine.json)
- [Trained results](comparison-nvfp4-trained-lora.json)
- [Runner validation repeat](verification-runner.json) — 801.73 ms, 1,000/1,000;
  recorded separately from the headline run.

The adapter and fixture remain in [Experiment 1](../lightning-systemone/README.md)
as the single source of truth. The runner checks their recorded hashes and uses
a temporary work area so new combination results do not land in Experiment 1.
The caller selects a fresh output file; existing results are not overwritten.

Model/adapter weights are not included. Set `MODEL_DIR`, `ADAPTER_DIR`, `RUN_DIR`,
and `GPU_UUID` for your environment, then follow the report's reproduction steps.
