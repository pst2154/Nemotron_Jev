# Small diffusion decision-training result

**Public JevBench accuracy: 173/231 → 183/231.** The adapter was selected using a separate validation set.
See [the recipe](../RECIPE.md) for the training and data configuration.

Nemotron-Labs-Diffusion-14B; rank-16 attention/MLP LoRA; frozen output head.
One training pass; mixed public-labeled and synthetic examples; shuffled positional codes.
Training and validation use the serving-style two-pass graph. No service was updated.

Selected update: **192** (validation only; zero means baseline).
Validation was used to select the checkpoint, so its score is not an untouched test estimate.
Public JevBench was evaluated only before and after training.

| Model | Split | Correct | Accuracy |
| --- | --- | --- | --- |
| Base | validation/all | 382/580 | 65.86% |
| Base | validation/BoolQ | 108/128 | 84.38% |
| Base | validation/PAWS | 64/128 | 50.00% |
| Base | validation/synthetic | 210/324 | 64.81% |
| Selected LoRA | validation/all | 488/580 | 84.14% |
| Selected LoRA | validation/BoolQ | 111/128 | 86.72% |
| Selected LoRA | validation/PAWS | 107/128 | 83.59% |
| Selected LoRA | validation/synthetic | 270/324 | 83.33% |
| Base | public JevBench/all | 173/231 | 74.89% |
| Selected LoRA | public JevBench/all | 183/231 | 79.22% |

Public JevBench changes: 19 wrong→correct; 9 correct→wrong.
These are public-set accuracy counts, not an official leaderboard composite score.

Optimizer-loop time: 42.0 minutes; loading/evaluation excluded.
Peak allocated GPU memory: 39.79 GiB.
Prompt-token range: 152–1045.
Warm paired median latency: base 89.9 ms; unmerged adapter 172.5 ms.

Latency is native model-side timing on one H100 PCIe 80 GB, not HTTP latency or vLLM throughput.
The paired test uses 18 questions, two warmups per mode/input, and three alternating-order measurements.
The existing vLLM container has not been validated with this adapter.

This bundled experiment does not isolate which recipe change caused an accuracy difference.
Exact source-group/state checks do not rule out original pretraining exposure to public datasets.
