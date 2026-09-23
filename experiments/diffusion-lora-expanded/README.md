# Expanded diffusion LoRA training

The larger run completed: **32,768 training examples, two epochs, and 65,536 example passes** on the same Nemotron-Labs-Diffusion-14B base model. It did not outperform the earlier 4,096-example adapter. This is a larger training experiment, not a larger model.

## Public JevBench results

All runs below contain 231 questions. Counts are recomputable from the accompanying per-question results; every response passed strict format validation without probability renormalization.

| Model / checkpoint | Correct | Accuracy | Evidence |
| --- | ---: | ---: | --- |
| Untrained base | 173/231 | 74.89% | [Results](results/base-jevbench.jsonl) |
| Earlier 4,096-example LoRA | 183/231 | 79.22% | [Results](results/previous-jevbench.jsonl) |
| Expanded run: original loss-selected checkpoint, step 1,536 | 175/231 | 75.76% | [Results](results/selected-jevbench.jsonl) |
| Same checkpoint, repeated evaluation | 175/231 | 75.76% | [Results](results/selected_repeat.jsonl) |
| Expanded run: validation-accuracy-selected checkpoint, step 3,328 | 181/231 | 78.35% | [Results](results/validation_accuracy_winner.jsonl) |
| Expanded run: final checkpoint, step 4,096 | 179/231 | 77.49% | [Results](results/final_two_epochs.jsonl) |

These are accuracy measurements of training checkpoints, not measurements of the separately optimized serving container. They do not establish the accuracy of a container or change the existing leaderboard submission.

## What changed

The earlier trainer selected the highest validation accuracy, breaking ties with cross-entropy. The expanded trainer instead selected the lowest validation cross-entropy. That change selected step 1,536 (504/580 validation answers correct; cross-entropy 0.3429).

Post-run diagnosis restored the earlier accuracy-first selection criterion. It selected step 3,328 (517/580 validation answers correct; cross-entropy 0.3868), which then scored 181/231 on public JevBench. This recovers six of the eight answers lost relative to the small adapter, but remains two answers below it. The checkpoint was identified from validation results, not by maximizing the public benchmark score. Its public evaluation is nevertheless a post-run diagnostic, not a new blind experiment. See the [diagnostic plan](results/plan.json).

This was therefore not a controlled comparison of dataset size alone. Checkpoint selection, data mixture, training budget, and other training details changed. The small remaining difference on 231 partly grouped questions is insufficient to establish that more data inherently makes the model worse.

## Training recipe

| Setting | Value |
| --- | --- |
| Base model | Nemotron-Labs-Diffusion-14B, dense |
| Training examples | 32,768: original 4,096 plus 28,672 new examples |
| Validation examples | 580, unchanged |
| Epochs / optimizer updates | 2 / 4,096 |
| Parallelism | Eight data-parallel replicas |
| Global batch size | 16 |
| LoRA | Rank 16, alpha 32 |
| Target modules | Attention Q/K/V/O and MLP gate/up/down projections |
| Frozen parameters | Base weights and output head |
| Objective | Candidate-only cross-entropy at one masked answer token |
| Learning rate | 2e-5, 5% warmup, cosine decay |
| Optimizer-loop time | 2,758 seconds, approximately 46 minutes |
| Peak allocated memory | 57.18 GiB per GPU |

The timing excludes model loading, validation, and checkpoint writes; it is not total experiment wall time. Training used differentiable causal prefill followed by a masked-answer pass, retaining the prefix cache in the autograd graph. Half the presentations used canonical serving order; half shuffled options and reassigned answer codes with corresponding target remapping.

New examples were generated with SOL and independently answered by GPT-5.4-mini without exposing the proposed labels or evidence. Matching, unambiguous answers with evidence were retained. This agreement filter reduces obvious label errors but does not prove correctness. The SargeDev distilled corpus was not used. Public JevBench was excluded from generation, training, and checkpoint selection; overlap filtering does not establish freedom from base-model pretraining contamination.

## Interpretation

The data mixture shifted substantially: new synthetic examples account for 87.5% of the expanded set. The same 1,420 public-labelled training examples went from 34.7% to 4.3% of the mixture. Median state length increased from 973 characters in the original data to 3,147 in the added data, versus 943 in validation. The new examples also came from 2,179 source groups, so they are not 28,672 independent scenarios.

Distribution mismatch is a plausible contributor, not a demonstrated sole cause. The original loss-selected checkpoint also made 18 wrong public predictions above 90% confidence, compared with four for the earlier adapter. That observation applies to step 1,536 and should not be attributed automatically to step 3,328. The final checkpoint scored below step 3,328, so this run does not support simply adding more epochs.

Both epochs covered all 32,768 examples; the completion receipt records 65,536 example passes and exact adapter reload verification. See [summary](results/summary.json), [completion receipt](results/complete.json), and coverage receipts for [epoch one](results/epoch-1-coverage.json) and [epoch two](results/epoch-2-coverage.json).

**Conclusion:** expanded training succeeded technically, but the earlier small adapter remains ahead in this evaluation. A useful next experiment would hold checkpoint selection and training budget constant while testing balanced data mixtures and multiple seeds. No checkpoint is promoted or deployed by this report.

The [small experiment](https://github.com/pst2154/Nemotron_Jev/tree/experiment/diffusion-lora-small/experiments/diffusion-lora-small) remains separate.
