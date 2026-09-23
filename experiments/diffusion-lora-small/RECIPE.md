# Small diffusion decision-training recipe

An AutoJev-inspired adaptation of Nemotron-Labs-Diffusion-14B, not a reproduction
of AutoJev's full-weight training run. Existing services and base weights stay unchanged.

## Data

Target 4,096 training rows: the existing 2,676 synthetic decisions plus 710 BoolQ
and 710 PAWS examples from pinned public training splits. Add 128 validation
examples from each public source to the existing 324 synthetic validation rows.
The exact accepted counts and source revisions are recorded by the data builder.
Labels from public datasets are used directly; no API key is needed.

The built corpus contains 1,566 Choice, 1,598 Noul, and 932 Score training rows,
covering 1,604 source groups. Validation covers 283 source groups. Several
synthetic questions share an underlying source, so row count is not a count of
independent source documents. The synthetic references are model-checked, not
human annotations.

Exclude exact state overlap with JevBench and cross-split state/source overlap.
Group PAWS pairs independent of sentence order and BoolQ records by passage so
the same source text cannot enter both splits. These checks do not establish that
the model's original pretraining excluded these public datasets.

Keep naturally varied input lengths; never silently truncate. Audit token lengths
and stop if a prompt exceeds 4,096 tokens. This is not a long-context training claim.
In the built corpus, complete training prompts range from 152 to 1,045 tokens
(median 472); the 4,096-token setting is a safety limit, not the observed input length.

## Training

- Fresh base checkpoint; do not continue from the earlier pilot adapter.
- Rank-16, alpha-32 LoRA on attention Q/K/V/O and MLP gate/up/down projections.
- Frozen base and output head; broader LoRA is the affordable first experiment.
- One pass over the training set, effective batch 16, about 256 updates.
- Candidate-only cross-entropy at one masked answer position; no generated rationale.
- Learning rate 2e-5, 5% warmup, cosine decay to 10%, weight decay 0.01, clipping 1.
- Half the training presentations retain the exact canonical serving format.
  Half shuffle the rendered options, reassign positional codes, and remap targets.
  Original label meanings and score-level semantics must not change.
- Evaluate every 64 updates and at completion. Select using validation accuracy,
  with validation cross-entropy as the tie-breaker. Keep the baseline as a candidate.
- Save adapters separately. JevBench is final evaluation only, never checkpoint selection.

Training uses the serving-style two-pass graph: causal prefix with differentiable
KV cache, then one masked answer. It does not detach the prefix. Activation
checkpointing is disabled for this small-context run to preserve cache gradients.
The original one-pass shortcut failed post-training argmax parity on a near-tied
smoke example and is not used in this recipe. Smoke-test gradients, save/reload,
target mapping, and finite loss before the full run. Measure warm paired latency
separately from accuracy evaluation.

## Run

Use the same trusted base checkpoint, native model dependencies, scorer, and
JevBench harness as the pilot. Supply their locations using the variables below.
Every output directory must be new.

Tested: Python 3.12, the packages pinned in `requirements-small.txt`, BF16, and
one H100 PCIe 80 GB. Base model revision:
`f8c3e2c078e193599b8882d965b1001c456ba738`. JevBench harness revision:
`51a8d73fa798aa337bb1b26abd10995c0ab847e9`.
Use a CUDA-enabled PyTorch installation compatible with the GPU driver.
The experiment uses native Transformers/PEFT; it does not enable LoRA in the
production vLLM server.

The existing synthetic pilot corpus is an explicit input to the data builder,
not bundled with these scripts. The public additions are fetched from the pinned
dataset revisions. Corpus checksums are supplied with the result artifacts.

```bash
python build_small_data.py --existing "$PILOT_DATA" --output "$DATA" \
  --recipe recipe-small.json --cache "$DATA_CACHE" --harness "$HARNESS"

python train.py --checkpoint "$CHECKPOINT" --app "$SCORER" \
  --harness "$HARNESS" --data "$DATA" --output "$AUDIT_OUTPUT" \
  --recipe recipe-small.json --audit-only

python train.py --checkpoint "$CHECKPOINT" --app "$SCORER" \
  --harness "$HARNESS" --data "$DATA" --output "$SMOKE_OUTPUT" \
  --recipe recipe-small.json --smoke --steps 2

python train.py --checkpoint "$CHECKPOINT" --app "$SCORER" \
  --harness "$HARNESS" --data "$DATA" --output "$TRAIN_OUTPUT" \
  --recipe recipe-small.json
```

`best-adapter` is the validation-selected adapter; `adapter` is the final-step
adapter. `selected.json` records the selected step. A selected step of zero means
the baseline won. Final accuracy evaluation loads `best-adapter`, not necessarily
the last checkpoint. No adapter is automatically deployed.

## Interpretation

This bundles several changes and is not an attribution study. Report dataset counts,
token-length distribution, selected step, synthetic/public validation separately,
JevBench accuracy, and latency. A validation improvement is not automatically a
general improvement or authorization to update a production endpoint.
