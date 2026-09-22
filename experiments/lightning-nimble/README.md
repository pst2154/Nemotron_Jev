# Experiment 2: Nimble fine-tuning and native candidate scoring

This experiment owns the BF16 training, holdout evaluation, merged checkpoint
tests, and native SemIf-style shared-prefix measurements. It is not the selected
NVFP4/vLLM combination; that is [Experiment 3](../lightning-combination/README.md).
The original fast untrained serving baseline is [Experiment 1](../lightning-systemone/README.md).

See [100-question measurements](HUNDRED_QUESTIONS.md) and the detailed
[training and SemIf report](SEMIF_LIGHTNING.md). This is a standalone research
runner, not a deployed HTTP service. No production endpoint was changed.

## Reproduce

Use an H100 80GB and the exact runtime in `kernel-wheels.json`: Python 3.12,
PyTorch 2.13.0+cu130, Transformers 5.17.0, PEFT 0.21.0. The optional
Mamba and causal-convolution kernels must be built against that runtime;
the recorded wheel hashes identify our tested artifacts, not public downloads.
Do not replace the installed Torch version with Nimble's generic requirements.

The scripts import the upstream Nimble prompt/data modules. Obtain that source
at the tested revision, then run with it on the Python import path:

```bash
git clone https://github.com/bespokelabsai/nimble.git
git -C nimble checkout f136b3f75721fda4ea961f73993cc50b08488835
export PYTHONPATH="$PWD/nimble${PYTHONPATH:+:$PYTHONPATH}"
```

Run the following from this experiment's directory, setting the checkpoint,
adapter, and output variables to your own locations. Model weights, adapters,
and compiled wheels are not included in this repository. The measured base
was an existing local Lightning BF16 checkpoint, not verified byte-identical
to a public release.

```bash
python3 test_shared_native.py --device cuda
python3 benchmark_100_native.py \
  --model "$BASE_CHECKPOINT" --adapter "$TRAINED_ADAPTER" \
  --label trained-lora --output "$ADAPTER_RESULTS"
python3 benchmark_100_native.py \
  --model "$MERGED_CHECKPOINT" \
  --label merged --output "$MERGED_RESULTS"
```

Output paths must not already exist. Each run verifies exactly 1,000 state
tokens, all 100 distinct question IDs, finite scores, and complete answers.
It tests 1/8/20/30/50/100 questions, then asks the same 100 individually.
Question definitions and chat framing add tokens beyond the state.

For training or frozen-holdout evaluation, run from the pinned Nimble checkout
so its audited data files resolve. Invoke the supplied `lightning_train.py`
with `--model "$BASE_CHECKPOINT" --output "$TRAINING_RUN" --full-train
--accumulation 8`. The report records the actual one-epoch configuration and
data fingerprints. `merge_lightning.py` takes `--run "$TRAINING_RUN"` and
`--output "$MERGED_CHECKPOINT"`; it creates a separate BF16 checkpoint.
Merging changed some predictions, so evaluate the merged artifact separately.

The 324-item holdout and 100-record lookup test are different workloads.
The lookup percentages must not be presented as Nimble semantic accuracy.
