# Small diffusion decision LoRA experiment

A 4,096-example adaptation of **Nemotron-Labs-Diffusion-14B** for single-token
typed decisions, inspired by AutoJev's classification-training recipe.

- [Measured results](results/REPORT.md): public JevBench, validation breakdown,
  training time, and paired warm latency.
- [Training recipe](RECIPE.md): data sources, augmentation, configuration, and commands.
- [Run configuration](recipe-small.json) and [audit](results/audit.json).
- [Validation checkpoint selection](results/validation-selection.jsonl).

This experiment does not change the production server, container, benchmark
submission configuration, or original base weights. Inference measurements use
native Transformers/PEFT, not the optimized vLLM service. The adapter and full
training corpus are not distributed in this branch; numeric evaluation records,
source revisions, corpus hashes, and the experiment scripts are included.

## Verification

```bash
python -m unittest discover -s experiments/diffusion-lora-small -p test_summary.py -v
python experiments/diffusion-lora-small/summarize_small.py \
  --run experiments/diffusion-lora-small/results
```

The report generator rejects incomplete training, a non-winning selected
checkpoint, missing/invalid public benchmark results, and incomplete latency pairs.

## Attribution

The recipe borrows choice permutation, classification loss, learning-rate
scheduling, and validation-based selection ideas from
[AutoJev](https://github.com/denis-pplx/autojev/tree/ee63c1515980491a742f0bd0685c8dc5ca1f00c3).
It uses LoRA rather than AutoJev's full-weight training and is not a reproduction
of its model or curated dataset. The NVIDIA model, public datasets, and
[JevBench](https://github.com/fstandhartinger/jevbench/tree/51a8d73fa798aa337bb1b26abd10995c0ab847e9)
retain their existing licenses.
