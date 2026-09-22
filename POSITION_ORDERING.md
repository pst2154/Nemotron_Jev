# Position-only option ordering

Current source enables `POSITION_ORDERING=1`. Set `POSITION_ORDERING=0` to restore
the original prompt order. Health and scoring metadata report
`length_overlap_rotate2_v1` or `off`.

This is an application-layer prompt-order change, not training, a vLLM kernel
change, extra generation, or permutation averaging. It affects Choice, Noul,
and Score while preserving their original label/code mapping and score levels.

For each description, rank by normalized tokenizer length minus twice its
content-word overlap fraction with state plus instructions. Sort descending,
then rotate left by two positions. Move each entire option, including its
original code. Output probabilities retain original label order. Structured
descriptions are serialized as JSON for ranking; ties are stable before rotation.

## Evidence and scope

On the exposed public JevBench subsets, position-only improved 23 to 25 of 32
development answers and 46 to 51 of 64 separate check answers. The diagnostic
repeated with identical probabilities. The check had five fixes and no
regressions. This is not a fresh holdout or a full-suite result for this variant.
The earlier 174/231 result used a different variant that also reassigned codes;
do not attribute that score to position-only.

Production prompt tokens match the tested position-only implementation exactly
on all 231 public questions. CPU tests cover stable sorting, code/label/meaning
preservation, structured input, original-order fallback, batched tokenization,
and Choice/Noul/Score decoding. No extra model inference is performed.

## Build without recompiling vLLM

The existing published image and its immutable JevBench submission remain
unchanged. To package the updated application locally, run from this checkout:

```bash
docker build \
  --build-arg VLLM_IMAGE=ghcr.io/pst2154/nemotron-jev@sha256:b8afc221e1ef9c8e74847e1a3d304a77123ac5114f1ef9f102048866fb286aa9 \
  -t nemotron-jev:position-ordering .
```

Use this local image with the deployment command in the README. It retains the
same UI, API, model-download entrypoint, and checkpoint version. Add
`-e POSITION_ORDERING=0` for original-order comparisons. Do not use results from
the modified image to describe the frozen benchmark container.
