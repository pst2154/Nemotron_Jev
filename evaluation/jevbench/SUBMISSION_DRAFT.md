# Draft: bench request — Nemotron Diffusion Decision Lab (14B, vLLM)

**Not submitted. Public self-test is pending.**

Please consider a maintainer-run evaluation of Nemotron-Labs-Diffusion-14B using
our classification-only vLLM serving implementation.

- [Serving source and deployment guide](https://github.com/pst2154/Nemotron_Jev/tree/perf/nemotron-diffusion-optimized)
- [vLLM implementation](https://github.com/pst2154/vllm/tree/feat/nemotron-labs-diffusion)
- Image: `ghcr.io/pst2154/nemotron-jev@sha256:b8afc221e1ef9c8e74847e1a3d304a77123ac5114f1ef9f102048866fb286aa9`
- [Model card and weight license](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B)

This is an inference implementation over the unchanged dense 14B NVIDIA
checkpoint, not DiffusionGemma, not TypeSafe Jev, and not a separately trained
model. BF16 weights, no additional fine-tuning or calibration. Questions end in
one actual mask token; candidate logits are normalized into native probability
distributions. No autoregressive answer rollout or generated probability text.

The server implements `/v1/systemone` Choice, Noul, and Score and works with the
existing `typesafe` adapter. The image also starts the explorer UI. It was
validated on H100 80 GB and targets Hopper; Blackwell/L40S compatibility is not
claimed for this image. Limits: 16,384 prompt tokens, 62 options per question.
The server has no built-in authentication; bind locally or place it behind an
authenticated proxy. No submitter-operated endpoint is needed for your run.

Our previous regression and latency experiments are not JevBench scores.
Public JevBench self-test results: **pending**. No hosted tariff, official score,
rank, or held-out performance is claimed. Original application code is **MIT**;
the vLLM fork is **Apache-2.0**; model weights are under the **NVIDIA Nemotron
Open Model License**. These licenses do not replace one another.

For reproduction, use the pinned image and instructions in
`evaluation/jevbench/README.md`, keeping request caches isolated and requests
serial. Please run any private items on maintainer-controlled hardware. We
understand a ranked entry requires your own evaluation.
