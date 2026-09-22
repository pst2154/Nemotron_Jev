# Bench request — Nemotron Diffusion Decision Lab (14B, vLLM)

Submitted by Codex on behalf of pst2154.

Please consider a maintainer-run evaluation of Nemotron-Labs-Diffusion-14B using
our classification-only vLLM serving implementation.

- [Frozen serving source](https://github.com/pst2154/Nemotron_Jev/tree/66b19e885e2f330711f69983ab9f5d182630ca26)
- [vLLM implementation](https://github.com/pst2154/vllm/tree/2c83d10caa71e3a9eac8fdd9f1288dac8c65596b)
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
The [public JevBench self-test and evidence](https://github.com/pst2154/Nemotron_Jev/blob/45da9ca376462f82f54d5b72036270bdfc70b020/evaluation/jevbench/PUBLIC_RESULTS.md) completed two unchanged passes:
**161/231 (69.70%)** both times: easy 48/48, standard 55/72, hard 58/111.
All responses passed strict schema checks. Pass 1 median/p95 were 21.45/122.69 ms;
pass 2 was 21.22/119.98 ms. These are serial same-node HTTP measurements on one
H100, not Internet or leaderboard-adjusted timings. Model startup completed before
testing, request caches were isolated, and all probability outputs were identical
across passes. Overall public-set Brier was 0.433044 and ECE 0.136736.
No hosted tariff, official score,
rank, or held-out performance is claimed. Original application code is **MIT**;
the vLLM fork is **Apache-2.0**; model weights are under the **NVIDIA Nemotron
Open Model License**. These licenses do not replace one another.

For reproduction, use the pinned image and [evaluation instructions](https://github.com/pst2154/Nemotron_Jev/blob/45da9ca376462f82f54d5b72036270bdfc70b020/evaluation/jevbench/README.md), keeping request caches isolated and requests
serial. Please run any private items on maintainer-controlled hardware. We
understand a ranked entry requires your own evaluation.

## Start the frozen candidate

Requires Docker, NVIDIA Container Toolkit, a CUDA 13-compatible driver and one
H100 80 GB. The image is publicly pullable without registry credentials. It
downloads the public checkpoint on first start, pinned to
`f8c3e2c078e193599b8882d965b1001c456ba738`; the volume preserves the download.

```bash
docker volume create nemotron-models
docker run -d --name nemotron-jev --gpus all --shm-size=8g \
  -p 127.0.0.1:8770:8770 -v nemotron-models:/models \
  -e ISOLATE_REQUEST_CACHE=1 \
  ghcr.io/pst2154/nemotron-jev@sha256:b8afc221e1ef9c8e74847e1a3d304a77123ac5114f1ef9f102048866fb286aa9
docker logs -f nemotron-jev
# Once startup completes, in another terminal:
curl --fail http://127.0.0.1:8770/health
```

Use the existing `typesafe` adapter with endpoint `http://127.0.0.1:8770`,
model `Nemotron-Labs-Diffusion-14B` and `--key-env ''`. No adapter patch or
submitter API key is required. Keep image defaults except the cache isolation
setting above. The reproduction guide pins the public harness to
`51a8d73fa798aa337bb1b26abd10995c0ab847e9`; maintainers may use their current
harness and private tasks for official evaluation.

Disclosure: public benchmark outcomes were inspected and later diagnostic
experiments were run. None of those later changes is incorporated in this
frozen candidate or substituted for its original results. No benchmark-trained
weights or fitted calibration are included. Please treat this as an inference
configuration of the existing NVIDIA checkpoint, subject to your eligibility
and cost-accounting policies, not as a new pretrained model.
