# Serve Lightning + LoRA with the System One API

**Use the pinned stock vLLM image below. No custom image build is required.**
The combined launcher runs vLLM and the existing System One adapter inside one
container. This is the Lightning recipe, not the DiffusionGemma container.

```text
vllm/vllm-openai:nightly-a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b
```

The model/adapter combination was measured on one **H100 80GB** with this image.
Do not assume an arbitrary older vLLM release supports the same Lightning,
NVFP4, Mamba-cache, and LoRA combination. No separate native-training kernel
wheels are needed for this serving path.

## Requirements

- Linux GPU host, Docker, NVIDIA Container Toolkit, and an already allocated
  H100 80GB. This guide does not allocate or release GPUs.
- The **original compatible Lightning NVFP4 base checkpoint**, including its
  tokenizer, already downloaded. The uploaded LoRA is not a complete model.
- Read access to the private adapter repository on Hugging Face.
- Free model and System One ports; writable storage for runtime caches.

The tested base was an existing NVFP4 artifact with its own quantization history.
Its precise equivalence to a public downloadable checkpoint has not been
established. This guide therefore does not invent a public base-model revision
or promise that the adapter alone reproduces the model. Use the same compatible
base as the experiment; MTP/speculative decoding is not enabled.

## 1. Get this branch and the private adapter

```bash
git clone --branch perf/lightning-100-question-shared-prefill \
  https://github.com/pst2154/Nemotron_Jev.git
cd Nemotron_Jev/experiments/lightning-combination

# Authenticate interactively with an account/token granted access to the repo.
python3 -m pip install -U huggingface_hub
hf auth login

# Choose your own absolute paths; do not reuse an occupied runtime location.
export MODEL_DIR=/absolute/path/to/compatible-lightning-nvfp4
export ADAPTER_DIR=/absolute/path/to/lightning-lora
export RUN_DIR=/absolute/path/to/new-serving-runtime

hf download einsteiner1983/nemotron-3.5-lightning-systemone-lora-r16 \
  --revision d879c16e51770ef5dfcf3e131419c1e19554d31e \
  --local-dir "$ADAPTER_DIR"
```

The pinned adapter revision contains the exact tested files. Verify on Linux:

```bash
sha256sum "$ADAPTER_DIR/adapter_model.safetensors" "$ADAPTER_DIR/adapter_config.json"
```

Expected hashes, in that order:

```text
fb862c8f2600a3e9ade87fe7139de94d837ca4474657b9159f3b8260f81de075
db20ccc562beacc1544b48b5151573ff0bbd149ef811054b575ecfc59c8a238c
```

No merge or fresh FP4 quantization is needed. Leave the base and adapter intact;
both are mounted read-only. Hub credentials are not passed into the container:
all model files must already be available locally.

## 2. Start both services in one container

```bash
nvidia-smi --query-gpu=uuid,name --format=csv
# Select only the GPU assigned to you.
export GPU_UUID=GPU-your-allocated-device
export CONTAINER_NAME=lightning-systemone-trained
export SERVER_PORT=8300
export SYSTEMONE_PORT=8790
export SYSTEMONE_HOST=127.0.0.1

bash launch_systemone.sh
docker logs -f "$CONTAINER_NAME"
```

`docker run` pulls the pinned image if it is missing. It starts vLLM with the
original tested flags, loads the LoRA as `trained`, waits for model readiness,
then starts `/v1/systemone`. If either serving process exits, the supervisor
stops the other and exits the container. It does not replace or stop a container
with a different name. Choose a fresh name rather than removing another service.

The wrapper retains these API settings:

```text
UPSTREAM_MODEL=trained
UPSTREAM_CONCURRENCY=100
STATE_FIRST=1
PERSISTENT_UPSTREAM=1
PRIME_SHARED_PREFIX=0
ISOLATE_REQUEST_CACHE=1
```

The model stays loopback-only. For access from other machines, explicitly set
`SYSTEMONE_HOST=0.0.0.0` and set a non-empty `SYSTEMONE_API_KEY` before launching.
The launcher rejects unauthenticated external binding. Use TLS and access
controls at your reverse proxy/firewall; a bearer key alone does not encrypt
HTTP. Do not expose the unauthenticated model port publicly.

This starts **the API, not a browser UI**. `launch_combination.sh` remains the
model-only launcher for users who want to run their own gateway.

## 3. Verify readiness and call System One

```bash
curl --fail "http://127.0.0.1:$SERVER_PORT/health"
curl --fail "http://127.0.0.1:$SERVER_PORT/v1/models"
curl --fail "http://127.0.0.1:$SYSTEMONE_PORT/health"

curl --fail-with-body "http://127.0.0.1:$SYSTEMONE_PORT/v1/systemone" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "The customer asks to swap blue shoes for the same size in black, and does not want a refund.",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What action is the customer requesting?",
        "criteria": {
          "exchange": "Swap the item for another variant",
          "refund": "Return the money already paid",
          "information": "Answer a question without changing the order"
        }
      },
      "wants_refund": {
        "type": "noul",
        "instructions": "Does the customer want their money returned?"
      }
    }
  }'
```

If `SYSTEMONE_API_KEY` is configured, add
`-H "Authorization: Bearer $SYSTEMONE_API_KEY"` to the POST. The expected
classification is `intent.choice = "exchange"` and `wants_refund.noul < 0.5`.
Probabilities are model outputs, not a fixed fixture response.

The gateway health endpoint reports only that its process is alive. Check both
health endpoints **and an actual classification call**. The model registry must
show `trained` as an adapter of `lightning`; otherwise you are not testing the
selected fine-tune.

## API behavior and limits

- `POST /v1/systemone`; model alias `jev-latest` or `lightning-systemone`.
  Both select the trained Lightning backend in this launcher, not TypeSafe Jev.
- Choice: 2–20 options. Noul: Boolean probability. Score: 2–10 ordered levels.
- 1–300 questions accepted by the adapter; this comparison measured up to 100.
- Each question uses one constrained answer token and candidate logprobs;
  this is not the separate native zero-generation runner.
- The API splits questions internally and shares their state prefix. Callers
  send one normal System One request and do not batch the questions themselves.
- Prototype `confidence` is a local concentration statistic, not TypeSafe's
  proprietary calibrated confidence. This is API-shaped compatibility, not a
  guarantee of every TypeSafe API feature or identical model behavior.
- At most 16 outer requests are admitted concurrently; internal workers are
  shared. This is not a claim of 16-request latency or throughput from the
  single-outer-request benchmark.

## Validation scope

The model, LoRA, original adapter, and pinned runtime were exercised in the
published endpoint benchmarks. The combined packaging wrapper was added after
those runs and checked with launch-argument/readiness/security tests; it was not
used to retroactively rerun all latency measurements. The running services were
left unchanged. Run the health checks and classification smoke test on your host.
