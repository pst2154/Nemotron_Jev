# Container verification

Image tag: `ghcr.io/pst2154/nemotron-jev:14b-v1`

Registry manifest digest: `sha256:9b189c07adb6ebced04c3f524f4556164e87aefb7ab50bb5e92fa7cd5d2537f4`

Local image/config ID: `sha256:8ec44184727dc03998a51268f97724e6f5782a0447bce4c7ba75d42c984aa8b0`

Runtime: Python 3.10, PyTorch 2.13.0+cu130, Transformers 5.17.0, huggingface-hub 1.32.0. Linux amd64, one H100 80 GB. Uncompressed image size: 16,542,216,472 bytes. Model weights are separate.

The deployed service runs code baked into this image. Only checkpoint/cache data are mounted; the application source is not bind-mounted. The same entrypoint starts the model and serves the UI/API. Container healthcheck passed. Live tests exercised Choice, Noul, weighted Score, normalized distributions, contrasting inputs, and 16 questions in one request. The deployed image also passed all 72 frozen accuracy cases and the original 306-question request-format regression; see the accuracy report for timings and caveats.

The base is a CUDA development image, not a minimal production runtime. A first inference for a new shape can include compiler-cache warmup. This release is a functional experimental server, not a hardened multi-tenant service.

GHCR package visibility is public. Anonymous registry manifest access returned HTTP 200 with the exact published digest above. No registry login is required.
