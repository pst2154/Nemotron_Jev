# Container verification

Image tag: `ghcr.io/pst2154/nemotron-jev:14b-v2`

Registry manifest digest: `sha256:a1bf099bb919461659339f5bb8c1e962b3d3367ffe40c74fcb92c4c8c2b170ea`

Local image/config ID: `sha256:7014c09647df578a3c0f9cc214958b1326b4549b3597ce4d63da058eba2b0c10`

Runtime: Python 3.10, PyTorch 2.13.0+cu130, Transformers 5.17.0, huggingface-hub 1.32.0. Linux amd64, one H100 80 GB. Uncompressed image size: 16,542,216,472 bytes. Model weights are separate.

The deployed service runs code baked into this image. Only checkpoint/cache data are mounted; the application source is not bind-mounted. The same entrypoint starts the model and serves the UI/API. Live tests exercised Choice, Noul, weighted Score, normalized distributions, contrasting inputs, and 16 questions in one request. The v1 image passed all 72 frozen accuracy cases and the original 306-question request-format regression; see the accuracy report for timings and caveats. V2 preserves the same application files and adds `OMP_NUM_THREADS=8`, matching the original deployment. This setting alone did not resolve the packaging latency difference; no speed improvement is claimed.

The base is a CUDA development image, not a minimal production runtime. A first inference for a new shape can include compiler-cache warmup. This release is a functional experimental server, not a hardened multi-tenant service.

Final v2 verification: container healthcheck passed, live typed-output tests passed, all 72 frozen cases passed with zero errors, and anonymous Docker pull by the digest above succeeded. Final-image accuracy and timing results are published in `evaluation/container-v2-results/`.

GHCR package visibility is public. Anonymous registry manifest access returned HTTP 200 with the exact published digest above. No registry login is required.
