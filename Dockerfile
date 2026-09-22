ARG VLLM_IMAGE=nemotron-masked-vllm:dev
FROM ${VLLM_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/pst2154/Nemotron_Jev" \
      org.opencontainers.image.title="Nemotron Diffusion Decision Lab"
WORKDIR /app
COPY app/ /app/
COPY launch.sh /app/launch.sh
RUN chmod 0755 /app/launch.sh
ENV CHECKPOINT_DIR=/models/checkpoint HF_HOME=/models/hf-cache PORT=8770 \
    TRITON_CACHE_DIR=/tmp/nemotron-triton TORCHINDUCTOR_CACHE_DIR=/tmp/nemotron-inductor \
    VLLM_CACHE_ROOT=/tmp/nemotron-vllm XDG_CACHE_HOME=/tmp/nemotron-cache \
    VLLM_CONFIG_ROOT=/tmp/nemotron-config FLASHINFER_WORKSPACE_BASE=/tmp/nemotron-flashinfer \
    CUDA_CACHE_PATH=/tmp/nemotron-cuda \
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=8 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CANDIDATE_ONLY=1 DIRECT_LOGITS=1 PRIME_SHARED_PREFIX=1 BATCH_TOKENIZE=1 \
    POSITION_ORDERING=1
EXPOSE 8770
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/health', timeout=4)" || exit 1
ENTRYPOINT ["/app/launch.sh"]
CMD []
