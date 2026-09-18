FROM dockerhub.nvidia.com/flashinfer/flashinfer-ci-cu130:20260822-9a0e83b
LABEL org.opencontainers.image.source="https://github.com/pst2154/Nemotron_Jev" \
      org.opencontainers.image.title="Nemotron Diffusion Decision Lab"
RUN python -m pip install --no-cache-dir transformers==5.17.0 huggingface-hub==1.32.0
WORKDIR /app
COPY app/ /app/
COPY launch.sh /app/launch.sh
RUN chmod 0755 /app/launch.sh
ENV CHECKPOINT_DIR=/models/checkpoint HF_HOME=/models/hf-cache PORT=8770 \
    TRITON_CACHE_DIR=/tmp/nemotron-triton TORCHINDUCTOR_CACHE_DIR=/tmp/nemotron-inductor
EXPOSE 8770
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/health', timeout=4)" || exit 1
ENTRYPOINT ["/app/launch.sh"]
