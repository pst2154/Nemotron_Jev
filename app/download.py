"""Fetch a pinned, locally cached model revision before starting the service."""
import os
from huggingface_hub import snapshot_download

snapshot_download('nvidia/Nemotron-Labs-Diffusion-14B',
    revision=os.environ.get('MODEL_REVISION', 'f8c3e2c078e193599b8882d965b1001c456ba738'),
    local_dir=os.environ.get('CHECKPOINT_DIR','/models/checkpoint'), max_workers=16)
