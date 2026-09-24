"""Fetch a pinned, locally cached model revision before starting the service."""
import os
from huggingface_hub import snapshot_download
from model_config import MODEL_ID, model_revision

snapshot_download(MODEL_ID,
    revision=model_revision(),
    local_dir=os.environ.get('CHECKPOINT_DIR','/models/checkpoint'), max_workers=16)
