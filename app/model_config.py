"""Checkpoint identity shared by download, API responses, and health."""
import os

MODEL_ID = os.environ.get('MODEL_ID', 'nvidia/Nemotron-Labs-Diffusion-8B')
MODEL_NAME = MODEL_ID.rsplit('/', 1)[-1]
PINNED_REVISIONS = {
    'nvidia/Nemotron-Labs-Diffusion-8B': '16c67f0560b912e93e0cabb6e0c4f5c3086d95fc',
    'nvidia/Nemotron-Labs-Diffusion-14B': 'f8c3e2c078e193599b8882d965b1001c456ba738',
}

def model_revision():
    revision = os.environ.get('MODEL_REVISION') or PINNED_REVISIONS.get(MODEL_ID)
    if not revision:
        raise ValueError('Set MODEL_REVISION for an unrecognized MODEL_ID')
    return revision
