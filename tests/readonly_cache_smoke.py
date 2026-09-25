"""Run inside the image with --read-only --network none and writable /tmp."""
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

checkpoint = os.environ['CHECKPOINT_DIR']
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True,
                                  local_files_only=True)
AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True,
                              local_files_only=True)
reference = config.auto_map['AutoModel']
get_class_from_dynamic_module(reference, checkpoint, local_files_only=True)
cache = Path(os.environ['HF_MODULES_CACHE'])
assert cache.is_dir() and any(cache.rglob('*.py'))
from usage import decision_usage
assert decision_usage(10, 1)['generated_text_tokens'] == 0
print('READONLY_OFFLINE_MODULE_CACHE_OK')
