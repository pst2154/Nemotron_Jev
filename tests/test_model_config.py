import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
import model_config

class ModelConfigTests(unittest.TestCase):
    def test_default_pin(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(model_config, 'MODEL_ID', 'nvidia/Nemotron-Labs-Diffusion-8B'):
            self.assertEqual(model_config.model_revision(), '16c67f0560b912e93e0cabb6e0c4f5c3086d95fc')

    def test_unknown_model_requires_revision(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(model_config, 'MODEL_ID', 'example/custom'):
            with self.assertRaises(ValueError): model_config.model_revision()

    def test_explicit_revision(self):
        with patch.dict(os.environ, {'MODEL_REVISION':'pinned'}):
            self.assertEqual(model_config.model_revision(), 'pinned')
