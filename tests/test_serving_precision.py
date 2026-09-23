"""Precision settings must fail before model loading for unsupported combinations."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from server import precision_settings


class PrecisionTests(unittest.TestCase):
    def test_default_retains_bf16_candidate_head(self):
        self.assertEqual(precision_settings({}), (None, True))

    def test_fp8_requires_explicit_full_head(self):
        for mode in ('fp8', 'fp8_per_channel', 'fp8_per_block'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, 'CANDIDATE_ONLY=0'):
                    precision_settings({'QUANTIZATION': mode})
                self.assertEqual(precision_settings({
                    'QUANTIZATION': mode, 'CANDIDATE_ONLY': '0'}), (mode, False))

    def test_unknown_settings_do_not_silently_fall_back(self):
        for settings in ({'QUANTIZATION': 'fp4'}, {'CANDIDATE_ONLY': 'yes'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                precision_settings(settings)


if __name__ == '__main__':
    unittest.main()
