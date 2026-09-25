import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from usage import decision_usage


class UsageTests(unittest.TestCase):
    def test_native_does_not_sample(self):
        usage = decision_usage(123, 1)
        self.assertEqual(usage['output_tokens'], 1)
        self.assertEqual(usage['input_tokens'], 123)
        self.assertEqual(usage['decision_positions'], 1)
        self.assertEqual(usage['generated_text_tokens'], 0)
        self.assertEqual(usage['discarded_sampled_tokens'], 0)

    def test_vllm_counts_decisions_not_json_response_tokens(self):
        usage = decision_usage(200, 100, sampled_tokens=100)
        self.assertEqual(usage['output_tokens'], 100)
        self.assertEqual(usage['discarded_sampled_tokens'], 100)
        self.assertEqual(usage['generated_text_tokens'], 0)
