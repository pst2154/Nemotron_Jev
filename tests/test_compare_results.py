"""Parity checks must fail closed on incomplete or malformed measurements."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from compare_results import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'payload_sha256': 'fixture', 'repeat': 0, 'answers': {
            'choice': {'type': 'choice', 'probabilities': {'a': 0.2, 'b': 0.8}},
            'noul': {'type': 'noul', 'noul': 0.7},
        }}]

    def test_identical_results(self):
        result = compare(self.rows, copy.deepcopy(self.rows))
        self.assertEqual(result['decisions'], 2)
        self.assertEqual(result['argmax_agreements'], 2)
        self.assertEqual(result['max_probability_delta'], 0)

    def test_missing_request_is_reported(self):
        result = compare(self.rows, [])
        self.assertEqual(result['matched_requests'], 0)
        self.assertEqual(result['missing_candidate_requests'], 1)

    def test_duplicate_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            compare(self.rows, self.rows + self.rows)

    def test_missing_question_is_rejected(self):
        candidate = copy.deepcopy(self.rows)
        del candidate[0]['answers']['noul']
        with self.assertRaisesRegex(ValueError, 'Question IDs'):
            compare(self.rows, candidate)

    def test_nonfinite_probability_is_rejected(self):
        candidate = copy.deepcopy(self.rows)
        candidate[0]['answers']['noul']['noul'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid candidate'):
            compare(self.rows, candidate)

    def test_changed_answer_and_probability_are_reported(self):
        candidate = copy.deepcopy(self.rows)
        candidate[0]['answers']['noul']['noul'] = 0.2
        result = compare(self.rows, candidate)
        self.assertEqual(result['argmax_agreements'], 1)
        self.assertEqual(len(result['mismatches']), 1)
        self.assertAlmostEqual(result['max_probability_delta'], 0.5)


if __name__ == '__main__':
    unittest.main()
