"""The engine boundary must preserve masks, options, and answer alignment."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from scoring import candidate_codes, prepare_questions
from vllm_scoring import evaluate


class Tokenizer:
    def __init__(self):
        self.code_encodes = 0

    def encode(self, text, **kwargs):
        if len(text) == 1:
            self.code_encodes += 1
            return [ord(text)]
        return [1, 2, 3]

    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def __call__(self, prompts, **kwargs):
        return {'input_ids': [self.encode(prompt, add_special_tokens=False)
                              for prompt in prompts]}


class ScoringTests(unittest.TestCase):
    def test_batch_tokenization_preserves_prepared_question_contract(self):
        payload = {'model': 'jev-latest', 'state': {'text': 'test'}, 'questions': {
            'q1': {'type': 'noul', 'instructions': 'True?'},
            'q2': {'type': 'choice', 'instructions': 'Pick',
                   'criteria': {'one': 'First', 'two': 'Second'}}}}
        tokenizer = Tokenizer()
        self.assertEqual(prepare_questions(tokenizer, payload),
                         prepare_questions(tokenizer, payload, batch_tokenize=True))

    def score_values(self, values):
        def generate(prompts, params, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
                token: SimpleNamespace(logprob=value) for token, value in values.items()
            }])])]

        payload = {'model': 'jev-latest', 'state': 'test', 'questions': {
            'q': {'type': 'noul', 'instructions': 'Is it true?'}}}
        with patch.dict(sys.modules, {'vllm': SimpleNamespace(
                SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))}):
            return evaluate(SimpleNamespace(generate=generate), Tokenizer(), payload)

    def test_direct_logits_and_logprobs_have_identical_candidate_softmax(self):
        logits = self.score_values({65: 10001.0, 66: 10002.0})
        logprobs = self.score_values({65: -2.0, 66: -1.0})
        self.assertEqual(logits['answers'], logprobs['answers'])

    def test_missing_candidate_is_not_silently_renormalized(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing candidate'):
            self.score_values({65: -1.0})

    def test_nonfinite_candidates_are_rejected(self):
        for bad in [float('nan'), float('inf'), float('-inf')]:
            with self.subTest(value=bad):
                with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
                    self.score_values({65: -1.0, 66: bad})

    def test_one_batch_has_masked_inputs_and_complete_candidate_probabilities(self):
        tokenizer = Tokenizer()
        seen = []

        def generate(prompts, params, **kwargs):
            seen.append((prompts, params))
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
                ord('A'): SimpleNamespace(logprob=-2.0),
                ord('B'): SimpleNamespace(logprob=-1.0),
            }])]) for _ in prompts]

        payload = {'model': 'jev-latest', 'state': 'example', 'questions': {
            'choice': {'type': 'choice', 'instructions': 'pick',
                       'criteria': {'first': 'one', 'second': 'two'}},
            'yes': {'type': 'noul', 'instructions': 'yes?'},
            'rating': {'type': 'score', 'instructions': 'rate',
                       'criteria': ['low', 'high']},
        }}
        with patch.dict(sys.modules, {'vllm': SimpleNamespace(
                SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))}):
            result = evaluate(SimpleNamespace(generate=generate), tokenizer, payload)
        self.assertEqual(len(seen), 1)
        self.assertTrue(all(p['prompt_token_ids'][-1] == 100 for p in seen[0][0]))
        for params in seen[0][1]:
            self.assertEqual(params.max_tokens, 1)
            self.assertEqual(params.logprob_token_ids, [65, 66])
        self.assertEqual(result['answers']['choice']['choice'], 'second')
        self.assertAlmostEqual(sum(result['answers']['choice']['probabilities'].values()), 1)
        self.assertAlmostEqual(result['answers']['yes']['noul'], 0.7310585786)
        self.assertAlmostEqual(result['answers']['rating']['score'], 0.7310585786)

    def test_candidate_mapping_is_cached_per_tokenizer(self):
        tokenizer = Tokenizer()
        first = candidate_codes(tokenizer)
        encodes = tokenizer.code_encodes
        self.assertIs(candidate_codes(tokenizer), first)
        self.assertEqual(tokenizer.code_encodes, encodes)

    def test_large_choice_set_keeps_every_candidate(self):
        class LargeTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                return [1000 + int(text)] if text.isdigit() else super().encode(text, **kwargs)

        seen = []

        def generate(prompts, params, **kwargs):
            seen.extend(params)
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
                token: SimpleNamespace(logprob=-5.0) for token in p.allowed_token_ids
            }])]) for p in params]

        payload = {'model': 'jev-latest', 'state': 'test', 'questions': {
            'many': {'type': 'choice', 'instructions': 'pick',
                     'criteria': {str(i): str(i) for i in range(255)}}}}
        with patch.dict(sys.modules, {'vllm': SimpleNamespace(
                SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))}):
            result = evaluate(SimpleNamespace(generate=generate), LargeTokenizer(), payload)
        self.assertEqual(seen[0].logprobs, 255)
        self.assertEqual(len(result['answers']['many']['probabilities']), 255)
        self.assertAlmostEqual(sum(result['answers']['many']['probabilities'].values()), 1)

    def test_priming_preserves_question_order_across_two_batches(self):
        batches = []

        def generate(prompts, params, **kwargs):
            batches.append(len(prompts))
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
                ord('A'): SimpleNamespace(logprob=-2),
                ord('B'): SimpleNamespace(logprob=-1),
            }])]) for _ in prompts]

        payload = {'model': 'jev-latest', 'state': 'test', 'questions': {
            str(i): {'type': 'noul', 'instructions': f'Condition {i}?'}
            for i in range(129)}}
        with patch.dict(sys.modules, {'vllm': SimpleNamespace(
                SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))}):
            result = evaluate(SimpleNamespace(generate=generate), Tokenizer(), payload,
                              prime_shared_prefix=True)
        self.assertEqual(batches, [1, 128])
        self.assertEqual(list(result['answers']), list(payload['questions']))


if __name__ == '__main__':
    unittest.main()
