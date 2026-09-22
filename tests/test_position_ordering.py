"""Frozen ordering must move options, not their label-to-code assignments."""
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from scoring import POSITION_ORDERING, order_options, prepare_questions
from vllm_scoring import evaluate


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, prompts, **kwargs):
        return {'input_ids': [self.encode(p) for p in prompts]}


def rendered(prepared):
    messages = json.loads(''.join(map(chr, prepared[4])))
    return json.loads(messages[1]['content'])


class PositionOrderingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.payload = {'model': 'jev-latest', 'state': {}, 'questions': {
            'choice': {'type': 'choice', 'instructions': 'Select',
                       'criteria': {'first': 'a', 'second': 'bb', 'third': 'ccc', 'fourth': 'dddd'}},
            'noul': {'type': 'noul', 'instructions': 'True?',
                     'criteria': {'false': 'a', 'true': 'bbbb'}},
            'score': {'type': 'score', 'instructions': 'Rate',
                      'criteria': ['a', 'bb', 'ccc']}}}

    def test_exact_rotation_and_stable_ties(self):
        options = [{'code': chr(65+i), 'label': str(i), 'meaning': 'x'*(i+1)} for i in range(4)]
        self.assertEqual([o['code'] for o in order_options(self.tokenizer, options, set())],
                         ['B', 'A', 'D', 'C'])
        for option in options:
            option['meaning'] = 'same'
        self.assertEqual([o['code'] for o in order_options(self.tokenizer, options, set())],
                         ['C', 'D', 'A', 'B'])

    def test_code_meaning_and_output_order_are_preserved(self):
        original = copy.deepcopy(self.payload)
        off = prepare_questions(self.tokenizer, self.payload, position_ordering=False)
        on = prepare_questions(self.tokenizer, self.payload, position_ordering=True)
        self.assertEqual(self.payload, original)
        for before, after in zip(off, on):
            self.assertEqual(before[:4], after[:4])
            old, new = rendered(before)['options'], rendered(after)['options']
            self.assertEqual({o['code']: o for o in old}, {o['code']: o for o in new})
            self.assertNotEqual(old, new)
        self.assertEqual([o['code'] for o in rendered(on[0])['options']], ['B', 'A', 'D', 'C'])

    def test_disabled_keeps_original_prompt_and_default_preparation(self):
        prepared = prepare_questions(self.tokenizer, self.payload, position_ordering=False)
        self.assertEqual(prepared, prepare_questions(self.tokenizer, self.payload))
        content = rendered(prepared[0])
        self.assertEqual(content, {'state': {}, 'question': 'Select', 'options': [
            {'code': chr(65+i), 'label': label, 'meaning': description}
            for i, (label, description) in enumerate(self.payload['questions']['choice']['criteria'].items())]})

    def test_batch_tokenization_and_structured_values(self):
        self.payload['state'] = {'text': 'CAFÉ café is a place'}
        q = self.payload['questions']['choice']
        q['instructions'] = {'ask': 'Select a café'}
        q['criteria'] = {'a': None, 'b': {'meaning': 'café'}, 'c': ['different', 'thing']}
        self.assertEqual(prepare_questions(self.tokenizer, self.payload, position_ordering=True),
                         prepare_questions(self.tokenizer, self.payload, batch_tokenize=True, position_ordering=True))

    def test_shared_state_and_instruction_words_affect_order(self):
        options = [{'code': 'A', 'label': 'x', 'meaning': 'alpha'},
                   {'code': 'B', 'label': 'y', 'meaning': 'omega'}]
        self.assertEqual([o['code'] for o in order_options(self.tokenizer, options, {'alpha'})], ['B', 'A'])

    def test_answers_use_original_codes_for_all_types(self):
        calls = []
        def generate(prompts, params, **kwargs):
            calls.append(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
                token: SimpleNamespace(logprob=0.0 if token == ord('B') else -30.0)
                for token in p.logprob_token_ids}])]) for p in params]
        with patch.dict(sys.modules, {'vllm': SimpleNamespace(
                SamplingParams=lambda **kw: SimpleNamespace(**kw))}):
            result = evaluate(SimpleNamespace(generate=generate), self.tokenizer, self.payload)
            off = evaluate(SimpleNamespace(generate=generate), self.tokenizer, self.payload, position_ordering=False)
        self.assertEqual(len(calls), 2)  # One engine call per three-question request.
        self.assertEqual(result['answers'], off['answers'])
        self.assertEqual(result['answers']['choice']['choice'], 'second')
        self.assertGreater(result['answers']['noul']['noul'], .999)
        self.assertAlmostEqual(result['answers']['score']['score'], 1)
        self.assertEqual(result['answers']['score']['legend'], {'0': 'a', '1': 'bb', '2': 'ccc'})
        self.assertEqual(result['scoring']['position_ordering'], POSITION_ORDERING)
        self.assertEqual(off['scoring']['position_ordering'], 'off')


if __name__ == '__main__':
    unittest.main()
