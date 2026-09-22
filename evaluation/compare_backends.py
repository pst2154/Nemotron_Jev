"""Reproducible native/vLLM parity and shared-state latency measurements.

Run each backend in a separate process on the same GPU. Result files contain
no endpoint, checkpoint path, or credentials. Timings include prompt preparation.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from functools import lru_cache, partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))


@lru_cache(maxsize=8)
def padded_state(tokenizer, target_tokens):
    colors = ['red', 'blue', 'green', 'yellow']
    state = '\n'.join(f'Item {i} is {colors[i % 4]}.' for i in range(100))
    padding = ' Unrelated background.'
    while len(tokenizer.encode(state + padding, add_special_tokens=False)) <= target_tokens:
        state += padding
    while len(tokenizer.encode(state, add_special_tokens=False)) < target_tokens:
        state += ' x'
    actual = len(tokenizer.encode(state, add_special_tokens=False))
    if actual != target_tokens:
        raise ValueError(f'Padding produced {actual} tokens, expected {target_tokens}')
    return state


def workload(tokenizer, count, target_tokens):
    colors = ['red', 'blue', 'green', 'yellow']
    state = padded_state(tokenizer, target_tokens)
    return {'model': 'jev-latest', 'state': state, 'questions': {
        str(i): {'type': 'choice', 'instructions': f'What color is Item {i}?',
                 'criteria': {color: color for color in colors}}
        for i in range(count)
    }}, {str(i): colors[i % 4] for i in range(count)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['native', 'vllm', 'http'], required=True)
    parser.add_argument('--url', help='System One base URL for an external comparison')
    parser.add_argument('--label', help='Public model label; do not put endpoints here')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[1000, 4000, 12000])
    parser.add_argument('--counts', type=int, nargs='+', default=[1, 4, 8, 16, 32, 64, 100])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--accuracy-limit', type=int, default=0)
    parser.add_argument('--skip-accuracy', action='store_true')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--no-prefix-cache', action='store_true')
    parser.add_argument('--candidate-only', action='store_true')
    parser.add_argument('--direct-logits', action='store_true')
    parser.add_argument('--max-batched-tokens', type=int, default=8192)
    parser.add_argument('--max-sequences', type=int, default=128)
    parser.add_argument('--prime-shared-prefix', action='store_true')
    parser.add_argument('--batch-tokenize', action='store_true')
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Choose a new output file; existing results are preserved')
    destination.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if args.backend == 'native':
        import torch
        from transformers import AutoModel
        from scoring import evaluate
        model = AutoModel.from_pretrained(
            args.checkpoint, trust_remote_code=True,
            dtype=torch.bfloat16).to('cuda').eval()
    elif args.backend == 'vllm':
        from vllm import LLM
        from vllm_scoring import evaluate
        evaluate = partial(evaluate, prime_shared_prefix=args.prime_shared_prefix,
                           batch_tokenize=args.batch_tokenize)
        from scoring import candidate_codes
        additional = ({'decision_token_ids': [token for _, token in candidate_codes(tokenizer)]}
                      if args.candidate_only else {})
        model = LLM(
            model=args.checkpoint, trust_remote_code=True, dtype='bfloat16',
            max_model_len=16386, max_num_seqs=args.max_sequences,
            max_num_batched_tokens=args.max_batched_tokens,
            gpu_memory_utilization=0.85, max_logprobs=1052,
            logprobs_mode='processed_logits' if args.direct_logits else 'processed_logprobs',
            enforce_eager=args.eager,
            enable_prefix_caching=not args.no_prefix_cache,
            additional_config=additional,
        )
    else:
        if not args.url:
            raise ValueError('--url is required for the http backend')
        model = None

        def evaluate(model, tokenizer, payload):
            request = urllib.request.Request(
                args.url.rstrip('/') + '/v1/systemone',
                data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
    def record(payload, metadata, gold):
        started = time.perf_counter()
        result = evaluate(model, tokenizer, payload)
        elapsed = 1000 * (time.perf_counter() - started)
        if set(result['answers']) != set(gold):
            raise RuntimeError('Response question IDs do not match the benchmark request')
        correct = 0
        for key, answer in result['answers'].items():
            if answer['type'] == 'noul':
                predicted = answer['noul'] >= 0.5
            elif answer['type'] == 'choice':
                predicted = answer['choice']
            else:
                predicted = int(max(answer['probabilities'], key=answer['probabilities'].get))
            correct += predicted == gold[key]
        row = dict(metadata, backend=args.label or args.backend, ms=elapsed,
                   candidate_only=None if args.backend == 'http' else args.candidate_only,
                   eager=None if args.backend == 'http' else args.eager,
                   direct_logits=None if args.backend == 'http' else args.direct_logits,
                   prefix_cache=(None if args.backend == 'http' else
                                 args.backend == 'vllm' and not args.no_prefix_cache),
                   scheduler=({'max_sequences': args.max_sequences,
                               'max_batched_tokens': args.max_batched_tokens}
                              if args.backend == 'vllm' else None),
                   prime_shared_prefix=None if args.backend == 'http' else args.prime_shared_prefix,
                   batch_tokenize=None if args.backend == 'http' else args.batch_tokenize,
                   correct=correct, total=len(gold), answers=result['answers'],
                   metrics=result.get('metrics', {}),
                   input_tokens=result['usage']['input_tokens'],
                   payload_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest())
        with destination.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'answers'}), flush=True)

    warmup, gold = workload(tokenizer, 1, 1000)
    evaluate(model, tokenizer, warmup)
    if not args.skip_accuracy:
        cases = json.loads(Path(__file__).with_name('cases.json').read_text())
        for case in cases[:args.accuracy_limit or None]:
            question = dict(case['question'])
            if question['type'] == 'boolean':
                question['type'] = 'noul'
            record({'model': 'jev-latest', 'state': case['state'],
                    'questions': {'answer': question}},
                   {'suite': 'accuracy', 'id': case['id']}, {'answer': case['gold']})
    for length in args.lengths:
        for count in args.counts:
            payload, gold = workload(tokenizer, count, length)
            if args.backend == 'vllm' and not args.no_prefix_cache:
                if not model.reset_prefix_cache():
                    raise RuntimeError('Could not reset prefix cache for cold measurement')
            for repeat in range(args.repeats):
                record(payload, {'suite': 'latency', 'state_tokens': length,
                                 'questions': count, 'repeat': repeat,
                                 'cache_state': ('gateway-managed' if args.backend == 'http'
                                                 else 'disabled' if args.backend == 'native'
                                                 or args.no_prefix_cache
                                                 else 'cold' if repeat == 0 else 'warm')}, gold)


if __name__ == '__main__':
    main()
