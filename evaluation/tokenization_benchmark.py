"""Measure prompt preparation and require exact serial/batched token identity."""
import argparse
import json
import time
from pathlib import Path

from compare_backends import workload
from scoring import prepare_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Existing tokenization measurements are preserved')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    rows = []
    for length in [1000, 4000, 12000]:
        payload, _ = workload(tokenizer, 100, length)
        reference = prepare_questions(tokenizer, payload)
        for batch in [False, True]:
            for repeat in range(3):
                start = time.perf_counter()
                actual = prepare_questions(tokenizer, payload, batch_tokenize=batch)
                elapsed = 1000 * (time.perf_counter() - start)
                if actual != reference:
                    raise RuntimeError('Batched tokenization changed a scoring input')
                rows.append({'state_tokens': length, 'questions': 100,
                             'batch_tokenize': batch, 'repeat': repeat,
                             'ms': elapsed, 'exact_match': True})
                print(json.dumps(rows[-1]), flush=True)
    destination.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
