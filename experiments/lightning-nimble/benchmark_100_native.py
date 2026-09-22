"""100 distinct record lookups: scaling diagnostic, not semantic holdout accuracy."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from nimble.scoring.parallel_schema import prepare_prompts
from shared_native import NativeScorer, load_native_model


def fixture(tokenizer, trial):
    facts = '\n'.join(f'Record {i}: {"on" if i % 2 == 0 else "off"}.' for i in range(100))
    text = f'Reference trial-{trial}.\n{facts}\nUnrelated padding: ' + 'neutral ' * 1000
    ids = tokenizer.encode(text, add_special_tokens=False)[:1000]
    state = tokenizer.decode(ids, skip_special_tokens=False)
    assert len(tokenizer.encode(state, add_special_tokens=False)) == 1000
    assert 'Record 99: off.' in state
    schema = {str(i): {'type': 'enum', 'description': f'What flag is recorded for Record {i}?',
                      'choices': ['enabled', 'disabled'],
                      'choice_descriptions': {'enabled': 'The record says on',
                                              'disabled': 'The record says off'}} for i in range(100)}
    return state, schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    scorer = NativeScorer(load_native_model(args.model, args.adapter))
    report = {'description': __doc__, 'checkpoint': args.label,
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'state_tokens': 1000, 'branch_batch_size': 8, 'client_concurrency': 1,
              'max_input_tokens': 16384, 'cross_request_cache': False,
              'timing': 'GPU synchronized scorer wall time; excludes tokenization and HTTP',
              'rows': []}

    def record(prepared, mode, trial, prep_ms=0):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        scores = scorer.shared(prepared, 8, optimized=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        assert len(scores) == len(prepared.names)
        assert all(torch.isfinite(s).all().item() for s in scores)
        predictions = [prepared.choices[i][s.argmax().item()] for i, s in enumerate(scores)]
        correct = sum(p == ('enabled' if int(k) % 2 == 0 else 'disabled')
                      for k, p in zip(prepared.names, predictions))
        row = {'mode': mode, 'questions': len(scores), 'trial': trial, 'ms': elapsed,
               'prepare_ms': prep_ms, 'correct': correct, 'names': prepared.names,
               'predictions': predictions, 'prefix_tokens': len(prepared.prefix_ids),
               'max_full_tokens': max(map(len, prepared.full_ids)),
               'peak_gib': torch.cuda.max_memory_allocated() / 2**30,
               'prompt_sha256': hashlib.sha256(json.dumps(prepared.full_ids).encode()).hexdigest()}
        report['rows'].append(row)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('predictions','names')}), flush=True)

    for count in [1, 8, 20, 30, 50, 100]:
        state, schema = fixture(tokenizer, 'warmup')
        subset = dict(list(schema.items())[:count])
        warm = prepare_prompts(tokenizer, state, subset, 16384)
        scorer.shared(warm, 8, optimized=True)
        torch.cuda.synchronize()
        for trial in range(10 if count == 100 else 3):
            state, schema = fixture(tokenizer, trial)
            start = time.perf_counter()
            prepared = prepare_prompts(tokenizer, state, dict(list(schema.items())[:count]), 16384)
            prep_ms = (time.perf_counter() - start) * 1000
            assert len(prepared.names) == count
            record(prepared, 'shared', trial, prep_ms)
    # Same 100 questions individually: separates long-schema effects from basic lookup ability.
    state, schema = fixture(tokenizer, 0)
    for key, field in schema.items():
        prepared = prepare_prompts(tokenizer, state, {key: field}, 16384)
        record(prepared, 'singleton_control', 0)
    report['complete'] = True
    out.write_text(json.dumps(report, indent=2))
    for count in [1, 8, 20, 30, 50, 100]:
        rows = [r for r in report['rows'] if r['mode'] == 'shared' and r['questions'] == count]
        print(json.dumps({'questions': count, 'median_ms': statistics.median(r['ms'] for r in rows),
                          'correct': sum(r['correct'] for r in rows), 'total': count*len(rows)}), flush=True)
    print('BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
