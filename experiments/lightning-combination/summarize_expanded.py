"""Validate paired benchmark evidence and render tables without private URLs."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def summarize(rows):
    valid = [r for r in rows if not r.get('error')]
    times = sorted(r['ms'] for r in valid)
    correct = sum(r['correct'] for r in rows)
    total = sum(r['total'] for r in rows)
    completed_total = sum(r['total'] for r in valid)
    return {'requests': len(rows), 'errors': len(rows)-len(valid), 'correct': correct,
            'total': total, 'completed_total': completed_total,
            'end_to_end_correct_pct': 100*correct/total,
            'accuracy_pct': 100*correct/completed_total if completed_total else None,
            'median_ms': statistics.median(times) if times else None,
            'p95_ms': times[math.ceil(.95*len(times))-1] if times else None}


def validate(stem):
    raw = (ROOT/f'{stem}-fixtures.json').read_bytes()
    fixtures = {r['id']: r for r in json.loads(raw)}
    rows = [json.loads(line) for line in (ROOT/f'{stem}-results.jsonl').read_text().splitlines()]
    seen = set()
    for row in rows:
        pair = (row['id'], row['backend'])
        assert pair not in seen, pair
        seen.add(pair)
        fixture = fixtures[row['id']]
        assert row['fixture_sha256'] == hashlib.sha256(raw).hexdigest()
        payload = {'model': 'jev-latest', **fixture['input']}
        assert row['payload_sha256'] == hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        assert row['total'] == len(fixture['gold'])
        if not row.get('error'):
            assert row['returned'] == row['total']
            assert row['correct'] == sum(d['prediction'] == d['gold'] and d['prediction'] is not None for d in row['decisions'])
    expected = {(f, b) for f in fixtures for b in ['lightning_lora', 'diffusiongemma']}
    assert seen == expected, f'Missing {len(expected-seen)} paired results'
    return rows


def main():
    rows = validate('expanded') + validate('grouped')
    groups = collections.defaultdict(list)
    for row in rows:
        # Natural semantic lengths vary; aggregate those rather than one cell per input.
        length = 'natural' if row['suite'] == 'semantic324' else row['state_tokens']
        groups[(row['suite'], length, row['questions_count'], row['backend'])].append(row)
    output = []
    for (suite, length, questions, backend), subset in groups.items():
        output.append(dict(suite=suite, state_tokens=length, questions=questions, backend=backend, **summarize(subset)))
    (ROOT/'expanded-summary.json').write_text(json.dumps(output, indent=2))
    lines = ['# Expanded paired endpoint results', '',
             'Correct denominators include all requested decisions. A failed request contributes no delivered correct answers; it is not a model misclassification. Latencies exclude failed requests. The summary JSON also provides completed-only denominators and accuracy.', '',
             '| Workload | State tokens | Questions | Backend | Correct / requested | Median ms | p95 ms | Errors |',
             '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in output:
        ms = lambda x: f'{x:,.0f}' if x is not None else '—'
        lines.append(f'| {r["suite"]} | {r["state_tokens"]} | {r["questions"]} | {r["backend"]} | {r["correct"]}/{r["total"]} | {ms(r["median_ms"])} | {ms(r["p95_ms"])} | {r["errors"]} |')
    lines += ['', '## Semantic 324 by output type', '', '| Backend | Type | Correct |', '| --- | --- | ---: |']
    for backend in ['lightning_lora', 'diffusiongemma']:
        for kind in ['choice', 'noul', 'score']:
            d = [d for r in rows if r['suite']=='semantic324' and r['backend']==backend for d in r.get('decisions', []) if d['type']==kind]
            lines.append(f'| {backend} | {kind} | {sum(x["correct"] for x in d)}/{len(d)} |')
    lines += ['', '## Lookup accuracy by evidence position', '', '| Backend | State tokens | Position | Correct |', '| --- | ---: | --- | ---: |']
    for backend in ['lightning_lora', 'diffusiongemma']:
        for length in [256,1000,4000,8000,12000]:
            for position in ['start','middle','end']:
                a = [r for r in rows if r['suite']=='random_lookup' and r['backend']==backend and r['state_tokens']==length and r['position']==position]
                lines.append(f'| {backend} | {length} | {position} | {sum(x["correct"] for x in a)}/{sum(x["total"] for x in a)} |')
    (ROOT/'EXPANDED_TABLES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verified_requests': len(rows), 'per_backend': {b: summarize([r for r in rows if r['backend']==b]) for b in ['lightning_lora','diffusiongemma']}}, indent=2))


if __name__ == '__main__':
    main()
