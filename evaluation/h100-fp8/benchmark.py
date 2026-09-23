"""Exercise the actual HTTP scorer, including long states and question fan-out."""
import argparse
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

p = argparse.ArgumentParser()
for key in ('url', 'checkpoint', 'harness', 'output'):
    p.add_argument('--' + key, required=True)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, a.harness)
from transformers import AutoTokenizer
from jevbench.tasks import load_jsonl
from jevbench.adapters.base import build_question
from jevbench.scoring import score_task

def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(a.url + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()

deadline = time.monotonic() + 600
while True:
    try:
        health = json.loads(request('/health'))
        break
    except (urllib.error.URLError, ConnectionError):
        if time.monotonic() >= deadline:
            raise
        time.sleep(2)
assert health['ready'] and health['backend'] == 'vllm'
assert b'<html' in request('/').lower()
smoke = json.loads(request('/v1/systemone', {'model': 'jev-latest',
    'state': 'All customer payouts have failed for three days.', 'questions': {
        'team': {'type': 'choice', 'instructions': 'Which team handles this?',
                 'criteria': {'billing': 'Payments and payouts', 'sales': 'New sales'}},
        'failed': {'type': 'noul', 'instructions': 'Have payouts failed?'},
        'severity': {'type': 'score', 'instructions': 'How severe is the payout issue?',
                     'criteria': ['No failure', 'Some failures', 'All payouts fail']}}}))
assert smoke['answers']['team']['choice'] == 'billing'
assert smoke['answers']['failed']['noul'] > 0.5
assert 0 <= smoke['answers']['severity']['score'] <= 2
(out / 'smoke.json').write_text(json.dumps(smoke, indent=2))
tasks = [task for tier in ('easy', 'original', 'hard')
         for task in load_jsonl(str(Path(a.harness) / 'datasets/public' / f'{tier}.jsonl'))]
rows = []
for task in tasks:
    q = build_question(task)
    result = json.loads(request('/v1/systemone', {'model': 'jev-latest',
        'state': task.state, 'questions': {'q': q}}))
    answer = result['answers']['q']
    probs = ({'no': 1-answer['noul'], 'yes': answer['noul']}
             if q['type'] == 'noul' else answer['probabilities'])
    scored = score_task(probs, task)
    assert scored['strict_valid'] and not scored['renormalized']
    rows.append({'id': task.id, **scored})
(out / 'accuracy.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True, local_files_only=True)
timings = []
for length in (128, 1000, 4096, 8192):
    state = ' hello' * length
    assert len(tok.encode(state, add_special_tokens=False)) == length
    for count in (1, 8, 32, 100):
        questions = {f'q{i}': {'type': 'choice',
            'instructions': f'Check {i}: Which word occurs in the state?',
            'criteria': {'hello': 'hello', 'goodbye': 'goodbye'}} for i in range(count)}
        payload = {'model': 'jev-latest', 'state': state, 'questions': questions}
        samples = []
        for repeat in range(4):
            started = time.perf_counter()
            result = json.loads(request('/v1/systemone', payload))
            elapsed = 1000 * (time.perf_counter() - started)
            assert set(result['answers']) == set(questions)
            if repeat:
                samples.append(elapsed)
        timings.append({'state_tokens': length, 'questions': count,
                        'median_ms': statistics.median(samples), 'samples_ms': samples})
        print('TIMING', json.dumps(timings[-1]), flush=True)
        (out / 'summary.json').write_text(json.dumps({'health': health,
            'accuracy': {'correct': sum(r['correct'] for r in rows), 'total': len(rows)},
            'timings': timings}, indent=2))
