"""Frozen paired accuracy/latency benchmark. Endpoints and secrets stay in env."""
import argparse
import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from tokenizers import Tokenizer


def encoded(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def make_fixtures(args):
    tok = Tokenizer.from_file(args.tokenizer)
    tokens = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    rng = random.Random(20260922)
    filler = 'Archive note: an unrelated office inventory contains paper, folders, chairs, and lamps. No target record is described here.\n'

    def pad(text, length, position, nonce):
        block = f'Trial {nonce}.\nRelevant evidence:\n{text}\nEnd of relevant evidence.\n'
        n = max(0, length - tokens(block))
        ids = tok.encode(filler * (length // 10 + 1), add_special_tokens=False).ids[:n]
        split = {'start': 0, 'middle': n // 2, 'end': n}[position]
        state = tok.decode(ids[:split]) + block + tok.decode(ids[split:])
        # Boundary merges can change count. Adjust neutral suffix only, never evidence.
        for _ in range(20):
            delta = length - tokens(state)
            if delta == 0:
                return state
            if delta > 0:
                state += ' x' * delta
            else:
                # Rebuild with a shorter padding stream, preserving evidence.
                n += delta
                ids = ids[:n]
                split = {'start': 0, 'middle': n // 2, 'end': n}[position]
                state = tok.decode(ids[:split]) + block + tok.decode(ids[split:])
        raise ValueError('Could not construct exact token length')

    fixtures = []
    if args.groups:
        diagnostic = json.loads(Path(args.groups).read_text())
        for length in [1000, 4000, 12000]:
            for count in [1, 6, 12, 24]:
                for trial in range(6):
                    position = ['start', 'middle', 'end'][trial % 3]
                    order = list(diagnostic['questions']); rng.shuffle(order)
                    selected = order[:count]
                    questions = {k: diagnostic['questions'][k] for k in selected}
                    state = pad(json.dumps(diagnostic['state']), length, position, f'group-{length}-{count}-{trial}')
                    fixtures.append({'id': f'group-{length}-{count}-{trial}', 'suite': 'semantic_grouped24',
                                     'position': position, 'state_tokens': tokens(state), 'questions_count': count,
                                     'input': {'state': state, 'questions': questions},
                                     'gold': {k: diagnostic['cases'][int(k)]['expected'] for k in selected}})
        Path(args.fixtures).write_text(json.dumps(fixtures, indent=2))
        print(json.dumps({'fixtures': len(fixtures), 'decisions_per_backend': sum(len(x['gold']) for x in fixtures)}))
        return
    for length in [256, 1000, 4000, 8000, 12000]:
        for count in [1, 10, 30, 100]:
            if length == 256 and count > 10:
                continue
            for trial in range(6):
                position = ['start', 'middle', 'end'][trial % 3]
                flags = [rng.choice(['on', 'off']) for _ in range(count)]
                order = list(range(count)); rng.shuffle(order)
                state = pad('\n'.join(f'Record {j}: {flags[j]}.' for j in order), length, position, f'{length}-{count}-{trial}')
                questions = {str(j): {'type': 'choice', 'instructions': f'What flag is recorded for Record {j}?',
                                     'criteria': {'enabled': 'The record says on', 'disabled': 'The record says off'}} for j in range(count)}
                fixtures.append({'id': f'lookup-{length}-{count}-{trial}', 'suite': 'random_lookup',
                                 'position': position, 'state_tokens': tokens(state), 'questions_count': count,
                                 'input': {'state': state, 'questions': questions},
                                 'gold': {str(j): 'enabled' if flags[j] == 'on' else 'disabled' for j in range(count)}})
    semantic = [json.loads(line) for line in Path(args.semantic).read_text().splitlines()]
    for row in semantic:
        inp = row['input']; name = next(iter(inp['questions']))
        fixtures.append({'id': row['id'], 'suite': 'semantic324', 'position': 'native',
                         'state_tokens': tokens(encoded(inp['state'])), 'questions_count': len(inp['questions']),
                         'input': inp, 'gold': {name: row['reference']['target']}})
    # Fixed balanced subset: eight questions per output type, no selection by model result.
    subset = []
    for kind in ['choice', 'noul', 'score']:
        candidates = [r for r in semantic if next(iter(r['input']['questions'].values()))['type'] == kind]
        subset.extend(random.Random(2718).sample(candidates, 8))
    for length in [4000, 12000]:
        for idx, row in enumerate(subset):
            position = ['start', 'middle', 'end'][idx % 3]
            inp = row['input']; name = next(iter(inp['questions']))
            state = pad(encoded(inp['state']), length, position, f'semantic-{length}-{idx}')
            fixtures.append({'id': f'long-{length}-{row["id"]}', 'suite': 'semantic_long',
                             'position': position, 'state_tokens': tokens(state), 'questions_count': 1,
                             'input': {'state': state, 'questions': inp['questions']},
                             'gold': {name: row['reference']['target']}})
    Path(args.fixtures).write_text(json.dumps(fixtures, indent=2))
    print(json.dumps({'fixtures': len(fixtures), 'decisions_per_backend': sum(len(x['gold']) for x in fixtures)}))


def score(fixture, data):
    answers = data['answers']; decisions = []
    for key, gold in fixture['gold'].items():
        q = fixture['input']['questions'][key]; kind = q['type']; a = answers.get(key, {})
        pred = None
        if kind == 'choice':
            pred = a.get('choice')
        elif kind == 'noul':
            p = a.get('noul')
            if isinstance(p, (float, int)) and math.isfinite(p) and 0 <= p <= 1:
                pred = p > 0.5
        elif kind == 'score':
            probs = a.get('probabilities', {})
            if probs:
                pred = int(max(probs, key=probs.get))
        decisions.append({'key': key, 'type': kind, 'gold': gold, 'prediction': pred,
                          'correct': pred is not None and pred == gold, 'answer': a})
    return decisions


def run(args):
    fixture_bytes = Path(args.fixtures).read_bytes()
    fixtures = json.loads(fixture_bytes)
    key = os.environ.get('DIFFUSION_API_KEY')
    if not key:
        cfg = configparser.ConfigParser(); cfg.read(Path.home()/'.ngc/config')
        key = next(cfg[s]['apikey'] for s in cfg.sections() if cfg[s].get('apikey'))
    clients = {'lightning_lora': httpx.Client(timeout=120),
               'diffusiongemma': httpx.Client(timeout=120, headers={'Authorization': 'Bearer '+key})}
    urls = {'lightning_lora': os.environ['LIGHTNING_URL'], 'diffusiongemma': os.environ['DIFFUSION_URL']}
    output = Path(args.output)
    done = set()
    if output.exists():
        done = {(r['id'], r['backend']) for r in map(json.loads, output.read_text().splitlines())}

    def call(backend, fixture):
        payload = {'model': 'jev-latest', **fixture['input']}
        row = {k: v for k, v in fixture.items() if k not in ['input', 'gold']}
        row.update(backend=backend, fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
                   payload_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
                   total=len(fixture['gold']))
        start = time.perf_counter()
        try:
            response = clients[backend].post(urls[backend], json=payload)
            row['http_status'] = response.status_code
            response.raise_for_status()
            data = response.json()
            row['decisions'] = score(fixture, data)
            row['correct'] = sum(d['correct'] for d in row['decisions'])
            row['returned'] = len(data['answers'])
            row['model_alias'] = data.get('model')
            row['usage'] = data.get('usage')
        except Exception as exc:
            # Never persist exception strings: HTTP exceptions can contain private URLs.
            row['error'] = type(exc).__name__
            row['correct'] = 0
        row['ms'] = (time.perf_counter()-start)*1000
        return row

    # One unmeasured warmup each. No concurrent outer requests on a given backend.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda b: call(b, fixtures[0]), clients))
        with output.open('a') as stream:
            for i, fixture in enumerate(fixtures):
                futures = [pool.submit(call, b, fixture) for b in clients if (fixture['id'], b) not in done]
                for future in futures:
                    row = future.result(); stream.write(json.dumps(row)+'\n'); stream.flush()
                    print(json.dumps({k: row[k] for k in ['id', 'backend', 'ms', 'correct', 'total']} | {'error': row.get('error')}), flush=True)
    for client in clients.values(): client.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build', 'run'])
    p.add_argument('--fixtures', required=True)
    p.add_argument('--tokenizer')
    p.add_argument('--semantic')
    p.add_argument('--groups', help='Build separate multi-question authored diagnostic fixtures')
    p.add_argument('--output')
    args = p.parse_args()
    make_fixtures(args) if args.action == 'build' else run(args)
