"""Evaluate frozen cases with deterministic scoring; endpoints are never saved."""
import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--url', required=True)
p.add_argument('--output', default='evaluation/results')
args = p.parse_args()
source = Path(__file__).with_name('cases.json').read_bytes()
cases = json.loads(source)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
rows = []
for case in cases:
    question = dict(case['question'])
    if question['type'] == 'boolean': question['type'] = 'noul'
    payload = {'model':'jev-latest', 'state':case['state'], 'questions':{'answer':question}}
    row = {k:case[k] for k in ('id','variant','gold')}
    row['type'] = question['type']
    started = time.perf_counter()
    try:
        req = urllib.request.Request(args.url.rstrip('/')+'/v1/systemone',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response: result=json.load(response)
        answer = result['answers']['answer']
        if question['type'] == 'choice': prediction = answer['choice']
        elif question['type'] == 'noul': prediction = answer['noul'] >= .5
        else: prediction = int(max(answer['probabilities'], key=answer['probabilities'].get))
        row.update(answer=answer, predicted=prediction, correct=prediction==case['gold'])
        if question['type']=='noul': row['brier']=(answer['noul']-float(case['gold']))**2
        if question['type']=='score': row['score_absolute_error']=abs(answer['score']-case['gold'])
    except Exception as exc:
        # Avoid persisting hostnames, credentials or transport URLs in public artifacts.
        row.update(error=type(exc).__name__, correct=False)
    row['elapsed_ms'] = 1000*(time.perf_counter()-started)
    rows.append(row)
    print(case['id'], row['correct'], flush=True)
    (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
summary = {'cases_sha256':hashlib.sha256(source).hexdigest(), 'total':len(rows),
           'correct':sum(r['correct'] for r in rows), 'errors':sum('error' in r for r in rows),
           'median_ms':statistics.median(r['elapsed_ms'] for r in rows)}
for field in ('type','variant'):
    summary['by_'+field] = {}
    for value in sorted({r[field] for r in rows}):
        group = [r for r in rows if r[field]==value]
        summary['by_'+field][value] = {'correct':sum(r['correct'] for r in group),'total':len(group)}
for metric in ('brier','score_absolute_error'):
    values = [r[metric] for r in rows if metric in r]
    summary[metric] = statistics.mean(values) if values else None
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
