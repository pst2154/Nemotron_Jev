"""Identical 1,000-Lightning-token inputs, one System One request per trial."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
fixture = ROOT / 'comparison-inputs.json'
if not fixture.exists():
    raise FileNotFoundError('The bundled comparison-inputs.json fixture is required')
fixtures = json.loads(fixture.read_text())
backend = os.environ['BENCH_BACKEND']
headers = {'Content-Type':'application/json'}
if os.environ.get('BENCH_API_KEY'):
    headers['Authorization'] = 'Bearer '+os.environ['BENCH_API_KEY']
report = {'backend':backend,'state_tokens':1000,'tokenizer':'Lightning tokenizer; identical text for both backends',
          'method':'client end-to-end HTTP, fresh state per trial, one warmup excluded, sequential outer requests', 'scenarios':[]}
for count in map(int, os.environ.get('BENCH_COUNTS','1,12,100').split(',')):
    rows=[]
    for i, entry in enumerate(fixtures):
        payload={'model':'jev-latest','state':entry['state'], 'questions':dict(list(entry['questions'].items())[:count])}
        start=time.perf_counter()
        try:
            request=urllib.request.Request(os.environ['BENCH_URL'],json.dumps(payload).encode(),headers)
            with urllib.request.urlopen(request,timeout=180) as response:
                result=json.load(response)
            elapsed=(time.perf_counter()-start)*1000
            answers=result['answers']
            correct=sum(answers.get(str(j),{}).get('choice') == ('enabled' if j%2==0 else 'disabled') for j in range(count))
            row={'ms':elapsed,'correct':correct,'returned':len(answers),'model':result.get('model'),'answers':answers}
        except urllib.error.HTTPError as exc:
            if exc.code in (401,403):
                raise SystemExit(f'Authentication failed with HTTP {exc.code}; no further requests sent')
            row={'ms':(time.perf_counter()-start)*1000,'error':f'HTTP {exc.code}'}
        except Exception as exc:
            row={'ms':(time.perf_counter()-start)*1000,'error':type(exc).__name__}
        if i: rows.append(row)
        else: warmup = row
        print(json.dumps({'backend':backend,'questions':count,'trial':i,**{k:v for k,v in row.items() if k!='answers'}}),flush=True)
    times=sorted(r['ms'] for r in rows if 'error' not in r)
    summary={'questions':count,'trials':len(rows),'successes':len(times),'p50_ms':round(statistics.median(times),2) if times else None,
             'p95_ms':round(max(times),2) if times else None,'correct':sum(r.get('correct',0) for r in rows),'total':count*len(rows),'warmup':warmup,'rows':rows}
    report['scenarios'].append(summary)
    (ROOT/f'comparison-{backend}{os.environ.get("BENCH_SUFFIX", "")}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
