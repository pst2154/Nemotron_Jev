"""Evaluate saved randomized states against an external reference service.

Agreement with the reference and correctness against known labels are distinct.
URLs and credentials are provided only through environment variables.
"""
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
saved=json.loads((ROOT/'bf16-unpadded-validation.json').read_text())
cases=next(d['rows'] for d in saved['datasets'] if d['dataset']=='randomized')
questions=json.loads((ROOT/'comparison-inputs.json').read_text())[0]['questions']
headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['REFERENCE_API_KEY']}
rows=[]
for i,case in enumerate(cases):
    payload={'model':os.environ.get('REFERENCE_MODEL','jev-latest'),'state':case['state'],'questions':questions}
    request=urllib.request.Request(os.environ['REFERENCE_URL'],json.dumps(payload).encode(),headers)
    start=time.perf_counter()
    with urllib.request.urlopen(request,timeout=180) as response: result=json.load(response)
    elapsed=(time.perf_counter()-start)*1000
    labels=[result['answers'][str(j)]['choice'] for j in range(100)]
    assert all(x in ('enabled','disabled') for x in labels)
    letters=['A' if x=='enabled' else 'B' for x in labels]
    row={'case':i,'ms':elapsed,'model':result.get('model'),'answers':result['answers'],
         'correct':sum(a==e for a,e in zip(letters,case['expected'])),
         'lightning_agreement':sum(a==b['choice'] for a,b in zip(letters,case['baseline']['answers']))}
    rows.append(row)
    report={'reference':os.environ['REFERENCE_NAME'],'dataset':'randomized bf16 validation states',
            'correct':sum(r['correct'] for r in rows),'total':len(rows)*100,
            'lightning_agreement':sum(r['lightning_agreement'] for r in rows),
            'median_ms':statistics.median(r['ms'] for r in rows),'rows':rows}
    (ROOT/f'reference-{os.environ["REFERENCE_NAME"]}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in row.items() if k!='answers'}),flush=True)
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
