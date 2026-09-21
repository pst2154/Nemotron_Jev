"""Same-server selected-label logprobs vs constrained one-token baseline.
Both paths generate one token internally; this is NOT a zero-generation benchmark.
"""
import concurrent.futures
import json
import math
from pathlib import Path
import statistics
import time
import httpx
from tokenizers import Tokenizer
from adapter import build_request

c=httpx.Client(base_url='http://127.0.0.1:8300',timeout=120,limits=httpx.Limits(max_connections=100,max_keepalive_connections=100))
t=Tokenizer.from_file('/model/tokenizer.json')
ids=[t.encode(x,add_special_tokens=False).ids for x in ['A','B']]
assert all(len(x)==1 for x in ids)
fixtures=json.loads(Path('/work/comparison-inputs.json').read_text())
report={'method':'alternating order; identical prompts; same server; 10 trials; both paths max_tokens=1; no zero-output support', 'scenarios':[]}
pool=concurrent.futures.ThreadPoolExecutor(max_workers=100)
def one(args):
    body,mode=args
    body=dict(body)
    if mode=='selected':
        body.pop('structured_outputs');body.pop('top_logprobs')
        body['logprob_token_ids']=[x[0] for x in ids]
    r=c.post('/v1/chat/completions',json=body);r.raise_for_status()
    out=r.json(); logs={p['token']:p['logprob'] for p in out['choices'][0]['logprobs']['content'][0]['top_logprobs']}
    vals=[logs[x] for x in ['A','B']]
    assert all(math.isfinite(x) and x > -9990 for x in vals)
    weights=[math.exp(x-max(vals)) for x in vals]
    return {'p_a':weights[0]/sum(weights),'choice':'A' if vals[0]>=vals[1] else 'B'}
for count in (1,12,100):
    rows=[]
    for trial, fixture in enumerate(fixtures):
        bodies=[build_request('lightning',fixture['state'],q,True) for q in list(fixture['questions'].values())[:count]]
        paired={}
        for mode in (['baseline','selected'] if trial%2 else ['selected','baseline']):
            start=time.perf_counter(); answers=list(pool.map(one,[(b,mode) for b in bodies])); elapsed=(time.perf_counter()-start)*1000
            paired[mode]={'ms':elapsed,'correct':sum(a['choice']==('A' if i%2==0 else 'B') for i,a in enumerate(answers)),'answers':answers}
        if trial:rows.append(paired)
    summary={'questions':count,'trials':len(rows),'rows':rows}
    for mode in ['baseline','selected']:
        times=sorted(r[mode]['ms'] for r in rows)
        summary[mode]={'p50_ms':round(statistics.median(times),2),'p95_ms':round(max(times),2),'correct':sum(r[mode]['correct'] for r in rows),'total':count*len(rows)}
    pairs=[(a,b) for r in rows for a,b in zip(r['baseline']['answers'],r['selected']['answers'])]
    summary['agreement']=sum(a['choice']==b['choice'] for a,b in pairs)
    summary['mean_abs_probability_difference']=sum(abs(a['p_a']-b['p_a']) for a,b in pairs)/len(pairs)
    report['scenarios'].append(summary)
    Path('/work/score-results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
pool.shutdown()
