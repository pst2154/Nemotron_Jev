"""Same-node, persistent-connection benchmark; validates every returned decision."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid
import httpx
from tokenizers import Tokenizer
from adapter import Adapter

os.environ['STATE_FIRST'] = '1'
tokenizer = Tokenizer.from_file('/model/tokenizer.json')
client = httpx.Client(base_url='http://127.0.0.1:8300/v1', timeout=120,
                     limits=httpx.Limits(max_connections=128, max_keepalive_connections=128), trust_env=False)


def call(body):
    r = client.post('/chat/completions', json=body)
    r.raise_for_status()
    return r.json()


def state(tokens, nonce):
    facts = '\n'.join(f'Record {i}: {"on" if i%2==0 else "off"}.' for i in range(100))
    content = f'Reference {nonce}.\n'+facts+'\nUnrelated padding: '+'neutral '*tokens
    ids = tokenizer.encode(content, add_special_tokens=False).ids[:tokens]
    value = tokenizer.decode(ids, skip_special_tokens=False)
    assert len(tokenizer.encode(value, add_special_tokens=False).ids) == tokens
    assert 'Record 99: off.' in value
    return value


def main():
    report = {'method':'same-node HTTP with persistent connections, independent constrained one-token questions',
              'model':'cached Lightning 3.5 NVFP4, modified MTP weights unused',
              'config_sha256':hashlib.sha256(Path('/model/config.json').read_bytes()).hexdigest(),
              'scenarios':[]}
    for tokens, count, workers in [(1000,1,1),(8000,1,1),(1000,12,12),
                                   (1000,100,16),(1000,100,32),(1000,100,64),(1000,100,100)]:
        only = os.environ.get('GPU_BENCH_ONLY')
        if only and only != f'{tokens},{count},{workers}':
            continue
        adapter = Adapter('unused','','lightning',workers=workers,call=call)
        questions = {str(i):{'type':'choice','instructions':f'What flag is recorded for Record {i}?',
                             'criteria':{'enabled':'The record says on','disabled':'The record says off'}}
                     for i in range(count)}
        for mode in ['fresh_state','repeated_state']:
            if os.environ.get('GPU_BENCH_CACHE') and mode != os.environ['GPU_BENCH_CACHE']:
                continue
            rows=[]
            shared=state(tokens,'warmup')
            warm = adapter.evaluate({'model':'lightning-systemone','state':shared,'questions':questions})
            for _ in range(10):
                text = state(tokens,uuid.uuid4().hex) if mode=='fresh_state' else shared
                start=time.perf_counter()
                try:
                    result=adapter.evaluate({'model':'lightning-systemone','state':text,'questions':questions})
                    correct=sum(a['choice']==('enabled' if int(k)%2==0 else 'disabled') for k,a in result['answers'].items())
                    row={'ms':(time.perf_counter()-start)*1000,'correct':correct,'usage':result['usage']}
                except Exception as exc:
                    row={'ms':(time.perf_counter()-start)*1000,'error':type(exc).__name__+': '+str(exc)[:150]}
                rows.append(row)
            good=[r for r in rows if 'error' not in r]
            times=sorted(r['ms'] for r in good)
            summary={'state_tokens':tokens,'questions':count,'workers':workers,'cache_case':mode,
                     'trials':len(rows),'successes':len(good),
                     'p50_ms':round(statistics.median(times),2) if times else None,
                     'p95_ms':round(times[math.ceil(.95*len(times))-1],2) if times else None,
                     'correct_answers':sum(r['correct'] for r in good),'total_answers':count*len(rows),
                     'rows':rows}
            report['scenarios'].append(summary)
            Path(os.environ.get('GPU_BENCH_OUTPUT', '/work/gpu-results.json')).write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
        adapter.pool.shutdown()


if __name__ == '__main__':
    main()
