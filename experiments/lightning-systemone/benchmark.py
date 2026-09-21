"""Reproducible HTTP latency/throughput screening, not GPU-only performance."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.request
from tokenizers import Tokenizer

ROUTE = {'type': 'choice', 'instructions': 'Which team should handle the current ticket?',
         'criteria': {'billing':'Payments, invoices or refunds', 'technical':'Application bugs or outages',
                      'sales':'Pricing, upgrades or new purchases'}}
QUESTIONS = [
    ROUTE,
    {'type':'noul','instructions':'Does the current ticket explicitly request a refund?'},
    {'type':'noul','instructions':'Does the current ticket explicitly describe an application crash?'},
    {'type':'noul','instructions':'Does the current ticket explicitly mention a duplicate charge?'},
    {'type':'noul','instructions':'Does the current ticket explicitly ask for a human agent?'},
    {'type':'noul','instructions':'Does the current ticket explicitly state that all users are blocked?'},
    {'type':'noul','instructions':'Does the current ticket explicitly mention an invoice?'},
    {'type':'noul','instructions':'Does the current ticket explicitly ask to buy a new subscription?'},
    {'type':'noul','instructions':'Does the current ticket explicitly request immediate action?'},
    {'type':'choice','instructions':'What does the customer explicitly request?',
     'criteria':{'refund':'Money back','repair':'Fix a broken application','purchase':'Buy something','other':'None of these'}},
    {'type':'score','instructions':'How frustrated is the customer in the current ticket?',
     'criteria':['Calm, factual request','Frustrated but civil','Abusive or threatening']},
    {'type':'score','instructions':'What scope of service outage is explicitly stated?',
     'criteria':['No service outage stated','Some users blocked','All users blocked']},
]


def call(url, payload):
    headers = {'Content-Type':'application/json'}
    if os.environ.get('SYSTEMONE_API_KEY'):
        headers['Authorization'] = 'Bearer '+os.environ['SYSTEMONE_API_KEY']
    t = time.perf_counter()
    req = urllib.request.Request(url.rstrip('/')+'/v1/systemone', data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            value = json.load(r)
        return {'ms':(time.perf_counter()-t)*1000, 'ok':True, 'response':value}
    except Exception as exc:
        return {'ms':(time.perf_counter()-t)*1000, 'ok':False, 'error_type':type(exc).__name__}


def state_with_tokens(tokenizer, count, serial):
    ticket = f'Current ticket {serial}: My invoice was charged twice. Please refund the duplicate charge.'
    filler = 'Archived unrelated note: the office has chairs and desks. '
    text = ticket+'\nBackground archive, not the current ticket:\n'+filler*count
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:count]
    state = tokenizer.decode(ids, skip_special_tokens=False)
    if len(tokenizer.encode(state, add_special_tokens=False).ids) != count:
        raise RuntimeError('State token round-trip did not match requested count')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8795')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', default='results.json')
    parser.add_argument('--requests', type=int, default=12)
    args = parser.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    report = {'date':'2026-09-21', 'model':'Nemotron 3.5 Lightning 30B-A3B',
              'method':'HTTP end-to-end; fixed closed-loop client concurrency; no retries; hosted backend hardware unknown',
              'upstream_concurrency':8, 'scenarios':[]}
    for tokens, n_questions in [(64,1),(1000,1),(8000,1),(1000,12)]:
        for concurrency in [1,4,8]:
            def payload(i):
                return {'model':'lightning-systemone','state':state_with_tokens(tok,tokens,i),
                        'questions':{f'q{j}':QUESTIONS[j] for j in range(n_questions)}}
            warmup = call(args.url,payload(999))
            payloads = [payload(i) for i in range(args.requests)]
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(lambda p:call(args.url,p),payloads))
            duration = time.perf_counter()-start
            good = [r for r in results if r['ok']]
            latencies = sorted(r['ms'] for r in good)
            scenario = {'state_tokens':tokens,'questions':n_questions,'concurrency':concurrency,
                        'attempted':len(results),'successful':len(good),'warmup_ok':warmup['ok'],
                        'p50_ms':round(statistics.median(latencies),1) if good else None,
                        'p95_ms':round(latencies[math.ceil(.95*len(latencies))-1],1) if good else None,
                        'requests_per_second':round(len(good)/duration,3),
                        'questions_per_second':round(len(good)*n_questions/duration,3),
                        'duration_seconds':duration,'results':results}
            report['scenarios'].append(scenario)
            Path(args.output).write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in scenario.items() if k!='results'}),flush=True)


if __name__ == '__main__':
    main()
