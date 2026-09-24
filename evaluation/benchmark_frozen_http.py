"""Evaluate an existing SystemOne endpoint on frozen questions, without editing inputs."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    blob = args.data.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == 'f9229a63d2418a3cd8b537791d6e9bacaf1dc045aafd8b6feeb66bb91e510877'
    rows = [json.loads(line) for line in blob.splitlines()]
    def call(body):
        request = urllib.request.Request(args.url+'/v1/systemone',
            json.dumps(dict(body, model='jev-latest')).encode(),
            {'Content-Type': 'application/json'})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        return result, time.perf_counter()-start
    with urllib.request.urlopen(args.url+'/health') as response:
        health = json.load(response)
    assert health['model'] == 'Nemotron-Labs-Diffusion-8B'
    assert health['cache_policy'] == 'isolated_request', 'Avoid cross-request cache bias'
    for _ in range(3):
        call(rows[0]['input'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'health.json').write_text(json.dumps(health, indent=2))
    with (args.output/'predictions.jsonl').open('w') as stream:
        for i,row in enumerate(rows):
            result, seconds = call(row['input'])
            assert result['model'] == 'Nemotron-Labs-Diffusion-8B'
            answer = result['answers']['decision']
            probs = ({'false':1-answer['noul'],'true':answer['noul']}
                     if row['kind']=='noul' else answer['probabilities'])
            stream.write(json.dumps({'id':row['id'],'probabilities':probs,
                'seconds':seconds,'engine_seconds':result['metrics']['engine_seconds'],
                'input_tokens':result['usage']['input_tokens']})+'\n')
            stream.flush()
            if (i+1)%100==0: print(json.dumps({'completed':i+1,'total':len(rows)}),flush=True)

if __name__ == '__main__':
    main()
