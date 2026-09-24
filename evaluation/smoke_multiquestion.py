"""Synthetic API smoke/latency test; not an accuracy benchmark."""
import argparse, json, statistics, time, urllib.request
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--url',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
results=[]
for count in (1,10,30,100):
    times=[]
    for repeat in range(4):
        body={'model':'jev-latest','state':'The available color is blue. Red is unavailable.',
              'questions':{f'q{i}':{'type':'choice','instructions':'Which color is available?',
                'criteria':{'blue':'Blue','red':'Red'}} for i in range(count)}}
        request=urllib.request.Request(a.url+'/v1/systemone',json.dumps(body).encode(),{'Content-Type':'application/json'})
        start=time.perf_counter()
        with urllib.request.urlopen(request,timeout=180) as response: result=json.load(response)
        elapsed=(time.perf_counter()-start)*1000
        assert result['model']=='Nemotron-Labs-Diffusion-8B'
        assert set(result['answers'])==set(body['questions'])
        assert all(answer['choice']=='blue' for answer in result['answers'].values())
        if repeat:times.append(elapsed)
    results.append({'questions':count,'median_ms':statistics.median(times),'samples_ms':times,'all_correct':True})
a.output.write_text(json.dumps({'description':'Repeated identical short questions; functionality smoke, not representative task throughput. One warmup and three measured calls per size.','results':results},indent=2))
print(json.dumps(results),flush=True)
