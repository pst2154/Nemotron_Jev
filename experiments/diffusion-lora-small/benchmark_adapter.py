"""Paired warm adapter-on/off latency; same loaded model, alternating order."""
import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--adapter',required=True)
p.add_argument('--app',required=True)
p.add_argument('--harness',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
sys.path[:0]=[a.app,a.harness]
from scoring import evaluate
from jevbench.tasks import load_jsonl
from jevbench.adapters.base import build_question
torch.set_num_threads(8)
tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,trust_remote_code=True)
base=AutoModel.from_pretrained(a.checkpoint,trust_remote_code=True,dtype=torch.bfloat16,
                             attn_implementation='sdpa').to('cuda')
model=PeftModel.from_pretrained(base,a.adapter).eval()
tasks=[task for tier in ('easy','original','hard') for task in
       load_jsonl(str(Path(a.harness)/'datasets/public'/f'{tier}.jsonl'))]
selected={}
for task in tasks:
    selected.setdefault(task.family,task)
tasks=list(selected.values())
random.Random(17).shuffle(tasks)
def call(task,enabled):
    payload={'model':'Nemotron-Labs-Diffusion-14B','state':task.state,
             'questions':{'decision':build_question(task)}}
    ctx=contextlib.nullcontext() if enabled else model.disable_adapter()
    with ctx,torch.inference_mode():
        torch.cuda.synchronize();start=time.perf_counter()
        result=evaluate(model.get_base_model(),tokenizer,payload,position_ordering=True)
        torch.cuda.synchronize()
        return result,(time.perf_counter()-start)*1000
with Path(a.output).open('x') as stream:
    for index,task in enumerate(tasks):
        for _ in range(2):
            for enabled in (False,True):call(task,enabled)
        for repeat in range(3):
            order=(False,True) if (index+repeat)%2==0 else (True,False)
            for enabled in order:
                result,ms=call(task,enabled)
                record={'id':task.id,'family':task.family,'repeat':repeat,
                        'adapter_enabled':enabled,'latency_ms':ms,'answer':result['answers']['decision']}
                stream.write(json.dumps(record)+'\n');stream.flush()
        print('PAIRED_LATENCY',index+1,len(tasks),flush=True)
print('PAIRED_LATENCY_COMPLETE',flush=True)
