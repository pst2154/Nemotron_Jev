import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from scoring import evaluate

checkpoint=os.environ.get('CHECKPOINT_DIR', '/models/checkpoint')
print('Loading Nemotron Labs Diffusion 14B in BF16',flush=True)
tokenizer=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
model=AutoModel.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16).to('cuda').eval()
lock=threading.Lock()

def generate(messages,mode='dlm',limit=128):
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=tokenizer(prompt,return_tensors='pt').input_ids.cuda()
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.inference_mode():
        if mode=='ar': out,nfe=model.ar_generate(ids,max_new_tokens=limit)
        elif mode=='linear_spec': out,nfe=model.linear_spec_generate(ids,max_new_tokens=limit,block_length=32,eos_token_id=tokenizer.eos_token_id)
        else: out,nfe=model.generate(ids,max_new_tokens=limit,block_length=32,threshold=.9,eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    tokens=out[0,ids.shape[1]:]
    return {'text':tokenizer.decode(tokens,skip_special_tokens=True),'mode':mode,'input_tokens':ids.shape[1],'output_tokens':len(tokens),'seconds':elapsed,'tokens_per_second':len(tokens)/elapsed,'nfe':int(nfe)}

class Handler(BaseHTTPRequestHandler):
    def send(self,status,value):
        data=json.dumps(value).encode(); self.send_response(status)
        self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path.split('?')[0] in ('/', '/explorer'):
            data=Path(__file__).with_name('explorer.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.end_headers(); self.wfile.write(data)
            return
        if self.path=='/health':return self.send(200,{'ready':True,'model':'Nemotron-Labs-Diffusion-14B','modes':['dlm','ar','linear_spec']})
        self.send(404,{'error':'Not found'})
    def do_POST(self):
        if self.path not in ('/v1/chat/completions', '/v1/systemone'): return self.send(404,{'error':'Not found'})
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=2000000:raise ValueError('Invalid request size')
            body=json.loads(self.rfile.read(length)); mode=body.get('mode','dlm')
            if self.path == '/v1/systemone':
                with lock: result = evaluate(model, tokenizer, body)
                return self.send(200, result)
            if mode not in ('dlm','ar','linear_spec'):raise ValueError('Unsupported mode')
            limit=int(body.get('max_tokens',128))
            if not 1<=limit<=1024:raise ValueError('max_tokens must be 1..1024')
            if mode == 'dlm': limit = ((limit + 31) // 32) * 32
            if body.get('stream'):raise ValueError('Streaming is not supported')
            with lock: result=generate(body['messages'],mode,limit)
            self.send(200,{'model':'Nemotron-Labs-Diffusion-14B','choices':[{'index':0,'message':{'role':'assistant','content':result['text']}}],'metrics':result})
        except (ValueError,KeyError,TypeError) as exc:self.send(400,{'error':{'message':str(exc),'type':'invalid_request_error'}})
        except Exception as exc:self.send(500,{'error':str(exc)})

port=int(os.environ.get('PORT','8770'))
print(f'Serving UI and API on port {port}',flush=True)
ThreadingHTTPServer(('0.0.0.0',port),Handler).serve_forever()
