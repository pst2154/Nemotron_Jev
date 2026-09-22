"""Single-GPU Lightning LoRA using Nimble's audited data and candidate loss.

Smoke mode never evaluates held-out answers. Base checkpoint is read-only.
"""
import argparse, hashlib, json, time, random, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model
from nimble.training.schema_data import prepare_data

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/model')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--full-train',action='store_true')
    p.add_argument('--experts',choices=['eager','grouped_mm'],default='grouped_mm')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    set_seed(17)
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=False)
    revision='local-config-sha256:'+hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest()
    training,heldout,audit=prepare_data('data','data/eval.jsonl',tokenizer,2048,args.model,revision)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'stage':'data_ready',**audit}),flush=True)
    started=time.perf_counter()
    model,loading=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    serious={k:v for k,v in loading.items() if v and k!='unexpected_keys'}
    unexpected=[k for k in loading.get('unexpected_keys',[]) if not k.startswith('mtp.')]
    if serious or unexpected:raise RuntimeError(f'Checkpoint mismatch: {serious}, unexpected={unexpected}')
    model.config.use_cache=False
    model.set_experts_implementation(args.experts)
    # Mamba out_proj is consumed directly by fused kernels; do not wrap it.
    targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
             and n.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','in_proj','up_proj','down_proj'}]
    assert targets
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=0)
    print(json.dumps({'stage':'loaded','seconds':time.perf_counter()-started,'trainable_parameters':sum(p.numel() for p in parameters),'targets':len(targets),'allocated_gib':torch.cuda.memory_allocated()/2**30}),flush=True)
    # Probe longest input as well as ordinary examples, not just tiny easy fits.
    ordered=sorted(training,key=lambda r:len(r['input_ids']),reverse=True)
    if args.full_train:
        ordered=list(training);random.Random(17).shuffle(ordered)
        args.steps=math.ceil(len(ordered)/args.accumulation)
    def evaluate(name):
        model.eval();results=[]
        with torch.inference_mode():
            for row in heldout:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    base=model.get_base_model()
                    hidden=base.model(input_ids=torch.tensor([row['input_ids']],device='cuda'),use_cache=False).last_hidden_state[:,-1,:]
                    head=base.get_output_embeddings().weight.index_select(0,torch.tensor(row['candidate_ids'],device='cuda'))
                logits=hidden.float() @ head.float().T
                prediction=logits.argmax(-1).item()
                results.append({'id':row['id'],'prediction':prediction,'label':row['labels'],'correct':prediction==row['labels'],'logits':logits[0].tolist()})
                if len(results)%50==0:print(json.dumps({'stage':name,'evaluated':len(results)}),flush=True)
        (out/(name+'.json')).write_text(json.dumps(results,indent=2))
        print(json.dumps({'stage':name,'correct':sum(r['correct'] for r in results),'total':len(results)}),flush=True)
        model.train()
    if args.full_train:evaluate('before')
    metrics=[];torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        tick=time.perf_counter(); losses=[]
        optimizer.zero_grad(set_to_none=True)
        if args.full_train:
            schedule_steps=math.ceil(len(ordered)/args.accumulation)*3
            warmup=math.ceil(schedule_steps*.1)
            scale=min((step+1)/warmup,(schedule_steps-step)/(schedule_steps-warmup))
            for group in optimizer.param_groups:group['lr']=5e-5*scale
        for micro in range(args.accumulation):
            index=step*args.accumulation+micro
            if args.full_train and index>=len(ordered):break
            row=ordered[index%len(ordered)]
            tokens=torch.tensor([row['input_ids']],device='cuda')
            base=model.get_base_model()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=base.model(input_ids=tokens,use_cache=False).last_hidden_state[:,-1,:]
                head=base.get_output_embeddings().weight
                candidates=head.index_select(0,torch.tensor(row['candidate_ids'],device='cuda'))
            logits=hidden.float() @ candidates.float().T
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor([row['labels']],device='cuda'))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            divisor=min(args.accumulation,len(ordered)-step*args.accumulation) if args.full_train else args.accumulation
            (loss/divisor).backward();losses.append(loss.item())
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.0)
        if not torch.isfinite(norm) or norm.item()==0:raise RuntimeError(f'Invalid gradient norm: {norm}')
        optimizer.step();torch.cuda.synchronize()
        record={'step':step+1,'seconds':time.perf_counter()-tick,'loss':sum(losses)/len(losses),'grad_norm':norm.item(),'peak_gib':torch.cuda.max_memory_allocated()/2**30,'tokens':len(row['input_ids'])}
        metrics.append(record);print(json.dumps(record),flush=True)
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        if args.full_train and (step+1)%25==0:
            model.save_pretrained(out/f'adapter-step-{step+1}')
            (out/'progress.json').write_text(json.dumps({'step':step+1,'total_steps':args.steps}))
    model.save_pretrained(out/'adapter');tokenizer.save_pretrained(out/'adapter')
    if args.full_train:evaluate('after')
    (out/'contract.json').write_text(json.dumps({'base':args.model,'revision':revision,'targets':targets,'rank':16,'candidate_loss':'FP32 candidate-only cross entropy','steps':args.steps,'accumulation':args.accumulation,'smoke_only':not args.full_train,'heldout_evaluated':args.full_train,'experts':args.experts},indent=2))
    print('TRAIN_COMPLETE' if args.full_train else 'SMOKE_COMPLETE',flush=True)

if __name__=='__main__': main()
