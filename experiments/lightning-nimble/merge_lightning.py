"""Merge a completed adapter into a new BF16 checkpoint, never into NVFP4."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from nimble.training.schema_data import prepare_data

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
run=Path(args.run);out=Path(args.output)
contract=json.loads((run/'contract.json').read_text())
assert not contract['smoke_only'] and contract['heldout_evaluated']
before=json.loads((run/'before.json').read_text());after=json.loads((run/'after.json').read_text())
if sum(x['correct'] for x in after)<=sum(x['correct'] for x in before):
    raise SystemExit('No held-out improvement; keep adapter for analysis, do not promote a merged serving checkpoint.')
out.mkdir(parents=True,exist_ok=False)
tokenizer=AutoTokenizer.from_pretrained(run/'adapter')
_,heldout,_=prepare_data('data','data/eval.jsonl',tokenizer,2048,contract['base'],contract['revision'])
base=AutoModelForCausalLM.from_pretrained(contract['base'],dtype=torch.bfloat16,device_map='cuda',trust_remote_code=False,attn_implementation='sdpa')
base.set_experts_implementation(contract['experts'])
model=PeftModel.from_pretrained(base,run/'adapter').eval()
model=model.merge_and_unload(safe_merge=True).eval()
results=[]
with torch.inference_mode():
 for row in heldout:
    with torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=model.model(input_ids=torch.tensor([row['input_ids']],device='cuda'),use_cache=False).last_hidden_state[:,-1,:]
        weights=model.get_output_embeddings().weight.index_select(0,torch.tensor(row['candidate_ids'],device='cuda'))
    logits=hidden.float()@weights.float().T
    pred=logits.argmax(-1).item();results.append({'id':row['id'],'prediction':pred,'label':row['labels'],'correct':pred==row['labels']})
    if len(results)%50==0:print(json.dumps({'merged_evaluated':len(results)}),flush=True)
model.save_pretrained(out,max_shard_size='5GB');tokenizer.save_pretrained(out)
report={'correct':sum(x['correct'] for x in results),'total':len(results),'unmerged_correct':sum(x['correct'] for x in after),'changed_predictions':sum(a['prediction']!=b['prediction'] for a,b in zip(after,results)),'base_precision':'BF16','quantized':False,'rows':results}
(out/'merge_evaluation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
print('MERGE_COMPLETE',flush=True)
