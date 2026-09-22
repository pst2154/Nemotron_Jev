"""Repeat identical-token decisions to separate run variance from cache effects."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from nimble.scoring.parallel_schema import prepare_prompts
from shared_native import NativeScorer, load_native_model

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--cases',required=True)
p.add_argument('--output',required=True);args=p.parse_args()
torch.set_num_threads(8)
tokenizer=AutoTokenizer.from_pretrained(args.model)
scorer=NativeScorer(load_native_model(args.model))
data=json.loads(Path(args.cases).read_text());report=[]
for offset in (0,8,16):
    keys=list(data['questions'])[offset:offset+8]
    schema={k:{'type':'enum','description':data['questions'][k]['instructions'],
               'choices':list(data['questions'][k]['criteria']),
               'choice_descriptions':data['questions'][k]['criteria']} for k in keys}
    prepared=prepare_prompts(tokenizer,json.dumps(data['state']),schema,2048)
    for mode in ('independent','shared'):
        trials=[]
        for repeat in range(10):
            scores=scorer.independent(prepared) if mode=='independent' else scorer.shared(prepared,8,optimized=True)
            trials.append([s.cpu().tolist() for s in scores])
        values=torch.tensor(trials)
        row={'offset':offset,'mode':mode,'repeats':10,
             'max_logit_delta_across_repeats':float((values-values[:1]).abs().max()),
             'changed_decisions_vs_first':int((values.argmax(-1)!=values[:1].argmax(-1)).sum()),
             'predictions':values.argmax(-1).tolist(),'logits':trials}
        report.append(row);Path(args.output).write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('logits','predictions')}),flush=True)
print('STABILITY_COMPLETE',flush=True)
