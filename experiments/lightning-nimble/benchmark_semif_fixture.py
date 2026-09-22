"""SemIf's owned fixtures, adapted to the unchanged Nimble prompt template.

This is NOT the upstream SemIf prompt or its published benchmark reproduction.
No labels/rationales/provenance enter prompts. All input rows remain in denominator.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from nimble.scoring.parallel_schema import prepare_prompts
from shared_native import NativeScorer, align_prompts, load_native_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/model')
    p.add_argument('--adapter')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--align-prefix', type=int)
    p.add_argument('--checkpoint-label', default='base')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError('Output must be new')
    source = Path(args.input).read_bytes()
    rows = [json.loads(line) for line in source.decode().splitlines() if line.strip()]
    assert len({r['id'] for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['group_id'], json.dumps(row['state'],sort_keys=True))].append(row)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_native_model(args.model,args.adapter)
    scorer = NativeScorer(model)
    report = {'description': __doc__, 'input_sha256': hashlib.sha256(source).hexdigest(),
              'source_revision': '1f2dea3e25379f9dfc98cb83c324f00ab5deda37',
              'total_rows': len(rows), 'groups': len(groups), 'max_input_tokens': args.max_tokens,
              'adapter': bool(args.adapter), 'runs': [], 'errors': [],
              'prefix_alignment': args.align_prefix,
              'checkpoint_label': args.checkpoint_label,
              'timing_scope': 'Loaded model, first-group warmup; forwards and selected-head readout; excludes tokenization/HTTP. Later novel shapes may still initialize kernels.'}
    warmed = False
    for group_number, group in enumerate(groups.values()):
        schema = {f'q{i}': {'type':'enum','description':r['question'],
                  'choices':[o['id'] for o in r['options']],
                  'choice_descriptions':{o['id']:o['description'] for o in r['options']}}
                  for i,r in enumerate(group)}
        state = group[0]['state']
        try:
            prepared = prepare_prompts(tokenizer,state if isinstance(state,str) else json.dumps(state),
                                       schema,args.max_tokens)
            if args.align_prefix:
                prepared = align_prompts(prepared,args.align_prefix)
        except ValueError as error:
            report['errors'].append({'ids':[r['id'] for r in group],'error':str(error)})
            out.write_text(json.dumps(report,indent=2));continue
        if not warmed:
            scorer.independent(prepared); scorer.shared(prepared,8,optimized=True);warmed=True
        reference = None
        for mode in ('independent','shared'):
            torch.cuda.synchronize();start=time.perf_counter()
            scores = scorer.independent(prepared) if mode=='independent' else scorer.shared(prepared,8,optimized=True)
            scores = [s.cpu() for s in scores]
            elapsed=(time.perf_counter()-start)*1000
            if not all(torch.isfinite(s).all().item() for s in scores):
                raise RuntimeError('Nonfinite candidate scores')
            if reference is None:reference=scores
            predictions = [{'id':r['id'],'prediction':s.argmax().item(),'label':r.get('label'),
                            'probabilities':s.softmax(-1).tolist(),
                            'correct':s.argmax().item()==r['label'] if 'label' in r else None,
                            'matches_independent':s.argmax().item()==ref.argmax().item(),
                            'max_logit_delta':float((s-ref).abs().max())}
                           for r,s,ref in zip(group,scores,reference)]
            record={'group':group_number,'mode':mode,'questions':len(group),'ms':elapsed,
                    'prefix_tokens':len(prepared.prefix_ids),'predictions':predictions}
            report['runs'].append(record);out.write_text(json.dumps(report,indent=2))
        print(json.dumps({'group':group_number+1,'groups':len(groups),
                          'questions':len(group),'shared_ms':elapsed,
                          'correct':sum(r['correct'] for r in predictions if r['correct'] is not None)
                                    if any(r['correct'] is not None for r in predictions) else None,
                          'agreement':sum(r['matches_independent'] for r in predictions)}),flush=True)
    print('FIXTURE_COMPLETE',flush=True)


if __name__=='__main__':main()
