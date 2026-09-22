"""Evaluate the frozen 324-row holdout under the selected inference kernels."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from nimble.training.schema_data import prepare_data
from shared_native import NativeScorer, load_native_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/model')
    p.add_argument('--adapter')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint-label', default='base')
    args = p.parse_args()
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    revision = 'local-config-sha256:' + hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest()
    _, heldout, audit = prepare_data('data', 'data/eval.jsonl', tokenizer, 2048, args.model, revision)
    model = load_native_model(args.model, args.adapter)
    scorer = NativeScorer(model)
    results = []
    with torch.inference_mode():
        for row in heldout:
            torch.cuda.synchronize(); start = time.perf_counter()
            hidden = scorer.base.model(input_ids=torch.tensor([row['input_ids']], device='cuda'),
                                       use_cache=False).last_hidden_state[0, -1]
            scores = scorer.project(hidden, row['candidate_ids'])
            if not torch.isfinite(scores).all().item():
                raise RuntimeError('Nonfinite candidate scores')
            pred = scores.argmax().item()
            results.append({'id': row['id'], 'prediction': pred, 'label': row['labels'],
                            'correct': pred == row['labels'], 'logits': scores.tolist(),
                            'ms': (time.perf_counter()-start)*1000})
            if len(results)%50 == 0:
                print(json.dumps({'evaluated': len(results)}),flush=True)
    report = {'correct': sum(r['correct'] for r in results), 'total': len(results),
              'autocast': False, 'adapter': bool(args.adapter), 'audit': audit, 'rows': results,
              'checkpoint_label': args.checkpoint_label}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('audit','rows')}),flush=True)


if __name__ == '__main__':
    main()
