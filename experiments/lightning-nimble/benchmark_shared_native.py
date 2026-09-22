"""Compare identical-token independent and shared Lightning execution.

This is an execution-equivalence/latency diagnostic, not a new accuracy benchmark.
Reports candidate-conditional probabilities, not calibrated confidence.
"""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from nimble.scoring.parallel_schema import prepare_prompts
from shared_native import NativeScorer, load_native_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/model')
    p.add_argument('--output', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--optimized', action='store_true')
    p.add_argument('--counts', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--first-group-only', action='store_true')
    p.add_argument('--adapter')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--checkpoint-label', default='base')
    p.add_argument('--batched-comparison', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_native_model(args.model, args.adapter)
    scorer = NativeScorer(model)
    dataset = json.loads(Path(args.cases).read_text())
    if any(n < 1 or n > len(dataset['questions']) for n in args.counts):
        raise ValueError('Question count must fit this fixture; use benchmark_100_native.py for 100 questions')
    report = {'description': __doc__, 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'optimized': args.optimized,
              'kernel_overlay': bool(os.environ.get('PYTHONPATH')),
              'adapter': bool(args.adapter), 'runs': []}
    report['checkpoint_label'] = args.checkpoint_label
    def save():
        Path(args.output).write_text(json.dumps(report, indent=2))
    for count in args.counts:
        for offset in ([0] if args.first_group_only else range(0, len(dataset['questions']), count)):
            keys = list(dataset['questions'])[offset:offset+count]
            schema = {k: {'type': 'enum', 'description': dataset['questions'][k]['instructions'],
                          'choices': list(dataset['questions'][k]['criteria']),
                          'choice_descriptions': dataset['questions'][k]['criteria']} for k in keys}
            prepared = prepare_prompts(tokenizer, json.dumps(dataset['state']), schema, 2048)
            reference = None
            modes = ([('independent',1), ('independent_batched',8), ('shared',8)]
                     if args.batched_comparison else [('independent',1),('shared',1),('shared',4),('shared',8)])
            for mode, batch in modes:
                def operation():
                    if mode == 'independent': return scorer.independent(prepared)
                    if mode == 'independent_batched': return scorer.batched_independent(prepared,batch)
                    return scorer.shared(prepared,batch,optimized=args.optimized)
                operation()  # warm up this shape; not included in timings
                times = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(args.repeats):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    scores = operation()
                    torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
                scores = [s.cpu() for s in scores]
                if not all(torch.isfinite(s).all().item() for s in scores):
                    raise RuntimeError('Nonfinite candidate scores')
                if reference is None:
                    reference = scores
                deltas = [float((a-b).abs().max()) for a,b in zip(scores,reference)]
                predictions = [prepared.choices[i][s.argmax().item()] for i,s in enumerate(scores)]
                row = {'questions': len(keys), 'offset': offset, 'mode': mode, 'batch': batch,
                       'ms': times, 'prefix_tokens': len(prepared.prefix_ids),
                       'suffix_tokens': list(map(len,prepared.suffix_ids)),
                       'max_logit_delta': max(deltas),
                       'agreement': sum(a.argmax()==b.argmax() for a,b in zip(scores,reference)).item(),
                       'correct': sum(pred==dataset['cases'][int(k)]['expected'] for k,pred in zip(keys,predictions)),
                       'predictions': predictions, 'logits': [s.tolist() for s in scores],
                       'peak_gib': torch.cuda.max_memory_allocated()/2**30}
                report['runs'].append(row); save(); print(json.dumps(row),flush=True)
                if args.profile and count == 8 and offset == 0 and mode == 'shared' and batch == 8:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        operation()
                        torch.cuda.synchronize()
                    Path(args.output + '.profile.txt').write_text(
                        prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
    print('BENCHMARK_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
