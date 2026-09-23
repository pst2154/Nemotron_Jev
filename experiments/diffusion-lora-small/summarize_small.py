"""Summarize a completed small-recipe run without selecting on JevBench."""
import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    audit = json.loads((run/'audit.json').read_text())
    selected = json.loads((run/'selected.json').read_text())
    training = read(run/'training.jsonl')
    assert len(training) == audit['steps']
    assert [row['step'] for row in training] == list(range(1, audit['steps']+1))
    assert sum(row['examples_in_update'] for row in training) == audit['training_rows']
    assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['grad_norm']>0
               for row in training)
    selection = read(run/'validation-selection.jsonl')
    assert selected['step'] in {row['step'] for row in selection}
    winner = min(selection, key=lambda row:(-row['correct'],row['cross_entropy']))
    assert selected['step'] == winner['step']
    summary = {'selected_step':selected['step'],
               'training_examples':sum(row['examples_in_update'] for row in training),
               'training_seconds':sum(row['seconds'] for row in training),
               'peak_gib':max(row['peak_gib'] for row in training),
               'training_tokens':{'min':min(audit['training_tokens']),
                                  'median':statistics.median(audit['training_tokens']),
                                  'max':max(audit['training_tokens'])},'accuracy':{}}
    table = []
    public_results = {}
    for split in ('heldout','jevbench'):
        for mode in ('base','trained'):
            rows = read(run/f'{mode}-{split}.jsonl')
            expected = audit['heldout_rows'] if split == 'heldout' else audit['jevbench_rows']
            assert len(rows) == expected
            assert len({row['id'] for row in rows}) == expected
            if split == 'jevbench':
                assert all(row['strict_valid'] and not row['renormalized'] for row in rows)
                assert all(row['probs'] and all(math.isfinite(p) and 0<=p<=1 for p in row['probs'].values())
                           and abs(sum(row['probs'].values())-1)<1e-5 for row in rows)
                public_results[mode] = {row['id']:row for row in rows}
                assert len(public_results[mode]) == expected
            groups = {'all':rows}
            if split == 'heldout':
                public_prefixes = ('google/boolq:', 'google-research-datasets/paws:')
                groups.update({name:[row for row in rows if row['id'].startswith(prefix)]
                    for name,prefix in zip(('BoolQ','PAWS'),public_prefixes)})
                groups['synthetic'] = [row for row in rows if not row['id'].startswith(public_prefixes)]
                assert sum(len(group) for name,group in groups.items() if name!='all') == len(rows)
            for name, group in groups.items():
                correct = sum(row['correct'] for row in group)
                key = f'{mode}/{split}/{name}'
                summary['accuracy'][key] = {'correct':correct,'total':len(group)}
                assert group, (split, name)
                display_split = 'validation' if split == 'heldout' else 'public JevBench'
                display_mode = 'Base' if mode == 'base' else 'Selected LoRA'
                table.append(f'| {display_mode} | {display_split}/{name} | {correct}/{len(group)} | {100*correct/len(group):.2f}% |')
    assert public_results['base'].keys() == public_results['trained'].keys()
    summary['jevbench_transitions'] = {
        'wrong_to_correct':sum(not before['correct'] and public_results['trained'][key]['correct']
                               for key,before in public_results['base'].items()),
        'correct_to_wrong':sum(before['correct'] and not public_results['trained'][key]['correct']
                               for key,before in public_results['base'].items())}
    latency = read(run/'paired-latency.jsonl')
    assert len(latency) == 108
    assert all(sum(row['id']==item['id'] and row['repeat']==item['repeat'] and
                   row['adapter_enabled']!=item['adapter_enabled'] for item in latency)==1
               for row in latency)
    summary['warm_latency_median_ms'] = {
        str(enabled):statistics.median(row['latency_ms'] for row in latency if row['adapter_enabled']==enabled)
        for enabled in (False,True)}
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    before = summary['accuracy']['base/jevbench/all']
    after = summary['accuracy']['trained/jevbench/all']
    report = '\n'.join([
        '# Small diffusion decision-training result','',
        f"**Public JevBench accuracy: {before['correct']}/{before['total']} → "
        f"{after['correct']}/{after['total']}.** The adapter was selected using a separate validation set.",
        'See [the recipe](../RECIPE.md) for the training and data configuration.','',
        'Nemotron-Labs-Diffusion-14B; rank-16 attention/MLP LoRA; frozen output head.',
        'One training pass; mixed public-labeled and synthetic examples; shuffled positional codes.',
        'Training and validation use the serving-style two-pass graph. No service was updated.','',
        f"Selected update: **{selected['step']}** (validation only; zero means baseline).",
        'Validation was used to select the checkpoint, so its score is not an untouched test estimate.',
        'Public JevBench was evaluated only before and after training.','',
        '| Model | Split | Correct | Accuracy |','| --- | --- | --- | --- |',*table,'',
        f"Public JevBench changes: {summary['jevbench_transitions']['wrong_to_correct']} wrong→correct; "
        f"{summary['jevbench_transitions']['correct_to_wrong']} correct→wrong.",
        'These are public-set accuracy counts, not an official leaderboard composite score.','',
        f"Optimizer-loop time: {summary['training_seconds']/60:.1f} minutes; loading/evaluation excluded.",
        f"Peak allocated GPU memory: {summary['peak_gib']:.2f} GiB.",
        f"Prompt-token range: {summary['training_tokens']['min']}–{summary['training_tokens']['max']}.",
        f"Warm paired median latency: base {summary['warm_latency_median_ms']['False']:.1f} ms; "
        f"unmerged adapter {summary['warm_latency_median_ms']['True']:.1f} ms.",'',
        'Latency is native model-side timing on one H100 PCIe 80 GB, not HTTP latency or vLLM throughput.',
        'The paired test uses 18 questions, two warmups per mode/input, and three alternating-order measurements.',
        'The existing vLLM container has not been validated with this adapter.','',
        'This bundled experiment does not isolate which recipe change caused an accuracy difference.',
        'Exact source-group/state checks do not rule out original pretraining exposure to public datasets.',''])
    (run/'REPORT.md').write_text(report)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
