"""Build the frozen small recipe corpus from pinned labeled public sources."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalized(value):
    return ' '.join(value.casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--harness', required=True)
    args = parser.parse_args()
    from datasets import load_dataset
    sys.path.insert(0, args.harness)
    from jevbench.tasks import load_jsonl
    recipe = json.loads(Path(args.recipe).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = {split: [json.loads(line) for line in (Path(args.existing)/f'{split}.jsonl').read_text().splitlines()]
            for split in ('train', 'eval')}
    benchmark = [task for tier in ('easy','original','hard') for task in
                 load_jsonl(str(Path(args.harness)/'datasets/public'/f'{tier}.jsonl'))]
    blocked = {digest(task.state) for task in benchmark}
    seen_states = {digest(row['input']['state']) for values in rows.values() for row in values}
    sources = {}
    for repository, revision in recipe['public_sources'].items():
        is_bool = repository == 'google/boolq'
        config = {} if is_bool else {'name': 'labeled_final'}
        dataset = load_dataset(repository, revision=revision, cache_dir=args.cache, **config)
        groups = set()
        counts = Counter()
        # Validation is allocated first. Shared source text is then excluded from training.
        for source_split, split, wanted in [('validation','eval',128), ('train','train',710)]:
            order = list(range(len(dataset[source_split])))
            random.Random(recipe['seed']).shuffle(order)
            for index in order:
                source = dataset[source_split][index]
                if is_bool:
                    state = source['passage']
                    group = digest(normalized(state))
                    question = {'type':'noul', 'instructions':
                        'Using only the supplied passage, answer this question: '+source['question']}
                    target = bool(source['answer'])
                else:
                    state = {'first':source['sentence1'], 'second':source['sentence2']}
                    group = digest(sorted(normalized(v) for v in state.values()))
                    question = {'type':'choice', 'instructions':
                        'Do the two sentences express the same meaning, including who did what to whom?',
                        'criteria': {'equivalent':'Both sentences express the same meaning.',
                                     'different':'The sentences differ in meaning.'}}
                    target = 'equivalent' if source['label'] == 1 else 'different'
                state_hash = digest(state)
                if group in groups or state_hash in seen_states or state_hash in blocked:
                    continue
                groups.add(group)
                seen_states.add(state_hash)
                row = {'id':f'{repository}:{source_split}:{index}',
                       'input':{'state':state,'questions':{'decision':question}},
                       'reference':{'target':target,'source':'public_dataset_label'},
                       'provenance':{'source_id':f'{repository}:{group}', 'repository':repository,
                                     'revision':revision,'source_split':source_split}}
                rows[split].append(row)
                counts[split] += 1
                if counts[split] == wanted:
                    break
            assert counts[split] == wanted, (repository, split, counts)
        sources[repository] = {'revision':revision, 'counts':dict(counts)}
    train_states = {digest(row['input']['state']) for row in rows['train']}
    eval_states = {digest(row['input']['state']) for row in rows['eval']}
    train_groups = {row['provenance']['source_id'] for row in rows['train']}
    eval_groups = {row['provenance']['source_id'] for row in rows['eval']}
    assert not train_states & eval_states
    assert not train_states & blocked
    assert not train_groups & eval_groups
    assert len(rows['train']) == recipe['training_rows']
    manifest = {'sources':sources,'counts':{key:len(value) for key,value in rows.items()},
                'exact_state_overlap':0,'source_group_overlap':0,'sha256':{}}
    for split, values in rows.items():
        path = out/f'{split}.jsonl'
        with path.open('x') as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        manifest['sha256'][split] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__ == '__main__':
    main()
