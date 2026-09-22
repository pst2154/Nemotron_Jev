"""Compare matched native/vLLM outputs without hiding unmatched test cases."""
import argparse
import json
import math
from pathlib import Path


def distribution(answer):
    if answer['type'] == 'noul':
        return {'false': 1 - answer['noul'], 'true': answer['noul']}
    return answer['probabilities']


def compare(reference, candidate):
    def index(rows):
        result = {}
        for row in rows:
            key = (row['payload_sha256'], row.get('repeat', 0))
            if key in result:
                raise ValueError('Duplicate payload/repeat in results')
            result[key] = row
        return result

    left, right = index(reference), index(candidate)
    common = left.keys() & right.keys()
    maximum_delta = 0.0
    agreements = decisions = 0
    mismatches = []
    for key in sorted(common):
        baseline, optimized = left[key], right[key]
        if baseline['answers'].keys() != optimized['answers'].keys():
            raise ValueError('Question IDs differ for an identical payload')
        for question, original in baseline['answers'].items():
            updated = optimized['answers'][question]
            if original['type'] != updated['type']:
                raise ValueError('Answer types differ')
            p, q = distribution(original), distribution(updated)
            if p.keys() != q.keys():
                raise ValueError('Candidate options differ')
            for probabilities in (p, q):
                if (not probabilities or
                        not all(math.isfinite(v) and 0 <= v <= 1
                                for v in probabilities.values()) or
                        abs(sum(probabilities.values()) - 1) > 1e-5):
                    raise ValueError('Invalid candidate probability distribution')
            maximum_delta = max(maximum_delta, max(abs(p[k] - q[k]) for k in p))
            same = max(p, key=p.get) == max(q, key=q.get)
            agreements += same
            decisions += 1
            if not same:
                mismatches.append({'payload_sha256': key[0], 'question': question})
    return {'matched_requests': len(common), 'decisions': decisions,
            'argmax_agreements': agreements, 'max_probability_delta': maximum_delta,
            'missing_candidate_requests': len(left.keys() - right.keys()),
            'extra_candidate_requests': len(right.keys() - left.keys()),
            'mismatches': mismatches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--max-probability-delta', type=float, default=0.02)
    args = parser.parse_args()
    rows = [list(map(json.loads, Path(path).read_text().splitlines()))
            for path in (args.reference, args.candidate)]
    result = compare(*rows)
    print(json.dumps(result, indent=2))
    failed = (not result['decisions'] or result['mismatches'] or
              result['missing_candidate_requests'] or result['extra_candidate_requests'] or
              result['max_probability_delta'] > args.max_probability_delta)
    raise SystemExit(int(bool(failed)))


if __name__ == '__main__':
    main()
