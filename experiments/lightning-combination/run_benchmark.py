"""Run the immutable original fixture without writing into Experiment 1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HASHES = {
    'adapter.py': '1b5745bc21a88b6ae4ef1c4431674d693087ed91f9b4d68d804a682463bdc1d6',
    'comparison-inputs.json': '64da9e7a96f4820be7a0f452d4116229d2b840aa13d42543cdef0df06e0775a9',
    'compare_endpoints.py': 'bec8bacfad899382865116e1b239ba1d3ec2c43e567e9a9cec4ee85c8137fa46',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='System One endpoint including /v1/systemone')
    parser.add_argument('--backend', choices=['base', 'trained'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = Path(__file__).resolve().parent.parent / 'lightning-systemone'
    for name, expected in HASHES.items():
        if hashlib.sha256((original/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Original experiment changed: {name}')
    with tempfile.TemporaryDirectory(prefix='lightning-combination-') as temp:
        root = Path(temp)
        for name in HASHES:
            shutil.copy2(original/name, root/name)
        env = {**os.environ, 'BENCH_BACKEND': args.backend, 'BENCH_COUNTS': '1,100',
               'BENCH_URL': args.url, 'BENCH_SUFFIX': ''}
        subprocess.run([sys.executable, str(root/'compare_endpoints.py')], env=env, check=True)
        result = json.loads((root/f'comparison-{args.backend}.json').read_text())
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
        if any(s['successes'] != s['trials'] for s in result['scenarios']):
            raise RuntimeError('Some requests failed; partial results preserved, not a successful run')


if __name__ == '__main__':
    main()
