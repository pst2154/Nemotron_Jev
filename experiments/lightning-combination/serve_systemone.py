"""Supervise vLLM and the immutable System One adapter in one stock container."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def adapter_environment():
    return {**os.environ,
            'UPSTREAM_BASE_URL': 'http://127.0.0.1:'+os.environ.get('SERVER_PORT', '8300')+'/v1',
            'UPSTREAM_MODEL': 'trained', 'UPSTREAM_CONCURRENCY': '100',
            'STATE_FIRST': '1', 'PERSISTENT_UPSTREAM': '1',
            'PRIME_SHARED_PREFIX': '0', 'ISOLATE_REQUEST_CACHE': '1',
            'HOST': os.environ.get('SYSTEMONE_HOST', '127.0.0.1'),
            'PORT': os.environ.get('SYSTEMONE_PORT', '8790')}


def await_model(process, url, timeout=1800):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('vLLM exited before readiness')
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError('vLLM did not become ready before startup timeout')


def main():
    children = []
    def stop(signum, frame):
        raise SystemExit(128+signum)
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, stop)
    try:
        model = subprocess.Popen(['vllm', 'serve', *sys.argv[1:]])
        children.append(model)
        env = adapter_environment()
        await_model(model, env['UPSTREAM_BASE_URL'].removesuffix('/v1')+'/health')
        api = subprocess.Popen([sys.executable, str(Path(__file__).with_name('adapter.py'))], env=env)
        children.append(api)
        print('Model ready; starting System One API with trained LoRA', flush=True)
        while True:
            for process in children:
                if process.poll() is not None:
                    raise RuntimeError('A serving process exited; stopping the container')
            time.sleep(1)
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
        for process in reversed(children):
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


if __name__ == '__main__':
    main()
