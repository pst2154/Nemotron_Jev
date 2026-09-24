"""Classification-only System One API and the existing decision explorer."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from model_config import MODEL_NAME


def precision_settings(environ):
    quantization = environ.get('QUANTIZATION', 'none')
    if quantization not in ('none', 'fp8', 'fp8_per_channel', 'fp8_per_block'):
        raise ValueError('QUANTIZATION must be none, fp8, fp8_per_channel or fp8_per_block')
    candidate = environ.get('CANDIDATE_ONLY', '1')
    if candidate not in ('0', '1'):
        raise ValueError('CANDIDATE_ONLY must be 0 or 1')
    if quantization != 'none' and candidate == '1':
        raise ValueError('FP8 requires CANDIDATE_ONLY=0 with this model backend')
    return (None if quantization == 'none' else quantization), candidate == '1'


def main():
    quantization, candidate_only = precision_settings(os.environ)
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm_scoring import evaluate
    from scoring import POSITION_ORDERING, candidate_codes

    ordering_setting = os.environ.get('POSITION_ORDERING', '1')
    if ordering_setting not in ('0', '1'):
        raise ValueError('POSITION_ORDERING must be 0 or 1')
    position_ordering = ordering_setting == '1'

    checkpoint = os.environ.get('CHECKPOINT_DIR', '/models/checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True)
    additional = ({'decision_token_ids': [token for _, token in candidate_codes(tokenizer)]}
                  if candidate_only else {})
    engine = LLM(
        model=checkpoint, trust_remote_code=True, dtype='bfloat16',
        quantization=quantization,
        max_model_len=16386,
        max_logprobs=1052,
        logprobs_mode=('processed_logits' if os.environ.get('DIRECT_LOGITS', '1') == '1'
                       else 'processed_logprobs'),
        max_num_seqs=int(os.environ.get('MAX_NUM_SEQS', '128')),
        max_num_batched_tokens=int(os.environ.get('MAX_BATCHED_TOKENS', '8192')),
        gpu_memory_utilization=float(os.environ.get('GPU_MEMORY_UTILIZATION', '0.85')),
        enable_prefix_caching=True,
        enforce_eager=os.environ.get('ENFORCE_EAGER', '0') == '1',
        additional_config=additional,
    )
    # The offline engine is not thread-safe. Questions within a request are
    # scheduled together; cross-request batching requires an async engine.
    lock = threading.Lock()
    isolate_cache = os.environ.get('ISOLATE_REQUEST_CACHE', '0') == '1'

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split('?')[0] in ('/', '/explorer'):
                data = Path(__file__).with_name('explorer.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)
            elif self.path == '/health':
                self.send(200, {'ready': True,
                               'model': MODEL_NAME,
                               'backend': 'vllm', 'modes': ['systemone'],
                               'quantization': quantization,
                               'candidate_only_head': candidate_only,
                               'cache_policy': 'isolated_request' if isolate_cache else 'shared',
                               'position_ordering': POSITION_ORDERING if position_ordering else 'off',
                               'limits': {'max_questions': 512,
                                          'max_choices': len(candidate_codes(tokenizer)),
                                          'max_prompt_tokens': 16384}})
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/v1/systemone':
                return self.send(404, {'error': 'Only /v1/systemone is supported'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 2000000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                with lock:
                    if isolate_cache and not engine.reset_prefix_cache():
                        raise RuntimeError('Could not isolate the request prefix cache')
                    result = evaluate(
                        engine, tokenizer, body,
                        prime_shared_prefix=os.environ.get('PRIME_SHARED_PREFIX', '1') == '1',
                        batch_tokenize=os.environ.get('BATCH_TOKENIZE', '1') == '1',
                        position_ordering=position_ordering)
                self.send(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {'error': {'message': str(exc),
                                        'type': 'invalid_request_error'}})
            except Exception as exc:
                self.send(500, {'error': str(exc)})

    port = int(os.environ.get('PORT', '8770'))
    print(f'Serving classification API and UI on port {port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()


if __name__ == '__main__':
    main()
