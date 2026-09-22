"""Batched single-mask scoring; no autoregressive answer rollout.

Each input ends with the checkpoint's mask token. vLLM returns the requested
candidate logprobs at that slot. Its single sampled token is discarded and is
never fed back into the model. All questions retain the native prompt format.
"""

import math
import time

from scoring import candidate_codes, prepare_questions


def evaluate(engine, tokenizer, payload, mask_token_id=100, prime_shared_prefix=False,
             batch_tokenize=False):
    from vllm import SamplingParams

    started = time.perf_counter()
    prepared = prepare_questions(tokenizer, payload, batch_tokenize=batch_tokenize)
    codes = candidate_codes(tokenizer)
    prompts = []
    params = []
    for _, _, labels, _, ids in prepared:
        prompts.append({'prompt_token_ids': ids + [mask_token_id]})
        candidate_ids = [token for _, token in codes[:len(labels)]]
        # vLLM's indexed-logprob path supports up to 128 ids. With larger
        # choice sets, masked top-k covers every candidate without omission.
        logprob_args = ({'logprob_token_ids': candidate_ids}
                        if len(candidate_ids) <= 128 else
                        {'logprobs': len(candidate_ids),
                         'allowed_token_ids': candidate_ids})
        params.append(SamplingParams(
            max_tokens=1, temperature=0.0, detokenize=False,
            **logprob_args,
        ))
    prepared_at = time.perf_counter()
    # A single engine call lets the scheduler batch question suffixes and reuse
    # already-computed prefix blocks. No prompt_logprobs: it defeats cache reads.
    if prime_shared_prefix and len(prompts) > 1:
        # Make common prefix blocks available before scheduling the fan-out.
        # Merely enabling APC need not avoid repeated prefill in the first batch.
        outputs = engine.generate(prompts[:1], params[:1], use_tqdm=False)
        outputs += engine.generate(prompts[1:], params[1:], use_tqdm=False)
    else:
        outputs = engine.generate(prompts, params, use_tqdm=False)
    inferred_at = time.perf_counter()
    if len(outputs) != len(prepared):
        raise RuntimeError('Incomplete scoring batch')
    answers = {}
    for item, output in zip(prepared, outputs):
        key, kind, labels, descriptions, _ = item
        if (len(output.outputs) != 1 or not output.outputs[0].logprobs
                or len(output.outputs[0].logprobs) != 1):
            raise RuntimeError('Expected exactly one scored mask position')
        logprobs = output.outputs[0].logprobs[0]
        candidate_ids = [token for _, token in codes[:len(labels)]]
        if logprobs is None or any(token not in logprobs for token in candidate_ids):
            raise RuntimeError('Missing candidate scores')
        values = [logprobs[token].logprob for token in candidate_ids]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('Non-finite candidate logprobs')
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        denominator = sum(weights)
        probs = [weight / denominator for weight in weights]
        top = max(range(len(probs)), key=probs.__getitem__)
        if kind == 'noul':
            answer = {'type': kind, 'noul': probs[1]}
        else:
            answer = {'type': kind, 'probabilities': dict(zip(labels, probs)),
                      'confidence': probs[top]}
            if kind == 'choice':
                answer['choice'] = labels[top]
            else:
                answer['score'] = sum(i * p for i, p in enumerate(probs))
                answer['legend'] = dict(zip(labels, descriptions))
        answers[key] = answer
    return {
        'model': 'Nemotron-Labs-Diffusion-14B', 'answers': answers,
        'usage': {'input_tokens': sum(len(item[4]) for item in prepared),
                  'output_tokens': len(answers)},
        'metrics': {'seconds': time.perf_counter() - started,
                    'prepare_seconds': prepared_at - started,
                    'engine_seconds': inferred_at - prepared_at,
                    'questions': len(answers)},
        'scoring': {'method': 'diffusion_masked_token_candidate_softmax',
                    'backend': 'vllm', 'confidence': 'maximum_candidate_probability',
                    'calibrated': False},
    }
